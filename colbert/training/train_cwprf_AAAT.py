from importlib import invalidate_caches
import os
import random
import time
import torch
import torch.nn as nn
import numpy as np
import pyterrier as pt
from transformers import AdamW

from torch.nn.functional import relu

from colbert.utils.runs import Run
from colbert.utils.amp import MixedPrecisionManager

from colbert.training.eager_batcher import SupervisedPRFEagerBatcherAAAT
from colbert.parameters import DEVICE

from colbert.modeling.colbert import ColBERT
from colbert.utils.utils import print_message
from colbert.training.utils import print_progress, manage_checkpoints


def train(args):
        
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if args.distributed:
        torch.cuda.manual_seed_all(args.random_seed)   
    if args.distributed:
        assert args.bsize % args.nranks == 0, (args.bsize, args.nranks)
        assert args.accumsteps == 1
        args.bsize = args.bsize // args.nranks

        print("Using args.bsize =", args.bsize, "(per process) and args.accumsteps =", args.accumsteps)
    
    


    
    print(">>>>>This is train in AAAT mode!!!")
    print(">>IBN training?",args.in_batch_negs)
    print(">>PRFs from E2E?",args.e2e)
    print(">>> regu lambda", args.reg_lambda)
    reader = SupervisedPRFEagerBatcherAAAT(args, (0 if args.rank == -1 else args.rank), args.nranks, num_prf=args.num_prf)
    
    # # load stopwords tokenids list
    # import os
    # if not os.path.exists("stopword-list.txt"):
    #     wget "https://raw.githubusercontent.com/terrier-org/terrier-core/5.x/modules/core/src/main/resources/stopword-list.txt"

    stops=[]
    with open("stopword-list.txt") as f:
        for l in f:
            stops.append(l.strip())
    
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    stop_ids = [x for x in tokenizer.convert_tokens_to_ids(stops) if x != 100]



    # load the document encoder from the ColBERT model
    colbert = ColBERT.from_pretrained('bert-base-uncased',
                                      query_maxlen=args.query_maxlen,
                                      doc_maxlen=args.doc_maxlen,
                                      dim=args.dim,
                                      similarity_metric=args.similarity,
                                      mask_punctuation=args.mask_punctuation)

    assert args.checkpoint is not None
    assert args.resume_optimizer is False, "TODO: This would mean reload optimizer too."
    print_message(f"#> Starting from checkpoint {args.checkpoint} -- but NOT the optimizer!")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    colbert.load_state_dict(checkpoint['model_state_dict'], strict=False) # load a checkpoint for colbertD
    
    from ..modeling.cwprf import CWPRFEncoder
    if args.checkpoint_init:
        cwprf = CWPRFEncoder.from_pretrained("castorini/unicoil-msmarco-passage")
        print_message("Training with initlization!!")
        
    else:
        print_message("Training from scratch!!")
        cwprf = CWPRFEncoder.from_pretrained('bert-base-uncased')
 

    cwprf = cwprf.to(DEVICE)

    if args.rank == 0:
        torch.distributed.barrier()

    colbert = colbert.to(DEVICE)
    cwprf.train()

    optimizer = AdamW(filter(lambda p: p.requires_grad, cwprf.parameters()), lr=args.lr, eps=1e-8)
    optimizer.zero_grad()

    amp = MixedPrecisionManager(args.amp)
    criterion = nn.MSELoss()
    
    start_time = time.time()
    train_loss = 0.0
    start_batch_idx = 0

    if args.resume:
        assert args.checkpoint is not None
        start_batch_idx = checkpoint['batch']
        reader.skip_to_batch(start_batch_idx, checkpoint['arguments']['bsize'])

#     if args.resume_unicoil:
#         print(f">>>>>>Initialisation parameters from the unicoil checkpoint:{args.resume_unicoil_path}")
#         unicoil_checkpoint = torch.load(args.resume_unicoil_path, map_location = 'cpu')
#         sprf.load_state_dict(unicoil_checkpoint['model_state_dict'])
#         start_batch_idx = unicoil_checkpoint['batch']
#         reader.skip_to_batch(start_batch_idx, unicoil_checkpoint['arguments']['bsize'])

    print(">>>start_batch_idx:",start_batch_idx)
    for batch_idx, BatchSteps in zip(range(start_batch_idx, args.maxsteps), reader):
        this_batch_loss = 0.0
        for queries, passages, prfpassages in BatchSteps: 
            #queries is 2x for each "query"
            #prfpassages is 3x for each "query"
            with amp.context():

                # get the embeddings of PRF and positive and negative passages
                # we dont backprop to change the embeddings
                with torch.no_grad():
                    prf_embs = colbert.doc(*prfpassages) #[2*bsize, PRFlen, Dim]
                    D_embs = colbert.doc(*passages)  #[2*bsize, Dlen, Dim]
                
                # separate positive and negative for easier torch ops
               
                D_embs_pos = D_embs[0:args.bsize,:,:] #[bsize, Dlen, Dim]
                D_embs_neg = D_embs[args.bsize:,:,:] #[bsize, Dlen, Dim]  
                prf_embs = prf_embs[:args.bsize,:,:] #[bsize, PRFlen, Dim]
               
                # examine the max-sims of the PRF embeddings vs the positive and negative passages
                if args.in_batch_negs:
                    prf_scores_pos = (prf_embs @ D_embs_pos.permute(0, 2, 1)).max(2).values
                    prf_scores_neg = torch.einsum('qie,dje->qdij', prf_embs, D_embs_neg).max(3).values
                    target_scores = prf_scores_pos - prf_scores_neg.max(0).values
                   

                else:
                    prf_scores_pos = (prf_embs @ D_embs_pos.permute(0, 2, 1)).max(2).values
                    prf_scores_neg = (prf_embs @ D_embs_neg.permute(0, 2, 1)).max(2).values
                    target_scores = prf_scores_pos - prf_scores_neg # shape is (batchsize * prf_passages, prf_passage length)
                
                queries_pos = ( queries[0][0:args.bsize,:], queries[1][0:args.bsize,:] ) #queries is 2x for each "query"
                prf_pos = ( prfpassages[0][0:args.bsize,:], prfpassages[1][0:args.bsize,:] )

                
                prf_ids = torch.cat([queries_pos[0], prf_pos[0]],dim=1)
                prf_mask = torch.cat([queries_pos[1], prf_pos[1]],dim=1)


                predict_weights = sprf(input_ids=prf_ids.to(DEVICE), attention_mask=prf_mask.to(DEVICE))# batch_size, prf_toks,1]
                # dont count loss on the usefulness of the original query embeddings, so start at 32nd embedding
                
                weights =predict_weights[:,32::].squeeze() 
                non_zero = torch.sum(weights!= 0).item()
#                 /torch.numel(weights)
#                 print_message(f"nbr non-zero: {non_zero}")
                
                mse_loss = criterion(predict_weights[:,32::].squeeze(), relu(target_scores))
#                 l1_lambda = 0.0005
#                 l1_lambda = 0.005
#                 l1_lambda = 0.008
#                 l1_lambda = 0.01
#                 l1_penalty = args.reg_lambda*(torch.sum(predict_weights[:,32::].squeeze(),dim=-1).mean())
                
#                 loss = mse_loss + l1_penalty
                loss = mse_loss
#                 print_message(f"MSE:{mse_loss} L1 Regu:{l1_penalty}")


                
                loss = loss / args.accumsteps

            amp.backward(loss)
            train_loss += loss.item()
            this_batch_loss += loss.item()

        amp.step(sprf, optimizer)

        if args.rank < 1:
            avg_loss = train_loss / (batch_idx+1)

            num_examples_seen = (batch_idx - start_batch_idx) * args.bsize * args.nranks
            elapsed = float(time.time() - start_time)

            log_to_mlflow = (batch_idx % 20 == 0)
            Run.log_metric('train/avg_loss_CWPRF', avg_loss, step=batch_idx, log_to_mlflow=log_to_mlflow)
            Run.log_metric('train/batch_loss_CWPRF', this_batch_loss, step=batch_idx, log_to_mlflow=log_to_mlflow)
            Run.log_metric('train/examples_CWPRF', num_examples_seen, step=batch_idx, log_to_mlflow=log_to_mlflow)
            Run.log_metric('train/throughput_CWPRF', num_examples_seen / elapsed, step=batch_idx, log_to_mlflow=log_to_mlflow)
            if args.in_batch_negs:
                print_message(f"batch_idx: {batch_idx} avg_loss_IBN: {avg_loss}")
            else:
                print_message(f"batch_idx: {batch_idx} avg_loss: {avg_loss}")
            # print("--------------------------------------------")
            manage_checkpoints(args, sprf, optimizer, batch_idx+1)
 
