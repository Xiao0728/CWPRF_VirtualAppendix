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

from colbert.training.eager_batcher import SupervisedPRFEagerBatcherOAAT
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

    ratio=0.5
    
    # load stopwords tokenids as a lst
    stops=[]
    with open("stopword-list.txt") as f:
        for l in f:
            stops.append(l.strip())
    
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    stop_ids = [x for x in tokenizer.convert_tokens_to_ids(stops) if x != 100]

    print(">>IBN training?",args.in_batch_negs)
    print(">>PRFs from E2E?",args.e2e)
    reader = SupervisedPRFEagerBatcherOAAT(args, (0 if args.rank == -1 else args.rank), args.nranks, args.num_prf)
    

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


    for batch_idx, BatchSteps in zip(range(start_batch_idx, args.maxsteps), reader):
        this_batch_loss = 0.0
        for queries, passages, prf1,prf2,prf3 in BatchSteps: # we get ids and mask for: Q, Doc, PRF1, PRF2, PRF3
            #queries is 2x bsizefor each "query"
            #same with each query, each prf is 2x bsize
            with amp.context():

                # get the embeddings of PRF and positive and negative passages
                # we dont backprop to change the embeddings
                with torch.no_grad():
                    prf1_embs = colbert.doc(*prf1) #[bsize, PRF1_len, Dim]
                    prf2_embs = colbert.doc(*prf2) #[bsize, PRF2_len, Dim]
                    prf3_embs = colbert.doc(*prf3) #[bsize, PRF3_len, Dim]
                    D_embs = colbert.doc(*passages) #[2xbsize, Dlen, Dim]
                
                # separate positive and negative for easier torch ops
                D_embs_pos = D_embs[0::2,:,:] #[bsize, Dlen, Dim]
                D_embs_neg = D_embs[1::2,:,:]#[bsize, Dlen, Dim]]
#                 print("====>prf_passages", prfpassages[0].size)
#                 print("prf_embs", prf_embs.shape)
#                 print("D_embs_pos", D_embs_pos.shape)
#                 print("D_embs_pos.permute", D_embs_pos.permute(0, 2, 1).shape)
                

                # examine the max-sims of each PRF embeddings vs the positive and negative passages
                if args.in_batch_negs:
                    prf1_scores_pos = (prf1_embs[0::2,:,:] @ D_embs_pos.permute(0, 2, 1)).max(2).values
                    prf1_scores_neg = torch.einsum('qie,dje->qdij',prf1_embs[0::2,:,:],  D_embs_neg).max(3).values #[Qbatch, Dbatch, Qlen (MaxSim score of this qtok)]
                    prf2_scores_pos = (prf2_embs[0::2,:,:] @ D_embs_pos.permute(0, 2, 1)).max(2).values
                    prf2_scores_neg = torch.einsum('qie,dje->qdij',prf2_embs[0::2,:,:],  D_embs_neg).max(3).values #[Qbatch, Dbatch, Qlen (MaxSim score of this qtok)]
                    prf3_scores_pos = (prf3_embs[0::2,:,:] @ D_embs_pos.permute(0, 2, 1)).max(2).values
                    prf3_scores_neg = torch.einsum('qie,dje->qdij',prf3_embs[0::2,:,:],  D_embs_neg).max(3).values #[Qbatch, Dbatch, Qlen (MaxSim score of this qtok)]


                    prf1_target_scores = prf1_scores_pos - prf1_scores_neg.max(0).values
                    prf2_target_scores = prf2_scores_pos - prf2_scores_neg.max(0).values 
                    prf3_target_scores = prf3_scores_pos - prf3_scores_neg.max(0).values

                else:
                    prf1_scores_pos = (prf1_embs[0::2,:,:] @ D_embs_pos.permute(0, 2, 1)).max(2).values
                    prf1_scores_neg = (prf1_embs[0::2,:,:] @ D_embs_neg.permute(0, 2, 1)).max(2).values
                    
                    prf2_scores_pos = (prf2_embs[0::2,:,:] @ D_embs_pos.permute(0, 2, 1)).max(2).values
                    prf2_scores_neg = (prf2_embs[0::2,:,:] @ D_embs_neg.permute(0, 2, 1)).max(2).values   
                    
                    prf3_scores_pos = (prf3_embs[0::2,:,:] @ D_embs_pos.permute(0, 2, 1)).max(2).values
                    prf3_scores_neg = (prf3_embs[0::2,:,:] @ D_embs_neg.permute(0, 2, 1)).max(2).values                
                    # compute how much each embedding helped bring positive above negative
                    # this is our "label" for each token for SPRF
                    prf1_target_scores = prf1_scores_pos - prf1_scores_neg #[batchsize * prf_passages, prf_passage_len)
                    prf2_target_scores = prf2_scores_pos - prf2_scores_neg
                    prf3_target_scores = prf3_scores_pos - prf3_scores_neg

                # drop the (duplicated) queries corresponding to the negative passages
                queries_pos = ( queries[0][0::2,:], queries[1][0::2,:] )
                prf1_pos = ( prf1[0][0::2,:], prf1[1][0::2,:] )
                prf2_pos = ( prf2[0][0::2,:], prf2[1][0::2,:] )
                prf3_pos = ( prf3[0][0::2,:], prf3[1][0::2,:] )

               

                # concatenate each query with the each prf passage 
                prf1_ids = torch.cat([queries_pos[0], prf1_pos[0]],dim=1)
                prf1_mask = torch.cat([queries_pos[1], prf1_pos[1]],dim=1)
            
                prf2_ids = torch.cat([queries_pos[0], prf2_pos[0]],dim=1)
                prf2_mask = torch.cat([queries_pos[1], prf2_pos[1]],dim=1)
                
                prf3_ids = torch.cat([queries_pos[0], prf3_pos[0]],dim=1)
                prf3_mask = torch.cat([queries_pos[1], prf3_pos[1]],dim=1)
#                 print("prf_ids.shape",prf_ids.shape)1

                # push the prf query formulation through the SPRF model, like [CLS] [Q] qt qt [MASK] [SEP] prf_t prf_t
                predict_weights_prf1 = sprf(input_ids=prf1_ids.to(DEVICE), attention_mask=prf1_mask.to(DEVICE))
                predict_weights_prf2 = sprf(input_ids=prf2_ids.to(DEVICE), attention_mask=prf2_mask.to(DEVICE))
                predict_weights_prf3 = sprf(input_ids=prf3_ids.to(DEVICE), attention_mask=prf3_mask.to(DEVICE))
#                 print("uc_weights:",uc_weights.shape)
#                 print("prf_scores_diff:",prf_scores_diff.shape)
                # dont count loss on the usefulness of the original query embeddings, so
                # start at 32nd embedding
             


                loss_prf1 = criterion(predict_weights_prf1[:,32::].squeeze(), relu(prf1_target_scores))
                loss_prf2 = criterion(predict_weights_prf2[:,32::].squeeze(), relu(prf2_target_scores))
                loss_prf3 = criterion(predict_weights_prf3[:,32::].squeeze(), relu(prf3_target_scores))
                
                loss = loss_prf1 + loss_prf2 + loss_prf3
                loss = loss / args.accumsteps

            #if args.rank < 1:
            #    print_progress(scores)

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

            manage_checkpoints(args, sprf, optimizer, batch_idx+1)
 
