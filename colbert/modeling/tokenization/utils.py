import torch


def tensorize_triples(query_tokenizer, doc_tokenizer, queries, positives, negatives, bsize):
    assert len(queries) == len(positives) == len(negatives)
    assert bsize is None or len(queries) % bsize == 0

    N = len(queries)
    Q_ids, Q_mask = query_tokenizer.tensorize(queries)
    D_ids, D_mask = doc_tokenizer.tensorize(positives + negatives)
    D_ids, D_mask = D_ids.view(2, N, -1), D_mask.view(2, N, -1)

    # Compute max among {length of i^th positive, length of i^th negative} for i \in N
    maxlens = D_mask.sum(-1).max(0).values

    # Sort by maxlens
    indices = maxlens.sort().indices
    Q_ids, Q_mask = Q_ids[indices], Q_mask[indices]
    D_ids, D_mask = D_ids[:, indices], D_mask[:, indices]

    (positive_ids, negative_ids), (positive_mask, negative_mask) = D_ids, D_mask

    query_batches = _split_into_batches(Q_ids, Q_mask, bsize)
    positive_batches = _split_into_batches(positive_ids, positive_mask, bsize)
    negative_batches = _split_into_batches(negative_ids, negative_mask, bsize)

    batches = []
    for (q_ids, q_mask), (p_ids, p_mask), (n_ids, n_mask) in zip(query_batches, positive_batches, negative_batches):
        Q = (torch.cat((q_ids, q_ids)), torch.cat((q_mask, q_mask)))
        D = (torch.cat((p_ids, n_ids)), torch.cat((p_mask, n_mask)))
        batches.append((Q, D))

    return batches


def tensorize_prftriples_OAAT(query_tokenizer, doc_tokenizer, prf_tokenizer, queries, prfpassages, positives, negatives, bsize,num_prf):
    assert len(queries) == len(positives) == len(negatives) ==len(prfpassages)
    assert bsize is None or len(queries) % bsize == 0
    
    
    #for handling prf psges seperately
    prf1,prf2,prf3 =[],[],[]
    for i in range(len(prfpassages)):
#         print(i)
    #     for text in prfpassages[0]:
        prf1.append(str(prfpassages[i][0]))
        prf2.append(str(prfpassages[i][1]))
        prf3.append(str(prfpassages[i][2]))

    N = len(queries)
    assert N==len(prf1)==len(prf2)==len(prf3)
    
    Q_ids, Q_mask = query_tokenizer.tensorize(queries)
    D_ids, D_mask = doc_tokenizer.tensorize(positives + negatives)
    D_ids, D_mask = D_ids.view(2, N, -1), D_mask.view(2, N, -1)
    # the input prf is individually, One-At-A-Time
    prf1_ids, prf1_mask = prf_tokenizer.tensorize(prf1)
    prf2_ids, prf2_mask = prf_tokenizer.tensorize(prf2)
    prf3_ids, prf3_mask = prf_tokenizer.tensorize(prf3)
    

    
#     print("--->>>>",prfpassages)
#     prf_ids, prf_mask = prf_tokenizer.tensorize(prfpassages) #the input prf should be prf1+prf2+prf3, like positives+negatives
#     prf_ids, prf_mask = prf_ids.view(num_prf,N,-1), prf_mask.view(num_prf,N,-1)
#     print("====>>>>>PRF", prf3_ids,"SHAPE:",prf3_ids.shape)
    # Compute max among {length of i^th positive, length of i^th negative} for i \in N
    maxlens = D_mask.sum(-1).max(0).values

    # Sort by maxlens
    indices = maxlens.sort().indices
    Q_ids, Q_mask = Q_ids[indices], Q_mask[indices]
    D_ids, D_mask = D_ids[:, indices], D_mask[:, indices]
    prf1_ids, prf1_mask = prf1_ids[indices], prf1_mask[indices]
    prf2_ids, prf2_mask = prf2_ids[indices], prf2_mask[indices]
    prf3_ids, prf3_mask = prf3_ids[indices], prf3_mask[indices]
#     print("====>>>>>PRF after indeces", prf_ids.shape)

    (positive_ids, negative_ids), (positive_mask, negative_mask) = D_ids, D_mask
#     (prf1_ids, prf2_ids, prf3_ids), (prf1_mask, prf2_mask, prf3_mask) = prf_ids, prf_mask
    
    

    query_batches = _split_into_batches(Q_ids, Q_mask, bsize)
    positive_batches = _split_into_batches(positive_ids, positive_mask, bsize)
    negative_batches = _split_into_batches(negative_ids, negative_mask, bsize)
    prf1_batches = _split_into_batches(prf1_ids, prf1_mask, bsize)
    prf2_batches = _split_into_batches(prf2_ids, prf2_mask, bsize)
    prf3_batches = _split_into_batches(prf3_ids, prf3_mask, bsize)
#     print(">>>>prf1 ids and masks",prf_ids.shape,prf_mask)
#     print("====>>>>>", prf_ids)
#     print("====>>>>>POS", positive_ids)

    batches = []
    for (q_ids, q_mask), (p_ids, p_mask), (n_ids, n_mask), (prf1_ids, prf1_mask), (prf2_ids, prf2_mask), (prf3_ids, prf3_mask) in zip(query_batches, positive_batches, negative_batches, prf1_batches, prf2_batches, prf3_batches):
        # how many Query ids should we append
        Q = (torch.cat((q_ids, q_ids)), torch.cat((q_mask, q_mask)))
        D = (torch.cat((p_ids, n_ids)), torch.cat((p_mask, n_mask)))
        
        #similar with Query, we need 2xbatch_size
        PRF1 = (torch.cat((prf1_ids,prf1_ids)), torch.cat((prf1_mask, prf1_mask)))
        PRF2 = (torch.cat((prf2_ids,prf2_ids)), torch.cat((prf2_mask, prf2_mask)))
        PRF3 = (torch.cat((prf3_ids,prf3_ids)), torch.cat((prf3_mask, prf3_mask)))
#         PRF = (prf_ids, prf_mask)
                   
        batches.append((Q, D, PRF1, PRF2, PRF3))

    return batches


def tensorize_prftriples_AAAT(query_tokenizer, doc_tokenizer, prf_tokenizer, queries, prfpassages, positives, negatives, bsize,num_prf):
    assert len(queries) == len(positives) == len(negatives) ==len(prfpassages)
    assert bsize is None or len(queries) % bsize == 0
    

    N = len(queries)
    Q_ids, Q_mask = query_tokenizer.tensorize(queries)
    D_ids, D_mask = doc_tokenizer.tensorize(positives + negatives)
    D_ids, D_mask = D_ids.view(2, N, -1), D_mask.view(2, N, -1)
    
#     print("--->>>>",prfpassages)
    prf_ids, prf_mask = prf_tokenizer.tensorize(prfpassages) #the input prf should be prf1+prf2+prf3, like positives+negatives
#     prf_ids, prf_mask = prf_ids.view(num_prf,N,-1), prf_mask.view(num_prf,N,-1)
#     print("====>>>>>PRF", prf_ids,"SHAPE:",prf_ids.shape)
    # Compute max among {length of i^th positive, length of i^th negative} for i \in N
    maxlens = D_mask.sum(-1).max(0).values

    # Sort by maxlens
    indices = maxlens.sort().indices
    Q_ids, Q_mask = Q_ids[indices], Q_mask[indices]
    D_ids, D_mask = D_ids[:, indices], D_mask[:, indices]
    prf_ids, prf_mask = prf_ids[indices], prf_mask[indices]
#     print("====>>>>>PRF after indeces", prf_ids.shape)

    (positive_ids, negative_ids), (positive_mask, negative_mask) = D_ids, D_mask
#     (prf1_ids, prf2_ids, prf3_ids), (prf1_mask, prf2_mask, prf3_mask) = prf_ids, prf_mask
    
    

    query_batches = _split_into_batches(Q_ids, Q_mask, bsize)
    positive_batches = _split_into_batches(positive_ids, positive_mask, bsize)
    negative_batches = _split_into_batches(negative_ids, negative_mask, bsize)
    prf_batches = _split_into_batches(prf_ids, prf_mask, bsize)
#     prf2_batches = _split_into_batches(prf1_ids, prf2_mask, bsize)
#     prf3_batches = _split_into_batches(prf1_ids, prf3_mask, bsize)
#     print(">>>>prf1 ids and masks",prf_ids.shape,prf_mask)
#     print("====>>>>>", prf_ids)
#     print("====>>>>>POS", positive_ids)

    batches = []
    for (q_ids, q_mask), (p_ids, p_mask), (n_ids, n_mask), (prf_ids, prf_mask) in zip(query_batches, positive_batches, negative_batches, prf_batches):
        # how many Query ids should we append
        Q = (torch.cat((q_ids, q_ids)), torch.cat((q_mask, q_mask)))
        D = (torch.cat((p_ids, n_ids)), torch.cat((p_mask, n_mask)))
        PRF = (torch.cat((prf_ids,prf_ids)), torch.cat((prf_mask, prf_mask)))
#         PRF = (prf_ids, prf_mask)
        
        
        batches.append((Q, D, PRF))

    return batches





def _sort_by_length(ids, mask, bsize):
    if ids.size(0) <= bsize:
        return ids, mask, torch.arange(ids.size(0))

    indices = mask.sum(-1).sort().indices
    reverse_indices = indices.sort().indices

    return ids[indices], mask[indices], reverse_indices


def _split_into_batches(ids, mask, bsize):
    batches = []
    for offset in range(0, ids.size(0), bsize):
        batches.append((ids[offset:offset+bsize], mask[offset:offset+bsize]))

    return batches
