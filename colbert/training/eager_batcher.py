import os
import ujson

from functools import partial
from colbert.utils.utils import print_message
from colbert.modeling.tokenization import QueryTokenizer, PRFQueryTokenizer, DocTokenizer, tensorize_triples, tensorize_prftriples_AAAT,tensorize_prftriples_OAAT
import os
# os.environ["JAVA_HOME"] = "/local/trmaster/opt/jdk-11.0.6/"
from colbert.utils.runs import Run

import pandas as pd

import re
def processQuery(query):
    query = re.sub(r"[^a-zA-Z0-9¿]+", " ", query)
    return query

def Q2(inputDF):
    inputDF['query']=inputDF['query'].apply(lambda query: processQuery(query))
    return inputDF


import pyterrier as pt
pt.init(boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"])
index_location = "/nfs/indices/passage_index/data.properties"
index = pt.IndexFactory.of(index_location)
# bm25 = pt.BatchRetrieve(index, meta=["docno", "text"])
bm25 = pt.BatchRetrieve(index, wmodel='BM25',metadata=["docno", "text"])
bm25_qe = bm25 >> pt.rewrite.RM3(index_location) >> bm25




class EagerBatcher():
    def __init__(self, args, rank=0, nranks=1):
        self.rank, self.nranks = rank, nranks
        self.bsize, self.accumsteps = args.bsize, args.accumsteps
        # self.query_tokenizer = QueryTokenizer(args.query_maxlen)
        self.query_tokenizer = PRFQueryTokenizer(args.query_maxlen, args.num_prf)
        self.doc_tokenizer = DocTokenizer(args.doc_maxlen)
        
        self.tensorize_triples = partial(tensorize_triples, self.query_tokenizer, self.doc_tokenizer)
        self.tensorize_prftriples_AAAT = partial(tensorize_prftriples_AAAT, self.query_tokenizer, self.doc_tokenizer)
        self.tensorize_prftriples_OAAT = partial(tensorize_prftriples_OAAT, self.query_tokenizer, self.doc_tokenizer)
        
        self.triples_path = args.triples
        self.num_prf = args.num_prf
        self._reset_triples()

    def _reset_triples(self):
        self.reader = open(self.triples_path, mode='r', encoding="utf-8")
        self.position = 0

    def __iter__(self):
        return self

    def __next__(self):
        # print("THIS IS THE ORITINAL BATCHER NEXT!")
        queries, positives, negatives = [], [], []

        for line_idx, line in zip(range(self.bsize * self.nranks), self.reader):
            if (self.position + line_idx) % self.nranks != self.rank:
                continue

            query, pos, neg = line.strip().split('\t')

            queries.append(query)
            positives.append(pos)
            negatives.append(neg)

        self.position += line_idx + 1

        if len(queries) < self.bsize:
            raise StopIteration

        return self.collate(queries, positives, negatives)

    def collate(self, queries, positives, negatives):
        assert len(queries) == len(positives) == len(negatives) == self.bsize

        return self.tensorize_triples(queries, positives, negatives, self.bsize // self.accumsteps)

    def skip_to_batch(self, batch_idx, intended_batch_size):
        self._reset_triples()

        Run.warn(f'Skipping to batch #{batch_idx} (with intended_batch_size = {intended_batch_size}) for training.')

        _ = [self.reader.readline() for _ in range(batch_idx * intended_batch_size)]

        return None



class SupervisedPRFEagerBatcherOAAT(EagerBatcher):
    def __init__(self, args, rank=0, nranks=1, num_prf=3):
        super().__init__(args, rank=rank, nranks=nranks)
        self.num_prf = num_prf
        self.prf_tokenizer = DocTokenizer(512 - args.query_maxlen)
        self.tensorize_prftriples_OAAT = partial(tensorize_prftriples_OAAT, self.query_tokenizer,self.doc_tokenizer, self.prf_tokenizer)
#         self.tensorize_prftriples = partial(tensorize_prftriples, self.query_tokenizer, self.doc_tokenizer, self.prf_tokenizer)
       
    
    def __next__(self):
        queries, prfpassages,prf1,prf2,prf3, positives, negatives = [], [], [], [], [],[],[]
     
        for line_idx, line in zip(range(self.bsize * self.nranks), self.reader):
            if (self.position + line_idx) % self.nranks != self.rank:
                continue

            query, pos, neg = line.strip().split('\t')
            positives.append(pos)
            negatives.append(neg)
            queries.append(query)

            df_query1 = pd.DataFrame(data={'qid': [1], 'query': [query]})
            df_query = Q2(df_query1)
#             res = (e2e_pipe%10)(df_query)
#             res = (bm25_qe%10)(df_query)
            res = (bm25%10)(df_query)
            res = res.sort_values('rank', ascending=True)
            
            if res.empty:
                print('RES DataFrame is empty!')
                prfpassages.append([".",".","."])
                continue
            else:
                prfpassages.append(res.head(3).text.tolist()) #

    
                
        self.position += line_idx + 1

        if len(queries) < self.bsize:
            raise StopIteration
        
        return self.collate(queries, prfpassages, positives, negatives)
#         return self.collate(prfqueries, positives, negatives)
    
    # here we split the query and prf using [SEP]
    def collate(self, queries, prfpassages, positives, negatives):
        assert len(queries) == len(positives) == len(negatives) == self.bsize == len(prfpassages)
        rtr = self.tensorize_prftriples_OAAT(queries, prfpassages, positives, negatives, self.bsize // self.accumsteps, self.num_prf)
        return rtr


####################3

  
class SupervisedPRFEagerBatcherAAAT(EagerBatcher):
    def __init__(self, args, rank=0, nranks=1, num_prf=3,e2e=False):
        super().__init__(args, rank=rank, nranks=nranks)
        self.num_prf = num_prf
        self.prf_tokenizer = DocTokenizer(512 - args.query_maxlen)
        self.tensorize_prftriples_AAAT = partial(tensorize_prftriples_AAAT, self.query_tokenizer, self.doc_tokenizer, self.prf_tokenizer)
        self.e2e = True
        
       
    
    def __next__(self):
        queries, prfpassages, positives, negatives = [], [], [], []
    
        for line_idx, line in zip(range(self.bsize * self.nranks), self.reader):
            if (self.position + line_idx) % self.nranks != self.rank:
                continue

            query, pos, neg = line.strip().split('\t')
            positives.append(pos)
            negatives.append(neg)
            queries.append(query)

            df_query1 = pd.DataFrame(data={'qid': [1], 'query': [query]})
            df_query = Q2(df_query1)
            
#             res = (bm25_qe%10)(df_query)
            res = (bm25%10)(df_query)
#             res = (e2e_pipe%10)(df_query)
            res = res.sort_values('rank', ascending=True)
            if res.empty:
                print('RES DataFrame is empty!')
                prfpassages.append(".")
                continue
            else:
                prfs = ' '.join([str(n) for n in res.head(3).text.values])
                prfpassages.append(str(prfs))

    
                
        self.position += line_idx + 1

        if len(queries) < self.bsize:
            raise StopIteration
        
        return self.collate(queries, prfpassages, positives, negatives)
#         return self.collate(prfqueries, positives, negatives)
    
    # here we split the query and prf using [SEP]
    def collate(self, queries, prfpassages, positives, negatives):
        assert len(queries) == len(positives) == len(negatives) == self.bsize == len(prfpassages)
        rtr = self.tensorize_prftriples_AAAT(queries, prfpassages, positives, negatives, self.bsize // self.accumsteps, self.num_prf)
        return rtr
    
        


class SupervisedPRFEagerBatcher(EagerBatcher):
    def __init__(self, args, rank=0, nranks=1, num_prf=3):
        super().__init__(args, rank=rank, nranks=nranks)
        self.num_prf = num_prf
       
    
    def __next__(self):
        queries, prfqueries, positives, negatives = [], [], [], [] 
    
#         e2e_res = pd.DataFrame(columns=["qid","query","docid","query_toks","query_embs","score","docno","rank","text"])
        for line_idx, line in zip(range(self.bsize * self.nranks), self.reader):
            if (self.position + line_idx) % self.nranks != self.rank:
                continue

            query, pos, neg = line.strip().split('\t')
            positives.append(pos)
            negatives.append(neg)

            
            df_query1 = pd.DataFrame(data={'qid': [1], 'query': [query]})
            df_query = Q2(df_query1)
            res = e2e_pipe(df_query)
#             res = (bm25_qe%10)(df_query)
            res = res.sort_values(by='rank')

            if res.empty:
                print('RES DataFrame is empty!')
                prfqueries.append(query+" [sep] ")
            else:
                prfpassages = ' [sep] '.join([str(n) for n in res.head(3).text.values])
                prfqueries.append(query+" [sep] " + prfpassages+" [sep] ")
                
        self.position += line_idx + 1

        if len(prfqueries) < self.bsize:
            raise StopIteration
        
        return self.collate(prfqueries, positives, negatives)
    
    def collate(self, prfqueries, positives, negatives):
        assert len(prfqueries) == len(positives) == len(negatives) == self.bsize
        return self.tensorize_triples(prfqueries, positives, negatives, self.bsize // self.accumsteps)


class PRFEagerBatcher(EagerBatcher):
   
    def __init__(self, args, rank=0, nranks=1):
        super().__init__(args, rank=rank, nranks=nranks)
        if not pt.started(): pt.init(boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"])
        index_location = "/nfs/indices/passage_index/data.properties"
        index = pt.IndexFactory.of(index_location)
        bm25 = pt.BatchRetrieve(index, wmodel='BM25',metadata=["docno", "text"])
        bm25_qe = bm25 >> pt.rewrite.RM3(index_location) >> bm25
        pyt_rankdiff = rankdiff(bm25 %5, bm25_qe%5 , 2)
        pyt = bm25_qe
        self.num_prf = args.num_prf


    def __next__(self):
        queries, positives, negatives = [], [], []
  
        #we should update hold outside batchsize constant, so we need to read less lines
        input_batch_size = self.bsize /self.num_prf
        for line_idx, line in zip(range(int(input_batch_size) * self.nranks), self.reader):
            print(line_idx)
            if (self.position + line_idx) % self.nranks != self.rank:
                continue

            query, pos, neg = line.strip().split('\t')
            queries.append(query)
            negatives.append(neg)
     
            res = self.pyt.search(query)
        
            if res.empty:
                positives.append(pos)
                print('RES DataFrame is empty!')
            else: 
                for i, row in enumerate( res.itertuples()):
                    positives.append(row.text)
                    if i == self.num_prf - 1:
                        break
                    



        self.position += line_idx + 1

        if len(queries) < self.bsize:
            raise StopIteration

        return self.collate(queries, positives, negatives)


class PRFEagerBatcherMix(EagerBatcher):
   
    def __init__(self, args, rank=0, nranks=1, num_prf=1, ratio=0.5):
        super().__init__(args, rank=rank, nranks=nranks)
#         self.pyt = pyt
        self.num_prf = num_prf
        self.ratio = ratio

    def __next__(self):
        queries, positives, negatives = [], [], []

        #we should update hold outside batchsize constant, so we need to read less lines
        #from here 
        input_batch_size_triples = self.bsize * self.ratio
        input_batch_size_prftriples1 = self.bsize * (1-self.ratio)
        input_batch_size_prftriples = input_batch_size_prftriples1 / self.num_prf
        
        # read ratio*bsize positive psges from triples 
        for line_idx, line in zip(range(int(input_batch_size_triples) * self.nranks), self.reader):

            if (self.position + line_idx) % self.nranks != self.rank:
                continue
            query, pos, neg = line.strip().split('\t')
            queries.append(query)
            positives.append(pos)
            negatives.append(neg)
        self.position += line_idx + 1
        
        # read (1-ratio)*bsize postive(PRF) psges from BM25 or BM25+RM3 while Negative psges from triples
        for line_idx, line in zip(range(int(input_batch_size_prftriples) * self.nranks), self.reader):
            if (self.position + line_idx) % self.nranks != self.rank:
                continue
            query, pos, neg = line.strip().split('\t')
            queries.append(query)
            negatives.append(neg)

            #TODO apply self.pyt on query
            df_query1 = pd.DataFrame(data={'qid': [1], 'query': [query]})
            df_query = Q2(df_query1)
            res = pyt(df_query)
            
            if res.empty:
                positives.append(pos)
                print('RES DataFrame is empty!')
            else: 
                for i, row in enumerate( res.itertuples()):
                    positives.append(row.text)
                    if i == self.num_prf - 1:
                        break

        self.position += line_idx + 1

        if len(queries) < self.bsize:
            raise StopIteration

        return self.collate(queries, positives, negatives)   
