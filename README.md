# CWPRF implementation


This repository provides the code for training and retrieval of our paper [ACL2023: Effective Contrastive Weighting for Dense Query Expansion](https://aclanthology.org/2023.acl-long.710.pdf). Before running the following experiments, please install the dependencies in the  environment.txt file.

The datasets as well as the query sets used in CWPRF can be downloaded from: https://microsoft.github.io/msmarco/TREC-Deep-Learning

# Training CWPRF
We have two training modes: AAAT and OAAT training modes to train CWPRF.

### CWPRF-AAAT training

- parameters: number of feedback passages for training: $f_p=3$; maximum input length for document encoder: 512; maximum inputlength for query encoder: 128.

- Training command for CWPRF-AAAT

```python
python -m colbert.train_AAAT --amp  --doc_maxlen 180  --mask-punctuation --bsize 24 --accum 1 --triples /path/to/train/triples.train.small.tsv --checkpoint /path/to/ColBERT/Checkpoints/colbert.dnn --root /path/to/save/checkpoint/CWPRF_AAAT --experiment psg --run CWPRF --num_prf 3 --in_batch_negs --checkpoint_init
```

### CWPRF-OAAT training
- parameters: number of feedback passages for training: $f_p=3$; maximum input length for document encoder: 512; maximum inputlength for query encoder: 128.

- Training command for CWPRF-OAAT

```python
python -m colbert.train_OAAT --amp  --doc_maxlen 180  --mask-punctuation --bsize 24 --accum 1 --triples /path/to/train/triples.train.small.tsv --checkpoint /path/to/ColBERT/Checkpoints/colbert.dnn --root /path/to/save/checkpoint/CWPRF_OAAT --experiment psg --run CWPRF --num_prf 3 --in_batch_negs --checkpoint_init
```

# Retrieval with CWPRF

Validation on TREC 2019 query set: [Validation Notebook](CWPRF_Inference.ipynb)

Main Results reported in the paper: Test on both TREC 2019 and TREC 2020 query set: [Test Notebook](CWPRF_Inference.ipynb)

# Hyperparameter Search

Hyperparameter search on validation set: [Hyper-parameter search Notebook](CWPRF_Inference.ipynb)


# Reproduce the results of CWPRF and various baselines

The result res files of CWPRF and various baselines for both TREC query sets are provided in the folder: [CWPRF VirtualAppendix](CWPRF_VirtualAppendix). 
