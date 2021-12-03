# GCM
This is our Tensorflow implementation for our paper based on [NeuRec](https://github.com/wubinzzu/NeuRec/):

>Jiancan Wu, Xiangnan He, Xiang Wang, Qifan Wang, Weijian Chen, Jianxun Lian, Xing Xie. 2021. Graph Convolution Machine for Context-aware Recommender System, [Paper in arXiv](https://arxiv.org/abs/2001.11402).

## Environment Requirement

The code runs well under python 3.8.10. The required packages are as follows:

- Tensorflow-gpu == 1.15.1
- numpy == 1.18.5
- scipy == 1.7.0
- pandas == 1.3.0
- cython == 0.29.21

## Quick Start
**Firstly**, compline the evaluator of cpp implementation with the following command line:

```bash
python setup.py build_ext --inplace
```

If the compilation is successful, the evaluator of cpp implementation will be called automatically.
Otherwise, the evaluator of python implementation will be called.

**Note that the cpp implementation is much faster than python.**

Further details, please refer to [NeuRec](https://github.com/wubinzzu/NeuRec/)

**Secondly**,  run [GCM.py](./GCM.py) in IDE or with command line:

### Yelp-NC

```bash
python GCM.py --dataset Yelp-NC --num_gcn_layers 2 --reg 1e-3 --decoder_type FM --adj_norm_type ls --num_negatives 4
```

### Yelp-OH
```bash
python GCM.py --dataset Yelp-OH --num_gcn_layers 2 --reg 1e-3 --decoder_type FM --adj_norm_type ls --num_negatives 4
```

### Amazon-Book
```bash
python GCM.py --dataset Amazon-Book --num_gcn_layers 2 --reg 1e-3 --decoder_type FM --adj_norm_type ls --num_negatives 2
```
