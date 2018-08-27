# AVID_pytorch

## Overview
PyTorch implementation of [AVID: Adversarial Visual Irregularity Detection](https://arxiv.org/abs/1805.09521).

## Preliminary

Download datasets & create directorys.
```
sh ./setup.sh
```

Preprocess UCSD dataset.

```
python prepocess_UCSD.py
```


## Training

Run `train.py`.

- example
```
CUDA_VISIBLE_DEVICES=0 python train.py --dataset IR-MNIST

```

## Test

See `example_UCSD.ipynb`
