# MIP Supervision

[![Open Word-Level In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/imadtoubal/MIP-Supervision/blob/master/demo.ipynb)

A simple PyTorch implementation of: Koziński M, Mosinska A, Salzmann M, Fua P. Tracing in 2D to reduce the annotation effort for 3D deep delineation of linear structures. Medical Image Analysis. 2020 Feb;60:101590. DOI: 10.1016/j.media.2019.101590.

## Install Environement

- Using conda:
  ```shell
  conda env create
  ```
- Using pip (TODO)

## Dataset

`demo.ipynb` uses a public segmentation dataset ([CHAOS](https://chaos.grand-challenge.org/Combined_Healthy_Abdominal_Organ_Segmentation/)) as an example. It is a good palce to start in this repository.

If you wish to use your own dataset, you can use `dataset.SegDataset3D` class as follows:

```python
from dataset import SegDataset3D

trainset = SegDataset3D(X, Y)
```

Where `X` and `Y` are numpy arrays of shape `(N, W, H, D)`, with:

- `N`: number of samples
- `W`: width of each sample
- `H`: height of each sample
- `D`: depth of each sample
