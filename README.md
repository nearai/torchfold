<img src="logo.jpg" width=30% align="right" />

[![PyPi version](https://pypip.in/v/torchfold/badge.png)](https://pypi.org/project/torchfold/) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1299387.svg)](https://doi.org/10.5281/zenodo.1299387)
# TorchFold

Blog post: http://near.ai/articles/2017-09-06-PyTorch-Dynamic-Batching/

Analogous to [TensorFlow Fold](https://github.com/tensorflow/fold), implements dynamic batching with super simple interface.
Replace every direct call in your computation to nn module with `f.add('function name', arguments)`.
It will construct an optimized version of computation and on `f.apply` will dynamically batch and execute the computation on given nn module.

## Installation
We recommend using pip package manager:
```
pip install torchfold
```

## Example

```
    f = torchfold.Fold()
   
    def dfs(node):
        if is_leaf(node):
            return f.add('leaf', node)
        else:
            prev = f.add('init')
            for child in children(node):
                prev = f.add('child', prev, child)
            return prev

    class Model(nn.Module):
        def __init__(self, ...):
            ...

        def leaf(self, leaf):
            ...

        def child(self, prev, child):
            ...

    res = dfs(my_tree)
    model = Model(...)
    f.apply(model, [[res]])
```


To cite this repository in publications:

    @misc{illia_polosukhin_2018_1299387,
      author       = {Illia Polosukhin and
                      Maksym Zavershynskyi},
      title        = {nearai/torchfold: v0.1.0},
      month        = jun,
      year         = 2018,
      doi          = {10.5281/zenodo.1299387},
      url          = {https://doi.org/10.5281/zenodo.1299387}
    }
