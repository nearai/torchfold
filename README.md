[![PyPi version](https://pypip.in/v/torchfold/badge.png)](https://pypi.org/project/torchfold/)
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
