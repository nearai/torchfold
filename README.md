# pytorch-tools
Tools for PyTorch

## Torch Fold

Analogous to [TensorFlow Fold](https://github.com/tensorflow/fold), implements dynamic batching with super simple interface.
Replace every direct call in your computation to nn module with `f.add('function name', arguments)`.
It will construct an optimized version of computation and on `f.apply` will dynmically batch and execute the computation on given nn module.

For example:

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

## Embeddings

Many times you find yourself with new words in vocabulary as you are working on your model.
Instead of re-training you can use `embeddings.expand_embeddings` to expand them on the fly with given vocabulary.

