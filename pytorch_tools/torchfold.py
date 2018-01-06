import collections

import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import torch.nn.functional as F


class Fold(object):

    class Node(object):
        def __init__(self, op, step, index, *args):
            self.op = op
            self.step = step
            self.index = index
            self.args = args
            self.split_idx = -1
            self.batch = True

        def split(self, num):
            """Split resulting node, if function returns multiple values."""
            nodes = []
            for idx in range(num):
                nodes.append(Fold.Node(
                    self.op, self.step, self.index, *self.args))
                nodes[-1].split_idx = idx
            return tuple(nodes)

        def nobatch(self):
            self.batch = False
            return self

        def get(self, values):
            if self.split_idx >= 0:
                return values[self.step][self.op][self.split_idx][self.index]
            else:
                return values[self.step][self.op][self.index]

        def __repr__(self):
            return "[%d:%d]%s" % (
                self.step, self.index, self.op)

    def __init__(self, volatile=False, cuda=False):
        self.steps = collections.defaultdict(
            lambda: collections.defaultdict(list))
        self.cached_nodes = collections.defaultdict(dict)
        self.total_nodes = 0
        self.volatile = volatile
        self._cuda = cuda

    def cuda(self):
        self._cuda = True
        return self

    def add(self, op, *args):
        """Add op to the fold."""
        self.total_nodes += 1
        if not all([isinstance(arg, (
            Fold.Node, int, torch.tensor._TensorBase, Variable)) for arg in args]):
            raise ValueError(
                "All args should be Tensor, Variable, int or Node, got: %s" % str(args))
        if args not in self.cached_nodes[op]:
            step = max([0] + [arg.step + 1 for arg in args
                              if isinstance(arg, Fold.Node)])
            node = Fold.Node(op, step, len(self.steps[step][op]), *args)
            self.steps[step][op].append(args)
            self.cached_nodes[op][args] = node
        return self.cached_nodes[op][args]

    def _batch_args(self, arg_lists, values):
        res = []
        for arg in arg_lists:
            r = []
            if isinstance(arg[0], Fold.Node):
                if arg[0].batch:
                    for x in arg:
                        r.append(x.get(values))
                    res.append(torch.cat(r, 0))
                else:
                    for i in range(2, len(arg)):
                        if arg[i] != arg[0]:
                            raise ValueError("Can not use more then one of nobatch argument, got: %s." % str(arg))
                    x = arg[0]
                    res.append(x.get(values))
            elif isinstance(arg[0], (torch.tensor._TensorBase, Variable)):
                res.append(torch.cat(arg, 0))
            else:
                try:
                    if self._cuda:
                        var = Variable(torch.cuda.LongTensor(arg), volatile=self.volatile)
                    else:
                        var = Variable(torch.LongTensor(arg), volatile=self.volatile)
                    res.append(var)
                except:
                    print("Constructing LongTensor from %s" % str(arg))
                    raise
        return res

    def apply(self, nn, nodes):
        """Apply current fold to given neural module."""
        values = {}
        for step in sorted(self.steps.keys()):
            values[step] = {}
            for op in self.steps[step]:
                func = getattr(nn, op)
                try:
                    batched_args = self._batch_args(
                        zip(*self.steps[step][op]), values)
                except Exception:
                    print("Error while executing node %s[%d] with args: %s" % (
                        op, step, self.steps[step][op][0]))
                    raise
                if batched_args:
                    arg_size = batched_args[0].size()[0]
                else:
                    arg_size = 1
                res = func(*batched_args)
                if isinstance(res, (tuple, list)):
                    values[step][op] = []
                    for x in res:
                        values[step][op].append(torch.chunk(x, arg_size))
                else:
                    values[step][op] = torch.chunk(res, arg_size)
        try:
            return self._batch_args(nodes, values)
        except Exception:
            print("Retrieving %s" % nodes)
            for lst in nodes:
                if isinstance(lst[0], Fold.Node):
                    print(', '.join([str(x.get(values).size()) for x in lst]))
            raise

    def __str__(self):
        result = ''
        for step in sorted(self.steps.keys()):
            result += '%d step:\n' % step
            for op in self.steps[step]:
                first_el = ''
                for arg in self.steps[step][op][0]:
                    if first_el: first_el += ', '
                    if isinstance(arg, (torch.Tensor, Variable)):
                        first_el += str(arg.size())
                    else:
                        first_el += str(arg)
                result += '\t%s = %d x (%s)\n' % (
                    op, len(self.steps[step][op]), first_el)
        return result

    def __repr__(self):
        return str(self)


class Unfold(object):
    """Replacement of Fold for debugging, where it does computation right away."""

    class Node(object):

        def __init__(self, tensor):
            self.tensor = tensor

        def __repr__(self):
            return str(self.tensor)

        def nobatch(self):
            return self

        def split(self, num):
            return [Unfold.Node(self.tensor[i]) for i in range(num)]

    def __init__(self, nn, volatile=False, cuda=False):
        self.nn = nn
        self.volatile = volatile
        self._cuda = cuda

    def cuda(self):
        self._cuda = True
        return self

    def _arg(self, arg):
        if isinstance(arg, Unfold.Node):
            return arg.tensor
        elif isinstance(arg, int):
            if self._cuda:
                return Variable(torch.cuda.LongTensor([arg]), volatile=self.volatile)
            else:
                return Variable(torch.LongTensor([arg]), volatile=self.volatile)
        else:
            return arg

    def add(self, op, *args):
        values = []
        for arg in args:
            values.append(self._arg(arg))
        res = getattr(self.nn, op)(*values)
        return Unfold.Node(res)

    def apply(self, nn, nodes):
        if nn != self.nn:
            raise ValueError("Expected that nn argument passed to constructor and passed to apply would match.")
        result = []
        for n in nodes:
            result.append(torch.cat([self._arg(a) for a in n]))
        return result
