import collections

import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import torchfold

import unittest


class TestEncoder(nn.Module):

    def __init__(self):
        super(TestEncoder, self).__init__()
        self.embed = nn.Embedding(10, 10)
        self.out = nn.Linear(20, 10)

    def concat(self, *nodes):
        return torch.cat(nodes, 0)

    def value(self, idx):
        return self.embed(idx)

    def value2(self, idx):
        return self.embed(idx), self.embed(idx)

    def attr(self, left, right):
        return self.out(torch.cat([left, right], 1))

    def logits(self, enc, embed):
        return torch.mm(enc, embed.t())


class TorchFoldTest(unittest.TestCase):

    def test_rnn(self):
        f = torchfold.Fold()
        v1, _ = f.add('value2', 1).split(2)
        v2, _ = f.add('value2', 2).split(2)
        r = v1
        for i in range(1000):
            r = f.add('attr', v1, v2)
            r = f.add('attr', r, v2)

        te = TestEncoder()
        enc = f.apply(te, [[r]])
        self.assertEqual(enc[0].size(), (1, 10))

    def test_nobatch(self):
        f = torchfold.Fold()
        v = []
        for i in range(15):
            v.append(f.add('value', i % 10))
        d = f.add('concat', *v).nobatch()
        res = []
        for i in range(100):
            res.append(f.add('logits', v[i % 10], d))

        te = TestEncoder()
        enc = f.apply(te, [res])
        self.assertEqual(len(enc), 1)
        self.assertEqual(enc[0].size(), (100, 15))


if __name__ == "__main__":
    unittest.main()

