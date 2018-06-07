import torch
from torch.autograd import Variable
import torch.nn as nn

import torchfold

import mock
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


class RNNEncoder(nn.Module):

    def __init__(self, num_units, input_size):
        super(RNNEncoder, self).__init__()
        self.num_units = num_units
        self.input_size = input_size
        self.encoder = nn.GRUCell(self.input_size, self.num_units)

    def encode(self, input_, state):
        return self.encoder(input_, state)


class TestRNNBatching(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.input_size = 5
        self.num_units = 4

    def _generate_variable(self, dim):
        t = torch.Tensor(1, dim).uniform_(0, 1)
        return Variable(t)

    def test_rnn_optimized_chunking(self):
        seq_lengths = [2, 3, 5]

        states = []
        for _ in xrange(len(seq_lengths)):
            states.append(self._generate_variable(self.num_units))

        f = torchfold.Fold()
        for seq_ind in xrange(len(seq_lengths)):
            for _ in xrange(seq_lengths[seq_ind]):
                states[seq_ind] = f.add('encode', self._generate_variable(self.input_size), states[seq_ind])

        enc = RNNEncoder(self.num_units, self.input_size)
        with mock.patch.object(torch, 'chunk', wraps=torch.chunk) as wrapped_chunk:
            result = f.apply(enc, [states])
            # torch.chunk is called 3 times instead of max(seq_lengths)=5.
            self.assertEquals(3, wrapped_chunk.call_count)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].size(), (len(seq_lengths), self.num_units))


if __name__ == "__main__":
    unittest.main()
