import os
import six

import array
import numpy as np

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F


def expand_embeddings(embed, vocab, new_words, optimizer,
                      initializer=None):
    embed_dim = embed.weight.size()[1]
    new_params = Parameter(torch.Tensor(
        len(vocab) + len(new_words), embed_dim
    ))
    new_params.data.normal_(0, 1)
    new_params.data[:len(vocab)] = embed.weight.data
    if hasattr(optimizer, 'state'):
        # Reset state of the optimizer.
        optimizer.state[embed.weight] = {}
        optimizer.state[new_params] = {}
    embed.weight = new_params
    new_vocab = {}
    for word in new_words:
        vocab[word] = len(vocab)
        new_vocab[word] = vocab[word]
    if initializer:
        initializer(embed, new_vocab)
    return vocab


def _load_glove(base_path, name, dim):
    filename = os.path.join(base_path, name, '%s.%dd.txt' % (name, dim))
    if not os.path.exists(filename):
        raise ValueError("Can not find %s filename to read" % filename)

    with open(filename, 'rb') as f:
        for line in f:
            tokens = line.strip().split(' ')
            word, entries = tokens[0], tokens[1:]
            try:
                if isinstance(word, six.binary_type):
                    word = word.decode('utf-8')
            except:
                print('non-UTF8 token', repr(word), 'ignored')
                continue
            yield word, [float(x) for x in entries]


def load_glove(base_path, name, dim):
    pt_filename = os.path.join(base_path, name, '%s.%dd.pt')
    if os.path.exists(pt_filename):
        return torch.load(pt_filename)

    words, arr = [], array.array('d')
    for word, entries in _load_glove(base_path, name, dim):
        arr.extend(entries)
        word.append(word)

    vocab = {word: i for i, word in enumerate(words)}
    arr = torch.Tensor(arr).view(-1, dim)
    torch.save((vocab, arr), pt_filename)
    return vocab, arr


def load_embed(embed, base_path, name, vocab):
    dim = embed.weight.size()[1]
    loaded = 0
    for word, entries in _load_glove(base_path, name, dim):
        idx = vocab.get(word, None)
        if not idx:
            continue
        embed.weight.data[idx, :] = torch.FloatTensor(entries)
        loaded += 1
    print("Loaded embeddings %s:%dd for %d words." % (
        name, dim, loaded))
