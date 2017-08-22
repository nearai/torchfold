import os
import time

import torch


def load_checkpoint(model, optimizer, model_dir):
    path = os.path.join(model_dir, 'checkpoint')
    if os.path.exists(path):
        print("Loading model from %s" % path)
        checkpoint = torch.load(path)
        old_state_dict = model.state_dict()
        for key in old_state_dict.keys():
            if key not in checkpoint['model']:
                checkpoint['model'][key] = old_state_dict[key]
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        return checkpoint.get('step', 0)
    return 0


def save_checkpoint(model, optimizer, step, model_dir, ignore=[]):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    path = os.path.join(model_dir, 'checkpoint')
    state_dict = model.state_dict()
    if ignore:
        for key in state_dict.keys():
            for item in ignore:
                if key.startswith(item):
                    state_dict.pop(key)
    torch.save({
        'model': state_dict,
        'optimizer': optimizer.state_dict(),
        'step': step
    }, path)


class Trainer(object):

    def __init__(
            self, model, optimizer, model_dir, batch_size,
            report_every_n=10, checkpoint_every_n=100, ignore=[]):
        self.model = model
        self.optimizer = optimizer
        self.model_dir = model_dir
        self.batch_size = batch_size
        self.report_every_n = report_every_n
        self.checkpoint_every_n = checkpoint_every_n

        self.start = time.time()
        self.count = 0
        self.accum_loss = 0

        self.step = load_checkpoint(model, optimizer, model_dir)
        self.optimizer.zero_grad()
        self.report_callbacks = []
        self.ignore = ignore

    def add_report_callback(self, func):
        self.report_callbacks.append(func)

    def report(self):
        for func in self.report_callbacks:
            func(self)
        self.start = time.time()

    def finish(self):
        save_checkpoint(
            self.model, self.optimizer, self.step, self.model_dir,
            ignore=self.ignore)

    def train(self, *args, **kwargs):
        self.count += 1
        result = self.model(*args, **kwargs)
        self.accum_loss += result[0]
        if self.count >= self.batch_size:
            self.accum_loss.backward()
            torch.nn.utils.clip_grad_norm(self.model.parameters(), 5.0)
            self.optimizer.step()
            self.optimizer.zero_grad()
            if self.step % self.report_every_n == 0:
                self.report()
            if self.step % self.checkpoint_every_n == 0:
                save_checkpoint(
                    self.model, self.optimizer, self.step, self.model_dir,
                    ignore=self.ignore)
            self.count = 0
            self.accum_loss = 0
            self.step += 1
        return result
