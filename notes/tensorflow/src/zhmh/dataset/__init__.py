import numpy as np


class BatchGenerator:
    def __init__(self, x_train, y_train, batch_size, step_size=1):
        from zhmh.magic import __assert__
        __assert__(x_train.shape[0] == y_train.shape[0], 'Feature dose not match label.')
        self.data_size = x_train.shape[0]
        self.batch_size = batch_size
        self.step_size = step_size
        self.X = x_train
        self.Y = y_train
        self.index = 0

    def count(self):
        return ((self.data_size - self.batch_size) // self.step_size) + 1

    def next(self):
        index = self.index if (self.data_size - self.index) > self.batch_size else 0
        d_x = self.X[index: index + self.batch_size]
        d_y = self.Y[index: index + self.batch_size]
        self.index = index + self.step_size
        return d_x, d_y
