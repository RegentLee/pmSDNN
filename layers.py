# coding: utf-8
import numpy as np


class SelectiveDesensitization:
    def __init__(self):
        pass


class FullyConnected:
    def __init__(self, input_size, output_size):
        self.x = None
        self.W = 0.01 * np.random.randn(input_size, output_size).astype('f')

    def forward(self, x):
        self.x = x
        y = np.dot(self.x, self.W)
        return y

    def backward(self, dy):
        grads = np.dot(self.x.T, dy)
        self.W -= grads


class SGN:
    def __init__(self):
        pass
