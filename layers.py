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
        self.y = None

    def forward(self, x):
        self.y = np.where(x < 0, -1, 1)
        return self.y

    def backward(self, dy):
        return dy


class PotentialLoss:
    def __init__(self):
        self.loss = None

    def forward(self, y, t):
        output_size = t.shape
        self.loss = (y - t) / 2
        return np.sum(np.abs(self.loss)) / (output_size[0]*output_size[1])

    def backward(self):
        return self.loss
