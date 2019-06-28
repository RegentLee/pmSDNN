# coding: utf-8
import numpy as np


class SelectiveDesensitization:
    def __init__(self, pattern, window_size, range_size):
        #ランダムパターンを作る
        pattern_size = pattern.shape
        random_pattern = np.empty((window_size, pattern_size[0], pattern_size[1]), dtype='int8')
        idx = np.arange(pattern_size[1])

        for i in window_size:
            np.random.shuffle(idx)
            random_pattern[i] = np.copy(pattern[idx])

        self.p = pattern
        self.rp = random_pattern
        self.ws = window_size
        self.rs = range_size

    def forward(self, contexts):
        sd = np.ones((len(contexts), 1))
        for i in range(self.ws - 1):
            for j in range(i + 1, min(i + 1 + self.rs, self.ws)):
                in0 = self.p[contexts[:, i]]
                in1 = self.p[contexts[:, j]]

                in0r = self.rp[i][contexts[:, i]]
                in1r = self.rp[j][contexts[:, j]]

                sd0 = (1 + in1r) * in0
                sd1 = (1 + in0r) * in1

                sd0 = np.where((sd0 == 1) | (sd0 == -1), 0, sd0 / 2)
                sd1 = np.where((sd1 == 1) | (sd1 == -1), 0, sd1 / 2)

                sd = np.hstack((sd, sd0, sd1))

        return sd

    def backward(self, dy):
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
