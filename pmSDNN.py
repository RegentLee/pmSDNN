# coding: utf-8
from layers import SelectiveDesensitization, FullyConnected, SGN, PotentialLoss


class pmSDNN:
    def __init__(self, pattern, window_size, range_size=None):
        if range_size is not None and window_size - 1 < range_size:
            raise ValueError('Range Size is over')
        if range_size is None:
            range_size = window_size - 1

        input_size = pattern.shape[1]*range_size*(window_size - range_size - 1) + 1

        sd = SelectiveDesensitization(pattern, window_size, range_size)
        fc = FullyConnected(input_size, pattern.shape[1])
        activation_func = SGN()
        loss_func = PotentialLoss()

        self.layers = [sd, fc, activation_func]
        self.loss_func = loss_func

    def forward(self, x, t):
        for layer in self.layers:
            x = layer.forward(x)
        loss = self.loss_func.forward(x, t)
        return loss

    def backward(self):
        dx = self.loss_func.backward()
        for layer in self.layers[::-1]:
            dx = layer.backward(dx)
