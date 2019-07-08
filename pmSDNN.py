# coding: utf-8
from layers import SelectiveDesensitization, FullyConnected, SGN, PotentialLoss


class pmSDNN:
    def __init__(self, pattern, input_num, range_num=None):
        if range_num is not None and input_num - 1 < range_num:
            raise ValueError('range_num is over input_num - 1')
        if range_num is None:
            range_num = input_num - 1

        input_size = pattern.shape[1] * range_num * (2 * input_num - range_num - 1) + 1

        sd = SelectiveDesensitization(pattern, input_num, range_num)
        fc = FullyConnected(input_size, pattern.shape[1])
        activation_func = SGN()
        loss_func = PotentialLoss()

        self.pattern = pattern
        self.layers = [sd, fc, activation_func]
        self.loss_func = loss_func

    def forward(self, x, t):
        for layer in self.layers:
            x = layer.forward(x)
        loss = self.loss_func.forward(x, self.pattern[t])
        return loss

    def backward(self):
        dx = self.loss_func.backward()
        for layer in self.layers[::-1]:
            dx = layer.backward(dx)
