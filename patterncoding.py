# coding: utf-8
import numpy as np


class PatternCoding:
    def __init__(self, n, pattern_size, reverse_size=1):
        self.n = n
        self.p = pattern_size
        self.r = reverse_size

    def sequence_pattern_coding(self):
        print('Making Pattern...')

        pattern = np.zeros((self.n, self.p), dtype='int8')

        # 0コ目
        pattern[0][: self.p//2] = 1
        pattern[0][self.p//2:] = -1
        np.random.shuffle(pattern[0])

        # 残り
        for i in range(1, self.n):
            while 1:
                p = np.copy(pattern[i - 1])
                a_temp = []
                b_temp = []
                for j in range(self.r):
                    while 1:
                        a = np.random.randint(self.p)
                        b = np.random.randint(self.p)
                        if p[a] * p[b] == -1 and a not in a_temp and b not in b_temp:
                            p[a] = -p[a]
                            p[b] = -p[b]
                            a_temp.append(a)
                            b_temp.append(b)
                            break
                if np.sum(np.all(p == pattern, axis=1)) == 0:
                    pattern[i] = p
                    break

        return pattern

    def random_pattern_coding(self):
        print('Making Pattern...')

        pattern = np.ones((self.n, self.p), dtype='int8')

        pattern[:, self.p // 2:] = -1

        # パターンを作る
        for i in range(self.n):
            while 1:
                np.random.shuffle(pattern[i])
                if np.sum(np.all(pattern[i] == pattern[:i], axis=1)) == 0:
                    break

        return pattern
