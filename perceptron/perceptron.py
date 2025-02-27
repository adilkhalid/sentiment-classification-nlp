import numpy as np


class Perceptron:
    def __init__(self, vocab_size):
        self.weights = [0] * vocab_size
        self.bias = 0

    def predict(self, x):
        weighted_sum = 0
        # dot product
        for i in range(len(x)):
            weighted_sum += self.weights[i] * x[i]

        weighted_sum += self.bias

        # HeaveSide step function (activation function)
        return 1 if weighted_sum >= 0 else 0
