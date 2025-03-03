import numpy as np


class LogisticRegression:
    def __init__(self, input_size=0):
        self.weights = np.zeros(input_size)
        self.bias = 0

    """
    σ(z)= 1/1+e^−z
    
    numbers between 0-1
    """

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def forward(self, X):
        return self.sigmoid(np.dot(X, self.weights) + self.bias)

    def predict(self, X):
        y_pred = self.forward(X)
        return y_pred >= .5
