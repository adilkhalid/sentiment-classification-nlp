import numpy as np


class MLP:
    def __init__(self, input_dim, hidden_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.W1 = np.random.uniform(-0.5, 0.5, (hidden_dim, input_dim))
        self.b1 = np.zeros((hidden_dim,))

        self.W2 = np.random.uniform(-0.5, 0.5, (1, hidden_dim))
        self.b2 = np.zeros(1, )

    def forward(self, X: np.ndarray):
        hidden_output = np.maximum(0, np.dot(self.W1, X) + self.b1)  # ReLU actication

        logits = np.dot(self.W2, hidden_output) + self.b2
        print(logits)
        y_pred = 1 / (1 + np.exp(-logits))  # Sigmoid
        print(y_pred)
        return y_pred, hidden_output
