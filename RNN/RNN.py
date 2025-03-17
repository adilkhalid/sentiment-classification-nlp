import numpy as np


class RNN:
    def __init__(self, input_dim, hidden_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Initialize weight matrices
        self.W_xh = np.random.uniform(-0.5, 0.5, (hidden_dim, input_dim))
        self.W_hh = np.random.uniform(-0.5, 0.5, (hidden_dim, hidden_dim))
        self.W_hy = np.random.uniform(-0.5, 0.5, (1, hidden_dim))

        # Initialize bias terms
        self.b_h = np.zeros(hidden_dim)
        self.b_y = np.zeros(1)

    def forward(self, X):
        h_t = np.zeros(self.hidden_dim)

        for x_t in X:
            h_t = np.tanh(np.dot(self.W_xh, x_t) + np.dot(self.W_hh, h_t) + self.b_h)

        logits = np.dot(self.W_hy, h_t) + self.b_y

        y_pred = 1 / (1 + np.exp(-logits))

        return y_pred, h_t