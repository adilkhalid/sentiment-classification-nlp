import numpy as np


class LSTM:
    def __init__(self, input_dim, hidden_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Initialize weights for Forget, Input, Candidate, and Output gates
        self.W_f = np.random.uniform(-0.5, 0.5, (hidden_dim, input_dim + hidden_dim))
        self.b_f = np.zeros(hidden_dim)

        self.W_i = np.random.uniform(-0.5, 0.5, (hidden_dim, input_dim + hidden_dim))
        self.b_i = np.zeros(hidden_dim)

        self.W_C = np.random.uniform(-0.5, 0.5, (hidden_dim, input_dim + hidden_dim))
        self.b_C = np.zeros(hidden_dim)

        self.W_o = np.random.uniform(-0.5, 0.5, (hidden_dim, input_dim + hidden_dim))
        self.b_o = np.zeros(hidden_dim)

        # Output layer weights (for sentiment classification)
        self.W_hy = np.random.uniform(-0.5, 0.5, (1, hidden_dim))
        self.b_y = np.zeros(1)

    def forward(self, X):
        h_t = np.zeros(self.hidden_dim)
        C_t = np.zeros(self.hidden_dim)

        for x_t in X:
            # Concatenate previous hidden state and input
            combined = np.concatenate((h_t, x_t))

            # Forget gate
            f_t = self.sigmoid(np.dot(self.W_f, combined) + self.b_f)

            # Input gate
            i_t = self.sigmoid(np.dot(self.W_i, combined) + self.b_i)
            C_tilde_t = np.tanh(np.dot(self.W_C, combined) + self.b_C)

            # Update cell state
            C_t = f_t * C_t + i_t * C_tilde_t

            # Output gate
            o_t = self.sigmoid(np.dot(self.W_o, combined) + self.b_o)
            h_t = o_t * np.tanh(C_t)

        # Compute final prediction
        logits = np.dot(self.W_hy, h_t) + self.b_y
        y_pred = self.sigmoid(logits)

        return y_pred, h_t, C_t

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
