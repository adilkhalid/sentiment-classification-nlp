import numpy as np

from utils.functions import softmax


class SelfAttention:
    def __init__(self, embed_dim):
        self.embed_dim = embed_dim
        self.d_k = embed_dim

        # Initialize weights
        self.W_q = np.random.randn(embed_dim, embed_dim) * 0.01
        self.W_k = np.random.randn(embed_dim, embed_dim) * 0.01
        self.W_v = np.random.randn(embed_dim, embed_dim) * 0.01

    def forward(self, X):
        """
        X: shape (seq_len, embed_dim)
        Returns: shape (seq_len, embed_dim)
        """
        Q = X @ self.W_q  # (seq_len, d_k)
        K = X @ self.W_k  # (seq_len, d_k)
        V = X @ self.W_v  # (seq_len, d_k)

        scores = Q @ K.T / np.sqrt(self.d_k)  # (seq_len, seq_len)
        weights = softmax(scores)  # (seq_len, seq_len)

        output = weights @ V  # (seq_len, d_k)
        return output
