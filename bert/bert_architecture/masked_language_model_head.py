import numpy as np

from utils.functions import softmax


class MaskedLanguageModelHead:
    def __init__(self, embed_dim, vocab_size):
        self.W = np.random.randn(embed_dim, vocab_size) * 0.01
        self.b = np.zeros(vocab_size)

    def forward(self, encoded_output):
        # encoded_output: (seq_len, embed_dim)
        logits = encoded_output @ self.W + self.b  # (seq_len, vocab_size)
        return softmax(logits)
