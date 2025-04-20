from typing import List

import numpy as np


class CBOW:
    def __init__(self, word2vec):
        self.word2vec = word2vec

    def forward(self, context_indices: List[int]) -> np.ndarray:
        """
        Forward pass: Predict target word from context words.
        - context_indices: List of indices representing context words.
        - Returns: Probability distribution over vocabulary.
        """
        # Step 1: Get word vectors from W1 (hidden layer)
        h = np.mean(self.word2vec.W1[context_indices], axis=0)  # (embedding_dim,)

        # Step 2: Compute raw scores (logits) for all words
        u = np.dot(self.word2vec.W2.T, h)  # (vocab_size,)

        # Step 3: Apply softmax to get probability distribution
        y_pred = np.exp(u) / np.sum(np.exp(u))  # Softmax activation

        return y_pred  # Probabilities for each word in vocabulary
