import numpy as np

from feature_extractors.word_embeddings.word_2_vec import Word2Vec


class SkipGram:
    def __init__(self, word2vec: Word2Vec):
        self.word2vec = word2vec

    def forward(self, target_index):
        h = self.word2vec.W1[target_index]  # Get target word vector
        u = np.dot(self.word2vec.W2.T, h)  # Compute scores for words

        y_pred = np.exp(u) / np.sum(np.exp(u))  # Softmax activation

        return y_pred
