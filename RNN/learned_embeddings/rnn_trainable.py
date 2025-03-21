import numpy as np


class RNNTrainable:
    def __init__(self, vocab_size, hidden_dim, embedding_dim):
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        # Trainable word embeddings (instead of using Word2Vec)
        self.embedding_matrix = np.random.uniform(-0.5, 0.5, (vocab_size, embedding_dim))


        # Initialize weight matrices
        self.W_xh = np.random.uniform(-0.5, 0.5, (hidden_dim, embedding_dim))
        self.W_hh = np.random.uniform(-0.5, 0.5, (hidden_dim, hidden_dim))
        self.W_hy = np.random.uniform(-0.5, 0.5, (1, hidden_dim))

        # Initialize bias terms
        self.b_h = np.zeros(hidden_dim)
        self.b_y = np.zeros(1)

    def forward(self, sentence_indices):
        h_t = np.zeros(self.hidden_dim)

        for idx in sentence_indices:
            x_t = self.embedding_matrix[idx]  # Lookup word embedding
            h_t = np.tanh(np.dot(self.W_xh, x_t) + np.dot(self.W_hh, h_t) + self.b_h)

        logits = np.dot(self.W_hy, h_t) + self.b_y
        y_pred = 1 / (1 + np.exp(-logits))  # Sigmoid

        return y_pred, h_t