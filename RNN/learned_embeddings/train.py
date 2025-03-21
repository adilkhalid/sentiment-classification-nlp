import numpy as np

from RNN.learned_embeddings.rnn_trainable import RNNTrainable
from data.dataloader import load_dataset


class RNNTrainer:
    def __init__(self, epochs=50, lr=0.001):
        self.dataset = load_dataset("../data/train.csv")
        self.epochs = epochs
        self.lr = lr
        # Build vocab
        unique_words = set(word for sentence, _ in self.dataset for word in sentence.split())
        self.word_to_index = {word: i for i, word in enumerate(unique_words)}
        vocab_size = len(self.word_to_index)

        # initialize RNN with trainable embeddings
        self.embedding_dim = 50
        self.hidden_dim = 16
        self.rnn = RNNTrainable(vocab_size, self.hidden_dim, self.embedding_dim)

    def train(self):
        for epoch in range(self.epochs):
            for sentence, y_true in self.dataset:
                sentence_indices = [self.word_to_index[word] for word in sentence]
                if not sentence_indices:
                    continue

                y_pred, h_states = self.rnn.forward(sentence_indices)
                self.backward(h_states, y_pred, y_true)

    def backward(self, h_states, y_pred, y_true, sentence_indices):
        # Loss gradient
        dL_dlogits = y_pred - y_true

        # Gradient for W_hy (output weight)
        dW_hy = np.outer(dL_dlogits, h_states[-1])
        db_y = dL_dlogits

        # Init hidden state gradient
        dL_dh_next = np.zeros(self.hidden_dim)

        dW_xh = np.zeros_like(self.rnn.W_xh)
        dW_hh = np.zeros_like(self.rnn.W_hh)
        db_h = np.zeros_like(self.rnn.b_h)

        d_embedding = np.zeros_like(self.rnn.embedding_dim)

        for t in reversed(range(len(sentence_indices))):
            word_index = sentence_indices[t]

            dL_dh = np.dot(self.rnn.W_hy, dL_dlogits) + dL_dh_next
            dL_dh_raw = (1 - h_states ** 2) * dL_dh

            dW_xh += np.outer(dL_dh_raw, self.rnn.embedding_matrix[word_index])  # input weights
            dW_hh += np.outer(dL_dh_raw, h_states[t - 1]) if t > 0 else np.zeros(
                self.hidden_dim)  # Reccurrence eights
            db_h += dL_dh_raw  # Bias Gradient

            dL_dh_next = np.dot(self.rnn.W_hh.T, dL_dh_raw)

            d_embedding[word_index] += np.dot(self.rnn.W_xh.T, dL_dh_raw)

            # Update weights using gradient descent
            self.rnn.W_xh -= self.lr * dW_xh
            self.rnn.W_hh -= self.lr * dW_hh
            self.rnn.W_hy -= self.lr * dW_hy
            self.rnn.b_h -= self.lr * db_h
            self.rnn.b_y -= self.lr * db_y

            # Update the word embeddings
            self.rnn.embedding_matrix -= self.lr * d_embedding


if __name__ == "__main__":
    RNNTrainer().train()
