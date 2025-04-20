import numpy as np
import pickle

from RNN.rnn import RNNPreTrained
from data.dataloader import load_dataset
from utils.model_io import load_word2vec_model, save_rnn_model


class RNNTrainer:
    def __init__(self, word2vec, hidden_dim=16, epochs=50, lr=0.001):
        self.dataset = load_dataset("../../dataset/train.csv")
        self.epochs = epochs
        self.lr = lr
        self.word2vec = word2vec  # Pretrained Word2Vec model

        # Initialize RNN (input size = embedding dimension of Word2Vec)
        self.rnn = RNNPreTrained(input_dim=self.word2vec.W1.shape[1], hidden_dim=hidden_dim)

    def train(self):
        for epoch in range(self.epochs):
            for sentence, y_true in self.dataset:
                # Convert sentence into a list of Word2Vec vectors
                sentence_vectors = [self.word2vec.W1[self.word2vec.word_to_index.get(word, 0)] for word in
                                    sentence.split()]
                if not sentence_vectors:  # Skip empty sentences
                    continue

                y_pred, h_states = self.rnn.forward(sentence_vectors)
                self.backward(h_states, y_pred, y_true, sentence_vectors)
        save_rnn_model(self.rnn)

    def backward(self, h_states, y_pred, y_true, sentence_vectors):
        """Compute gradients using Backpropagation Through Time (BPTT)"""
        dL_dlogits = (y_pred - y_true) * y_pred * (1 - y_pred)

        dW_hy = np.outer(dL_dlogits, h_states[-1])
        db_y = dL_dlogits

        dL_dh_next = np.zeros(self.rnn.hidden_dim)
        dW_xh = np.zeros_like(self.rnn.W_xh)
        dW_hh = np.zeros_like(self.rnn.W_hh)
        db_h = np.zeros_like(self.rnn.b_h)

        # No embedding updates since Word2Vec is frozen
        for t in reversed(range(len(sentence_vectors))):
            x_t = sentence_vectors[t]

            dL_dh = np.dot(self.rnn.W_hy.T, dL_dlogits) + dL_dh_next
            dL_dh_raw = (1 - h_states[t] ** 2) * dL_dh  # tanh derivative

            dW_xh += np.outer(dL_dh_raw, x_t)
            dW_hh += np.outer(dL_dh_raw, h_states[t - 1]) if t > 0 else np.zeros_like(self.rnn.W_hh)
            db_h += dL_dh_raw

            dL_dh_next = np.dot(self.rnn.W_hh.T, dL_dh_raw)

        self.rnn.W_xh -= self.lr * dW_xh
        self.rnn.W_hh -= self.lr * dW_hh
        self.rnn.W_hy -= self.lr * dW_hy
        self.rnn.b_h -= self.lr * db_h
        self.rnn.b_y -= self.lr * db_y


if __name__ == "__main__":
    word2vec_model = load_word2vec_model('../../cbow/cbow_model')
    trainer = RNNTrainer(word2vec_model)
    trainer.train()
