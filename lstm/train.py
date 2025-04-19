import numpy as np
from data.dataloader import load_dataset
from lstm import LSTM
from utils.model_io import load_word2vec_model, save_lstm_model


class LSTMTrainer:
    def __init__(self, word2vec, hidden_dim=16, epochs=50, lr=0.001):
        self.dataset = load_dataset("../dataset/train.csv")
        self.epochs = epochs
        self.lr = lr
        self.word2vec = word2vec

        self.lstm = LSTM(input_dim=self.word2vec.W1.shape[1], hidden_dim=hidden_dim)

    def train(self):
        for epoch in range(self.epochs):
            total_loss = 0
            for sentence, y_true in self.dataset:
                sentence_vectors = [
                    self.word2vec.W1[self.word2vec.word_to_index.get(word, 0)]
                    for word in sentence.split()
                ]
                if not sentence_vectors:
                    continue

                y_pred, h_t, C_t = self.lstm.forward(sentence_vectors)
                loss = self.binary_cross_entropy(y_true, y_pred)
                total_loss += loss

                self.backward(y_pred, y_true, h_t, sentence_vectors)

            print(f"Epoch {epoch + 1}/{self.epochs} - Loss: {total_loss:.4f}")

        save_lstm_model(self.lstm, file_prefix="lstm_model")

    def backward(self, y_pred, y_true, h_t, sentence_vectors):
        dL_dlogits = (y_pred - y_true) * y_pred * (1 - y_pred)  # sigmoid derivative

        # Output layer gradients
        dW_hy = np.outer(dL_dlogits, h_t)
        db_y = dL_dlogits

        # Gradient wrt h_t
        dL_dh = np.dot(self.lstm.W_hy.T, dL_dlogits)

        # You can optionally backprop through LSTM layers here
        # For simplicity, we just update final layer and W_hy
        self.lstm.W_hy -= self.lr * dW_hy
        self.lstm.b_y -= self.lr * db_y

    @staticmethod
    def binary_cross_entropy(y_true, y_pred):
        epsilon = 1e-9
        return -(
                y_true * np.log(y_pred + epsilon) +
                (1 - y_true) * np.log(1 - y_pred + epsilon)
        ).item()


if __name__ == "__main__":
    word2vec_model = load_word2vec_model('../cbow/cbow_model')
    trainer = LSTMTrainer(word2vec_model)
    trainer.train()
