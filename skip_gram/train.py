import numpy as np

from data.dataloader import load_dataset
from features.word_embeddings.word_2_vec import Word2Vec
from skip_gram.SkipGram import SkipGram
from utils.model_io import save_word2vec_model


def train():
    dataset = load_dataset("../data/test.csv")
    word2vec = Word2Vec()
    sentences = [sentence.split() for sentence, label in dataset]
    word2vec.build_vocabulary(sentences)
    learning_rate = 0.01
    epochs = 10

    training_data = word2vec.generate_skipgram_training_data(sentences, 2)
    model = SkipGram(word2vec)
    for epochs in range(epochs):
        for target_index, context_index in training_data:

            # Forward Pass
            y_pred = model.forward(target_index)

            y_true = np.zeros(word2vec.vocab_size)
            y_true[context_index] = 1

            error = y_pred - y_true

            # Compute gradients

            dW2 = np.outer(word2vec.W1[target_index].mean(axis=0), error)
            dW1 = np.dot(word2vec.W2, error)

            # update weights
            word2vec.W2 -= learning_rate * dW2
            word2vec.W1[target_index] -= learning_rate * dW1

    save_word2vec_model(word2vec, file_prefix="skipgram")


if __name__ == "__main__":
    train()
