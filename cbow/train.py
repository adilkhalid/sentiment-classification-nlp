import numpy as np

from cbow import CBOW
from data.dataloader import load_dataset
from features.word_embeddings.word_2_vec import Word2Vec
from utils.model_io import save_json_model, save_word2vec_model


def train():
    dataset = load_dataset("../data/test.csv")
    word2vec = Word2Vec()
    sentences = [sentence.split() for sentence, label in dataset]
    word2vec.build_vocabulary(sentences)
    learning_rate = 0.01
    epochs = 10

    training_data = word2vec.generate_cbow_training_data(sentences, mode="cbow")
    model = CBOW(word2vec)
    for epochs in range(epochs):
        total_loss = 0
        for context_indices, target_index in training_data:

            # Forward Pass
            y_pred = model.forward(context_indices)

            y_true = np.zeros(word2vec.vocab_size)
            y_true[target_index] = 1

            error = y_pred - y_true

            # Compute gradients

            dW2 = np.outer(word2vec.W1[context_indices].mean(axis=0), error)
            dW1 = np.dot(word2vec.W2, error)
            print(word2vec.W1[context_indices].mean(axis=0))
            print("stop")
            # update weights
            word2vec.W2 -= learning_rate * dW2
            for index in context_indices:
                word2vec.W1[index] -= learning_rate * dW1

            # compute loss
            total_loss -= np.log(y_pred[target_index] + 1e-9)

    save_word2vec_model(word2vec)


if __name__ == "__main__":
    train()
