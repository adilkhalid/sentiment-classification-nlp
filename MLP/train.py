import numpy as np

from MLP import MLP
from data.dataloader import load_dataset
from feature_extractors.word_embeddings.word_2_vec import Word2Vec
from utils.functions import binary_cross_entropy
from utils.model_io import save_mlp_model, save_word2vec_model


def train():
    dataset = load_dataset("../dataset/test.csv")
    word2vec = Word2Vec()
    sentences = [sentence.split() for sentence, label in dataset]
    word2vec.build_vocabulary(sentences)

    mlp = MLP(input_dim=word2vec.embedding_dim, hidden_dim=16)
    learning_rate = .001
    epochs = 1
    for epoch in range(epochs):
        for sentence, label in dataset:
            X = word2vec.sentence_to_vector(sentence.split())

            y_pred, hidden_output = mlp.forward(X)

            error = y_pred - label

            dW2 = np.outer(error, hidden_output)
            dW1 = np.outer(mlp.W2.T * error, X)

            mlp.W2 -= learning_rate * dW2
            mlp.W1 -= learning_rate * dW1
    save_word2vec_model(word2vec, "mlp")
    save_mlp_model(mlp)


if __name__ == "__main__":
    train()
