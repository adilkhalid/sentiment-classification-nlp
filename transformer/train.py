import numpy as np
from data.dataloader import load_dataset
from TransformerClassifier import TransformerClassifier
from transformer import Transformer
from utils.model_io import save_transformer_model


def train():
    dataset = load_dataset("../dataset/train.csv")
    sentences = [s.split() for s, _ in dataset]
    labels = [label for _, label in dataset]

    vocab = set(word for sentence in sentences for word in sentence)
    word_to_index = {word: i for i, word in enumerate(vocab)}
    vocab_size = len(word_to_index)

    X_train = []
    max_len = 20
    embed_dim = 32
    for sentence, label in dataset:
        input_ids = [word_to_index.get(w, 0) for w in sentence.split()]
        input_ids = input_ids[:max_len] + [0] * max(0, max_len - len(input_ids))
        X_train.append((input_ids, label))

    ff_hidden_dim = 64
    num_layers = 2
    transformer = Transformer(vocab_size, max_len, embed_dim, ff_hidden_dim, num_layers)
    classifier = TransformerClassifier(transformer, embed_dim)

    lr = 0.01
    epochs = 10

    for epoch in range(epochs):
        for input_ids, label in X_train:
            y_pred, pooled = classifier.forward(input_ids)
            grad = y_pred - label

            dW = np.outer(pooled, grad)
            db = grad

            classifier.output_weights -= lr * dW
            classifier.output_bias -= lr * db

    save_transformer_model(transformer, classifier, word_to_index)


if __name__ == "__main__":
    train()
