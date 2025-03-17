import numpy as np

from data.dataloader import load_dataset
from features.bag_of_words.n_gram_feature_extractor import NGramFeatureExtractor
from logistic_regression import LogisticRegression
from utils.model_io import save_model_npz


def train_model():
    dataset = load_dataset("../data/test.csv")
    n_gram_extractor = NGramFeatureExtractor()
    sentences = [sentence.split() for sentence, _ in dataset]
    n_gram_extractor.build_vocab(sentences)

    model = LogisticRegression(n_gram_extractor.vocab_size)
    learning_rate = 0.01
    epochs = 20

    for epoch in range(epochs):
        total_loss = 0
        for sentence, label in dataset:
            feature_vector = n_gram_extractor.extract_feature(sentence.split())
            y_pred = model.forward(feature_vector)

            error = label - y_pred

            # Compute Gradients
            for index, count in enumerate(feature_vector):
                model.weights[index] += learning_rate * error * count  # Gradient update

            loss = - (label * np.log(y_pred + 1e-9) + (1 - label) * np.log(1 - y_pred + 1e-9))
            total_loss += loss
    save_model_npz(model, extra=n_gram_extractor.model_details())


if __name__ == "__main__":
    train_model()

