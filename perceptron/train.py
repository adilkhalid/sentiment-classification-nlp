import random

from data.dataloader import load_dataset
from feature_extractors.bag_of_words.indexer import IndexerFeatureExtractor
from feature_extractors.bag_of_words.n_gram_feature_extractor import NGramFeatureExtractor
from feature_extractors.bag_of_words.one_hot_encoding import OneHotEncodingFeatureExtractor
from feature_extractors.bag_of_words.tfidf import TFIDF
from perceptron import Perceptron
from utils.model_io import save_json_model


def train_with_tfidf():
    tfidf = TFIDF()
    dataset = load_dataset("../data/train.csv")

    sentences = [sentence.split() for sentence, _ in dataset]
    labels = [label for _, label in dataset]
    tfidf.compute_idf(sentences)

    lr = 0.01
    epochs = 20
    vocab_list = list(tfidf.idf.keys())  # Fixed word order
    perceptron = Perceptron(len(vocab_list))
    X_train = [
        [tfidf.compute_tfidf(sentence).get(word, 0) for word in vocab_list]
        for sentence in sentences
    ]
    for _ in range(epochs):
        errors = 0
        random.shuffle(X_train)
        for x, target in zip(X_train, labels):
            prediction = perceptron.predict(x)
            error = target - prediction
            if error != 0:
                update = lr * (2 * error)

                # Update weights
                for i in range(len(perceptron.weights)):
                    perceptron.weights[i] += update * x[i]

                perceptron.bias += update
                errors += 1

        if errors == 0:
            break  # Stop early if no errors

    # Save trained model
    save_json_model(perceptron, file_path="perceptron_tfidf_model.json",
                    extra={"idf": tfidf.idf, "vocab_list": vocab_list})


def train_with_n_gram():
    n_gram_extractor = NGramFeatureExtractor()
    dataset = load_dataset("../data/train.csv")

    for sentence, label in dataset:
        n_gram_extractor.add_to_vocab(sentence.split())

    perceptron = Perceptron(n_gram_extractor.vocab_size)
    lr = 0.01
    epochs = 20

    for _ in range(epochs):
        errors = 0
        for x, target in dataset:
            feature = n_gram_extractor.extract_feature(x.split())
            print(feature)
            prediction = perceptron.predict(feature)
            error = target - prediction
            if error != 0:
                update = lr * (2 * error)
                for i in range(n_gram_extractor.vocab_size):
                    perceptron.weights[i] += update * feature[i]
                print(f"After Update: Weights: {perceptron.weights[:10]} Bias: {perceptron.bias}")
                perceptron.bias += update
                errors += 1
    print(n_gram_extractor.ngram_to_index)
    save_json_model(perceptron, file_path="perceptron_ngram_model.json",
                    extra={"ngram_to_index": {str(k): v for k, v in n_gram_extractor.ngram_to_index.items()},
                           "n": n_gram_extractor.n,
                           "vocab_size": n_gram_extractor.vocab_size})


def train_with_indexer():
    indexer = IndexerFeatureExtractor()
    dataset = load_dataset("../data/train.csv")

    for sentence, label in dataset:
        indexer.add_to_indexer(sentence.split())

    perceptron = Perceptron(len(indexer.objs_to_ints))
    lr = 0.001
    epochs = 20
    for epochs in range(epochs):
        errors = 0
        for x, target in dataset:
            feature = indexer.extract_feature(x.split())
            prediction = perceptron.predict(feature)
            error = target - prediction
            if error != 0:
                update = lr * (2 * error)

                # update weights
                for i in range(len(perceptron.weights)):
                    perceptron.weights[i] += update * feature[i]
                perceptron.bias += update
                errors += 1
        if errors == 0:
            break

    save_json_model(perceptron, file_path="perceptron_indexer_model.json",
                    extra={"objs_to_ints": indexer.objs_to_ints})


def train():
    oneHotEncoding = OneHotEncodingFeatureExtractor()
    dataset = load_dataset("../data/train.csv")

    for sentence, label in dataset:
        oneHotEncoding.add_to_vocab(sentence.split())

    perceptron = Perceptron(len(oneHotEncoding.vocabulary))
    learning_rate = 0.02
    epochs = 50
    for _ in range(epochs):
        errors = 0
        for x, target in dataset:
            feature = oneHotEncoding.extract_features(x.split())
            prediction = perceptron.predict(feature)
            error = target - prediction

            if error != 0:
                update = learning_rate * (2 * error)
                # update weights
                for i in range(len(feature)):
                    perceptron.weights[i] += update * feature[i]
                perceptron.bias += update
                errors += 1
        if errors == 0:
            break

    save_json_model(perceptron, extra={"vocabulary": oneHotEncoding.vocabulary})


if __name__ == "__main__":
    train_with_tfidf()
