
from data.dataloader import load_dataset
from features.indexer import IndexerFeatureExtractor
from features.one_hot_encoding import OneHotEncodingFeatureExtractor
from perceptron import Perceptron
from utils import save_json_model


def train_with_indexer():
    indexer = IndexerFeatureExtractor()
    dataset = load_dataset("../data/train.csv")

    for sentence, label in dataset:
        indexer.add_to_indexer(sentence)

    perceptron = Perceptron(len(indexer.ints_to_objs))
    lr = 0.001
    epochs = 20
    for epochs in range(epochs):
        errors = 0
        for x, target in dataset:
            feature = indexer.extract_feature(x)
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
                    extra={"ints_to_objs": indexer.ints_to_objs, "objs_to_ints": indexer.objs_to_ints})


def train():
    oneHotEncoding = OneHotEncodingFeatureExtractor()
    dataset = load_dataset("../data/train.csv")

    for sentence, label in dataset:
        oneHotEncoding.add_to_vocab(sentence)

    perceptron = Perceptron(len(oneHotEncoding.vocabulary))
    learning_rate = 0.02
    epochs = 50
    for _ in range(epochs):
        errors = 0
        for x, target in dataset:
            feature = oneHotEncoding.extract_features(x)
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
    train_with_indexer()
