
from data.dataloader import load_dataset
from features.indexer import IndexerFeatureExtractor
from features.one_hot_encoding import OneHotEncodingFeatureExtractor
from perceptron import Perceptron
from utils import load_json_model


def test_perceptron():
    model_data = load_json_model()
    perceptron = Perceptron(vocab_size=len(model_data["weights"]))
    perceptron.weights = model_data["weights"]
    perceptron.bias = model_data["bias"]
    dataset = load_dataset("../data/test.csv")
    oneHotEncoding = OneHotEncodingFeatureExtractor()
    oneHotEncoding.vocabulary = model_data["vocabulary"]

    predictions = [perceptron.predict(oneHotEncoding.extract_features(x)) for x, _ in dataset]

    # Print predictions vs. actual labels
    for i in range(len(dataset)):
        print(f"Text: {i + 1} | Prediction: {predictions[i]} | Actual: {dataset[i][1]}")


def test_indexer_perceptron():
    model_data = load_json_model(file_path="perceptron_indexer_model.json")
    perceptron = Perceptron(vocab_size=len(model_data["weights"]))
    perceptron.weights = model_data["weights"]
    perceptron.bias = model_data["bias"]
    dataset = load_dataset("../data/test.csv")
    indexer = IndexerFeatureExtractor()
    indexer.ints_to_objs = model_data["ints_to_objs"]
    indexer.objs_to_ints = model_data["objs_to_ints"]

    predictions = [perceptron.predict(indexer.extract_feature(x)) for x, _ in dataset]

    # Print predictions vs. actual labels
    for i in range(len(dataset)):
        print(f"Text: {i + 1} | Prediction: {predictions[i]} | Actual: {dataset[i][1]}")


if __name__ == "__main__":
    test_indexer_perceptron()
