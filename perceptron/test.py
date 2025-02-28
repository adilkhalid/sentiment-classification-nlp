import ast

from data.dataloader import load_dataset
from features.indexer import IndexerFeatureExtractor
from features.n_gram_feature_extractor import NGramFeatureExtractor
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

    predictions = [perceptron.predict(oneHotEncoding.extract_features(x.split())) for x, _ in dataset]

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
    indexer.objs_to_ints = model_data["objs_to_ints"]

    predictions = [perceptron.predict(indexer.extract_feature(x.split())) for x, _ in dataset]

    # Print predictions vs. actual labels
    for i in range(len(dataset)):
        print(f"Text: {i + 1} | Prediction: {predictions[i]} | Actual: {dataset[i][1]}")


def test_ngram_perceptron():
    model_data = load_json_model(file_path="perceptron_ngram_model.json")
    perceptron = Perceptron(vocab_size=len(model_data["weights"]))
    perceptron.weights = model_data["weights"]
    perceptron.bias = model_data["bias"]
    dataset = load_dataset("../data/test.csv")
    nGramFeatureExtractor = NGramFeatureExtractor()
    print(model_data)
    nGramFeatureExtractor.vocab_size = model_data["vocab_size"]
    nGramFeatureExtractor.n = model_data["n"]
    nGramFeatureExtractor.ngram_to_index ={ast.literal_eval(k): v for k, v in model_data["ngram_to_index"].items()}

    predictions = [perceptron.predict(nGramFeatureExtractor.extract_feature(x.split())) for x, _ in dataset]

    # Print predictions vs. actual labels
    for i in range(len(dataset)):
        print(f"Text: {i + 1} | Prediction: {predictions[i]} | Actual: {dataset[i][1]}")


if __name__ == "__main__":
    test_perceptron()
