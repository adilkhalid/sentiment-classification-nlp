

import ast
from data.dataloader import load_dataset
from feature_extractors.n_gram_feature_extractor import NGramFeatureExtractor
from logistic_regression import LogisticRegression
from utils.model_io import load_model_npz


def test_logistic_regression():
    """Test the trained logistic regression model."""
    # Load model
    model_data = load_model_npz("logistic_regression.npz")
    if model_data is None:
        return

    # Initialize model
    logistic_regression = LogisticRegression(input_size=len(model_data["weights"]))
    logistic_regression.weights = model_data["weights"]
    logistic_regression.bias = model_data["bias"]

    # Load test dataset
    dataset = load_dataset("../data/test.csv")

    # Load feature extractor details
    extractor_data = model_data["extra"]
    nGramFeatureExtractor = NGramFeatureExtractor()
    nGramFeatureExtractor.vocab_size = extractor_data["vocab_size"]
    nGramFeatureExtractor.n = extractor_data["n"]
    nGramFeatureExtractor.ngram_to_index = {ast.literal_eval(k): v for k, v in extractor_data["ngram_to_index"].items()}

    # Run predictions
    predictions = [logistic_regression.predict(nGramFeatureExtractor.extract_feature(x.split())) for x, _ in dataset]

    # Print predictions vs actual labels
    for i in range(len(dataset)):
        print(f"Text {i + 1}: Prediction: {predictions[i]} | Actual: {dataset[i][1]}")


if __name__ == "__main__":
    test_logistic_regression()

