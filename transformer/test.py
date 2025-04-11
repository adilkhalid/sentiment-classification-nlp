import numpy as np
import pickle

from TransformerClassifier import TransformerClassifier
from transformer import Transformer
from data.dataloader import load_dataset
from utils.model_io import load_transformer_model


def test():
    # Load raw components
    model_data = load_transformer_model()

    vocab_size = len(model_data["word_to_idx"])
    max_len = model_data["max_len"]
    embed_dim = model_data["embed_dim"]

    # Rebuild transformer
    transformer = Transformer(
        vocab_size=vocab_size,
        max_len=max_len,
        embed_dim=embed_dim,
        num_layers=2,
        ff_hidden_dim=64
    )
    transformer.token_embedding = model_data["token_embedding"]
    transformer.position_embedding = model_data["position_embedding"]

    # Rebuild classifier
    classifier = TransformerClassifier(transformer, embed_dim)
    classifier.output_weights = model_data["output_weights"]
    classifier.output_bias = model_data["output_bias"]

    # Load and run test
    dataset = load_dataset("../data/test.csv")
    for sentence, label in dataset:
        tokens = sentence.split()
        input_ids = [model_data["word_to_idx"].get(word, 0) for word in tokens]
        input_ids = input_ids[:max_len] + [0] * max(0, max_len - len(input_ids))

        y_pred, _ = classifier.forward(input_ids)
        sentiment = "Positive" if y_pred >= 0.5 else "Negative"
        actual = "Positive" if label == 1 else "Negative"

        print(f"Sentence: {sentence}")
        print(f"Predicted: {sentiment}, Actual: {actual}")
        print("---")


if __name__ == "__main__":
    test()
