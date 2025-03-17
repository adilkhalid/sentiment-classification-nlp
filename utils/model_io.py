import json
import logging
import pickle

import numpy as np

from MLP import MLP
from features.word_embeddings.word_2_vec import Word2Vec


def save_json_model(model, file_path="perceptron_model.json", extra=None):
    """Save the trained perceptron model and vocabulary."""

    model_data = {
        "weights": model.weights,
        "bias": model.bias
    }
    if extra:
        model_data.update(extra)
    with open(file_path, "w") as f:
        json.dump(model_data, f)
    print("Model saved successfully.")


def load_json_model(file_path="perceptron_model.json"):
    """Load perceptron model from a JSON file."""
    try:
        with open(file_path, "r") as f:
            model_data = json.load(f)
        logging.info("Model loaded successfully.")
        return model_data
    except FileNotFoundError:
        logging.error(f"Model file {file_path} not found.")
        return None
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        return None


def save_model_npz(model, file_path="logistic_regression.npz", extra=None):
    """Save logistic regression model weights and bias using NumPy."""
    np.savez(file_path, weights=model.weights, bias=model.bias, extra=extra)
    print(f"Model saved successfully at {file_path}")


def load_model_npz(file_path="logistic_regression.npz"):
    """Load logistic regression model from a file."""
    try:
        data = np.load(file_path, allow_pickle=True)
        return {
            "weights": data["weights"],
            "bias": data["bias"],
            "extra": data["extra"].item() if "extra" in data else None
        }
    except FileNotFoundError:
        print(f"Error: Model file {file_path} not found.")
        return None


def save_word2vec_model(word2vec, file_prefix="cbow_model"):
    np.save(f"{file_prefix}_W1.npy", word2vec.W1)
    np.save(f"{file_prefix}_W2.npy", word2vec.W2)

    with open(f"{file_prefix}_vocab.pkl", "wb") as f:
        pickle.dump({
            "word_to_index": word2vec.word_to_index,
            "index_to_word": word2vec.index_to_word,
            "embedding_dim": word2vec.embedding_dim,
            "vocab_size": word2vec.vocab_size
        }, f)
    print(f"Model saved as {file_prefix}_W1.npy, {file_prefix}_W2.npy, and {file_prefix}_vocab.pkl")


def load_word2vec_model(file_prefix="cbow_model"):
    """
    Loads a trained Word2Vec model.
    - Loads W1 and W2 from `.npy`
    - Loads vocabulary and metadata from `.pkl`
    """
    with open(f"{file_prefix}_vocab.pkl", "rb") as f:
        metadata = pickle.load(f)

    word2vec = Word2Vec(window_size=2, embedding_dim=metadata["embedding_dim"])
    word2vec.word_to_index = metadata["word_to_index"]
    word2vec.index_to_word = metadata["index_to_word"]
    word2vec.vocab_size = metadata["vocab_size"]

    word2vec.W1 = np.load(f"{file_prefix}_W1.npy")
    word2vec.W2 = np.load(f"{file_prefix}_W2.npy")

    print(f"Model loaded from {file_prefix}")
    return word2vec


def save_mlp_model(model: MLP, file_prefix="mlp_model"):
    """
    Saves the trained MLP classifier.

    Args:
        model (MLPClassifier): The trained MLP model.
        file_prefix (str): Prefix for the saved files.
    """
    # Save weights and biases as NumPy arrays
    np.save(f"{file_prefix}_W1.npy", model.W1)
    np.save(f"{file_prefix}_W2.npy", model.W2)
    np.save(f"{file_prefix}_b1.npy", model.b1)
    np.save(f"{file_prefix}_b2.npy", model.b2)

    # Save metadata as a Pickle file
    metadata = {
        "input_dim": model.input_dim,
        "hidden_dim": model.hidden_dim
    }
    with open(f"{file_prefix}_metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    print(
        f"Model saved as {file_prefix}_W1.npy, {file_prefix}_W2.npy, {file_prefix}_b1.npy, {file_prefix}_b2.npy, and {file_prefix}_metadata.pkl")


def load_mlp_model(file_prefix="mlp_model"):
    """
    Loads a trained MLP classifier.

    Args:
        file_prefix (str): Prefix of the saved files.

    Returns:
        MLPClassifier: The loaded MLP model.
    """
    # Load metadata
    with open(f"{file_prefix}_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    # Initialize a new model with saved dimensions
    model = MLP(input_dim=metadata["input_dim"], hidden_dim=metadata["hidden_dim"])

    # Load weights and biases
    model.W1 = np.load(f"{file_prefix}_W1.npy")
    model.W2 = np.load(f"{file_prefix}_W2.npy")
    model.b1 = np.load(f"{file_prefix}_b1.npy")
    model.b2 = np.load(f"{file_prefix}_b2.npy")

    print(f"Model loaded from {file_prefix}")

    return model
