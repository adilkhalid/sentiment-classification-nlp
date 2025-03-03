import json
import logging
import numpy as np


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
