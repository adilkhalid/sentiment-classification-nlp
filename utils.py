import json
import logging


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
