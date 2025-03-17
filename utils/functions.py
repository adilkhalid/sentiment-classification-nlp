import numpy as np


def binary_cross_entropy(y_true, y_pred):
    """
    Computes binary cross-entropy loss.

    Args:
        y_true (int): Actual sentiment label (0 or 1).
        y_pred (float): Predicted probability (between 0 and 1).

    Returns:
        float: Binary cross-entropy loss.
    """
    return - (y_true * np.log(y_pred + 1e-9) + (1 - y_true) * np.log(1 - y_pred + 1e-9))
