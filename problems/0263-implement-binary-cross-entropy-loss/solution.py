import numpy as np


def binary_cross_entropy(
    y_true: list[float],
    y_pred: list[float],
    epsilon: float = 1e-15,
) -> float:
    """
    Compute binary cross-entropy loss.

    Args:
        y_true: True binary labels (0 or 1)
        y_pred: Predicted probabilities (between 0 and 1)
        epsilon: Small value for numerical stability

    Returns:
        Mean binary cross-entropy loss
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)

    loss = -np.mean(
        y_true * np.log(y_pred) + (1.0 - y_true) * np.log1p(-y_pred)
    )
    return float(loss)