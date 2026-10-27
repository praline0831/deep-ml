import numpy as np

def compute_cross_entropy_loss(predicted_probs: np.ndarray, true_labels: np.ndarray, epsilon = 1e-15) -> float:
    # Your code here
    return -np.mean(np.sum(true_labels * np.log(predicted_probs + epsilon), axis=1))
