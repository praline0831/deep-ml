import numpy as np

def log_softmax(scores: np.ndarray) -> np.ndarray:
    # Your code here
    x = scores - np.max(scores, keepdims=True)
    return x - np.log(np.sum(np.exp(x),keepdims=True))