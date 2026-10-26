import torch

def batch_normalization(X: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, epsilon: float = 1e-5) -> torch.Tensor:
    """Perform Batch Normalization on a 4D tensor in BCHW format."""
    # X: (B, C, H, W)；gamma/beta: (C,) —— 每个通道一组缩放/平移参数
    mean = X.mean(dim=(0, 2, 3), keepdim=True)     # (1, C, 1, 1)
    var = X.var(dim=(0, 2, 3), keepdim=True, unbiased=False)  # (1, C, 1, 1)
    X_norm = (X - mean) / torch.sqrt(var + epsilon)
    return gamma.view(1, -1, 1, 1) * X_norm + beta.view(1, -1, 1, 1)