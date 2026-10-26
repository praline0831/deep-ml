import torch

def group_normalization(X: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, num_groups: int, epsilon: float = 1e-5) -> torch.Tensor:
    """
    Perform Group Normalization on a 4D input tensor.
    
    Args:
        X: torch tensor of shape (B, C, H, W), input data
        gamma: torch tensor of shape (1, C, 1, 1), scale parameter
        beta: torch tensor of shape (1, C, 1, 1), shift parameter
        num_groups: number of groups for normalization
        epsilon: small constant to avoid division by zero
    
    Returns:
        norm_X: torch tensor of shape (B, C, H, W), normalized output
    """
    B, C, H, W = X.shape
    # 通道维拆成 num_groups 组，每组 C//num_groups 个通道
    # reshape: (B, C, H, W) → (B, num_groups, C//num_groups, H, W)
    G = num_groups
    X = X.view(B, G, C // G, H, W)

    # 对每组的 (C//G, H, W) 算均值和方差
    mean = X.mean(dim=(2, 3, 4), keepdim=True)     # (B, G, 1, 1, 1)
    var = X.var(dim=(2, 3, 4), keepdim=True, unbiased=False)
    X_norm = (X - mean) / torch.sqrt(var + epsilon)

    # 还原形状 → (B, C, H, W)
    X_norm = X_norm.view(B, C, H, W)
    # 仿射变换：gamma/beta 形状 (1, C, 1, 1) 直接广播
    return gamma * X_norm + beta