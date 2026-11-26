import torch

def pca(data, k) -> torch.Tensor:
    """
    Perform PCA on `data`, returning the top `k` principal components as a tensor.
    Input: Tensor or convertible of shape (n_samples, n_features).
    Returns: a torch.Tensor of shape (n_features, k), with floats rounded to 4 decimals.
    Note: If an eigenvector's first non-zero value is negative, flip its sign.
    """
    x = torch.as_tensor(data, dtype=torch.float64)

    # 1. 标准化：每个特征减均值、除标准差（零均值单位方差）
    x = x - x.mean(dim=0)
    x = x / x.std(dim=0)

    # 2. 计算协方差矩阵（此时即相关矩阵，实对称）
    n = x.shape[0]
    cov = (x.T @ x) / (n - 1)

    # 3. 特征值分解，eigh 返回升序的特征值与对应的特征向量
    eigvals, eigvecs = torch.linalg.eigh(cov)

    # 4. 取最大的 k 个特征值对应的特征向量作为主成分
    idx = torch.argsort(eigvals, descending=True)[:k]
    components = eigvecs[:, idx]

    # 5. 符号约定：让每个特征向量的首个非零元素为正
    for j in range(components.shape[1]):
        vec = components[:, j]
        nonzero = torch.nonzero(vec.abs() > 1e-10).flatten()
        if nonzero.numel() > 0 and vec[nonzero[0]] < 0:
            components[:, j] = -vec

    return components.round(decimals=4)