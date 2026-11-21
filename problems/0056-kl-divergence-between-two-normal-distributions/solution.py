import numpy as np

def kl_divergence_normal(mu_p, sigma_p, mu_q, sigma_q, eps: float = 1e-8):
    """
    KL( N(mu_p, sigma_p^2) || N(mu_q, sigma_q^2) )
    纯 NumPy 实现，支持标量 / 数组广播。

    Args:
        mu_p, sigma_p: P 的均值、标准差
        mu_q, sigma_q: Q 的均值、标准差
        eps: 防止 sigma=0 时 log/除法产生 inf/nan

    Returns:
        np.ndarray 或 np.float64：逐元素 KL 值
    """
    # 统一转成 float64 ndarray，标量也能参与运算并保持广播
    mu_p    = np.asarray(mu_p,    dtype=np.float64)
    sigma_p = np.asarray(sigma_p, dtype=np.float64)
    mu_q    = np.asarray(mu_q,    dtype=np.float64)
    sigma_q = np.asarray(sigma_q, dtype=np.float64)

    var_p = sigma_p ** 2
    var_q = sigma_q ** 2

    kl = 0.5 * (
        np.log(var_q + eps) - np.log(var_p + eps)
        + (var_p + (mu_p - mu_q) ** 2) / (var_q + eps)
        - 1.0
    )
    return kl