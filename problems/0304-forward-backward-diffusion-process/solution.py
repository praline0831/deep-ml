import torch

def diffusion_process(x_0: torch.Tensor,
                      betas: torch.Tensor,
                      timestep: int,
                      forward_noise: torch.Tensor,
                      predicted_noise: torch.Tensor,
                      backward_noise: torch.Tensor = None) -> tuple:
    """
    Implement forward and backward diffusion processes.

    Args:
        x_0: Original clean data (any shape), torch.Tensor
        betas: Noise schedule tensor of shape (T,)
        timestep: Current timestep t (1-indexed)
        forward_noise: Noise epsilon for forward diffusion
        predicted_noise: Model's predicted noise for backward diffusion
        backward_noise: Random noise z for stochastic backward step

    Returns:
        tuple: (x_t, x_t_minus_1) - noisy and denoised samples as torch.Tensors
    """
    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
    alphas_t = alphas_cumprod[timestep - 1]
    x_t = torch.sqrt(alphas_t) * x_0 + torch.sqrt(1.0 - alphas_t) * forward_noise

    # ---------- 反向过程：从 x_t 去噪一步得到 x_{t-1} ----------
    beta_t = betas[timestep - 1]          # β_t
    alpha_t = 1.0 - beta_t                # α_t
    alpha_bar_t = alphas_cumprod[timestep - 1]        # ᾱ_t
    alpha_bar_prev = alphas_cumprod[timestep - 2] if timestep > 1 else torch.tensor(1.0)  # ᾱ_{t-1}，约定 ᾱ_0 = 1

    # 后验均值：用模型预测的噪声代替真实的 forward_noise
    mean = (x_t - (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t) * predicted_noise) \
        / torch.sqrt(alpha_t)

    # 方差 σ_t² = β_t · (1 - ᾱ_{t-1}) / (1 - ᾱ_t)；t == 1 时 ᾱ_0 = 1，方差为 0，不再加噪
    if timestep > 1 and backward_noise is not None:
        std = torch.sqrt(beta_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t))
        x_t_minus_1 = mean + std * backward_noise
    else:
        x_t_minus_1 = mean

    return x_t, x_t_minus_1