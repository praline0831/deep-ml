import torch

def diffusion_loss(x_0: torch.Tensor, t: int, beta_start: float, beta_end: float, num_timesteps: int, noise: torch.Tensor, predicted_noise: torch.Tensor) -> float:
    """
    Compute the reconstruction loss for diffusion model training.
    
    Args:
        x_0: Original input data (torch.Tensor)
        t: Timestep (1-indexed, from 1 to num_timesteps)
        beta_start: Starting value of linear beta schedule
        beta_end: Ending value of linear beta schedule
        num_timesteps: Total number of diffusion timesteps
        noise: True noise tensor (same shape as x_0)
        predicted_noise: Model's predicted noise (same shape as x_0)
    
    Returns:
        Mean squared error loss (float)
    """
    # 1. 线性 beta 调度: betas[0] 对应 t=1, betas[T-1] 对应 t=T
    betas = torch.linspace(beta_start, beta_end, num_timesteps)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)  # alpha_bars[k] = 累乘到第 k+1 步

    # 2. 取出第 t 步的 alpha_bar (t 为 1-indexed)
    alpha_bar_t = alpha_bars[t - 1]

    # 3. 前向加噪: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
    #    (真实训练中, 这个 x_t 就是喂给模型的输入, 模型输出 predicted_noise)
    x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1.0 - alpha_bar_t) * noise

    # 4. 重建损失: 用预测噪声从 x_t 反解出 x_0, 再与原图做 MSE
    #    x_0_hat = (x_t - sqrt(1 - alpha_bar_t) * predicted_noise) / sqrt(alpha_bar_t)
    x_0_hat = (x_t - torch.sqrt(1.0 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)
    loss = torch.mean((x_0_hat - x_0) ** 2)
    return float(loss)
