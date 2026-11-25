import torch

def ema_update(ema_params, model_params_list, decay: float) -> torch.Tensor:
    """
    Compute the Exponential Moving Average of model parameters over training steps.

    Args:
        ema_params: torch.Tensor, initial EMA parameters
        model_params_list: list of torch.Tensors, model params at each training step
        decay: float, EMA decay rate in [0, 1]
    Returns:
        Final EMA parameters as a torch.Tensor, rounded to 4 decimal places
    """
    for step_params in model_params_list:
        ema_params = decay * ema_params + (1 - decay) * step_params
    return torch.round(ema_params, decimals=4)