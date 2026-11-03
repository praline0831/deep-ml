import math

import torch

def pos_encoding(position: int, d_model: int):
    """
    Compute positional encodings for Transformer models.

    PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

    Args:
        position: sequence length (number of positions)
        d_model: model dimensionality

    Returns:
        torch.Tensor of shape (position, d_model) with dtype float16,
        or -1 if position == 0 or d_model <= 0.
    """
    if position == 0 or d_model <= 0:
        return -1

    pe = torch.zeros(position, d_model)                              # (position, d_model)

    # 每个位置的序号，形状 (position, 1)，方便后面广播
    pos = torch.arange(position, dtype=torch.float32).unsqueeze(1)

    # 10000^(2i/d_model)，用 exp-log 形式数值上更稳定
    # 形状 (ceil(d_model/2),)，对应波长从 2π 递增到 10000*2π
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32)
        * (-math.log(10000.0) / d_model)
    )

    pe[:, 0::2] = torch.sin(pos * div_term)                          # 偶数维 -> sin
    pe[:, 1::2] = torch.cos(pos * div_term[: d_model // 2])          # 奇数维 -> cos

    return pe.half()
