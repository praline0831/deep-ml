import torch

def fp8_block_quantize(
    tensor: torch.Tensor,
    block_size: int = 128
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a tensor to FP8-E4M3 format using block-wise scaling.
    
    Args:
        tensor: Input tensor of shape (N,) where N is divisible by block_size
        block_size: Number of elements per quantization block
        
    Returns:
        quantized: Quantized values of shape (N,), clipped to [-448, 448]
        scales: Per-block scale factors of shape (N // block_size,)
    """
    N = tensor.shape[0]
    num_blocks = N // block_size
    # 把 reshape 成 (num_blocks, block_size)，每个块一行
    blocks = tensor.reshape(num_blocks, block_size)
    # 每个块的最大绝对值 M_b
    abs_max = blocks.abs().amax(dim=1, keepdim=True)
    # 比例因子 s_b = (M_b + ε) / 448，ε=1e-12 保证全零块也有正的 scale
    eps = 1e-12
    scales = (abs_max + eps) / 448.0                      # (num_blocks, 1)
    # 缩放、四舍五入到整数、手动 clip 到 [-448, 448]
    quantized_blocks = (blocks / scales).round().clamp(-448, 448)
    quantized = quantized_blocks.reshape(N)
    scales = scales.squeeze(1)                            # (num_blocks,)
    return quantized, scales


def fp8_block_dequantize(
    quantized: torch.Tensor,
    scales: torch.Tensor,
    block_size: int = 128
) -> torch.Tensor:
    """
    Dequantize FP8-E4M3 values back to full precision.
    
    Args:
        quantized: Quantized values of shape (N,)
        scales: Per-block scale factors of shape (N // block_size,)
        block_size: Number of elements per quantization block
        
    Returns:
        Dequantized tensor of shape (N,)
    """
    N = quantized.shape[0]
    num_blocks = N // block_size
    # scales 扩展回 (num_blocks, block_size)，每块里每个元素共享同一个 scale
    expanded_scales = scales.unsqueeze(1).expand(num_blocks, block_size)
    # 反量化 = 量化值 × scale
    dequantized = quantized.reshape(num_blocks, block_size) * expanded_scales
    return dequantized.reshape(N)