import torch

def sliding_window_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    Compute sliding window attention.
    
    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_v)
        window_size: Number of positions to the left and right each query can attend to
    
    Returns:
        Output matrix of shape (seq_len, d_v), rounded to 4 decimal places.
    """
    score = Q @ K.transpose(-2, -1) / (K.size(-1) ** 0.5)
    mask = torch.ones_like(score, dtype=torch.bool)
    mask = mask.triu(-window_size)
    mask = mask.tril(window_size)
    score = score.masked_fill(~mask, float('-inf'))
    attn = torch.softmax(score, dim=-1)
    output = torch.round(attn @ V, decimals=4)
    return output

