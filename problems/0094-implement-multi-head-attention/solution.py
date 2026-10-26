import torch
import torch.nn.functional as F
from typing import Tuple

def compute_qkv(X: torch.Tensor, W_q: torch.Tensor, W_k: torch.Tensor, W_v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute Query, Key, and Value matrices.

    Args:
        X: Input matrix of shape (seq_len, d_model)
        W_q, W_k, W_v: Weight matrices of shape (d_model, d_model)

    Returns:
        Q, K, V matrices each of shape (seq_len, d_model)
    """
    # Your code here
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    return Q, K, V

def self_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product self-attention.

    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_k)

    Returns:
        Attention output of shape (seq_len, d_k)
    """
    # Your code here
    score = Q @ K.transpose(-2, -1) / (Q.shape[-1])**0.5
    attention = F.softmax(score, dim=-1)
    return attention @ V

def multi_head_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, n_heads: int) -> torch.Tensor:
    """
    Compute multi-head attention.

    Args:
        Q, K, V: Matrices of shape (seq_len, d_model)
        n_heads: Number of attention heads

    Returns:
        Attention output of shape (seq_len, d_model)
    """
    # Your code here
    d_model = Q.shape[-1]
    d_k = d_model // n_heads

    # 拆分多头: (seq_len, d_model) → (n_heads, seq_len, d_k)
    Q = Q.view(-1, n_heads, d_k).transpose(0, 1)
    K = K.view(-1, n_heads, d_k).transpose(0, 1)
    V = V.view(-1, n_heads, d_k).transpose(0, 1)

    # 并行做缩放点积注意力
    attn = self_attention(Q, K, V)   # (n_heads, seq_len, d_k)

    # 拼回: (n_heads, seq_len, d_k) → (seq_len, d_model)
    attn = attn.transpose(0, 1).contiguous().view(-1, d_model)
    return attn
