import torch

def multiquery_attention(X: torch.Tensor, W_queries: list, W_key: torch.Tensor, W_value: torch.Tensor, W_out: torch.Tensor) -> torch.Tensor:
    """
    Compute Multi-Query Attention.
    
    Args:
        X: Input tensor of shape (seq_len, d_model)
        W_queries: List of query weight tensors, each (d_model, d_k), one per head
        W_key: Shared key weight tensor of shape (d_model, d_k)
        W_value: Shared value weight tensor of shape (d_model, d_v)
        W_out: Output projection tensor of shape (num_heads * d_v, d_model)
    
    Returns:
        Output tensor of shape (seq_len, d_model), rounded to 4 decimal places
    """
    n_heads = len(W_queries)
    seq_len = X.shape[0]
    d_k = W_key.shape[1]
    d_v = W_value.shape[1]

    # Q：每个头各自投影 -> (seq_len, n_heads, d_k)
    Q = torch.stack([X @ Wq for Wq in W_queries], dim=1)

    # K/V：只投影一份，所有头共享 -> (seq_len, d_k) / (seq_len, d_v)
    K = X @ W_key
    V = X @ W_value

    # 每个头的 Q 与同一份 K 做点积：(seq_len, n_heads, d_k) @ (d_k, seq_len)
    # 广播后 -> (seq_len, n_heads, seq_len)
    scores = Q @ K.T / (d_k ** 0.5)
    attn = torch.softmax(scores, dim=-1)

    # 所有头对同一份 V 加权求和：(seq_len, n_heads, seq_len) @ (seq_len, d_v)
    # 广播后 -> (seq_len, n_heads, d_v)
    out = attn @ V

    # 拼接所有头并做输出投影
    return out.reshape(seq_len, n_heads * d_v) @ W_out