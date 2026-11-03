import torch

def gated_attention(
    X: torch.Tensor,
    W_q: torch.Tensor,
    W_k: torch.Tensor,
    W_v: torch.Tensor,
    W_g: torch.Tensor
) -> torch.Tensor:
    """
    Compute Gated Attention output.
    
    Args:
        X: Input tensor of shape (seq_len, d_model)
        W_q: Query projection of shape (d_model, d_k)
        W_k: Key projection of shape (d_model, d_k)
        W_v: Value projection of shape (d_model, d_v)
        W_g: Gate projection of shape (d_model, d_v)
    
    Returns:
        Gated attention output of shape (seq_len, d_v), rounded to 4 decimal places
    
    Hint: First compute standard scaled dot-product attention, then apply
    a sigmoid gate to modulate the output.
    """
    q, k, v = X @ W_q, X @ W_k, X @ W_v
    scores = q @ k.transpose(-2, -1) / (k.shape[-1]**0.5)
    attn_output = torch.softmax(scores, dim=-1) @ v
    gate = torch.sigmoid(X @ W_g)
    return gate * attn_output
   