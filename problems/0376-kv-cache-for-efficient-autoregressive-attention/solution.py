import torch

def kv_cache_attention_step(x_new: torch.Tensor, W_Q: torch.Tensor, W_K: torch.Tensor, W_V: torch.Tensor, cache: tuple) -> tuple:
    """
    Perform a single attention step with KV caching.
    
    Args:
        x_new: New token embedding, shape (d_model,)
        W_Q: Query projection matrix, shape (d_model, d_k)
        W_K: Key projection matrix, shape (d_model, d_k)
        W_V: Value projection matrix, shape (d_model, d_v)
        cache: Tuple (K_cache, V_cache) of tensors or None if first step
    
    Returns:
        Tuple (output, updated_cache) where output is shape (d_v,)
        and updated_cache is (K_new, V_new)
    """
    # Project new token to query, key, value space (均为 1-D 向量)
    q = torch.matmul(x_new, W_Q)
    k = torch.matmul(x_new, W_K)
    v = torch.matmul(x_new, W_V)

    # 把新 token 的 K/V 拼进缓存 (第一步时缓存为空, unsqueeze 成 (1, d) 再拼)
    if cache is None:
        K = k.unsqueeze(0)
        V = v.unsqueeze(0)
    else:
        K = torch.cat((cache[0], k.unsqueeze(0)), dim=0)
        V = torch.cat((cache[1], v.unsqueeze(0)), dim=0)

    # 新 token 的 query 对缓存里的所有 key 做注意力: scores 形状 (seq_len,)
    scores = torch.matmul(K, q) / (q.shape[-1] ** 0.5)

    # Softmax 得到注意力权重, 再对 V 加权求和: 输出形状 (d_v,)
    p = torch.softmax(scores, dim=-1)
    output = torch.matmul(p, V)

    return output, (K, V)