import torch

def flash_attention_forward(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, 
                           block_size: int = 2) -> torch.Tensor:
    """
    Compute attention output using Flash Attention v1 algorithm.
    
    Args:
        Q: Query matrix (seq_len, d_model)
        K: Key matrix (seq_len, d_model)
        V: Value matrix (seq_len, d_model)
        block_size: Size of blocks for tiled computation
    
    Returns:
        Output matrix (seq_len, d_model)
    """
    N, d = Q.shape
    scale = d ** -0.5

    num_q = (N + block_size - 1) // block_size
    num_k = (N + block_size - 1) // block_size

    O = torch.zeros_like(Q)
    m = torch.full((N,), float('-inf'), device=Q.device)
    l = torch.zeros(N, device=Q.device)

    for i in range(num_q):
        q_s, q_e = i * block_size, min((i + 1) * block_size, N)
        Qb = Q[q_s:q_e]
        Ob = O[q_s:q_e]
        mb = m[q_s:q_e]
        lb = l[q_s:q_e]

        for j in range(num_k):
            k_s, k_e = j * block_size, min((j + 1) * block_size, N)
            Kb = K[k_s:k_e]
            Vb = V[k_s:k_e]

            S = Qb @ Kb.T * scale                                  # (Br, Bc)
            m_new = torch.maximum(mb, S.max(dim=-1).values)        # (Br,)

            # 重缩放已累加项：exp(old_m - new_m)
            Ob = Ob * torch.exp(mb - m_new).unsqueeze(-1)          # (Br, d)
            P = torch.exp(S - m_new.unsqueeze(-1))                 # (Br, Bc)
            Ob = Ob + P @ Vb                                       # (Br, d)
            lb = lb * torch.exp(mb - m_new) + P.sum(dim=-1)        # (Br,)
            mb = m_new

        O[q_s:q_e] = Ob / lb.unsqueeze(-1)
        m[q_s:q_e] = mb
        l[q_s:q_e] = lb

    return O
