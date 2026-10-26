import torch

def moe(x: torch.Tensor, We: torch.Tensor, Wg: torch.Tensor, n_experts: int, top_k: int) -> torch.Tensor:
    """
    Args:
        x: Input tensor of shape (n_batch, l_seq, d_model)
        We: Expert weights of shape (n_experts, d_model, d_model)
        Wg: Gating weights of shape (d_model, n_experts)
        n_experts: Number of experts
        top_k: Number of experts to route each token to
    Returns:
        Output tensor of shape (n_batch, l_seq, d_model)
    """
    n_batch, l_seq, d_model = x.shape
    total = n_batch * l_seq
    x_flat = x.view(total, d_model)               # 拍平: (total, d_model)

    # 门控打分 + softmax
    gate_logits = x_flat @ Wg                      # (total, n_experts)
    gate = torch.softmax(gate_logits, dim=-1)      # 每行归一化

    # Top-k 路由
    topk_val, topk_idx = torch.topk(gate, k=top_k, dim=-1)
    gate = torch.zeros_like(gate).scatter_(1, topk_idx, topk_val)  # 只保留 top_k 的权重
    gate = gate / gate.sum(dim=-1, keepdim=True)   # 重新归一化

    # 稀疏计算：对每个专家，收集分给它的 token，批量处理
    out = torch.zeros_like(x_flat)
    for j in range(n_experts):
        mask_j = gate[:, j] > 0                    # 哪些 token 给了专家 j
        if not mask_j.any():
            continue
        x_j = x_flat[mask_j]                       # (n_j, d_model)
        g_j = gate[mask_j, j:j+1]                  # (n_j, 1)
        out[mask_j] += (x_j @ We[j]) * g_j         # 专家输出 × 门控权重

    return out.view(n_batch, l_seq, d_model)