import torch

def moe_shared_expert_forward(
    X: torch.Tensor,
    W_gate: torch.Tensor,
    W_shared: torch.Tensor,
    W_experts: list,
    top_k: int = 2
) -> dict:
    """
    Forward pass of a Mixture of Experts layer with a shared expert.
    
    Args:
        X: Input tokens, shape (num_tokens, d_model)
        W_gate: Gating weights, shape (d_model, num_routed_experts)
        W_shared: Shared expert weights, shape (d_model, d_out)
        W_experts: List of routed expert weight tensors, each (d_model, d_out)
        top_k: Number of routed experts per token
    
    Returns:
        Dictionary with keys: 'output', 'shared_output', 'routed_output',
                              'routing_indices', 'routing_weights'
    """
    num_tokens = X.shape[0]
    d_out = W_shared.shape[1]

    # 1. 共享专家：所有 token 都经过，提供通用基础表示
    shared_output = X @ W_shared  # (num_tokens, d_out)

    # 2. 路由：计算 logits 并选出 top-k 专家
    router_logits = X @ W_gate  # (num_tokens, num_experts)
    routing_weights, routing_indices = torch.topk(router_logits, top_k, dim=-1)
    # 权重只对被选中的 expert logits 做 softmax，保留 router 的置信度差异
    routing_weights = torch.softmax(routing_weights, dim=-1)  # (num_tokens, top_k)

    # 3. 路由专家：按 token 加权聚合被选中的专家输出
    routed_output = torch.zeros(num_tokens, d_out, device=X.device, dtype=X.dtype)
    for k in range(top_k):
        expert_idx = routing_indices[:, k]              # (num_tokens,)
        w = routing_weights[:, k].unsqueeze(-1)         # (num_tokens, 1)
        for e in range(len(W_experts)):
            mask = expert_idx == e
            if mask.any():
                routed_output[mask] += w[mask] * (X[mask] @ W_experts[e])

    # 4. 共享专家 + 路由专家
    output = shared_output + routed_output

    return {
        "output": output,
        "shared_output": shared_output,
        "routed_output": routed_output,
        "routing_indices": routing_indices,
        "routing_weights": routing_weights,
    }