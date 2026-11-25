import torch

def moe_load_balancing_loss(gate_logits, num_experts: int, alpha: float = 0.01) -> float:
    """
    Compute the load balancing auxiliary loss for a Mixture of Experts layer.
    
    Args:
        gate_logits: torch.Tensor of shape (num_tokens, num_experts), raw gating scores
        num_experts: int, number of experts
        alpha: float, scaling coefficient for the loss
    
    Returns:
        float: load balancing loss rounded to 4 decimal places
    """
    probs = torch.softmax(gate_logits, dim=-1)
    assignments = probs.argmax(dim=-1)
    one_hot = torch.nn.functional.one_hot(assignments, num_experts).to(probs.dtype)

    # f_i: fraction of tokens whose top-1 expert is i
    f = one_hot.mean(dim=0)
    # P_i: mean router probability assigned to expert i over all tokens
    p = probs.mean(dim=0)

    loss = num_experts * (f * p).sum()
    return round(alpha * loss.item(), 4)