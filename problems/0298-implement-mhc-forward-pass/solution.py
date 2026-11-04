import torch

def mhc_forward(
    x: torch.Tensor,
    H_pre_raw: torch.Tensor,
    H_post_raw: torch.Tensor,
    H_res_raw: torch.Tensor,
    layer_output: torch.Tensor,
    sinkhorn_iters: int = 5
) -> torch.Tensor:
    """
    Compute mHC forward pass with manifold-constrained mappings.
    
    The mHC formula is:
        x_out = H_res @ x + H_post.T @ layer_output
    
    Where:
        - H_pre (not used here, already applied to get layer_output)
        - H_post = 2 * sigmoid(H_post_raw)  (non-negative constraint)
        - H_res = Sinkhorn(exp(H_res_raw))  (doubly stochastic constraint)
    
    Args:
        x: Hidden states, shape (n, C) where n is num streams
        H_pre_raw: Raw pre-mapping coefficients, shape (1, n) - not used in this step
        H_post_raw: Raw post-mapping coefficients, shape (1, n)
        H_res_raw: Raw residual mapping coefficients, shape (n, n)
        layer_output: Output from layer F, shape (1, C)
        sinkhorn_iters: Number of Sinkhorn iterations
    
    Returns:
        x_out: Updated hidden states, shape (n, C)
    """
    # Post-mapping: non-negative coefficients in (0, 2)
    H_post = 2 * torch.sigmoid(H_post_raw)  # (1, n)

    # Residual mapping: doubly stochastic via Sinkhorn(exp(raw))
    H_res = torch.exp(H_res_raw)  # (n, n), strictly positive
    for _ in range(sinkhorn_iters):
        H_res = H_res / H_res.sum(dim=1, keepdim=True)  # row-normalize
        H_res = H_res / H_res.sum(dim=0, keepdim=True)  # col-normalize

    x_out = H_res @ x + H_post.T @ layer_output  # (n,n)@(n,C) + (n,1)@(1,C)
    return x_out