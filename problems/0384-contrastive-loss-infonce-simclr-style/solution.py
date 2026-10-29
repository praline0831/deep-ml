import torch
import torch.nn.functional as F

def contrastive_loss(embeddings: torch.Tensor, temperature: float) -> float:
    """
    Compute the NT-Xent (SimCLR-style) contrastive loss.

    Args:
        embeddings: Tensor of shape (2N, d) where consecutive pairs
                    (2i, 2i+1) are positive pairs.
        temperature: Temperature scaling parameter (tau > 0).

    Returns:
        The mean contrastive loss as a float.
    """
    n = embeddings.shape[0]  # 2N

    # 1. L2 归一化: 归一化后点积 == 余弦相似度
    z = F.normalize(embeddings, dim=-1)

    # 2. 相似度矩阵 (2N, 2N), 除以温度 tau
    sim = z @ z.T / temperature

    # 3. 屏蔽对角线 (自己和自己的相似度): exp(-inf)=0, 等价于从分母中排除
    sim.fill_diagonal_(float('-inf'))

    # 4. 每个样本的正样本下标: 相邻配对 (2i, 2i+1), XOR 1 即翻转末位
    pos_idx = torch.arange(n) ^ 1

    # 5. NT-Xent: L_i = -log( exp(s_i,pos) / sum_{k!=i} exp(s_i,k) )
    pos_sim = sim[torch.arange(n), pos_idx]          # (2N,) 各自正对的相似度
    loss = -pos_sim + torch.logsumexp(sim, dim=-1)   # logsumexp 数值稳定

    return float(loss.mean())