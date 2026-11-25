import torch


def compute_dual_residuals_with_degree(h: torch.Tensor, edge_index: torch.Tensor, *, stable=False):
    """
    Dual residuals and node degree for adaptive gating.

    Returns:
        r_global: [N, D] deviation from graph-wide mean
        r_local:  [N, D] deviation from neighbor mean
        deg:      [N, 1] node degree
    """
    if h.dim() != 2:
        raise ValueError(f"h should be [N, D], got {tuple(h.shape)}")
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError(f"edge_index should be [2, E], got {tuple(edge_index.shape)}")

    if stable:
        # Center before the reduction so constant embeddings give exact zeros.
        values = h.double()
        centered = values - values[:1]
        global_residual = centered - centered.mean(dim=0, keepdim=True)
        src, dst = edge_index
        degree = torch.bincount(dst, minlength=h.size(0)).unsqueeze(1)
        differences = torch.zeros_like(values)
        chunk_size = max(1, 1_000_000 // max(1, h.size(1)))
        for start in range(0, src.numel(), chunk_size):
            targets = dst[start:start + chunk_size]
            sources = src[start:start + chunk_size]
            differences.index_add_(0, targets, values[targets] - values[sources])
        # The historical empty neighborhood has mean zero and local residual h_i.
        local_residual = torch.where(degree > 0, differences / degree.clamp_min(1), values)
        return global_residual.to(h.dtype), local_residual.to(h.dtype), degree.to(h.dtype)

    global_mean = torch.mean(h, dim=0, keepdim=True)
    r_global = h - global_mean

    src, dst = edge_index[0], edge_index[1]
    n, d = h.size(0), h.size(1)
    neigh_sum = torch.zeros((n, d), device=h.device, dtype=h.dtype)
    neigh_sum.index_add_(0, dst, h[src])

    deg_val = torch.zeros((n,), device=h.device, dtype=h.dtype)
    deg_val.index_add_(0, dst, torch.ones_like(dst, dtype=h.dtype))

    deg_clamped = deg_val.clamp_min(1.0).unsqueeze(1)
    neigh_mean = neigh_sum / deg_clamped
    r_local = h - neigh_mean

    return r_global, r_local, deg_val.unsqueeze(1)
