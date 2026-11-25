"""Small graph operations shared by training and scoring."""

import torch


def smooth_scores_by_graph(
    score: torch.Tensor,
    edge_index: torch.Tensor,
    alpha: float,
    device: torch.device,
    *,
    empty_neighbor_policy: str = "legacy_zero",
) -> torch.Tensor:
    """Blend scores with incoming neighbors, declaring missing-data semantics.

    ``self`` preserves a node's score when no neighbor is observed. It also
    preserves constant signals on every graph. ``legacy_zero`` reproduces the
    historical attenuation of isolated nodes and is not recommended for new
    normal-reference models.
    """
    if empty_neighbor_policy not in {"self", "legacy_zero"}:
        raise ValueError("empty_neighbor_policy must be 'self' or 'legacy_zero'")
    if alpha <= 0.0 or edge_index.numel() == 0:
        return score
    src, dst = edge_index[0], edge_index[1]
    n = score.size(0)
    neigh_sum = torch.zeros(n, device=device, dtype=score.dtype)
    neigh_sum.index_add_(0, dst, score[src])
    deg = torch.zeros(n, device=device, dtype=score.dtype)
    deg.index_add_(0, dst, torch.ones_like(score[src]))
    neigh_mean = neigh_sum / deg.clamp_min(1.0)
    if empty_neighbor_policy == "self":
        neigh_mean = torch.where(deg > 0, neigh_mean, score)
    return (1.0 - alpha) * score + alpha * neigh_mean


def add_virtual_knn_edges(
    edge_index: torch.Tensor,
    h: torch.Tensor,
    degree_threshold: int,
    k: int,
    device: torch.device,
) -> torch.Tensor:
    """Append embedding-space kNN incoming edges for low in-degree nodes."""
    n = h.size(0)
    if n > 50000:
        return edge_index
    with torch.no_grad():
        src, dst = edge_index[0], edge_index[1]
        deg = torch.zeros(n, device=device, dtype=torch.long)
        deg.scatter_add_(0, dst, torch.ones(edge_index.size(1), device=device, dtype=torch.long))
        low_deg_mask = deg < degree_threshold
        if low_deg_mask.sum() == 0:
            return edge_index

        h_norm = torch.nn.functional.normalize(h, p=2, dim=1)
        sim = torch.mm(h_norm, h_norm.t())
        sim.fill_diagonal_(-1e9)
        _, idx = sim.topk(min(k, n - 1), dim=1)
        # Preserve the historical top-k candidate pool and tie order. Only
        # replace per-node CUDA synchronization and Python set membership.
        target = torch.arange(n, device=device)[:, None].expand_as(idx)
        existing = torch.unique(dst * n + src, sorted=True)
        candidate = target * n + idx
        if existing.numel():
            loc = torch.searchsorted(existing, candidate.contiguous())
            present = (loc < existing.numel()) & (existing[loc.clamp_max(existing.numel() - 1)] == candidate)
        else:
            present = torch.zeros_like(idx, dtype=torch.bool)
        valid = (~present) & (idx != target) & low_deg_mask[:, None]
        need = (degree_threshold - deg).clamp(min=0, max=k)
        take = valid & (valid.cumsum(dim=1) <= need[:, None])
        if not bool(take.any()):
            return edge_index
        new_edges = torch.stack([idx[take], target[take]]).to(edge_index.dtype)
        combined = torch.cat([edge_index, new_edges], dim=1)
        return torch.unique(combined, dim=1)
