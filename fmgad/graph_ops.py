"""Small graph operations shared by training and scoring."""

import torch


def smooth_scores_by_graph(
    score: torch.Tensor,
    edge_index: torch.Tensor,
    alpha: float,
    device: torch.device,
) -> torch.Tensor:
    """Blend a score with the mean score of incoming neighbors."""
    if alpha <= 0.0 or edge_index.numel() == 0:
        return score
    src, dst = edge_index[0], edge_index[1]
    n = score.size(0)
    neigh_sum = torch.zeros(n, device=device, dtype=score.dtype)
    neigh_sum.index_add_(0, dst, score[src])
    deg = torch.zeros(n, device=device, dtype=score.dtype)
    deg.index_add_(0, dst, torch.ones_like(score[src]))
    neigh_mean = neigh_sum / deg.clamp_min(1.0)
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
        dst = edge_index[1]
        deg = torch.zeros(n, device=device, dtype=torch.long)
        deg.scatter_add_(0, dst, torch.ones(edge_index.size(1), device=device, dtype=torch.long))
        low_degree_nodes = [
            (node, degree) for node, degree in enumerate(deg.tolist())
            if degree < degree_threshold
        ]
        if not low_degree_nodes:
            return edge_index

        incoming = [set() for _ in range(n)]
        sources, destinations = edge_index[:2].tolist()
        for s, d in zip(sources, destinations):
            incoming[d].add(s)

        h_norm = torch.nn.functional.normalize(h, p=2, dim=1)
        sim = torch.mm(h_norm, h_norm.t())
        sim.fill_diagonal_(-1e9)
        _, idx = sim.topk(min(k, n - 1), dim=1)
        neighbors = idx.tolist()

        new_edges = []
        for i, degree in low_degree_nodes:
            need = min(k, degree_threshold - degree)
            if need <= 0:
                continue
            added = 0
            for j in neighbors[i]:
                if j == i or j in incoming[i]:
                    continue
                new_edges.append([j, i])
                added += 1
                if added >= need:
                    break
        if not new_edges:
            return edge_index

        new_edges = torch.tensor(new_edges, device=device, dtype=edge_index.dtype).t()
        combined = torch.cat([edge_index, new_edges], dim=1)
        return torch.unique(combined, dim=1)
