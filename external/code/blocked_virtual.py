"""Keep the original virtual edges while bounding the intermediate cosine matrix."""
import torch


def blocked_edges(edge_index, h, degree_threshold, k, device):
    n = h.shape[0]
    if n > 50000:
        return edge_index
    with torch.no_grad():
        src, dst = edge_index
        deg = torch.bincount(dst, minlength=n)
        if not bool((deg < degree_threshold).any()):
            return edge_index

        normalized = torch.nn.functional.normalize(h, p=2, dim=1)
        candidates = []
        for start in range(0, n, 512):
            stop = min(start + 512, n)
            sim = normalized[start:stop] @ normalized.T
            local = torch.arange(stop - start, device=device)
            sim[local, local + start] = -1e9
            candidates.append(sim.topk(min(k, n - 1), dim=1).indices)
        idx = torch.cat(candidates)

        target = torch.arange(n, device=device)[:, None].expand_as(idx)
        existing = torch.unique(dst * n + src, sorted=True)
        candidate = target * n + idx
        if existing.numel():
            loc = torch.searchsorted(existing, candidate.contiguous())
            present = (loc < existing.numel()) & (existing[loc.clamp_max(existing.numel() - 1)] == candidate)
        else:
            present = torch.zeros_like(idx, dtype=torch.bool)
        valid = (~present) & (idx != target) & (deg < degree_threshold)[:, None]
        need = (degree_threshold - deg).clamp(min=0, max=k)
        take = valid & (valid.cumsum(dim=1) <= need[:, None])
        if not bool(take.any()):
            return edge_index
        new_edges = torch.stack([idx[take], target[take]]).to(edge_index.dtype)
        return torch.unique(torch.cat([edge_index, new_edges], dim=1), dim=1)


def install():
    from fmgad import detector
    detector.add_virtual_knn_edges = blocked_edges


def selftest():
    from fmgad.graph_ops import add_virtual_knn_edges
    torch.manual_seed(83)
    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for n in (3, 811, 1579, 2459):
        e = torch.randint(n, (2, 4 * n), device=device)
        for h in (torch.randn(n, 64, device=device), torch.zeros(n, 11, device=device)):
            expected = add_virtual_knn_edges(e, h, 5, 5, device)
            actual = blocked_edges(e, h, 5, 5, device)
            if not torch.equal(actual, expected):
                raise ValueError(f'Virtual edge mismatch n={n} {device}')
    print(f'OK virtual-edge vectors, {device}')


if __name__ == '__main__':
    selftest()
