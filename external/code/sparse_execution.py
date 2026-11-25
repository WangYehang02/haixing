"""Cache static graph operators and aggregate on sparse matrices for wide graphs."""
import weakref

import torch

import sparse_gcn


class GraphCache:
    def __init__(self):
        self.edge_ref = None
        self.key = None
        self.value = None

    def get(self, edges, flags, build):
        try:
            key = (edges._version, flags)
        except RuntimeError:
            return build()
        if self.edge_ref is not None and self.edge_ref() is edges and self.key == key:
            return self.value
        self.value = build()
        self.edge_ref = weakref.ref(edges)
        self.key = key
        return self.value


_gcn = GraphCache()
_residual = GraphCache()


def adjacency(edges, weights, n):
    return torch.sparse_coo_tensor(edges, weights, (n, n), device=edges.device).coalesce().to_sparse_csr()


def cache_gcn(layer, x, edges, edge_weight=None):
    if (edge_weight is not None or layer.cached or not layer.normalize or
            not sparse_gcn._supported(layer, x, edges, edge_weight)):
        return sparse_gcn._csr_forward(layer, x, edges, edge_weight)

    n = x.shape[0]

    def build():
        index, weight = sparse_gcn.gcn_norm(edges, None, n, layer.improved,
                                            layer.add_self_loops, layer.flow, x.dtype)
        if layer.flow == 'source_to_target':
            index = index.flip(0)
        return adjacency(index, weight, n)

    flags = (n, x.dtype, layer.improved, layer.add_self_loops, layer.flow)
    mat = _gcn.get(edges, flags, build)
    result = torch.sparse.mm(mat, layer.lin(x))
    return result if layer.bias is None else result + layer.bias


def residual(h, edges):
    n = h.shape[0]

    def build():
        weights = torch.ones(edges.shape[1], dtype=h.dtype, device=h.device)
        mat = adjacency(edges.flip(0), weights, n)
        degree = torch.bincount(edges[1], minlength=n).to(h.dtype).unsqueeze(1)
        return mat, degree

    mat, degree = _residual.get(edges, (n, h.dtype), build)
    global_residual = h - h.mean(dim=0, keepdim=True)
    local_residual = h - torch.sparse.mm(mat, h) / degree.clamp_min(1)
    return global_residual, local_residual, degree


def install():
    from fmgad import detector
    sparse_gcn.GCNConv.forward = cache_gcn
    detector.compute_dual_residuals_with_degree = residual


def selftest():
    from fmgad.residuals import compute_dual_residuals_with_degree
    from torch_geometric.nn import GCNConv
    torch.manual_seed(19)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    edges = torch.randint(94, (2, 1502), device=device)
    layer = GCNConv(17, 11).to(device)
    inputs = torch.randn(96, 17, device=device, requires_grad=True)
    for _ in range(2):
        for x, y in zip(compute_dual_residuals_with_degree(inputs, edges), residual(inputs, edges)):
            torch.testing.assert_close(x, y, atol=5e-7, rtol=2e-6)
        expected = sparse_gcn._ORIGINAL_FORWARD(layer, inputs, edges)
        actual = cache_gcn(layer, inputs, edges)
        torch.testing.assert_close(expected, actual, atol=8e-7, rtol=2e-6)
        ex = torch.autograd.grad(expected.square().sum(), inputs)[0]
        ac = torch.autograd.grad(actual.square().sum(), inputs)[0]
        torch.testing.assert_close(ex, ac, atol=1e-6, rtol=1e-5)
        edges[0,0] = (edges[0,0]+1) % 94
    print(f'OK cache, direction, duplicates, inputs and gradients on {device}')


if __name__ == '__main__':
    selftest()
