"""Opt-in memory-only replacement for vanilla edge-list PyG GCNConv.

Call install() before running a model. Normalization is the installed PyG's
unchanged edge-list gcn_norm, BEFORE duplicate summation. Only propagation is
replaced by CSR @ self.lin(x), avoiding an E-by-output-features message tensor.
Floating-point reduction order changes; mathematical, not bitwise, equivalence
is intended. This module does not modify graph inputs or model parameters.

Reference: https://pytorch-geometric.readthedocs.io/en/2.6.1/_modules/
torch_geometric/nn/conv/gcn_conv.html
"""
import argparse
import copy
import json

import torch
from torch_geometric.nn import GCNConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm

_ORIGINAL_FORWARD = getattr(GCNConv.forward, "_fmgad_original_forward", GCNConv.forward)
_HOOKS = ("_propagate_forward_pre_hooks", "_propagate_forward_hooks",
          "_message_forward_pre_hooks", "_message_forward_hooks",
          "_aggregate_forward_pre_hooks", "_aggregate_forward_hooks",
          "_message_and_aggregate_forward_pre_hooks", "_message_and_aggregate_forward_hooks",
          "_edge_update_forward_pre_hooks", "_edge_update_forward_hooks")


def _supported(layer, x, edge_index, edge_weight):
    if type(layer) is not GCNConv or not isinstance(x, torch.Tensor):
        return False
    if x.layout != torch.strided or x.ndim != 2 or x.dtype not in (torch.float32, torch.float64):
        return False
    if x.device.type not in ("cpu", "cuda") or layer.node_dim not in (-2, 0):
        return False
    if layer.aggr not in ("add", "sum") or layer.flow not in ("source_to_target", "target_to_source"):
        return False
    if getattr(layer, "explain", False) or any(getattr(layer, h, {}) for h in _HOOKS):
        return False
    if torch.is_autocast_enabled() or (x.device.type == "cpu" and torch.is_autocast_cpu_enabled()):
        return False
    if not isinstance(edge_index, torch.Tensor) or edge_index.layout != torch.strided:
        return False
    if edge_index.dtype != torch.long or edge_index.ndim != 2 or edge_index.shape[0] != 2:
        return False
    if edge_index.device != x.device:
        return False
    if edge_weight is not None:
        if not isinstance(edge_weight, torch.Tensor) or edge_weight.requires_grad:
            return False
        if (edge_weight.layout != torch.strided or edge_weight.ndim != 1 or
                len(edge_weight) != edge_index.shape[1] or edge_weight.dtype != x.dtype or
                edge_weight.device != x.device):
            return False
    cache = getattr(layer, "_cached_edge_index", None)
    if layer.normalize and cache is not None:
        cached_edge, cached_weight = cache
        if (cached_edge.layout != torch.strided or cached_edge.device != x.device or
                cached_weight is None or cached_weight.requires_grad or
                cached_weight.dtype != x.dtype or cached_weight.device != x.device):
            return False
    return True


def _csr_forward(self, x, edge_index, edge_weight=None):
    if not _supported(self, x, edge_index, edge_weight):
        return _ORIGINAL_FORWARD(self, x, edge_index, edge_weight)

    n_nodes = x.size(self.node_dim)
    if self.normalize:
        cache = self._cached_edge_index
        if cache is None:
            # Deliberately use the exact Tensor/edge-list branch. Converting to
            # sparse first would change improved/self-loop/duplicate semantics.
            edge_index, edge_weight = gcn_norm(
                edge_index, edge_weight, n_nodes, self.improved,
                self.add_self_loops, self.flow, x.dtype)
            if self.cached:
                self._cached_edge_index = (edge_index, edge_weight)
        else:
            edge_index, edge_weight = cache
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.shape[1], dtype=x.dtype, device=x.device)

    # PyG source_to_target sends x[source] to destination. CSR rows therefore
    # index destinations. Reverse flow swaps those roles. Summation of parallel
    # weighted edges retains their arithmetic contribution, including loops.
    indices = edge_index.flip(0) if self.flow == "source_to_target" else edge_index
    adjacency = torch.sparse_coo_tensor(
        indices, edge_weight, size=(n_nodes, n_nodes),
        dtype=x.dtype, device=x.device).coalesce().to_sparse_csr()
    transformed = self.lin(x)
    out = torch.sparse.mm(adjacency, transformed)
    if self.bias is not None:
        out = out + self.bias
    return out


_csr_forward._fmgad_original_forward = _ORIGINAL_FORWARD


def install():
    """Explicit, idempotent process-local patch; importing alone changes nothing."""
    already = getattr(GCNConv.forward, "_fmgad_original_forward", None) is not None
    if not already:
        GCNConv.forward = _csr_forward
    return {"installed": True, "already_installed": already,
            "implementation": "edge-list gcn_norm then coalesced CSR @ linear(x) + bias",
            "adjacency_gradients": "unsupported: original GCNConv fallback",
            "equivalence": "same arithmetic; floating-point reduction order may differ"}


def selftest(device="cpu"):
    """Tiny forward/backward QA; no files, no labels, no implicit GPU use."""
    import torch_geometric
    torch.manual_seed(1729)
    device = torch.device(device)
    if device.type == "cuda":
        # The memory-stat APIs may reject an otherwise valid device before its
        # CUDA context exists. Initialize explicitly before the tiny QA meter.
        torch.cuda.set_device(device)
        torch.empty(0, device=device)
        torch.cuda.reset_peak_memory_stats(device)
    n, incoming, outgoing = 11, 7, 5
    # Directed random graph, parallel edges, repeated explicit loops; node 10
    # is isolated before normalization. Positive weights keep degrees valid.
    edge = torch.randint(0, n - 1, (2, 39), device=device)
    edge = torch.cat([edge, edge[:, :5], torch.tensor([[0, 0, 3], [0, 0, 3]], device=device)], dim=1)
    all_cases, worst = [], {"forward": 0., "input_grad": 0., "parameter_grad": 0.}
    for dtype in (torch.float64, torch.float32):
        tol = dict(atol=2e-11, rtol=2e-10) if dtype == torch.float64 else dict(atol=3e-6, rtol=3e-5)
        configs = [dict(normalize=True, add_self_loops=True, improved=False),
                   dict(normalize=True, add_self_loops=True, improved=True),
                   dict(normalize=True, add_self_loops=False, improved=True),
                   dict(normalize=False, add_self_loops=False, improved=False)]
        for config in configs:
            for flow in ("source_to_target", "target_to_source"):
                for weighted in (False, True):
                    for training in (False, True):
                        original = GCNConv(incoming, outgoing, flow=flow, **config).to(device=device, dtype=dtype)
                        original.train(training)
                        with torch.no_grad():
                            original.bias.uniform_(-.2, .2)
                        patched = copy.deepcopy(original)
                        x1 = torch.randn(n, incoming, device=device, dtype=dtype, requires_grad=True)
                        x2 = x1.detach().clone().requires_grad_()
                        weight = torch.rand(edge.shape[1], dtype=dtype, device=device) + .2 if weighted else None
                        a = _ORIGINAL_FORWARD(original, x1, edge, weight)
                        b = _csr_forward(patched, x2, edge, weight)
                        torch.testing.assert_close(a, b, **tol)
                        worst["forward"] = max(worst["forward"], float((a - b).abs().max()))
                        # Arbitrary shared upstream gradient, checks dense x,
                        # self.lin.weight and nonzero bias, in train/eval modes.
                        upstream = torch.randn_like(a)
                        (a * upstream).sum().backward()
                        (b * upstream).sum().backward()
                        torch.testing.assert_close(x1.grad, x2.grad, **tol)
                        worst["input_grad"] = max(worst["input_grad"], float((x1.grad - x2.grad).abs().max()))
                        for p, q in zip(original.parameters(), patched.parameters()):
                            torch.testing.assert_close(p.grad, q.grad, **tol)
                            worst["parameter_grad"] = max(worst["parameter_grad"], float((p.grad - q.grad).abs().max()))
                        all_cases.append(dict(dtype=str(dtype), flow=flow, weighted=weighted, training=training, **config))
    # Test the original cache policy over repeated calls and changing x.
    a = GCNConv(incoming, outgoing, cached=True, bias=False).to(device)
    b = copy.deepcopy(a)
    for _ in range(2):
        x = torch.randn(n, incoming, device=device)
        torch.testing.assert_close(_ORIGINAL_FORWARD(a, x, edge), _csr_forward(b, x, edge), atol=3e-6, rtol=3e-5)
    torch.testing.assert_close(a._cached_edge_index[0], b._cached_edge_index[0])
    torch.testing.assert_close(a._cached_edge_index[1], b._cached_edge_index[1])
    # Differentiable adjacency must execute the original implementation, with
    # gradients retained for its edge weights as well as input features.
    a = GCNConv(incoming, outgoing).to(device)
    b = copy.deepcopy(a)
    x1 = torch.randn(n, incoming, device=device, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_()
    w1 = (torch.rand(edge.shape[1], device=device) + .2).requires_grad_()
    w2 = w1.detach().clone().requires_grad_()
    assert not _supported(b, x2, edge, w2)
    out1 = _ORIGINAL_FORWARD(a, x1, edge, w1)
    out2 = _csr_forward(b, x2, edge, w2)
    out1.square().sum().backward()
    out2.square().sum().backward()
    torch.testing.assert_close(out1, out2)
    torch.testing.assert_close(w1.grad, w2.grad)
    torch.testing.assert_close(x1.grad, x2.grad)
    # The unsupported mean aggregator must not silently become a sum.
    mean_layer = GCNConv(incoming, outgoing, aggr="mean").to(device)
    assert not _supported(mean_layer, x1, edge, None)
    torch.testing.assert_close(_ORIGINAL_FORWARD(mean_layer, x1, edge), _csr_forward(mean_layer, x1, edge))
    # Verify empty edge lists and isolated-node behavior with and without loops.
    empty = torch.empty((2, 0), dtype=torch.long, device=device)
    for norm in (False, True):
        conv = GCNConv(incoming, outgoing, normalize=norm).to(device)
        torch.testing.assert_close(_ORIGINAL_FORWARD(conv, x1, empty), _csr_forward(conv, x1, empty))
    previous = GCNConv.forward
    try:
        first = install()
        assert install()["already_installed"]
        torch.testing.assert_close(mean_layer(x1, edge), _ORIGINAL_FORWARD(mean_layer, x1, edge))
        # Regular Module.__call__ still wraps the installed forward.
        conv = GCNConv(incoming, outgoing).to(device)
        torch.testing.assert_close(conv(x1, edge), _ORIGINAL_FORWARD(conv, x1, edge), atol=3e-6, rtol=3e-5)
    finally:
        GCNConv.forward = previous
    return {"status": "passed", "device": str(device), "torch": str(torch.__version__),
            "pyg": str(torch_geometric.__version__), "main_forward_backward_cases": len(all_cases),
            "max_absolute_errors": worst, "cache_and_edge_gradient_fallback": "passed",
            "empty_edges_and_unsupported_aggregation": "passed", "install_idempotence": "passed",
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if not args.self_test:
        parser.error("Use --self-test; production code calls install() explicitly")
    torch.set_num_threads(2)
    print(json.dumps(selftest(args.device), indent=2), flush=True)
