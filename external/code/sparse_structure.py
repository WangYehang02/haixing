"""Exact all-pairs squared structure loss without storing an N-by-N matrix.

For the original unsigmoided DotProductDecoder, Q = decoder.nn(h, edges),
the squared reconstruction error of source row i is

    sum_j A_ij**2 + q_i.T @ (Q.T @ Q) @ q_i - 2*q_i.T @ (A @ Q)_i.

A is precisely the unweighted to_dense_adj target: parallel edges SUM, original
self loops remain, rows are sources, and absent edges are zero. This is not a
negative-edge sampler or a normalized-adjacency objective. Both occurrences of
Q retain autograd. Float64 accumulation changes rounding, not the objective.
The final structure error is cast to the decoder's dtype before the original
attribute/structure fusion. Opt in from detector via FMGAD_STRUCTURE_GRAM=1.
Separately, FMGAD_ZERO_NORM_SUBGRADIENT=1 selects the valid zero subgradient
of both reconstruction norms at exact zero; its forward values are unchanged.

This implementation is restricted to the original double_recon_loss defaults
pos_weight_a=pos_weight_s=.5, bce_s=False, and sigmoid_s=False. Near cancellation
is recomputed as an explicit squared residual in bounded blocks, never masked
by an epsilon or by clipping a negative squared error to zero.
"""
import argparse
import copy
import inspect
import json
import os
import time

import torch


class _ZeroNormSubgradient(torch.autograd.Function):
    """Unchanged sqrt forward; zero subgradient for the composed norm at zero.

    This is an explicit optimization convention for ||r||_2 when r=0, not an
    epsilon, a positive residual floor, or a remedy for nonfinite inputs.
    """
    @staticmethod
    def forward(ctx, squared):
        value = squared.sqrt()
        ctx.save_for_backward(value)
        return value

    @staticmethod
    def backward(ctx, gradient):
        value, = ctx.saved_tensors
        zero = value == 0
        denominator = torch.where(zero, torch.ones_like(value), 2 * value)
        return torch.where(zero, torch.zeros_like(gradient), gradient / denominator)


def norm_from_squared(squared):
    """Default is the original derivative; opt-in changes only exact zeros."""
    if os.environ.get("FMGAD_ZERO_NORM_SUBGRADIENT", "0") == "1":
        return _ZeroNormSubgradient.apply(squared)
    return squared.sqrt()


class StructureTarget:
    """Constant coalesced source-row target; no labels or graph preprocessing."""

    def __init__(self, edge_index, n_nodes):
        n_nodes = int(n_nodes)
        if n_nodes < 1:
            raise ValueError("Structure target needs at least one node")
        if (edge_index.dtype != torch.long or edge_index.ndim != 2 or
                edge_index.shape[0] != 2 or edge_index.layout != torch.strided):
            raise ValueError("Expected a strided long edge_index of shape (2,E)")
        if edge_index.numel() and (int(edge_index.min()) < 0 or int(edge_index.max()) >= n_nodes):
            raise ValueError("Edge endpoint outside graph")
        values = torch.ones(edge_index.shape[1], dtype=torch.float64, device=edge_index.device)
        coo = torch.sparse_coo_tensor(edge_index, values, (n_nodes, n_nodes)).coalesce()
        # Above 2**24, the legacy float32 to_dense_adj summation is no longer an
        # exact integer count. Reject rather than silently redefine that target.
        if coo.values().numel() and float(coo.values().max()) > 2**24:
            raise ValueError("Parallel edge multiplicity exceeds exact float32 target range")
        self.adjacency = coo.to_sparse_csr()
        self.row_squared_norm = torch.zeros(n_nodes, dtype=torch.float64, device=edge_index.device)
        self.row_squared_norm.index_add_(0, coo.indices()[0], coo.values().square())
        self.n_nodes = n_nodes
        self._cpu_csr = None

    def _direct_rows(self, q, rows, row_block=16, col_block=8192):
        """Rare cancellation repair, bounded memory and fully differentiable."""
        if self._cpu_csr is None:
            self._cpu_csr = (self.adjacency.crow_indices().cpu(),
                             self.adjacency.col_indices().cpu(),
                             self.adjacency.values().cpu())
        crow, columns, values = self._cpu_csr
        selected = rows.detach().cpu().tolist()
        chunks = []
        for start in range(0, len(selected), row_block):
            row_ids = selected[start:start + row_block]
            selected_q = q[row_ids]
            total = torch.zeros(len(row_ids), dtype=q.dtype, device=q.device)
            for left in range(0, self.n_nodes, col_block):
                right = min(self.n_nodes, left + col_block)
                target = torch.zeros((len(row_ids), right - left), dtype=q.dtype, device=q.device)
                for local_row, source_row in enumerate(row_ids):
                    lo, hi = int(crow[source_row]), int(crow[source_row + 1])
                    row_columns = columns[lo:hi]
                    mask = (row_columns >= left) & (row_columns < right)
                    target[local_row, (row_columns[mask] - left).to(q.device)] = values[lo:hi][mask].to(q.device)
                residual = target - selected_q @ q[left:right].t()
                total = total + residual.square().sum(dim=1)
            chunks.append(total)
        return torch.cat(chunks)

    def squared_error(self, q):
        if (q.ndim != 2 or q.shape[0] != self.n_nodes or q.shape[1] < 1 or
                q.device != self.adjacency.device or q.dtype not in (torch.float32, torch.float64)):
            raise ValueError("Q must be float32/64 N-by-D on the target device")
        q64 = q.to(torch.float64)
        gram = q64.t() @ q64
        quadratic = ((q64 @ gram) * q64).sum(dim=1)
        aq = torch.sparse.mm(self.adjacency, q64)
        cross = (q64 * aq).sum(dim=1)
        squared = self.row_squared_norm + quadratic - 2 * cross
        scale = self.row_squared_norm + quadratic.abs() + 2 * cross.abs()
        tolerance = 256 * torch.finfo(torch.float64).eps * scale
        if bool((squared < -tolerance).any()):
            raise FloatingPointError("Structure Gram residual is substantially negative")
        # A perfectly empty target with Q==0 has zero scale and needs no repair.
        repair = torch.nonzero((squared.abs() <= tolerance) & (scale > 0), as_tuple=False).flatten()
        if repair.numel():
            squared = squared.index_copy(0, repair, self._direct_rows(q64, repair))
        return squared

    def error(self, q):
        return norm_from_squared(self.squared_error(q)).to(q.dtype)


def structure_target_for(owner, edge_index, n_nodes):
    """Cache only the same unchanged edge tensor, never just matching N/E."""
    try:
        version = edge_index._version
    except RuntimeError:
        version = None
    cached = getattr(owner, "_gram_structure_cache", None)
    if (version is not None and cached is not None and cached[0] is edge_index and
            cached[1] == version and cached[2] == int(n_nodes)):
        return cached[3]
    target = StructureTarget(edge_index, n_nodes)
    if version is not None:
        owner._gram_structure_cache = (edge_index, version, int(n_nodes), target)
    return target


def reconstruction_score(ae, x, emb, edge_index, target, weight):
    """Same decoder order, attribute error, alpha, and nodewise square roots."""
    from pygod.nn.functional import double_recon_loss
    if ae.loss_func is not double_recon_loss:
        raise ValueError("Gram backend only implements original double_recon_loss")
    defaults = inspect.signature(double_recon_loss).parameters
    if any(defaults[name].default != val for name, val in
           (("pos_weight_a", .5), ("pos_weight_s", .5), ("bce_s", False))):
        raise ValueError("Installed double_recon_loss defaults differ from supported objective")
    if getattr(ae.struct_decoder, "sigmoid_s", None) is not False:
        raise ValueError("Gram backend requires an unsigmoided dot-product decoder")
    if not 0 <= weight <= 1:
        raise ValueError("Reconstruction weight must be in [0,1]")
    # Keep the dense GraphAE.decode order to preserve dropout RNG semantics.
    x_hat = ae.attr_decoder(emb, edge_index)
    q = ae.struct_decoder.nn(emb, edge_index)
    attribute_error = norm_from_squared(torch.sum(torch.pow(x - x_hat, 2), 1))
    return weight * attribute_error + (1 - weight) * target.error(q)


def zero_norm_self_test(device="cpu"):
    """Positive rows stay exact; zero residuals get a finite zero subgradient."""
    from fmgad.models.autoencoder import GraphAE
    from torch_geometric.utils import to_dense_adj
    torch.set_num_threads(2)
    torch.manual_seed(20261121)
    rows = []
    previous = os.environ.get("FMGAD_ZERO_NORM_SUBGRADIENT")
    try:
        for dtype in (torch.float32, torch.float64):
            positive1 = torch.tensor([.01, .3, 1., 25., 1e5], dtype=dtype, device=device, requires_grad=True)
            positive2 = positive1.detach().clone().requires_grad_()
            os.environ["FMGAD_ZERO_NORM_SUBGRADIENT"] = "0"
            old = norm_from_squared(positive1)
            os.environ["FMGAD_ZERO_NORM_SUBGRADIENT"] = "1"
            new = norm_from_squared(positive2)
            upstream = torch.linspace(-1., 1., len(old), dtype=dtype, device=device)
            old_grad = torch.autograd.grad(old, positive1, upstream)[0]
            new_grad = torch.autograd.grad(new, positive2, upstream)[0]
            assert torch.equal(old, new) and torch.equal(old_grad, new_grad)
            residual = torch.tensor([[0., 0.], [3., 4.]], dtype=dtype, device=device, requires_grad=True)
            norm = norm_from_squared(residual.square().sum(1))
            grad = torch.autograd.grad(norm.sum(), residual)[0]
            torch.testing.assert_close(norm, torch.tensor([0., 5.], device=device, dtype=dtype), rtol=0, atol=0)
            torch.testing.assert_close(grad, torch.tensor([[0., 0.], [.6, .8]], device=device, dtype=dtype))
            for all_zero in (False, True):
                edge = torch.tensor([[], []] if all_zero else [[0, 1], [1, 0]], device=device, dtype=torch.long)
                q = torch.zeros(4, 3, device=device, dtype=dtype)
                if not all_zero:
                    q[:2] = torch.randn(2, 3, device=device, dtype=dtype)
                q.requires_grad_()
                target = StructureTarget(edge, 4)
                score = target.error(q)
                dense = to_dense_adj(edge, max_num_nodes=4)[0].to(dtype)
                expected = (dense - q @ q.t()).square().sum(1).sqrt()
                torch.testing.assert_close(score, expected)
                score.sum().backward()
                assert bool(torch.isfinite(q.grad).all())
                assert bool((q.grad[2:] == 0).all())
                rows.append(dict(kind="zero_structure_rows", all_zero=all_zero, dtype=str(dtype)))
            edge = torch.tensor([[0, 1], [1, 0]], device=device, dtype=torch.long)
            x = torch.zeros(4, 5, device=device, dtype=dtype)
            x[:2] = torch.randn(2, 5, device=device, dtype=dtype)
            model = GraphAE(5, 3, dropout=0).to(device=device, dtype=dtype)
            score = reconstruction_score(model, x, model.encode(x, edge), edge, StructureTarget(edge, 4), .8)
            score.mean().backward()
            assert bool(torch.isfinite(score).all())
            assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
            rows.append(dict(kind="both_GraphAE_terms_zero_at_isolates", dtype=str(dtype)))
            rows.append(dict(kind="positive_forward_and_gradient_exact", dtype=str(dtype)))
        bad = torch.tensor([float("nan")], device=device, requires_grad=True)
        norm_from_squared(bad).sum().backward()
        assert bool(torch.isnan(bad.grad).all())
        rows.append(dict(kind="nonfinite_input_not_hidden"))
    finally:
        if previous is None:
            os.environ.pop("FMGAD_ZERO_NORM_SUBGRADIENT", None)
        else:
            os.environ["FMGAD_ZERO_NORM_SUBGRADIENT"] = previous
    return dict(status="pass", device=str(device), n_cases=len(rows), cases=rows, labels_read=False)


def self_test(device="cpu"):
    """Check scores, all Q gradients, decoder/network gradients and Adam steps."""
    from torch_geometric.utils import to_dense_adj
    from fmgad.models.autoencoder import GraphAE
    started = time.perf_counter()
    torch.manual_seed(20261121)
    torch.set_num_threads(2)
    if str(device).startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
    report = {"device": str(device), "torch": torch.__version__, "cases": [], "labels_read": False}
    graphs = {
        "directed_duplicate_loops_isolates": [[0, 0, 0, 1, 2, 2, 4], [1, 1, 0, 3, 1, 2, 4]],
        "empty": [[], []],
        "asymmetric": [[0, 0, 1, 3], [1, 2, 2, 0]],
        "duplicate_loops": [[0, 0, 0, 1, 1, 3], [0, 0, 0, 1, 1, 3]],
    }
    for dtype in (torch.float64, torch.float32):
        rtol, atol = (2e-11, 2e-11) if dtype == torch.float64 else (4e-5, 4e-5)
        for name, edges in graphs.items():
            edge = torch.tensor(edges, dtype=torch.long, device=device)
            dense = to_dense_adj(edge, max_num_nodes=7)[0].to(dtype)
            target = StructureTarget(edge, 7)
            for scale in (.01, 1., 30.):
                q1 = (torch.randn(7, 5, dtype=dtype, device=device) * scale).requires_grad_()
                q2 = q1.detach().clone().requires_grad_()
                expected = (dense - q1 @ q1.t()).square().sum(1).sqrt()
                actual = target.error(q2)
                torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
                upstream = torch.linspace(-.8, 1.2, 7, device=device, dtype=dtype)
                (expected * upstream).sum().backward()
                (actual * upstream).sum().backward()
                torch.testing.assert_close(q2.grad, q1.grad, rtol=rtol, atol=atol)
                report["cases"].append({"kind": "Q_score_and_gradient", "graph": name,
                                        "dtype": str(dtype), "scale": scale,
                                        "score_max_abs": float((actual - expected).detach().abs().max()),
                                        "gradient_max_abs": float((q2.grad - q1.grad).abs().max())})
        # Dense target I and Q almost I stress subtractive cancellation.
        identity_edge = torch.arange(7, device=device).repeat(2, 1)
        near = torch.eye(7, dtype=dtype, device=device) * (1 + 1e-8)
        expected = (torch.eye(7, dtype=dtype, device=device) - near @ near.t()).square().sum(1)
        actual = StructureTarget(identity_edge, 7).squared_error(near)
        torch.testing.assert_close(actual, expected.to(torch.float64), rtol=1e-6, atol=1e-20)
        report["cases"].append({"kind": "near_exact_reconstruction", "dtype": str(dtype)})
        # Tiny positive error on empty rows has no subtractive cancellation.
        # In particular it must not turn into a quadratic-time direct repair.
        tiny_target = StructureTarget(torch.empty((2, 0), dtype=torch.long, device=device), 7)
        def unexpected_repair(*args, **kwargs):
            raise AssertionError("Non-cancelling tiny Q triggered direct row repair")
        tiny_target._direct_rows = unexpected_repair
        tiny = torch.full((7, 5), 1e-8, dtype=dtype, device=device, requires_grad=True)
        tiny_score = tiny_target.error(tiny)
        tiny_dense = (tiny @ tiny.t()).square().sum(1).sqrt()
        torch.testing.assert_close(tiny_score, tiny_dense, rtol=4e-5, atol=0)
        tiny_score.sum().backward()
        assert bool(torch.isfinite(tiny.grad).all())
        report["cases"].append({"kind": "tiny_Q_empty_target_no_direct_repair", "dtype": str(dtype)})
        for dropout in (0., .25):
            edge = torch.tensor(graphs["directed_duplicate_loops_isolates"], dtype=torch.long, device=device)
            x = torch.randn(7, 9, dtype=dtype, device=device)
            dense = to_dense_adj(edge, max_num_nodes=7)[0].to(dtype)
            model1 = GraphAE(9, 5, dropout=dropout).to(device=device, dtype=dtype)
            model2 = copy.deepcopy(model1)
            target = StructureTarget(edge, 7)
            optim1 = torch.optim.Adam(model1.parameters(), lr=.001, weight_decay=.01)
            optim2 = torch.optim.Adam(model2.parameters(), lr=.001, weight_decay=.01)
            for step in range(3):
                optim1.zero_grad()
                optim2.zero_grad()
                torch.manual_seed(100 + step)
                x_hat, s_hat, _ = model1(x, edge)
                expected = model1.loss_func(x, x_hat, dense, s_hat, .8)
                torch.manual_seed(100 + step)
                actual = reconstruction_score(model2, x, model2.encode(x, edge), edge, target, .8)
                torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
                upstream = torch.linspace(.2, 1.2, 7, dtype=dtype, device=device)
                (expected * upstream).mean().backward()
                (actual * upstream).mean().backward()
                for (name, p1), (_, p2) in zip(model1.named_parameters(), model2.named_parameters()):
                    torch.testing.assert_close(p2.grad, p1.grad, rtol=rtol, atol=atol, msg=name)
                torch.nn.utils.clip_grad_norm_(model1.parameters(), 1.)
                torch.nn.utils.clip_grad_norm_(model2.parameters(), 1.)
                optim1.step()
                optim2.step()
                for p1, p2 in zip(model1.parameters(), model2.parameters()):
                    torch.testing.assert_close(p2, p1, rtol=rtol, atol=atol)
                report["cases"].append({"kind": "GraphAE_gradient_Adam", "dtype": str(dtype),
                                        "dropout": dropout, "step": step})
    # In-place edge mutation and same-size different graphs must invalidate cache.
    owner = type("Owner", (), {})()
    edge = torch.tensor([[0], [1]], dtype=torch.long, device=device)
    first = structure_target_for(owner, edge, 3)
    assert structure_target_for(owner, edge, 3) is first
    edge[1, 0] = 2
    second = structure_target_for(owner, edge, 3)
    assert second is not first
    assert structure_target_for(owner, edge.clone(), 3) is not second
    report["cases"].append({"kind": "cache_invalidates_mutated_edges"})
    if str(device).startswith("cuda"):
        import os
        import tempfile
        from torch_geometric.data import Data
        from fmgad.detector import ResFlowGAD
        from fmgad.models import FlowMatchingModel, MLPFlowMatching
        edge = torch.tensor(graphs["directed_duplicate_loops_isolates"], dtype=torch.long, device=device)
        x = torch.randn(7, 9, device=device)
        data = Data(x=x, edge_index=edge)  # Deliberately has no y.
        dense_model = ResFlowGAD(hid_dim=5, ae_epochs=2, ae_dropout=.25, verbose=False,
                                use_virtual_neighbors=False, use_proto=False, polarity_enabled=False)
        dense_model.ae = GraphAE(9, 5, dropout=.25).to(device)
        gram_model = copy.deepcopy(dense_model)
        old_switch = os.environ.get("FMGAD_STRUCTURE_GRAM")
        try:
            with tempfile.TemporaryDirectory(prefix="fmgad_gram_qa_") as checkpoint_root:
                for name, model, switch in (("dense", dense_model, "0"), ("gram", gram_model, "1")):
                    path = os.path.join(checkpoint_root, name)
                    os.mkdir(path)
                    os.environ["FMGAD_STRUCTURE_GRAM"] = switch
                    torch.manual_seed(223)
                    assert model._train_ae_once(data, path) in (1, 2)
            for p1, p2 in zip(dense_model.ae.parameters(), gram_model.ae.parameters()):
                torch.testing.assert_close(p2, p1, rtol=4e-5, atol=4e-5)
            # Inference equivalence at exactly the same checkpoint isolates the
            # score rewrite from the harmless optimizer rounding differences.
            gram_model.ae.load_state_dict(dense_model.ae.state_dict())
            free = FlowMatchingModel(MLPFlowMatching(10, dim_t=512), 10).to(device)
            os.environ["FMGAD_STRUCTURE_GRAM"] = "0"
            expected = dense_model.score_samples(None, free, data, apply_polarity=False)
            os.environ["FMGAD_STRUCTURE_GRAM"] = "1"
            actual = gram_model.score_samples(None, free, data, apply_polarity=False)
            torch.testing.assert_close(actual, expected, rtol=4e-5, atol=4e-5)
            report["cases"].append({"kind": "detector_training_and_inference_switch",
                                    "score_max_abs": float((actual - expected).abs().max())})
        finally:
            if old_switch is None:
                os.environ.pop("FMGAD_STRUCTURE_GRAM", None)
            else:
                os.environ["FMGAD_STRUCTURE_GRAM"] = old_switch
        torch.cuda.synchronize()
        report["gpu_name"] = torch.cuda.get_device_name()
        report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    report.update(status="pass", n_cases=len(report["cases"]), seconds=time.perf_counter() - started)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    tests = parser.add_mutually_exclusive_group(required=True)
    tests.add_argument("--self-test", action="store_true")
    tests.add_argument("--zero-norm-self-test", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--memory-fraction", type=float, default=.1)
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device=args.device)
    result = zero_norm_self_test(args.device) if args.zero_norm_self_test else self_test(args.device)
    result["memory_fraction"] = args.memory_fraction if args.device.startswith("cuda") else None
    output = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        from pathlib import Path
        Path(args.output).write_text(output, encoding="utf-8")
    print(output)
