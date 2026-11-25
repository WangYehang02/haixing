"""Opt-in fixed-CSR matmul with a cached CSR transpose for dense gradients.

For constant A and Y=A@X, dX=A.T@dY exactly. The graph, duplicate weights,
normalization, dense parameters, loss, and forward sparse.mm operation are
unchanged. This avoids rebuilding/converting A.T in each backward. Reduction
order can differ from PyTorch's generic sparse backward; bitwise equivalence
is not promised. No adjacency gradients are approximated: calls that need
them use the original implementation.

Call install() explicitly in a NEW, declared execution backend. Importing
alone has no effect. It intercepts torch.sparse.mm at the GCN, residual and
structure-AQ call sites without editing those scientific modules. A weak,
bounded cache retains only A.T, not old A graphs. A.T uses one extra CSR's
worth of storage; its initial CSC-to-CSR conversion can need workspace.

The cache checks the original tensor's mutation version. Backward also saves
A for PyTorch's version check, so mutating it between forward and backward
raises rather than silently using the wrong graph. Inference does not build
the transpose. CPU/CUDA float32/64, first-order training, sum reduction only.
"""
import argparse
from collections import OrderedDict
import json
import time
import weakref

import torch


_ORIGINAL_MM = getattr(torch.sparse.mm, '_fmgad_static_original_mm', torch.sparse.mm)
_CACHE = OrderedDict()
_MAX_ENTRIES = 8
_BUILDS = 0
_HITS = 0


def _autocast_enabled():
    if torch.is_autocast_enabled():
        return True
    try:
        return torch.is_autocast_enabled('cpu')
    except TypeError:
        return torch.is_autocast_cpu_enabled()


def _signature(matrix):
    try:
        return (matrix._version, tuple(matrix.shape), matrix.dtype, matrix.device)
    except RuntimeError:
        return None  # Inference tensors cannot supply a reliable version.


def _supported(matrix, dense, reduce):
    return (reduce == 'sum' and isinstance(matrix, torch.Tensor) and
            isinstance(dense, torch.Tensor) and matrix.layout == torch.sparse_csr and
            dense.layout == torch.strided and matrix.ndim == dense.ndim == 2 and
            matrix.device == dense.device and matrix.dtype == dense.dtype and
            dense.dtype in (torch.float32, torch.float64) and
            dense.device.type in ('cpu', 'cuda') and not matrix.requires_grad and
            not _autocast_enabled() and _signature(matrix) is not None)


def _transpose(matrix):
    global _BUILDS, _HITS
    key, signature = id(matrix), _signature(matrix)
    cached = _CACHE.get(key)
    if cached is not None and cached[0]() is matrix and cached[1] == signature:
        _HITS += 1
        _CACHE.move_to_end(key)
        return cached[2]
    # transpose returns a CSC view. Convert once to CSR so backward is CSR@X.
    transposed = matrix.transpose(0, 1).to_sparse_csr()
    _BUILDS += 1
    def release(reference):
        present = _CACHE.get(key)
        if present is not None and present[0] is reference:
            _CACHE.pop(key, None)
    _CACHE[key] = (weakref.ref(matrix, release), signature, transposed)
    _CACHE.move_to_end(key)
    while len(_CACHE) > _MAX_ENTRIES:
        _CACHE.popitem(last=False)
    return transposed


class _FixedCSRMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, matrix, dense, transposed):
        ctx.save_for_backward(matrix, transposed)
        return _ORIGINAL_MM(matrix, dense)

    @staticmethod
    def backward(ctx, gradient):
        matrix, transposed = ctx.saved_tensors  # Access enforces version checks.
        return None, _ORIGINAL_MM(transposed, gradient), None


def static_mm(mat1, mat2, reduce='sum'):
    """Drop-in sum SpMM; unsupported cases retain the original operation."""
    matrix, dense = mat1, mat2
    if not _supported(matrix, dense, reduce):
        # Passing reduce='sum' explicitly selects a different ATen path which
        # does not support COO/CUDA on some PyTorch versions. Preserve the
        # ordinary sum dispatch for unsupported layouts / learned adjacency.
        return (_ORIGINAL_MM(matrix, dense) if reduce == 'sum' else
                _ORIGINAL_MM(matrix, dense, reduce=reduce))
    if not torch.is_grad_enabled() or not dense.requires_grad:
        return _ORIGINAL_MM(matrix, dense)
    return _FixedCSRMM.apply(matrix, dense, _transpose(matrix))


static_mm._fmgad_static_original_mm = _ORIGINAL_MM


def install(max_entries=8):
    """Explicit, process-local and idempotent; no changes to frozen source."""
    global _MAX_ENTRIES
    if int(max_entries) != max_entries or max_entries < 1:
        raise ValueError('max_entries must be a positive integer')
    _MAX_ENTRIES = int(max_entries)
    while len(_CACHE) > _MAX_ENTRIES:
        _CACHE.popitem(last=False)
    torch.sparse.mm = static_mm
    return dict(installed=True, implementation='fixed_CSR_forward_cached_transpose_backward',
                max_entries=_MAX_ENTRIES, learned_adjacency='original_sparse_mm_fallback',
                arithmetic='Y=A@X; dX=A.T@dY; original forward operation')


def uninstall():
    if torch.sparse.mm is static_mm:
        torch.sparse.mm = _ORIGINAL_MM
    clear_cache()


def clear_cache():
    global _BUILDS, _HITS
    _CACHE.clear()
    _BUILDS = _HITS = 0


def cache_stats():
    storage = 0
    for _, _, matrix in _CACHE.values():
        storage += sum(v.numel() * v.element_size() for v in
                       (matrix.crow_indices(), matrix.col_indices(), matrix.values()))
    return dict(entries=len(_CACHE), builds=_BUILDS, hits=_HITS,
                retained_transpose_bytes=storage, max_entries=_MAX_ENTRIES)


def _sync(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def self_test(device='cpu'):
    from torch_geometric.nn.conv.gcn_conv import gcn_norm
    torch.manual_seed(2711)
    torch.set_num_threads(2)
    device = torch.device(device)
    cases = []
    clear_cache()
    for dtype in (torch.float32, torch.float64):
        indices = torch.tensor([[0, 0, 0, 1, 3, 3], [1, 1, 0, 4, 2, 3]], device=device)
        values = torch.tensor([1., 2., .5, -1., 2., 1.], dtype=dtype, device=device)
        matrices = [torch.sparse_coo_tensor(indices, values, (5, 7)).coalesce().to_sparse_csr(),
                    torch.sparse_coo_tensor(torch.empty((2, 0), dtype=torch.long, device=device),
                        torch.empty(0, dtype=dtype, device=device), (5, 7)).to_sparse_csr()]
        # Keep literal parallel CSR entries as well as already-coalesced input.
        matrices.append(torch.sparse_csr_tensor(torch.tensor([0,3,4,4,6,6], device=device),
            indices[1], values, size=(5,7)))
        positive = torch.tensor([[0,0,0,1,3,3], [1,1,0,4,2,3]], device=device)
        normalized_index, normalized_weight = gcn_norm(positive, None, 7, False, True,
                                                        'source_to_target', dtype)
        matrices.append(torch.sparse_coo_tensor(normalized_index.flip(0), normalized_weight,
                                                (7,7)).coalesce().to_sparse_csr())
        for index, matrix in enumerate(matrices):
            for noncontiguous in (False, True):
                x1 = torch.randn(matrix.shape[1], 8, dtype=dtype, device=device)
                if noncontiguous:
                    x1 = x1[:, ::2]
                x1 = x1.detach().requires_grad_()
                x2 = x1.detach().clone().requires_grad_()
                old, new = _ORIGINAL_MM(matrix, x1), static_mm(matrix, x2)
                upstream = (torch.randn(old.shape[0], old.shape[1]*2, dtype=dtype, device=device)[:, ::2]
                            if noncontiguous else torch.randn_like(old))
                g1, = torch.autograd.grad(old, x1, upstream)
                g2, = torch.autograd.grad(new, x2, upstream)
                atol = 2e-6 if dtype == torch.float32 else 2e-12
                torch.testing.assert_close(new, old, rtol=atol, atol=atol)
                torch.testing.assert_close(g2, g1, rtol=atol, atol=atol)
                cases.append(dict(kind='directed_duplicate_empty_normalized', matrix=index,
                    dtype=str(dtype), noncontiguous=noncontiguous,
                    forward_max_abs=float((new-old).detach().abs().max()),
                    gradient_max_abs=float((g1-g2).abs().max())))
        # Verify gradients propagated through learned dense W and a row norm.
        matrix = matrices[-1]
        x = torch.randn(7, 3, dtype=dtype, device=device)
        w1 = torch.randn(3, 4, dtype=dtype, device=device, requires_grad=True)
        w2 = w1.detach().clone().requires_grad_()
        loss1 = _ORIGINAL_MM(matrix, x@w1).square().sum(1).sqrt().mean()
        loss2 = static_mm(matrix, x@w2).square().sum(1).sqrt().mean()
        g1, = torch.autograd.grad(loss1, w1)
        g2, = torch.autograd.grad(loss2, w2)
        torch.testing.assert_close(g1, g2, rtol=2e-6 if dtype==torch.float32 else 2e-12,
                                   atol=2e-6 if dtype==torch.float32 else 2e-12)
        cases.append(dict(kind='dense_parameters_and_norm', dtype=str(dtype)))
    matrix = matrices[-1]
    x = torch.randn(7, 4, device=device, dtype=matrix.dtype, requires_grad=True)
    clear_cache()
    static_mm(matrix, x)
    assert cache_stats()['builds'] == 1
    static_mm(matrix, x)
    assert cache_stats()['hits'] == 1
    matrix.values().mul_(2)
    new = static_mm(matrix, x)
    assert cache_stats()['builds'] == 2
    matrix.values().add_(1)
    try:
        new.sum().backward()
    except RuntimeError as e:
        assert 'modified by an inplace operation' in str(e)
    else:
        raise AssertionError('Mutation after forward did not block stale transpose')
    cases.append(dict(kind='mutation_invalidation_and_backward_version_check'))
    # A graph requiring adjacency gradients must keep PyTorch's original path.
    learned = matrix.detach().clone().requires_grad_()
    y = static_mm(learned, x)
    assert 'FixedCSRMM' not in type(y.grad_fn).__name__
    grad_a, grad_x = torch.autograd.grad(y.sum(), (learned, x))
    assert grad_a is not None and bool(torch.isfinite(grad_x).all())
    cases.append(dict(kind='learned_adjacency_fallback'))
    coo = matrix.to_sparse_coo()
    torch.testing.assert_close(static_mm(coo, x), _ORIGINAL_MM(coo, x), rtol=2e-12, atol=2e-12)
    cases.append(dict(kind='COO_fallback_keeps_original_sum_dispatch'))
    clear_cache()
    with torch.no_grad():
        static_mm(matrix, x)
    assert cache_stats()['entries'] == 0
    cases.append(dict(kind='inference_does_not_allocate_transpose'))
    # Dead input matrices must not be retained by the cache itself.
    def ephemeral():
        temporary = matrix.detach().clone()
        static_mm(temporary, x)
        return weakref.ref(temporary)
    reference = ephemeral()
    import gc
    gc.collect()
    assert reference() is None and cache_stats()['entries'] == 0
    cases.append(dict(kind='weak_cache_releases_dead_graph'))
    # Exercise the three actual call sites with the same unmodified modules.
    from torch_geometric.nn import GCNConv
    from sparse_execution import cache_gcn, residual
    from sparse_structure import StructureTarget
    edges = torch.tensor([[0,0,0,1,2,3], [0,1,1,2,0,3]], device=device)
    layer = GCNConv(5,4).to(device=device,dtype=torch.float64)
    h = torch.randn(7,5,device=device,dtype=torch.float64,requires_grad=True)
    q = torch.randn(7,4,device=device,dtype=torch.float64,requires_grad=True)
    target = StructureTarget(edges,7)
    inputs = (h,q,*layer.parameters())
    outputs, gradients = [], []
    for enabled in (False, True):
        if enabled:
            install()
        else:
            uninstall()
        y = cache_gcn(layer,h,edges)
        global_r, local_r, _ = residual(h,edges)
        error = target.error(q)
        loss = y.square().mean()+global_r.square().mean()+local_r.square().mean()+error.mean()
        gradients.append(torch.autograd.grad(loss,inputs))
        outputs.append((y.detach(),global_r.detach(),local_r.detach(),error.detach()))
    for before,after in zip(outputs[0],outputs[1]):
        torch.testing.assert_close(after,before,rtol=2e-12,atol=2e-12)
    for before,after in zip(gradients[0],gradients[1]):
        torch.testing.assert_close(after,before,rtol=2e-11,atol=2e-11)
    uninstall()
    cases.append(dict(kind='original_GCN_residual_and_structure_AQ_integration'))
    install()
    assert torch.sparse.mm is static_mm
    uninstall()
    assert torch.sparse.mm is _ORIGINAL_MM
    cases.append(dict(kind='explicit_install_uninstall'))
    _sync(device)
    return dict(status='pass', n_cases=len(cases), cases=cases, device=str(device),
                torch=torch.__version__, labels_read=False)


def benchmark(device='cpu', n=30000, edges=300000, width=64, repeats=7):
    """Same frozen synthetic A and upstream gradients for both implementations."""
    torch.manual_seed(1889)
    torch.set_num_threads(2)
    device = torch.device(device)
    indices = torch.randint(n, (2, edges), device=device)
    # Row-normalized, asymmetric weighted graph with retained parallel weights.
    degree = torch.bincount(indices[0], minlength=n).clamp_min(1)
    weights = degree[indices[0]].to(torch.float32).reciprocal()
    matrix = torch.sparse_coo_tensor(indices, weights, (n,n)).coalesce().to_sparse_csr()
    del indices, weights, degree
    x = torch.randn(n, width, device=device, requires_grad=True)
    upstream = torch.randn(n, width, device=device)
    clear_cache()
    _sync(device)
    start = time.perf_counter()
    _transpose(matrix)
    _sync(device)
    setup = time.perf_counter()-start
    timings = {'original': [], 'static': []}
    maxima = dict(forward_max_abs=0., gradient_max_abs=0., original_repeat_forward_max_abs=0.)
    previous_original = None
    for repeat in range(repeats+2):
        # Alternate order; synchronize around only backward. No gradient
        # accumulation, optimizer, different graph, or stochastic model.
        outputs, gradients = {}, {}
        order = ('original','static') if repeat%2==0 else ('static','original')
        for name in order:
            function = _ORIGINAL_MM if name=='original' else static_mm
            y = function(matrix, x)
            _sync(device)
            start = time.perf_counter()
            gradient, = torch.autograd.grad(y, x, upstream)
            _sync(device)
            elapsed = time.perf_counter()-start
            if repeat >= 2:
                timings[name].append(elapsed)
            outputs[name], gradients[name] = y.detach(), gradient
        # Even two ordinary CUDA sparse.mm calls can differ in the last bit.
        torch.testing.assert_close(outputs['static'],outputs['original'],rtol=3e-6,atol=3e-7)
        torch.testing.assert_close(gradients['static'],gradients['original'],rtol=3e-5,atol=3e-6)
        maxima['forward_max_abs'] = max(maxima['forward_max_abs'],float((outputs['static']-outputs['original']).abs().max()))
        maxima['gradient_max_abs'] = max(maxima['gradient_max_abs'],float((gradients['static']-gradients['original']).abs().max()))
        if previous_original is not None:
            maxima['original_repeat_forward_max_abs'] = max(maxima['original_repeat_forward_max_abs'],
                float((outputs['original']-previous_original).abs().max()))
        previous_original = outputs['original']
    import statistics
    medians = {name:statistics.median(vals) for name,vals in timings.items()}
    return dict(status='pass',device=str(device),torch=torch.__version__,n_nodes=n,
                input_edges=edges,coalesced_entries=matrix.values().numel(),width=width,
                seed=1889,transpose_setup_seconds=setup,backward_seconds=timings,
                backward_median_seconds=medians,backward_speedup=medians['original']/medians['static'],
                cache=cache_stats(),**maxima,labels_read=False,
                scope='dense-input backward only; excludes transpose setup and other AE/flow work')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--benchmark',action='store_true')
    parser.add_argument('--nodes',type=int,default=30000)
    parser.add_argument('--edges',type=int,default=300000)
    parser.add_argument('--width',type=int,default=64)
    parser.add_argument('--repeats',type=int,default=7)
    parser.add_argument('--memory-fraction',type=float,default=.1)
    parser.add_argument('--output')
    args = parser.parse_args()
    if not args.self_test and not args.benchmark:
        parser.error('Choose --self-test and/or --benchmark')
    if args.device.startswith('cuda'):
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction,device=args.device)
    result = dict(labels_read=False,device=args.device)
    if args.self_test:
        result['self_test'] = self_test(args.device)
    if args.benchmark:
        result['benchmark'] = benchmark(args.device,args.nodes,args.edges,args.width,args.repeats)
    if args.device.startswith('cuda'):
        result.update(gpu=torch.cuda.get_device_name(),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      memory_fraction=args.memory_fraction)
    import hashlib
    from pathlib import Path
    result['source_sha256'] = hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()
    text = json.dumps(result,indent=2,allow_nan=False)
    if args.output:
        Path(args.output).write_text(text,encoding='utf-8')
    print(text)
