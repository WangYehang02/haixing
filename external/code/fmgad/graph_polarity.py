"""Label-free polarity from predictable graph context and marginal feature rarity."""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.stats import rankdata
from sklearn.random_projection import SparseRandomProjection


def midrank(scores):
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if not np.isfinite(scores).all():
        raise ValueError("Polarity inputs must be finite")
    if scores.size < 2:
        return np.full(scores.shape, 0.5)
    return (rankdata(scores, method="average") - 1) / (scores.size - 1)


def graph_reference(features, edge_index):
    """Return a fixed graph reference. Use distinct incoming non-self neighbors."""
    x = np.asarray(features)
    if x.ndim != 2 or len(x) == 0:
        raise ValueError("Expected a nonempty node-by-feature matrix")
    n, width = x.shape
    dims = min(width, 96)
    projection = None
    if width > dims:
        projector = SparseRandomProjection(n_components=dims, random_state=431)
        projection = projector.fit(np.zeros((1, width), dtype=np.float32)).components_.tocsr()
    z = np.zeros((n, dims), dtype=np.float32)
    energy = np.zeros(n, dtype=np.float64)
    lengths_squared = np.zeros(n, dtype=np.float64)
    total_variance = 0.
    active = 0
    # Bound peak allocation on wide sparse-feature benchmarks stored as dense tensors.
    for start in range(0, width, 96):
        stop = min(start + 96, width)
        block = x[:, start:stop].astype(np.float64, copy=True)
        if not np.isfinite(block).all():
            raise ValueError("Nonfinite node attribute")
        varying = np.max(block, axis=0) != np.min(block, axis=0)
        active += int(varying.sum())
        original = block[:, varying]
        lengths_squared += np.einsum("ij,ij->i", original, original)
        total_variance += np.var(original, axis=0).sum()
        block -= np.median(block, axis=0)
        scale = np.sqrt(np.mean(block * block, axis=0))
        block /= np.where(scale > 0, scale, 1)
        np.arcsinh(block, out=block)
        block[:, ~varying] = 0
        energy += np.einsum("ij,ij->i", block, block)
        if projection is None:
            z[:, start:stop] = block
        else:
            z += (projection[:, start:stop] @ block.astype(np.float32).T).T
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    z /= np.where(norms > 0, norms, 1)
    src, dst = np.asarray(edge_index)
    real = src != dst
    adj = csr_matrix((np.ones(np.count_nonzero(real), np.float32), (dst[real], src[real])), shape=(n, n))
    adj.data.fill(1)
    degree = np.diff(adj.indptr)
    has_neighbors = degree > 0
    context = (adj @ z) / np.maximum(1, degree[:, None]).astype(np.float32)
    observed = np.einsum("ij,ij->i", z, context, dtype=np.float64)
    expected = np.zeros(n)
    comparable = np.zeros(n, dtype=bool)
    bins = np.floor(np.log2(np.maximum(1, degree))).astype(int)
    for group in np.unique(bins[has_neighbors]):
        ix = np.flatnonzero(has_neighbors & (bins == group))
        if len(ix) < 2:
            continue
        # Exact mean compatibility with other nodes' neighborhoods in this degree bin.
        mean_other = (context[ix].sum(axis=0, dtype=np.float64) - context[ix]) / (len(ix) - 1)
        expected[ix] = np.einsum("ij,ij->i", z[ix], mean_other)
        comparable[ix] = True
    advantage = observed - expected
    h = (max(0., float(np.sign(advantage[comparable]).mean())) if comparable.any() else 0.)
    ctx_rank = np.full(n, .5)
    ctx_rank[comparable] = midrank(-advantage[comparable])
    rarity = midrank(energy)
    weight = h * comparable * degree / (degree + 1.)
    anchor = midrank((1 - weight) * rarity + weight * ctx_rank)
    radial_share = float(np.clip(np.var(np.sqrt(lengths_squared)) / total_variance, 0, 1)) if total_variance else 1.
    return anchor, {"radial_share": radial_share, "n_comparable": int(comparable.sum()),
                    "graph_predictability": h, "mean_context_weight": float(weight.mean()),
                    "comparable_fraction": float(comparable.mean()), "varying_features": active}


def calibrate_graph_polarity(raw, anchor):
    r = midrank(raw)
    a = np.asarray(anchor, dtype=np.float64).reshape(-1)
    if len(r) != len(a) or not np.isfinite(a).all():
        raise ValueError("Raw scores and graph reference must have matching finite inputs")
    if np.ptp(a) == 0:
        return r, {"agreement": 0., "decision": "raw_fallback", "model_weight": 1.}
    left, right = r - r.mean(), a - a.mean()
    norm = np.linalg.norm(left) * np.linalg.norm(right)
    rho = float(np.clip(left @ right / norm, -1., 1.)) if norm > 0 else 0.
    out = (a + rho * (r - .5) + abs(rho) * .5) / (1 + abs(rho))
    return out, {"agreement": rho, "decision": "flip" if rho < 0 else "keep",
                 "model_weight": abs(rho) / (1 + abs(rho))}
