"""Graph-conditioned score orientation with magnitude and angular references."""
from dataclasses import dataclass

import numpy as np
import torch
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

from fmgad.graph_polarity import graph_reference, midrank


def _units(x):
    norm = np.sqrt(np.einsum("ij,ij->i", x, x, dtype=np.float64))
    return x / np.where(norm > 0, norm, 1.).astype(np.float32)[:, None]


def _angular_gap(x, edge_index, degree):
    n, width = x.shape
    if n <= 1:
        return np.zeros(n)
    z = _units(np.asarray(x, dtype=np.float32))
    dim = min(64, width-1, n-1)
    if dim > 0 and np.any(z):
        z = _units(TruncatedSVD(n_components=dim, n_iter=7, random_state=1729).fit_transform(csr_matrix(z)))
    src, dst = edge_index
    valid_edges = src != dst
    src, dst = src[valid_edges], dst[valid_edges]
    affinity = np.zeros(n, dtype=np.float64)
    for start in range(0, len(src), 8192):
        ix = slice(start, start + 8192)
        cos = np.einsum("ij,ij->i", z[src[ix]], z[dst[ix]])
        np.add.at(affinity, dst[ix], np.clip(cos, -1, 1))
    k = min(10, n-1)
    top = np.empty(n, dtype=np.float64)
    batch = max(1, min(512, 8_000_000 // n))
    for start in range(0, n, batch):
        stop = min(start + batch, n)
        sim = np.clip(z[start:stop] @ z.T, -1, 1)
        sim[np.arange(stop-start), np.arange(start, stop)] = -np.inf
        sim.partition(n-k, axis=1)
        top[start:stop] = sim[:, -k:].mean(axis=1)
    valid = degree > 0
    gap = top - affinity / np.maximum(1, degree)
    gap[~valid] = np.median(gap[valid]) if valid.any() else 0.
    return gap


def _magnitude_prior(x, edge_index):
    # Match the original torch-float accumulation while bounding intermediate storage.
    x = torch.as_tensor(x, dtype=torch.float32, device="cpu")
    edges = torch.as_tensor(edge_index, dtype=torch.long, device="cpu")
    total = torch.zeros_like(x)
    batch = max(1, min(4096, 1_000_000 // max(1, x.shape[1])))
    for src, dst in edges.split(batch, dim=1):
        total.index_add_(0, dst, x[src])
    degree = torch.bincount(edges[1], minlength=len(x)).float().clamp_min(1)
    return torch.linalg.vector_norm(x - total / degree[:, None], dim=1).numpy()


@dataclass
class GeometryState:
    reference: np.ndarray
    degree: np.ndarray
    degree_rank: np.ndarray
    diagnostics: dict
    feature_anchor: np.ndarray


def prepare_geometry(x, edge_index, *, gap=None, local_prior=None):
    """Optional precomputed graph probes must be from the identical graph and node order."""
    x, e = np.asarray(x), np.asarray(edge_index)
    if x.ndim != 2 or not len(x) or e.ndim != 2 or e.shape[0] != 2 or e.dtype.kind not in "iu":
        raise ValueError("Expected features and an integer edge index of shape (2,E)")
    if e.size and (e.min() < 0 or e.max() >= len(x)):
        raise ValueError("Edge outside of feature matrix")
    degree = np.bincount(e[1, e[0] != e[1]], minlength=len(x))
    feature_anchor, info = graph_reference(x, e)
    cutoff = 2 / np.sqrt(max(1, info["n_comparable"]))
    angular = info["radial_share"] < .5 and info["graph_predictability"] > cutoff
    if angular:
        probe = _angular_gap(x, e, degree) if gap is None else gap
    else:
        probe = _magnitude_prior(x, e) if local_prior is None else local_prior
    probe = midrank(probe)
    if len(probe) != len(x):
        raise ValueError("Graph probe has an unexpected node count")
    return GeometryState(probe, degree, midrank(degree),
                         {**info, "reference_type": "angular" if angular else "magnitude",
                          "predictability_cutoff": float(cutoff)}, feature_anchor)


def _corr(a, b):
    ac, bc = a-a.mean(), b-b.mean()
    norm = np.linalg.norm(ac) * np.linalg.norm(bc)
    return float(np.clip(ac @ bc / norm,-1,1)) if norm else 0.


def score_geometry(raw, state, *, angular_threshold=-.1, magnitude_refinement=True, shrink_angular=False):
    r, p = midrank(raw), state.reference
    if len(r) != len(p):
        raise ValueError("Graph and model scores have different node counts")
    diag = {**state.diagnostics, "feature_weight":0.}
    if np.ptp(r) == 0 or np.ptp(p) == 0:
        model_constant = np.ptp(r) == 0
        return (p if model_constant else r), {**diag, "decision": "reference_fallback" if model_constant else "raw_fallback",
                                               "flipped":False, "agreement":0., "model_weight":0. if model_constant else 1.}
    rho = _corr(r, p)
    angular = diag["reference_type"] == "angular"
    flip = bool(rho <= angular_threshold if angular else rho < np.sqrt(.5))
    oriented = (midrank(-np.asarray(raw, dtype=np.float64)) if angular and not shrink_angular else 1-r) if flip else r
    if angular:
        w = state.degree / (2.*state.degree+1) if shrink_angular else .5
        out = (1-w)*oriented + w*p
        model_weight = float(np.mean(1-w))
    else:
        degree_rho = _corr(oriented, state.degree_rank)
        w = degree_rho**2
        degree_score = 1-state.degree_rank if degree_rho < 0 else state.degree_rank
        out = (oriented + .05*p + w*degree_score) / (1.05+w)
        model_weight = 1/(1.05+w)
        diag["degree_agreement"] = degree_rho
        if magnitude_refinement and diag["graph_predictability"] > diag["predictability_cutoff"]:
            feature_weight = max(0., 2*diag["radial_share"]-1) * (1-abs(rho))**2
            if feature_weight > 0:
                out = (1-feature_weight)*midrank(out) + feature_weight*state.feature_anchor
                model_weight *= 1-feature_weight
                diag["feature_weight"] = feature_weight
    return out, {**diag,"agreement":rho,"flipped":flip,"decision":"flip" if flip else "keep","model_weight":model_weight}
