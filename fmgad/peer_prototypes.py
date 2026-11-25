"""Cross-fitted neighborhood prototypes for conditioning FMGAD anomaly ranks."""

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.stats import rankdata
from sklearn.cluster import KMeans


def midranks(values):
    return (rankdata(values, method="average") - 1) / max(1, len(values) - 1)


@dataclass
class PeerFold:
    reference: np.ndarray
    query: np.ndarray
    reference_role: np.ndarray
    query_role: np.ndarray
    n_roles: int
    eligible_neighbors: csr_matrix


def fit_peer_roles(features, edges):
    """A query node's role uses topology and features of out-of-fold neighbors.

    Fold membership and KMeans initialization use fixed seeds across graphs.
    Node features in a query fold are excluded from fitting and role assignment.
    """
    x = np.asarray(features, dtype=np.float32)
    if len(x) < 2 or x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Expected at least two nodes with finite attributes")
    src, dst = np.asarray(edges)
    real = src != dst
    n = len(x)
    adj = csr_matrix(
        (np.ones(real.sum(), dtype=np.float32), (dst[real], src[real])), shape=(n, n)
    )
    adj.data[:] = 1
    degree = np.asarray(adj.sum(axis=1)).ravel()
    neighbor_degree = (adj @ np.log1p(degree)) / np.maximum(1, degree)
    topology = np.column_stack([np.log1p(degree), neighbor_degree])
    order = np.random.default_rng(991).permutation(n)
    folds = []
    for query in np.array_split(order, min(5, n)):
        ref = np.setdiff1d(np.arange(n), query)
        med = np.median(x[ref], axis=0)
        q25, q75 = np.quantile(x[ref], [0.25, 0.75], axis=0)
        scale = np.where(
            q75 - q25 > 1e-8, q75 - q25, np.maximum(np.std(x[ref], axis=0), 1e-8)
        )
        source_features = (
            np.arcsinh((x[ref] - med) / scale).clip(-8, 8).astype(np.float32)
        )
        obs = adj[:, ref]
        count = np.asarray(obs.sum(axis=1))
        neighbor = (obs @ source_features + 3 * source_features.mean(axis=0)) / (
            count + 3
        )
        structure = (topology - topology[ref].mean(axis=0)) / np.maximum(
            topology[ref].std(axis=0), 0.1
        )
        norm = max(np.sqrt(np.mean(np.sum(neighbor[ref] ** 2, axis=1))), 1e-4)
        signature = np.column_stack([neighbor / norm, structure / np.sqrt(2)]).astype(
            np.float32
        )
        groups = min(len(ref), 16, max(2, int(np.sqrt(len(ref) / 8))))
        kmeans = KMeans(n_clusters=groups, n_init=3, random_state=431).fit(
            signature[ref]
        )
        folds.append(
            PeerFold(
                ref,
                query,
                kmeans.labels_,
                kmeans.predict(signature[query]),
                groups,
                obs[query],
            )
        )
    return folds


def role_prior(folds, base_rank, shuffle_seed=None):
    """Estimate each role's expected baseline anomaly rank using eight prior observations."""
    risk = np.empty(len(base_rank))
    generator = (
        np.random.default_rng(shuffle_seed) if shuffle_seed is not None else None
    )
    for fold in folds:
        role = fold.reference_role
        count = np.bincount(role, minlength=fold.n_roles)
        sums = np.bincount(
            role, weights=base_rank[fold.reference], minlength=fold.n_roles
        )
        centers = (sums + 4.0) / (count + 8.0)
        assigned = fold.query_role
        if generator is not None:
            assigned = generator.permutation(assigned)
        risk[fold.query] = centers[assigned]
    return risk


def adjacent_prior(folds, base_rank):
    """Average distinct out-of-fold neighbors' anomaly ranks with eight prior observations."""
    prior = np.empty(len(base_rank))
    for fold in folds:
        edges = fold.eligible_neighbors
        sums = edges @ base_rank[fold.reference]
        counts = np.asarray(edges.sum(axis=1)).ravel()
        prior[fold.query] = (sums + 4.0) / (counts + 8.0)
    return prior


def peer_conditioned_score(features, edges, scores, *, weight=0.2, mode="role"):
    """Fuse the role-specific mean anomaly rank with a published FMGAD score."""
    if not (0.0 <= weight <= 1.0):
        raise ValueError("Expected weight between 0 and 1")
    if len(features) != len(scores):
        raise ValueError("Expected one score per node")
    if mode not in ("role", "role_and_neighbors"):
        raise ValueError(f"Unknown peer prototype mode: {mode}")
    if weight == 0 or len(scores) < 2:
        return np.asarray(scores).copy()
    base_rank = midranks(scores)
    folds = fit_peer_roles(features, edges)
    risk = role_prior(folds, base_rank)
    if mode == "role_and_neighbors":
        risk = (risk + adjacent_prior(folds, base_rank)) / 2
    return (1 - weight) * base_rank + weight * risk
