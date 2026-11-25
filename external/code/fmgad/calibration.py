"""Label-free orientation relative to an explicitly assumed anomaly prior.

This is a conditional validity check, not a proof that unlabeled anomalies
have been identified. Resampling diagnostics are sensitivity checks, not
independent-node confidence intervals on graph data.
"""
from typing import Any, Dict, List, Tuple

import torch


def unit_rank(value: torch.Tensor) -> torch.Tensor:
    """Average ranks in [0, 1]; equal observations always receive equal ranks."""
    flat = value.detach().reshape(-1).to(torch.float64)
    if not bool(torch.isfinite(flat).all()):
        raise ValueError('Scores must be finite.')
    n = flat.numel()
    if n <= 1:
        return torch.full_like(flat, 0.5)
    _, inverse, counts = torch.unique(flat, sorted=True, return_inverse=True, return_counts=True)
    ends = counts.cumsum(0).to(flat.dtype)
    means = ends - (counts.to(flat.dtype) + 1) / 2
    return means[inverse] / (n - 1)


def rank_correlation(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a - a.mean(), b - b.mean()
    denominator = a.norm() * b.norm()
    return float((a @ b / denominator).clamp(-1, 1)) if denominator > 1e-12 else 0.0


def calibrate_polarity_evidence(
    score: torch.Tensor,
    probes: List[torch.Tensor],
    *,
    min_abs_correlation: float = 0.70,
    score_weight: float = 0.95,
    min_nodes: int = 32,
    stability_bins: int = 8,
) -> Tuple[torch.Tensor, bool, Dict[str, Any]]:
    """Use a main ranking only when its signed alignment with priors is stable.

    Strong positive alignment: keep. Strong negative alignment: reverse the
    ranking, explicitly changing its interpretation from distance to prior-
    aligned evidence. Weak, conflicting or unstable evidence: use the prior.
    If no informative prior exists, abstain and preserve the input score.

    The threshold is a minimum *absolute* rank correlation, not a positive
    cutoff below which every ranking is inverted. All valid probes must agree.
    Leaving out each prior-rank stratum must preserve the proposed sign.
    """
    if not 0 < min_abs_correlation <= 1:
        raise ValueError('min_abs_correlation must be in (0, 1].')
    if not 0 <= score_weight <= 1:
        raise ValueError('score_weight must be in [0, 1].')
    if min_nodes < 3 or stability_bins < 2:
        raise ValueError('Need min_nodes >= 3 and stability_bins >= 2.')
    sr = unit_rank(score)
    diag = {'mode': 'signed_evidence', 'decision': 'abstain', 'flipped': False,
            'min_abs_correlation': float(min_abs_correlation), 'score_weight': float(score_weight),
            'n_nodes': int(sr.numel()), 'num_probes': 0,
            'assumption': 'larger prior values indicate stronger anomaly evidence',
            'stability_kind': 'leave_one_prior_rank_stratum_out_not_a_confidence_interval'}
    valid = []
    for p in probes:
        if p is None or p.numel() != sr.numel():
            continue
        flat = p.detach().reshape(-1).to(device=sr.device, dtype=sr.dtype)
        if not bool(torch.isfinite(flat).all()) or flat.numel() < 2 or bool(flat.max() == flat.min()):
            continue
        valid.append(unit_rank(flat))
    diag['num_probes'] = len(valid)
    if not valid:
        diag['reason'] = 'no_informative_prior'
        return score, False, diag
    prior = torch.stack(valid).mean(0)
    correlations = [rank_correlation(sr, p) for p in valid]
    diag['probe_correlations'] = correlations
    diag['agreement'] = rank_correlation(sr, prior)
    sign = 1 if all(r >= min_abs_correlation for r in correlations) else (
        -1 if all(r <= -min_abs_correlation for r in correlations) else 0)
    reason = 'weak_or_conflicting_alignment'
    if sr.numel() < min_nodes:
        sign, reason = 0, 'insufficient_nodes'
    diagnostics = []
    if sign:
        for p in valid:
            strata = (p * stability_bins).long().clamp_max(stability_bins - 1)
            for block in torch.unique(strata):
                mask = strata != block
                if int(mask.sum()) < min_nodes:
                    continue
                rho = rank_correlation(unit_rank(sr[mask]), unit_rank(p[mask]))
                diagnostics.append(rho)
        if len(diagnostics) < 2 or any(sign * r <= 0 for r in diagnostics):
            sign, reason = 0, 'unstable_direction'
    diag['stratum_correlations'] = diagnostics
    if sign == 0:
        diag.update(decision='prior_fallback', reason=reason)
        return prior.to(dtype=torch.float32).reshape(score.shape), False, diag
    oriented = sr if sign > 0 else 1 - sr
    out = score_weight * oriented + (1 - score_weight) * prior
    diag.update(decision='keep' if sign > 0 else 'flip', flipped=sign < 0,
                reason='strong_stable_signed_alignment')
    return out.to(dtype=torch.float32).reshape(score.shape), sign < 0, diag
