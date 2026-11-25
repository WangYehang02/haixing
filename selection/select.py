"""Select one configuration of one detector on one unlabeled graph."""
import argparse
import csv
import itertools
import json
import math
import re
from pathlib import Path

import numpy as np


TAILS = (0.01, 0.05, 0.10)


def midrank(values):
    """Percentile in [0, 1]; tied values receive their average rank."""
    if len(values) == 1:
        return np.array([0.5])
    _, indices, counts = np.unique(values, return_inverse=True, return_counts=True)
    return (np.cumsum(counts) - (counts + 1) / 2)[indices] / (len(values) - 1)


def margin(scores, count, *, historical=False):
    """AutoGAD Eq. (3), with sample variance in each of two disjoint groups."""
    values = np.asarray(scores, dtype=np.float64)
    if not np.isfinite(values).all() or not 2 <= count <= len(values) - 2:
        raise ValueError("CSM requires finite scores and two nodes per group")
    min_variance = 0.
    if historical:
        # Archived external grid used affine rescaling and a near-constant guard.
        scale = np.max(np.abs(values))
        if scale == 0:
            raise ValueError("constant scores")
        values = values / scale
        span = np.ptp(values)
        if span <= 8 * np.finfo(np.float64).eps:
            raise ValueError("numerically indistinguishable scores")
        values = (values - values.min()) / span
        min_variance = np.finfo(np.float64).eps
    ordered = np.sort(values)
    high, rest = ordered[-count:], ordered[:-count]
    variance = high.var(ddof=1) + rest.var(ddof=1)
    if not np.isfinite(variance) or variance <= min_variance:
        raise ValueError("undefined CSM within-group variance")
    result = float((high.mean() - rest.mean()) / math.sqrt(variance))
    if not np.isfinite(result):
        raise ValueError("nonfinite CSM")
    return result


def membership(scores, count):
    """Top-k set weights, with fractional membership for cutoff ties."""
    cut = np.partition(scores, len(scores) - count)[len(scores) - count]
    above = scores > cut
    tied = scores == cut
    return above.astype(np.float64) + tied * ((count - int(above.sum())) / int(tied.sum()))


def q_statistics(vectors):
    n = vectors.shape[1]
    sizes = [int(n * q) for q in TAILS]
    pairs = list(itertools.combinations(range(len(vectors)), 2))
    centered = []
    for scores in vectors:
        r = midrank(scores)
        centered.append(r - r.mean())
    spearman = []
    for i, j in pairs:
        norm = np.linalg.norm(centered[i]) * np.linalg.norm(centered[j])
        corr = np.dot(centered[i], centered[j]) / norm if norm > 1e-12 else 0.
        spearman.append(float(np.clip(corr, -1, 1)))
    csm, overlap = [], []
    for k in sizes:
        csm.append(float(np.mean([margin(s, k, historical=True) for s in vectors])))
        memberships = [membership(s, k) for s in vectors]
        chance = k / n
        overlap.append(np.mean([(np.dot(memberships[i], memberships[j]) / k - chance) / (1 - chance)
                                for i,j in pairs]))
    return dict(zip(("csm_1","csm_5","csm_10"), csm),
                spearman=float(np.mean(spearman)), overlap=float(np.mean(overlap)))


def objectives(eligible, rule):
    if not eligible:
        raise ValueError("no eligible candidates")
    if rule != "q":
        return {key: stats["mean_csm"] for key, stats in eligible.items()}
    ids = sorted(eligible)
    ranks = {field: midrank([eligible[c][field] for c in ids])
             for field in ("csm_1","csm_5","csm_10","spearman","overlap")}
    multi = np.mean([ranks[f] for f in ("csm_1","csm_5","csm_10")], axis=0)
    total = (multi + ranks["spearman"] + ranks["overlap"]) / 3
    return {cid: float(q) for cid, q in zip(ids, total)}


def winner(scores):
    return min(scores, key=lambda cid: (-scores[cid], cid))


def select(directory, rule, seeds):
    ids = {}
    excluded = {}
    training_masks = {}
    nodes = None
    required = {"raw", "seeds", "collapsed"}
    if rule == "csm-final":
        required.add("final")
    for path in sorted(directory.glob("*.npz")):
        cid = path.stem
        if not re.fullmatch(r"t\d+_i\d+", cid):
            raise ValueError(f"{path.name}: expected a configuration name such as t00_i00")
        with np.load(path, allow_pickle=False) as pack:
            if not required <= set(pack.files) or set(pack.files) - {"raw", "final", "seeds", "collapsed"}:
                raise ValueError(f"{cid}: unexpected candidate fields")
            if not np.array_equal(pack["seeds"], seeds):
                raise ValueError(f"{cid}: seed order differs from the declared search seeds")
            raw = np.asarray(pack["raw"], dtype=np.float64)
            flags = pack["collapsed"]
            if flags.dtype != bool or flags.shape != (len(seeds),):
                raise ValueError(f"{cid}: supply one boolean collapse flag per seed")
            if raw.ndim != 2 or len(raw) != len(seeds) or raw.shape[1] < 4:
                raise ValueError(f"{cid}: expected a seed-by-node raw score matrix")
            if nodes is None:
                nodes = raw.shape[1]
            if nodes != raw.shape[1]:
                raise ValueError(f"{cid}: node count differs from other configurations")
            train = cid.split("_")[0]
            if tuple(flags) != training_masks.setdefault(train, tuple(flags)):
                raise ValueError(f"{cid}: conflicting encoder diagnostics for the same training run")
            if flags.any():
                excluded[cid] = "collapsed encoder"
                continue
            scores = pack["final"] if rule == "csm-final" else raw
            if scores.shape != raw.shape:
                raise ValueError(f"{cid}: final score shape differs from raw score shape")
            try:
                if not np.isfinite(raw).all() or (np.ptp(raw, axis=1) == 0).any():
                    raise ValueError("nonfinite or constant raw scores")
                if rule == "q":
                    ids[cid] = q_statistics(raw)
                else:
                    vals = [margin(s, max(1,int(nodes * .05))) for s in scores]
                    ids[cid] = {"mean_csm": float(np.mean(vals))}
            except ValueError as exc:
                excluded[cid] = str(exc)
    score = objectives(ids, rule)
    return dict(rule=rule, winner=winner(score), seeds=seeds, nodes=nodes,
                eligible=len(ids), excluded=excluded, objective=score)


def replay(csv_file):
    groups = {}
    expected = {}
    with csv_file.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            name, cid = row["dataset"], row["config"]
            group = groups.setdefault(name, {})
            if cid in group:
                raise ValueError(f"duplicate configuration {name}/{cid}")
            group[cid] = {field:float(row[field]) for field in ("csm_1","csm_5","csm_10","spearman","overlap")}
            if row["historical_winner"] == "1":
                if name in expected:
                    raise ValueError(f"multiple historical winners for {name}")
                expected[name] = cid
    if not groups:
        raise ValueError("empty archive")
    result = {name:winner(objectives(rows, "q")) for name,rows in groups.items()}
    if result != expected:
        raise ValueError("recomputed Q winners differ from the archived choices")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="select using trial predictions")
    run.add_argument("rule", choices=("q","csm-raw","csm-final"))
    run.add_argument("directory", type=Path)
    run.add_argument("--seeds", type=int, nargs="+", help="declared order, default 0 1 2 for Q or 10 11 12 for CSM")
    check = sub.add_parser("replay", help="repeat the external Q ranking from archived candidate statistics")
    check.add_argument("file", nargs="?", type=Path, default=Path(__file__).with_name("external_q_statistics.csv"))
    args = parser.parse_args()
    if args.command == "replay":
        result = replay(args.file)
    else:
        seeds = args.seeds or ([0,1,2] if args.rule == "q" else [10,11,12])
        if len(seeds) < 2 or len(set(seeds)) != len(seeds):
            parser.error("supply at least two distinct search seeds")
        result = select(args.directory, args.rule, seeds)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
