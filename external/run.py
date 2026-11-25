"""Train the external FMGAD-stable backbone using a released dataset configuration."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code"))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_graph(path, config, reference):
    import numpy as np

    with np.load(path, allow_pickle=False) as pack:
        if set(pack.files) != {"x", "edge_index"}:
            raise ValueError("Expected an NPZ containing exactly x and edge_index")
        x, edges = pack["x"], pack["edge_index"]
    graph = config["graph"]
    if x.shape != (graph["nodes"], graph["features"]) or edges.shape != (2, graph["edge_entries"]):
        raise ValueError("Graph dimensions differ from the selected graph version")
    # Recompressing the arrays may change the NPZ hash. Raw tensor bytes maintain
    # the registration between the data and the published node-wise graph prior.
    for key, values in (("x", x), ("edge_index", edges)):
        digest = hashlib.sha256(np.ascontiguousarray(values).view(np.uint8)).hexdigest()
        if digest != reference["tensor_sha256"][key]:
            raise ValueError(f"{key} does not match the features, edges or node order of the published graph")
    if edges.dtype.kind not in "iu" or not np.isfinite(x).all():
        raise ValueError("Expected finite node attributes and integer edges")
    return x, edges


def load_geometry(ds, edges, cfg, reference):
    import numpy as np
    from fmgad.geometry_polarity import GeometryState
    from fmgad.graph_polarity import midrank

    path = ROOT / "geometry" / (ds + ".npz")
    if sha(path) != reference["geometry_sha256"]:
        raise ValueError(f"Geometry reference checksum mismatch for {ds}")
    n = cfg["graph"]["nodes"]
    degree = np.zeros(n, dtype=np.int64)
    for left in range(0, edges.shape[1], 1_000_000):
        a, b = edges[:, left:left+1_000_000]
        degree += np.bincount(b[a != b], minlength=n)
    with np.load(path, allow_pickle=False) as pack:
        return GeometryState(pack["reference"], degree, midrank(degree), reference["geometry"], pack["feature"])


def set_backend(name):
    if name.startswith("gram"):
        import sparse_execution
        import blocked_virtual
        sparse_execution.install()
        blocked_virtual.install()
    elif name == "sparse_gcn":
        import sparse_gcn
        sparse_gcn.install()
    elif name != "dense":
        raise ValueError(f"Unknown backend {name}")
    os.environ.update(FMGAD_STRUCTURE_GRAM=str(int(name.startswith("gram"))),
                      FMGAD_ZERO_NORM_SUBGRADIENT=str(int(name.endswith("zero_subgradient"))))
    if "static_sparse" in name:
        import static_sparse_mm
        static_sparse_mm.install()
    import chunked_flow
    chunked_flow.install(block_size=8192)


def set_seed(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def archive_encoder(folder, cfg, seed, reference):
    if folder is None or seed not in (0, 1, 2):
        return None
    weights = folder / cfg["dataset"] / f"s{seed}.pt"
    info = read_json(weights.with_suffix(".json"))
    expected = read_json(ROOT/"encoder_signatures.json")[cfg["dataset"]][str(seed)]
    if (info["seed"], info["train"], info["graph_sha256"]) != (
        seed, cfg["selection"]["train_index"], reference["archive_graph_sha256"]
    ) or any(info[key] != value for key, value in expected.items()) or sha(weights) != expected["ae_sha256"]:
        raise ValueError(f"Encoder provenance does not match {cfg['dataset']}, seed {seed}")
    return weights, int(info["epoch"]), info["ae_sha256"]


def train(cfg, seed, x, edges, state, encoder, work):
    import numpy as np
    import torch
    from torch_geometric.data import Data
    from fmgad import detector
    from fmgad.detector import ResFlowGAD
    from fmgad.geometry_polarity import score_geometry
    from fmgad.graph_ops import smooth_scores_by_graph
    from stable_residuals import compute_dual_residuals_with_degree

    if not torch.cuda.is_available():
        raise RuntimeError("The released training backbone requires CUDA")
    runtime = cfg["runtime"]
    torch.set_num_threads(1)
    os.environ.update(FMGAD_INFERENCE_SEED=str(runtime["inference_seed"]), FMGAD_REUSE_CHECKPOINTS="0")
    set_backend(runtime["backend"])
    graph = Data(x=torch.from_numpy(x).float().cuda(), edge_index=torch.from_numpy(edges).long().cuda())
    fit_ae, fit_free = ResFlowGAD._train_ae_once, ResFlowGAD._train_dm_free
    encoder_info = {}

    def autoencoder(model, g, path):
        if encoder:
            weights, epoch, digest = encoder
            shutil.copyfile(weights, Path(path)/f"{epoch}.pt")
        else:
            epoch = fit_ae(model, g, path)
            if epoch <= 0:
                raise RuntimeError("Encoder did not produce a finite training update")
            digest = sha(Path(path)/f"{epoch}.pt")
        encoder_info.update(source="archived" if encoder else "fresh", epoch=epoch, sha256=digest)
        model.ae.requires_grad_(False)
        set_seed(seed + runtime["flow_seed_offset"])
        return epoch

    def free_flow(model, g, path):
        value = fit_free(model, g, path)
        set_seed(seed + runtime["conditional_seed_offset"])
        return value

    def residual(h, edge_index):
        return compute_dual_residuals_with_degree(h, edge_index, stable=True)

    def normalize(model, z):
        value = z.double()
        anchor = value[:1]
        shifted = value-anchor
        center = shifted.mean(dim=0, keepdim=True)
        std = shifted.std(dim=0, keepdim=True, unbiased=len(z)>1) + 1e-8
        normalized = ((shifted-center)/std).clamp(-10,10)
        return normalized.to(z.dtype), (anchor+center).to(z.dtype), std.to(z.dtype)

    model = ResFlowGAD(**cfg["model"])
    set_seed(seed)
    with patch.object(detector, "compute_dual_residuals_with_degree", residual), \
         patch.object(ResFlowGAD, "_normalize_clip_with_stats", normalize), \
         patch.object(ResFlowGAD, "_train_ae_once", autoencoder), \
         patch.object(ResFlowGAD, "_train_dm_free", free_flow):
        model.fit_unlabeled(graph, str(work))
        inference = cfg["inference"]
        model.weight = inference["weight"]
        model.sample_steps = inference["sample_steps"]
        model.proto_alpha = inference["proto_alpha"]
        model.use_score_smoothing = False
        with torch.no_grad():
            raw_tensor = model.score_samples(model.dm_proto, model.dm, graph, apply_polarity=False).cuda()
            raw = smooth_scores_by_graph(raw_tensor, graph.edge_index, inference["score_smoothing_alpha"],
                                         raw_tensor.device).cpu().numpy()
    final, decisions = score_geometry(raw, state)
    if raw.shape != final.shape or raw.shape != (len(x),) or not np.isfinite([raw, final]).all():
        raise ValueError("Unexpected or non-finite output scores")
    return raw, final, decisions, encoder_info


def main():
    references = read_json(ROOT/"geometry/index.json")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=sorted(references))
    parser.add_argument("seed", type=int, choices=(0,1,2,3,42))
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing <dataset>.npz")
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    parser.add_argument("--encoder-dir", type=Path, help="Archived AE root for seeds 0, 1, 2")
    parser.add_argument("--check-only", action="store_true", help="Validate input and geometry without GPU training")
    args = parser.parse_args()

    ds, seed = args.dataset, args.seed
    cfg_path = ROOT.parent/"configs/main"/(ds+".json")
    cfg = read_json(cfg_path)
    if cfg["family"] != "external" or cfg["dataset"] != ds or seed not in cfg["runtime"]["seeds"]:
        raise ValueError("Configuration does not match the selected graph and seed")
    source = args.data_dir / (ds+".npz")
    reference = references[ds]
    x, edges = load_graph(source, cfg, reference)
    state = load_geometry(ds, edges, cfg, reference)
    encoder = archive_encoder(args.encoder_dir, cfg, seed, reference)
    if args.check_only:
        print(f"Verified {ds}: {x.shape[0]} nodes, {x.shape[1]} features, {edges.shape[1]} edges.")
        return

    import numpy as np
    parent = args.output_dir/ds
    parent.mkdir(parents=True, exist_ok=True)
    out = parent/f"s{seed}"
    if out.exists():
        raise FileExistsError(f"Run directory already exists: {out}. Choose a new --output-dir.")
    with TemporaryDirectory(dir=parent) as temporary:
        stage = Path(temporary)/"run"
        stage.mkdir()
        raw, score, decisions, ae = train(cfg, seed, x, edges, state, encoder, stage/"weights")
        name = stage/"stable.npz"
        np.savez_compressed(name,raw=raw,score=score)
        record = dict(dataset=ds,seed=seed,model=cfg["model"],inference=cfg["inference"],runtime=cfg["runtime"],
                      config_sha256=sha(cfg_path),input_sha256=sha(source), encoder=ae,
                      geometry_sha256=reference["geometry_sha256"],diagnostics=decisions,
                      scores_sha256=sha(name),labels_used=False)
        (stage/"stable.json").write_text(json.dumps(record,indent=2)+"\n",encoding="utf-8")
        stage.rename(out)
    print(out/"stable.npz")


if __name__ == "__main__":
    main()
