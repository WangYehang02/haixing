"""Train FMGAD-stable on one native graph without using evaluation labels."""
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
NATIVE = ("books", "disney", "enron", "reddit", "weibo")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset", choices=NATIVE)
    p.add_argument("seed", type=int)
    p.add_argument("--graph", type=Path, help="NPZ with x and edge_index; default: native PyGOD loader")
    p.add_argument("--encoder", type=Path, help="Optional archived AE checkpoint for this graph and seed")
    p.add_argument("--output-dir", type=Path, default=Path("runs"))
    p.add_argument("--arm", choices=("stable","resid_center","old_agg","zero","beta2"), default="stable")
    args = p.parse_args()
    import numpy as np
    import torch
    from torch_geometric.data import Data
    from fmgad import detector
    from fmgad.detector import ResFlowGAD

    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if not torch.cuda.is_available():
        p.error("FMGAD training requires a CUDA GPU")
    cfg = json.loads((ROOT/"configs/main"/(args.dataset+".json")).read_text(encoding="utf-8"))
    training = dict(cfg["model"])
    if args.arm == "zero":
        training["residual_target"] = "zero"
    elif args.arm == "beta2":
        training["residual_loss_weight"] = 2.0
    set_seed(args.seed)
    model = ResFlowGAD(**training)
    if args.graph:
        with np.load(args.graph,allow_pickle=False) as f:
            graph = Data(x=torch.from_numpy(f["x"]).float(), edge_index=torch.from_numpy(f["edge_index"]).long())
    else:
        data = model._load_dataset(args.dataset)
        graph = Data(x=data.x.float(), edge_index=data.edge_index.long())
    reference = json.loads((ROOT/"data/native_signatures.json").read_text())
    signatures = reference[args.dataset]
    for key in ("x","edge_index"):
        array = getattr(graph,key).contiguous().numpy()
        if hashlib.sha256(memoryview(array).cast("B")).hexdigest() != signatures[key]:
            p.error(f"{args.dataset}: {key} differs from the graph used in the main results (values or order)")
    encoder_hash = None
    encoder_item = signatures["encoders"].get(str(args.seed))
    if args.encoder:
        encoder_hash = hashlib.sha256(args.encoder.read_bytes()).hexdigest()
        if encoder_item is None or encoder_hash != encoder_item["sha256"]:
            p.error("--encoder must match the archived autoencoder for this dataset and seed")

    target = args.output_dir/args.dataset/f"s{args.seed}"
    target.mkdir(parents=True,exist_ok=True)
    orig_ae, orig_free = model._train_ae_once, model._train_dm_free
    normalize = model._normalize_clip_with_stats
    aggregate = detector.compute_dual_residuals_with_degree

    def train_ae(data, directory):
        nonlocal encoder_hash
        if args.encoder:
            epoch = encoder_item["epoch"]
            shutil.copyfile(args.encoder, Path(directory)/f"{epoch}.pt")
        else:
            epoch = orig_ae(data, directory)
            encoder_hash = hashlib.sha256((Path(directory)/f"{epoch}.pt").read_bytes()).hexdigest()
        set_seed(args.seed)
        return epoch

    def train_free(data, directory):
        result = orig_free(data, directory)
        set_seed(10000+args.seed)
        return result

    def center_residual(values):
        normalized, mean, scale = normalize(values)
        width = model.hid_dim
        if values.shape[1] == 2 * width:
            r = values[:, width:].double()
            shifted = r - r[:1]
            centered = (shifted - shifted.mean(dim=0, keepdim=True)).clamp(-10,10)
            normalized = torch.cat([normalized[:, :width],centered.to(values.dtype)],dim=1)
            scale = torch.cat([scale[:, :width],torch.ones_like(scale[:, width:])],dim=1)
        return normalized, mean, scale

    with tempfile.TemporaryDirectory(dir=target) as working, ExitStack() as overrides:
        overrides.enter_context(patch.dict(os.environ,FMGAD_MODEL_ROOT=working,FMGAD_REUSE_CHECKPOINTS="0",
                                           FMGAD_INFERENCE_SEED=str(cfg["runtime"]["inference_seed"])))
        overrides.enter_context(patch.object(model,"_train_ae_once",train_ae))
        overrides.enter_context(patch.object(model,"_train_dm_free",train_free))
        if args.arm == "resid_center":
            overrides.enter_context(patch.object(model,"_normalize_clip_with_stats",center_residual))
        elif args.arm == "old_agg":
            overrides.enter_context(patch.object(detector,"compute_dual_residuals_with_degree",
                                                 lambda h, e, **kwargs: aggregate(h,e,stable=False)))
        raw, score = model.forward(args.dataset,graph=graph,predict_only=True)
        if not (raw.shape == score.shape == (len(graph.x),) and torch.isfinite(score).all() and torch.isfinite(raw).all()):
            raise ValueError("Invalid prediction vector")
        file = Path(working)/"predictions.npz"
        np.savez_compressed(file,raw=raw.numpy(),score=score.numpy(),seed=args.seed,arm=args.arm,
                            encoder_sha256=encoder_hash,encoder_source="archived" if args.encoder else "fresh")
        out = target/(args.arm+".npz")
        os.replace(file,out)
    print(out)


if __name__ == "__main__":
    main()
