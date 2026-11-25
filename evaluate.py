"""Evaluate saved FMGAD scores against labels in the original graph node order."""
import argparse
import json

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("predictions", help="NPZ from run_native.py or external/run.py")
    p.add_argument("labels", help="1D integer NPY array: -1 unlabeled, 0 normal, 1 anomalous")
    args = p.parse_args()
    labels = np.load(args.labels, allow_pickle=False)
    if labels.ndim != 1 or not np.isin(labels, (-1,0,1)).all():
        p.error("Expected one label from {-1,0,1} per node")
    mask = labels >= 0
    if len(np.unique(labels[mask])) != 2:
        p.error("ROC-AUC requires both classes on evaluated nodes")
    result = {"nodes":len(labels),"evaluated":int(mask.sum()),"positives":int((labels==1).sum())}
    with np.load(args.predictions, allow_pickle=False) as pred:
        for key in ("raw","score"):
            values = pred[key]
            if values.shape != labels.shape or not np.isfinite(values).all():
                p.error(f"{key} must contain one finite value per graph node")
            result[key] = {"roc_auc":roc_auc_score(labels[mask],values[mask]),
                           "average_precision":average_precision_score(labels[mask],values[mask])}
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
