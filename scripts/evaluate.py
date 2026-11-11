"""Evaluate a YOLOv8-cls model on a split, reporting per-class precision/recall/F1.

Top-1 accuracy alone is misleading on SDNET2018 (~15% cracked), so the headline
metric for this project is F1 on the `cracked` class.

By default a patch is `cracked` when that class has the highest probability
(equivalent to threshold 0.5 for two classes). Pass --threshold to flag a patch
as cracked whenever P(cracked) >= threshold; scripts/tune_threshold.py picks one.

Usage:
    python scripts/evaluate.py --weights runs/classify/baseline/weights/best.pt --data data/processed --split test
    python scripts/evaluate.py ... --threshold 0.25
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from ultralytics import YOLO

POSITIVE = "cracked"


def load_split(split_dir: Path, names: dict[int, str]) -> tuple[list[str], np.ndarray]:
    paths, y_true = [], []
    for idx, name in names.items():
        for p in sorted((split_dir / name).iterdir()):
            paths.append(str(p))
            y_true.append(idx)
    return paths, np.array(y_true)


def predict_probs(model: YOLO, paths: list[str], imgsz: int, batch: int) -> np.ndarray:
    """Return an (N, num_classes) array of class probabilities."""
    probs = []
    for i in range(0, len(paths), batch):
        results = model.predict(paths[i:i + batch], imgsz=imgsz, verbose=False)
        probs += [r.probs.data.cpu().numpy() for r in results]
    return np.stack(probs)


def apply_threshold(probs: np.ndarray, pos_idx: int, threshold: float | None) -> np.ndarray:
    if threshold is None:
        return probs.argmax(axis=1)
    neg_idx = 1 - pos_idx
    return np.where(probs[:, pos_idx] >= threshold, pos_idx, neg_idx)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--threshold", type=float, help=f"P({POSITIVE}) cut-off; default argmax")
    ap.add_argument("--out", type=Path, help="optional JSON file for the metrics")
    args = ap.parse_args()

    model = YOLO(str(args.weights))
    names = model.names  # {idx: class_name}
    pos_idx = next(i for i, n in names.items() if n == POSITIVE)

    paths, y_true = load_split(args.data / args.split, names)
    probs = predict_probs(model, paths, args.imgsz, args.batch)
    y_pred = apply_threshold(probs, pos_idx, args.threshold)

    labels = sorted(names)
    target_names = [names[i] for i in labels]
    print(f"threshold: {args.threshold if args.threshold is not None else 'argmax'}")
    print(classification_report(y_true, y_pred, labels=labels, target_names=target_names, digits=4))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("confusion matrix (rows=true, cols=pred):", target_names)
    print(cm)

    if args.out:
        report = classification_report(y_true, y_pred, labels=labels, target_names=target_names, output_dict=True)
        report["threshold"] = args.threshold
        report["confusion_matrix"] = cm.tolist()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
        print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
