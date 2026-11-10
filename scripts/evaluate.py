"""Evaluate a YOLOv8-cls model on a split, reporting per-class precision/recall/F1.

Top-1 accuracy alone is misleading on SDNET2018 (~15% cracked), so the headline
metric for this project is F1 on the `cracked` class.

Usage:
    python scripts/evaluate.py --weights runs/classify/baseline/weights/best.pt --data data/processed --split test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.metrics import classification_report, confusion_matrix
from ultralytics import YOLO


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--out", type=Path, help="optional JSON file for the metrics")
    args = ap.parse_args()

    model = YOLO(str(args.weights))
    names = model.names  # {idx: class_name}

    split_dir = args.data / args.split
    paths, y_true = [], []
    for idx, name in names.items():
        for p in sorted((split_dir / name).iterdir()):
            paths.append(str(p))
            y_true.append(idx)

    y_pred = []
    for i in range(0, len(paths), args.batch):
        results = model.predict(paths[i:i + args.batch], imgsz=args.imgsz, verbose=False)
        y_pred += [int(r.probs.top1) for r in results]

    labels = sorted(names)
    target_names = [names[i] for i in labels]
    print(classification_report(y_true, y_pred, labels=labels, target_names=target_names, digits=4))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("confusion matrix (rows=true, cols=pred):", target_names)
    print(cm)

    if args.out:
        report = classification_report(y_true, y_pred, labels=labels, target_names=target_names, output_dict=True)
        report["confusion_matrix"] = cm.tolist()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
        print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
