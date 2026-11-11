"""Pick a decision threshold for the `cracked` class on val, then report it on test.

For inspection a missed crack costs more than a false alarm, so the default policy
is: the highest-precision threshold that still reaches --target-recall on val.
The F1-optimal threshold and the default argmax (0.5) are reported alongside.

The threshold is chosen on val only; test is touched once, for reporting.

Usage:
    python scripts/tune_threshold.py --weights <run>/weights/best.pt --data data/processed --out-dir <run>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

GRID = np.round(np.arange(0.01, 1.0, 0.005), 3)


def metrics_at(y_pos: np.ndarray, scores: np.ndarray, t: float) -> dict[str, float]:
    pred = scores >= t
    tp = int(np.sum(pred & y_pos))
    fp = int(np.sum(pred & ~y_pos))
    fn = int(np.sum(~pred & y_pos))
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"threshold": float(t), "precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def sweep(y_pos: np.ndarray, scores: np.ndarray) -> list[dict[str, float]]:
    return [metrics_at(y_pos, scores, t) for t in GRID]


def pick_f1(curve: list[dict[str, float]]) -> float:
    return max(curve, key=lambda m: (m["f1"], m["threshold"]))["threshold"]


def pick_recall(curve: list[dict[str, float]], target: float) -> float:
    ok = [m for m in curve if m["recall"] >= target]
    if not ok:  # unreachable target: fall back to the most permissive threshold
        return float(GRID[0])
    return max(ok, key=lambda m: (m["precision"], m["threshold"]))["threshold"]


def save_plot(curve: list[dict[str, float]], marks: dict[str, float], path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    t = [m["threshold"] for m in curve]
    fig, ax = plt.subplots(figsize=(7, 4))
    for key in ("precision", "recall", "f1"):
        ax.plot(t, [m[key] for m in curve], label=key)
    for name, x in marks.items():
        ax.axvline(x, ls="--", lw=1, color="gray")
        ax.text(x, 0.02, f" {name}={x:.3f}", rotation=90, va="bottom", fontsize=8)
    ax.set(xlabel="P(cracked) threshold", ylabel="score (val)", ylim=(0, 1.02), title="Cracked class vs. threshold (val)")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--target-recall", type=float, default=0.90)
    ap.add_argument("--policy", choices=("recall", "f1"), default="recall", help="which threshold to recommend")
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()

    from ultralytics import YOLO
    from evaluate import POSITIVE, load_split, predict_probs

    model = YOLO(str(args.weights))
    names = model.names
    pos_idx = next(i for i, n in names.items() if n == POSITIVE)

    scores, y_pos = {}, {}
    for split in ("val", "test"):
        paths, y_true = load_split(args.data / split, names)
        scores[split] = predict_probs(model, paths, args.imgsz, args.batch)[:, pos_idx]
        y_pos[split] = y_true == pos_idx

    val_curve = sweep(y_pos["val"], scores["val"])
    candidates = {
        "argmax": 0.5,
        "f1_opt": pick_f1(val_curve),
        f"recall>={args.target_recall:.2f}": pick_recall(val_curve, args.target_recall),
    }
    chosen_key = "f1_opt" if args.policy == "f1" else f"recall>={args.target_recall:.2f}"

    rows = []
    for name, t in candidates.items():
        rows.append({
            "name": name,
            "val": metrics_at(y_pos["val"], scores["val"], t),
            "test": metrics_at(y_pos["test"], scores["test"], t),
        })

    print(f"{'policy':<16}{'thr':>7} | {'val P':>7}{'val R':>7}{'val F1':>8} | {'test P':>7}{'test R':>7}{'test F1':>8}{'missed':>8}{'false+':>8}")
    for r in rows:
        v, te = r["val"], r["test"]
        mark = "  <- chosen" if r["name"] == chosen_key else ""
        print(f"{r['name']:<16}{v['threshold']:>7.3f} | {v['precision']:>7.3f}{v['recall']:>7.3f}{v['f1']:>8.3f} | "
              f"{te['precision']:>7.3f}{te['recall']:>7.3f}{te['f1']:>8.3f}{te['fn']:>8d}{te['fp']:>8d}{mark}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = {"chosen": chosen_key, "threshold": candidates[chosen_key], "target_recall": args.target_recall, "candidates": rows}
    (args.out_dir / "threshold.json").write_text(json.dumps(out, indent=2))
    save_plot(val_curve, {k: v for k, v in candidates.items() if k != "argmax"}, args.out_dir / "threshold_curve.png")
    print(f"\nsaved -> {args.out_dir / 'threshold.json'}, {args.out_dir / 'threshold_curve.png'}")


if __name__ == "__main__":
    main()
