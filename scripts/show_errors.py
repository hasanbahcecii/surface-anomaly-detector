"""Save image grids of the model's worst mistakes on a split.

    missed_cracks.png   cracked patches with the lowest P(cracked)  (false negatives)
    false_alarms.png    uncracked patches with the highest P(cracked) (false positives)

Looking at these tells whether misses are hard cases (hairline cracks, shadows,
texture) or label noise, which decides what to try next.

Usage:
    python scripts/show_errors.py --weights <run>/weights/best.pt --data data/processed --out-dir <run>/errors
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw


def make_grid(paths: list[str], captions: list[str], out: Path, cols: int = 8, size: int = 160) -> None:
    rows = max(1, -(-len(paths) // cols))
    cap_h = 18
    grid = Image.new("RGB", (cols * size, rows * (size + cap_h)), "white")
    draw = ImageDraw.Draw(grid)
    for i, (p, cap) in enumerate(zip(paths, captions)):
        x, y = (i % cols) * size, (i // cols) * (size + cap_h)
        with Image.open(p) as im:
            grid.paste(im.convert("RGB").resize((size, size)), (x, y))
        draw.text((x + 4, y + size + 3), cap, fill="black")
    out.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=32, help="images per grid")
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()

    import numpy as np
    from ultralytics import YOLO
    from evaluate import POSITIVE, load_split, predict_probs

    model = YOLO(str(args.weights))
    pos_idx = next(i for i, n in model.names.items() if n == POSITIVE)
    paths, y_true = load_split(args.data / args.split, model.names)
    p_crack = predict_probs(model, paths, args.imgsz, args.batch)[:, pos_idx]
    is_crack = y_true == pos_idx

    crack_ids = np.flatnonzero(is_crack)
    missed = crack_ids[np.argsort(p_crack[crack_ids])][:args.n]
    clean_ids = np.flatnonzero(~is_crack)
    alarms = clean_ids[np.argsort(-p_crack[clean_ids])][:args.n]

    for name, ids in (("missed_cracks", missed), ("false_alarms", alarms)):
        out = args.out_dir / f"{name}.png"
        make_grid([paths[i] for i in ids], [f"P={p_crack[i]:.3f} {Path(paths[i]).name[:14]}" for i in ids], out)
        print(f"saved -> {out}")

    for t in (0.5, 0.1, 0.01):
        print(f"cracked patches with P(cracked) < {t}: {int(np.sum(p_crack[crack_ids] < t))} / {len(crack_ids)}")


if __name__ == "__main__":
    main()
