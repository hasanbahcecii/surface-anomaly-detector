"""Prepare SDNET2018 for YOLOv8 classification training.

SDNET2018 ships as 256x256 patches grouped by surface and label:
    D/{CD,UD}  bridge decks   (C = cracked, U = uncracked)
    P/{CP,UP}  pavements
    W/{CW,UW}  walls
Kaggle mirrors sometimes rename these (e.g. Decks/Cracked, Decks/Non-cracked),
so labels are inferred from directory names rather than a fixed layout.

Output (Ultralytics classification layout):
    <out>/{train,val,test}/{cracked,uncracked}/<surface>_<filename>
    <out>/split_manifest.csv   # reproducible record of every file's split

The split is stratified by (surface, label) so each split keeps the same mix of
decks/pavements/walls and the same crack ratio. --oversample-cracked K writes each
cracked *train* image K times (val/test untouched) to counter the ~15% crack ratio;
augmentation makes the copies differ at train time. The split itself does not
depend on K, so val/test are identical across K for the same seed.

Usage:
    python scripts/prepare_data.py --src data/raw --out data/processed
    python scripts/prepare_data.py --src data/raw --out data/processed_os2 --oversample-cracked 2
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

CRACKED_DIRS = {"cd", "cp", "cw", "cracked", "positive"}
UNCRACKED_DIRS = {"ud", "up", "uw", "non-cracked", "noncracked", "uncracked", "negative"}

SURFACE_ALIASES = {
    "d": "deck", "decks": "deck", "deck": "deck",
    "p": "pavement", "pavements": "pavement", "pavement": "pavement",
    "w": "wall", "walls": "wall", "wall": "wall",
}


def infer_label(path: Path) -> str | None:
    for part in reversed(path.parent.parts):
        key = part.lower()
        if key in CRACKED_DIRS:
            return "cracked"
        if key in UNCRACKED_DIRS:
            return "uncracked"
    return None


def infer_surface(path: Path) -> str:
    for part in reversed(path.parent.parts):
        surface = SURFACE_ALIASES.get(part.lower())
        if surface:
            return surface
    # Fall back to the label dir's first letter's sibling: CD -> D, UP -> P, ...
    label_dir = path.parent.name.lower()
    if len(label_dir) == 2 and label_dir[1] in "dpw":
        return SURFACE_ALIASES[label_dir[1]]
    return "unknown"


def collect(src: Path) -> list[tuple[Path, str, str]]:
    samples, skipped = [], 0
    for path in sorted(src.rglob("*")):
        if path.suffix.lower() not in IMAGE_EXTS:
            continue
        label = infer_label(path.relative_to(src))
        if label is None:
            skipped += 1
            continue
        samples.append((path, infer_surface(path.relative_to(src)), label))
    if skipped:
        print(f"[warn] skipped {skipped} images with no recognizable label dir", file=sys.stderr)
    return samples


def stratified_split(samples, ratios, seed):
    groups = defaultdict(list)
    for s in samples:
        groups[(s[1], s[2])].append(s)

    rng = random.Random(seed)
    splits = {"train": [], "val": [], "test": []}
    for key in sorted(groups):
        items = groups[key]
        rng.shuffle(items)
        n_train = round(len(items) * ratios[0])
        n_val = round(len(items) * ratios[1])
        splits["train"] += items[:n_train]
        splits["val"] += items[n_train:n_train + n_val]
        splits["test"] += items[n_train + n_val:]
    return splits


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True, help="extracted SDNET2018 root")
    ap.add_argument("--out", type=Path, required=True, help="output dir (YOLO-cls layout)")
    ap.add_argument("--ratios", type=float, nargs=3, default=(0.70, 0.15, 0.15), metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--link", action="store_true", help="symlink instead of copying (saves disk)")
    ap.add_argument("--oversample-cracked", type=int, default=1, metavar="K", help="copies of each cracked train image")
    args = ap.parse_args()

    if abs(sum(args.ratios) - 1.0) > 1e-6:
        ap.error("ratios must sum to 1")
    if args.oversample_cracked < 1:
        ap.error("--oversample-cracked must be >= 1")

    samples = collect(args.src)
    if not samples:
        sys.exit(f"no labeled images found under {args.src}")

    if args.out.exists():
        shutil.rmtree(args.out)

    splits = stratified_split(samples, args.ratios, args.seed)
    manifest_rows = []
    for split, items in splits.items():
        for path, surface, label in items:
            copies = args.oversample_cracked if split == "train" and label == "cracked" else 1
            for k in range(copies):
                suffix = f"_dup{k}" if k else ""
                dst = args.out / split / label / f"{surface}_{path.stem}{suffix}{path.suffix}"
                dst.parent.mkdir(parents=True, exist_ok=True)
                if args.link:
                    os.symlink(path.resolve(), dst)
                else:
                    shutil.copy2(path, dst)
                manifest_rows.append((split, label, surface, str(path.relative_to(args.src)), str(dst.relative_to(args.out))))

    with open(args.out / "split_manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "label", "surface", "source", "dest"])
        w.writerows(manifest_rows)

    print(f"{len(samples)} images -> {args.out}")
    counts = {split: Counter() for split in splits}
    for split, label, *_ in manifest_rows:
        counts[split][label] += 1
    for split, c in counts.items():
        n = c["cracked"] + c["uncracked"]
        print(f"  {split:5s} {n:6d}  cracked={c['cracked']:5d}  uncracked={c['uncracked']:6d}  crack_ratio={c['cracked'] / max(n, 1):.3f}")


if __name__ == "__main__":
    main()
