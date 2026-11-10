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
decks/pavements/walls and the same crack ratio.

Usage:
    python scripts/prepare_data.py --src data/raw --out data/processed
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
    args = ap.parse_args()

    if abs(sum(args.ratios) - 1.0) > 1e-6:
        ap.error("ratios must sum to 1")

    samples = collect(args.src)
    if not samples:
        sys.exit(f"no labeled images found under {args.src}")

    if args.out.exists():
        shutil.rmtree(args.out)

    splits = stratified_split(samples, args.ratios, args.seed)
    manifest_rows = []
    for split, items in splits.items():
        for path, surface, label in items:
            dst = args.out / split / label / f"{surface}_{path.name}"
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
    for split, items in splits.items():
        counts = Counter(label for _, _, label in items)
        ratio = counts["cracked"] / max(len(items), 1)
        print(f"  {split:5s} {len(items):6d}  cracked={counts['cracked']:5d}  uncracked={counts['uncracked']:6d}  crack_ratio={ratio:.3f}")


if __name__ == "__main__":
    main()
