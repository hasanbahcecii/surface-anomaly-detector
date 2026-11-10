# surface-anomaly-detector

Crack detection on concrete surfaces (bridge decks, pavements, walls) for construction-tech inspection workflows.
A lightweight YOLOv8-nano model is trained on SDNET2018, optimized for edge inference (ONNX / TensorRT), and run as a
sliding window over large surface images to localize cracks.

> Status: **v0.1 — dataset + baseline** (in progress). See [ROADMAP.md](ROADMAP.md).

## Dataset

[SDNET2018](https://digitalcommons.usu.edu/all_datasets/48/) — ~56k 256×256 concrete patches labeled cracked / uncracked,
across three surface types. It is **image-level labeled** (no bounding boxes), so the baseline is a patch classifier
(`yolov8n-cls`); localization comes from the sliding-window pipeline in v0.3.

Split: 70 / 15 / 15, stratified by surface × label, seed 42 (`data/processed/split_manifest.csv`).

## Training (Google Colab)

All training runs on Colab — open [`notebooks/01_train_baseline_colab.ipynb`](notebooks/01_train_baseline_colab.ipynb),
set the runtime to T4 GPU, add `KAGGLE_USERNAME` / `KAGGLE_KEY` as Colab secrets, and run all cells.

## Repo layout

```
scripts/prepare_data.py   SDNET2018 -> YOLO-cls train/val/test folders
scripts/evaluate.py       per-class precision / recall / F1 on a split
notebooks/                Colab notebooks
data/                     (gitignored) raw + processed data
```

## Results

| Model | Split | Cracked F1 | Top-1 acc |
|---|---|---|---|
| yolov8n-cls baseline | test | – | – |
