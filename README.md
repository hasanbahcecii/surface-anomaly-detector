# surface-anomaly-detector

Crack detection on concrete surfaces (bridge decks, pavements, walls) for construction-tech inspection workflows.
A lightweight YOLOv8-nano model is trained on SDNET2018, optimized for edge inference (ONNX / TensorRT), and run as a
sliding window over large surface images to localize cracks.

> Status: **v0.1 — dataset + baseline** done; next up v0.2 (ONNX / TensorRT). See [ROADMAP.md](ROADMAP.md).

## Dataset

[SDNET2018](https://digitalcommons.usu.edu/all_datasets/48/) — ~56k 256×256 concrete patches labeled cracked / uncracked,
across three surface types. It is **image-level labeled** (no bounding boxes), so the baseline is a patch classifier
(`yolov8n-cls`); localization comes from the sliding-window pipeline in v0.3.

Split: 70 / 15 / 15, stratified by surface × label, seed 42 (`data/processed/split_manifest.csv`).

## Training (Google Colab)

All training runs on Colab — open [`notebooks/01_train_baseline_colab.ipynb`](notebooks/01_train_baseline_colab.ipynb)
(baseline) or [`notebooks/02_improve_recall_colab.ipynb`](notebooks/02_improve_recall_colab.ipynb) (v2: 256px + cracked oversampling),
set the runtime to T4 GPU, add `KAGGLE_USERNAME` / `KAGGLE_KEY` as Colab secrets, and run all cells.

## Repo layout

```
scripts/prepare_data.py   SDNET2018 -> YOLO-cls train/val/test folders
scripts/evaluate.py       per-class precision / recall / F1 on a split (optional --threshold)
scripts/tune_threshold.py pick the cracked-class threshold on val, report on test
scripts/show_errors.py    image grids of the worst missed cracks / false alarms
notebooks/                Colab notebooks
data/                     (gitignored) raw + processed data
```

## Results

Held-out test split (8,413 patches, 15.1% cracked). Trained 30 epochs on a Colab T4 (~2.1 h), imgsz 224.

| Model | Params | Cracked P | Cracked R | **Cracked F1** | Top-1 acc |
|---|---|---|---|---|---|
| Majority class ("never cracked") | – | – | 0.000 | 0.000 | 0.849 |
| yolov8n-cls baseline | 1.44M | 0.905 | 0.740 | **0.814** | 0.949 |

Confusion matrix (rows = true, cols = predicted):

|  | cracked | uncracked |
|---|---|---|
| **cracked** | 940 | 331 |
| **uncracked** | 99 | 7,043 |

### Decision threshold

Thresholds on P(cracked) are chosen on **val** and reported on **test** (`scripts/tune_threshold.py`).
Val and test agree closely, so the chosen points generalize.

| Operating point | Threshold | Test P | Test R | Test F1 | Missed cracks | False alarms |
|---|---|---|---|---|---|---|
| Argmax (default) | 0.500 | 0.905 | 0.740 | 0.814 | 331 | 99 |
| F1-optimal | 0.465 | 0.894 | 0.748 | 0.815 | 320 | 113 |
| High sensitivity (val recall ≥ 0.90) | 0.095 | 0.568 | 0.898 | 0.696 | 130 | 869 |

Takeaway: a threshold alone can't buy recall cheaply. Reaching 90% recall cuts missed cracks from 331 to 130 but
raises false alarms ~9× (99 → 869), because a quarter of the cracked patches get very low scores. Recall has to come
from the model itself (see roadmap), not the threshold.
