# Roadmap

## v0.1 — Dataset + Baseline
- [x] SDNET2018 download (Kaggle, via Colab) — 56,092 patches
- [x] train/val/test split (70/15/15, stratified) — `scripts/prepare_data.py`
- [x] YOLOv8n-cls fine-tune (Google Colab, T4, 30 epochs)
- [x] Target: **cracked-class F1 > 0.75** on test — achieved **0.814** (P 0.905 / R 0.740)
      (SDNET2018 has no boxes, so mAP@0.5 is not measurable; F1 replaces it)
- [x] First commit: data prep script + notebook
- [ ] Tune decision threshold for higher crack recall — `scripts/tune_threshold.py` (notebook §7), awaiting Colab run

## v0.2 — Model Optimization
- [ ] ONNX export
- [ ] TensorRT FP16 conversion
- [ ] Inference on Jetson TX2 (fallback: Colab GPU / laptop CPU benchmark)
- [ ] Latency table: PyTorch vs ONNX vs TRT

## v0.3 — Inference Pipeline
- [ ] CLI: `python detect.py --input image/video`
- [ ] Sliding window over large surfaces → crack heatmap / boxes
- [ ] Save annotated output

## v0.4 — Polish + Portfolio
- [ ] Modular code (`src/` layout)
- [ ] Unit tests (pytest)
- [ ] README: problem, dataset, results, demo GIF
- [ ] GitHub Actions: lint + test CI
- [ ] Model weights → Hugging Face Hub or GitHub release

**DONE:** publishable portfolio project with a ConTech narrative.
