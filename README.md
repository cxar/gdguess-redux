# gdguess2

Accuracy-first audio identification pipeline with intermingled sources. Milestones M0–M10 with docs-drift CI guard.

## Quickstart

```
# 1) Build manifest & windows
python -m src.data.build_manifest --root data/raw --out data/manifest.parquet
python -m src.data.chunk_windows --manifest data/manifest.parquet --out data/windows.parquet --win_s 30 --hop_s 10

# 2) Train (baseline)
python -m src.train.train --config configs/train.yaml

# 3) Build prototypes & index
python -m src.index.build_prototypes --ckpt ckpts/best.pt --windows data/windows.parquet --out data/proto.npz
python -m src.index.faiss_index --protos data/proto.npz --out data/index.faiss

# 4) Stream inference
python -m src.infer.streaming --ckpt ckpts/best.pt --index data/index.faiss --audio path/to/file.wav

# 5) Evaluate
python -m src.eval.evaluate_show_level --preds runs/latest/preds.json --out reports/metrics.json
```

## Smoke test (one command)

```
python -c "import src; print('ok')"
```

## Architecture (high level)

- Data ingestion → windows → dataset/sampler
- Encoder (OpenBEATs-Large) → heads (ArcFace/SupCon/Classifier)
- Training with robust augs; prototypes + FAISS for retrieval
- Streaming inference with posterior and early-exit
- Evaluation suite and serving endpoints

See docs/INDEX.md for full documentation.
