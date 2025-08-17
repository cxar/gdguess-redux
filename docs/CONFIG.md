# Configuration and CLI Examples

Canonical commands:

```
python -m src.data.build_manifest --root data/raw --out data/manifest.parquet
python -m src.data.chunk_windows --manifest data/manifest.parquet --out data/windows.parquet --win_s 30 --hop_s 10
python -m src.train.train --config configs/train.yaml
python -m src.index.build_prototypes --ckpt ckpts/best.pt --windows data/windows.parquet --out data/proto.npz
python -m src.index.faiss_index --protos data/proto.npz --out data/index.faiss
python -m src.infer.streaming --ckpt ckpts/best.pt --index data/index.faiss --audio path/to/file.wav
python -m src.eval.evaluate_show_level --preds runs/latest/preds.json --out reports/metrics.json
```

Future: Hydra/YAML configs under `configs/`.
