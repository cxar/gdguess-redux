# Configuration and CLI Examples

Canonical commands:

```
python -m src.data.build_manifest --root data/raw --out data/manifest.parquet --format parquet --log-errors data/manifest_errors.txt
python -m src.data.chunk_windows --manifest data/manifest.parquet --out data/windows.parquet --win_s 30 --hop_s 10 --format parquet
python -m src.train.train --config configs/train.yaml
python -m src.index.build_prototypes --ckpt ckpts/best.pt --windows data/windows.parquet --out data/proto.npz
python -m src.index.faiss_index --protos data/proto.npz --out data/index.faiss
python -m src.infer.streaming --ckpt ckpts/best.pt --index data/index.faiss --audio path/to/file.wav
python -m src.eval.evaluate_show_level --preds runs/latest/preds.json --out reports/metrics.json
```

Flags:
- build_manifest: --exts flac,wav,mp3; --num-workers N; --log-errors errors.txt
- chunk_windows: --win_s, --hop_s, --drop_last (default True)

## Augmentation Configuration

Example augmentation pipeline config (YAML):
```yaml
augmentations:
  room_ir:
    enabled: true
    p: 0.7  # probability of application
    rt60_range: [0.2, 1.2]
  
  device_color:
    enabled: true
    p: 0.5
    modes: ["phone", "laptop", "random"]
  
  noise:
    enabled: true
    p: 0.6
    snr_range: [5, 20]
    types: ["babble", "hvac", "traffic"]
  
  spec_augment:
    enabled: true
    p: 0.8
    time_masks: 2
    freq_masks: 2
    max_time_width: 10
    max_freq_width: 8
  
  mixstyle:
    enabled: true
    p: 0.5
    alpha: 0.1
```

Future: Hydra/YAML configs under `configs/`.
