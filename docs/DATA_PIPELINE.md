# Data Pipeline

## Overview
Raw audio → manifest → windows → dataset → batched training

```
data/raw/YYYY-MM-DD/*.mp3
    ↓ (build_manifest.py)
data/manifest.parquet
    ↓ (chunk_windows.py)  
data/windows.parquet
    ↓ (AudioWindowDataset)
mel spectrograms + metadata
    ↓ (LineageAwareBatchSampler)
class-balanced batches
```

## Components

### 1. Dataset (`AudioWindowDataset`)
- **Input**: windows.parquet + audio files
- **Output**: mel spectrograms + metadata
- **Features**:
  - Loads 30s audio windows on-demand
  - Applies audio standardization (24kHz, mono, -23 LUFS)
  - Extracts mel spectrograms (80 bins, configurable)
  - Supports augmentation pipeline integration
  - Returns: `{audio, mel, date_id, file_id, window_idx, track_offset_s}`

### 2. Sampler (`LineageAwareBatchSampler`)
- **Purpose**: Ensures contrastive learning constraints
- **Key constraint**: Positives from different `file_id` within same `date_id`
- **Features**:
  - Class-balanced batching (equal date representation)
  - Configurable samples per date
  - Deterministic with seeds
  - Validates no duplicate file_ids per date

### 3. Alternative (`BalancedRandomSampler`)
- **Purpose**: Simple balanced sampling without lineage constraints
- **Use case**: Non-contrastive training or debugging

## Configuration

```python
from src.data.dataset import AudioWindowDataset
from src.data.sampler import LineageAwareBatchSampler
from torch.utils.data import DataLoader

# Dataset setup
dataset = AudioWindowDataset(
    windows_path="data/windows.parquet",
    audio_root="/path/to/audio",
    target_sr=24000,
    n_mels=80,
    apply_augmentations=True,
    augmentation_config={
        'room_ir': {'enabled': True, 'p': 0.7},
        'device_color': {'enabled': True, 'p': 0.5},
        'noise': {'enabled': True, 'p': 0.6, 'snr_range': [5, 20]},
        'spec_augment': {'enabled': True, 'p': 0.8},
    }
)

# Sampler setup
sampler = LineageAwareBatchSampler(
    windows_df=pd.read_parquet("data/windows.parquet"),
    batch_size=32,
    samples_per_date=4,  # 8 dates per batch
    shuffle=True,
    seed=42
)

# DataLoader
dataloader = DataLoader(dataset, batch_sampler=sampler, num_workers=4)
```

## Validation
- `validate_batch_lineage()` ensures sampler constraints
- Tests verify determinism and correctness
- Integration tests confirm end-to-end pipeline

## Performance Notes
- On-demand audio loading (only needed windows)
- Efficient indexing via pandas groupby
- Parallel loading with DataLoader num_workers
- Mel extraction cached per window (future optimization)
