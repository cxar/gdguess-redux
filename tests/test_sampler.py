import pandas as pd
import torch
from pathlib import Path
import tempfile
import numpy as np


def test_import_sampler():
    import src.data.sampler  # noqa: F401


def create_test_windows_df() -> pd.DataFrame:
    """Create test windows DataFrame with proper structure."""
    data = []
    
    # Create multiple dates with multiple files each
    for date_idx in range(3):  # 3 dates
        date_id = f"2025-01-{date_idx+1:02d}"
        
        for file_idx in range(4):  # 4 files per date
            file_id = f"file_{date_idx}_{file_idx}"
            
            for window_idx in range(5):  # 5 windows per file
                data.append({
                    'date_id': date_id,
                    'file_id': file_id,
                    'window_idx': window_idx,
                    'track_offset_s': window_idx * 10.0,
                    'win_s': 30.0,
                    'hop_s': 10.0,
                })
    
    return pd.DataFrame(data)


def test_lineage_aware_sampler():
    """Test LineageAwareBatchSampler ensures no duplicate file_ids per date."""
    from src.data.sampler import LineageAwareBatchSampler, validate_batch_lineage
    
    windows_df = create_test_windows_df()
    
    # Test with batch_size=6, samples_per_date=2 (so 3 dates per batch - matches our 3 test dates)
    sampler = LineageAwareBatchSampler(
        windows_df=windows_df,
        batch_size=6,
        samples_per_date=2,
        drop_last=True,
        shuffle=False,  # Disable shuffle for deterministic testing
        seed=42
    )
    
    assert len(sampler) > 0, "Sampler should produce at least one batch"
    
    # Get first batch
    batches = list(sampler)
    assert len(batches) > 0, "Should produce at least one batch"
    
    first_batch_indices = batches[0]
    assert len(first_batch_indices) == 6, f"Batch size should be 6, got {len(first_batch_indices)}"
    
    # Extract batch data
    batch_data = [windows_df.iloc[idx].to_dict() for idx in first_batch_indices]
    
    # Validate lineage constraints
    validation = validate_batch_lineage(batch_data)
    assert validation['no_duplicate_file_ids_per_date'], "No duplicate file_ids within same date"
    assert validation['has_multiple_dates'], "Batch should contain multiple dates"


def test_balanced_random_sampler():
    """Test BalancedRandomSampler provides balanced date representation."""
    from src.data.sampler import BalancedRandomSampler
    
    windows_df = create_test_windows_df()
    
    sampler = BalancedRandomSampler(
        windows_df=windows_df,
        batch_size=6,
        shuffle=False,
        seed=42,
        drop_last=True
    )
    
    assert len(sampler) > 0, "Sampler should produce batches"
    
    # Collect all samples from multiple batches
    all_batch_data = []
    for batch_indices in sampler:
        batch_data = [windows_df.iloc[idx] for idx in batch_indices]
        all_batch_data.extend(batch_data)
    
    # Check date distribution
    date_counts = {}
    for sample in all_batch_data:
        date_id = sample['date_id']
        date_counts[date_id] = date_counts.get(date_id, 0) + 1
    
    # Should have relatively balanced representation
    date_values = list(date_counts.values())
    assert len(set(date_values)) <= 2, "Date representation should be balanced (within 1 sample difference)"


def test_sampler_determinism():
    """Test that samplers are deterministic with same seed."""
    from src.data.sampler import LineageAwareBatchSampler
    
    windows_df = create_test_windows_df()
    
    # Create two samplers with same seed
    sampler1 = LineageAwareBatchSampler(windows_df, batch_size=6, samples_per_date=2, seed=42)
    sampler2 = LineageAwareBatchSampler(windows_df, batch_size=6, samples_per_date=2, seed=42)
    
    batches1 = list(sampler1)
    batches2 = list(sampler2)
    
    assert len(batches1) == len(batches2), "Same number of batches"
    
    for b1, b2 in zip(batches1, batches2):
        assert b1 == b2, "Batches should be identical with same seed"


def test_dataset_basic_functionality():
    """Test SimpleAudioDataset basic functionality."""
    from src.data.dataset import SimpleAudioDataset
    
    windows_df = create_test_windows_df()
    
    # Create temporary parquet file
    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
        windows_df.to_parquet(tmp.name, index=False)
        tmp_path = tmp.name
    
    try:
        dataset = SimpleAudioDataset(tmp_path, target_sr=16000, n_mels=64)
        
        assert len(dataset) == len(windows_df), "Dataset length should match windows"
        
        # Test single sample
        sample = dataset[0]
        
        required_keys = {'audio', 'mel', 'date_id', 'file_id', 'window_idx', 'track_offset_s', 'idx'}
        assert set(sample.keys()) >= required_keys, f"Sample should contain keys: {required_keys}"
        
        # Check tensor shapes
        assert isinstance(sample['audio'], torch.Tensor), "Audio should be tensor"
        assert isinstance(sample['mel'], torch.Tensor), "Mel should be tensor"
        assert sample['mel'].dim() == 2, "Mel should be 2D (freq, time)"
        assert sample['mel'].shape[0] == 64, "Should have 64 mel bins"
        
        # Check metadata types
        assert isinstance(sample['date_id'], str), "date_id should be string"
        assert isinstance(sample['file_id'], str), "file_id should be string"
        assert isinstance(sample['window_idx'], int), "window_idx should be int"
        
    finally:
        Path(tmp_path).unlink()  # Clean up


def test_dataset_determinism():
    """Test that dataset returns consistent results."""
    from src.data.dataset import SimpleAudioDataset
    
    windows_df = create_test_windows_df()
    
    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
        windows_df.to_parquet(tmp.name, index=False)
        tmp_path = tmp.name
    
    try:
        dataset = SimpleAudioDataset(tmp_path)
        
        # Get same sample twice
        sample1 = dataset[0]
        sample2 = dataset[0]
        
        # Should be identical (deterministic generation based on date_id)
        assert torch.allclose(sample1['audio'], sample2['audio']), "Audio should be deterministic"
        assert torch.allclose(sample1['mel'], sample2['mel']), "Mel should be deterministic"
        assert sample1['date_id'] == sample2['date_id'], "Metadata should match"
        
    finally:
        Path(tmp_path).unlink()


def test_integration_dataset_sampler():
    """Test dataset and sampler working together."""
    from src.data.dataset import SimpleAudioDataset
    from src.data.sampler import LineageAwareBatchSampler
    from torch.utils.data import DataLoader
    
    windows_df = create_test_windows_df()
    
    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
        windows_df.to_parquet(tmp.name, index=False)
        tmp_path = tmp.name
    
    try:
        dataset = SimpleAudioDataset(tmp_path)
        sampler = LineageAwareBatchSampler(
            windows_df=windows_df,
            batch_size=6,
            samples_per_date=2,
            seed=42
        )
        
        # Create DataLoader
        dataloader = DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=0,  # Single threaded for testing
        )
        
        # Test loading one batch
        batch = next(iter(dataloader))
        
        assert 'audio' in batch, "Batch should contain audio"
        assert 'mel' in batch, "Batch should contain mel"
        assert 'date_id' in batch, "Batch should contain date_id"
        
        # Check batch dimensions
        assert batch['audio'].shape[0] == 6, "Batch size should be 6"
        assert batch['mel'].shape[0] == 6, "Mel batch size should be 6"
        
        # Validate lineage constraints
        batch_samples = [
            {
                'date_id': batch['date_id'][i],
                'file_id': batch['file_id'][i],
            }
            for i in range(len(batch['date_id']))
        ]
        
        from src.data.sampler import validate_batch_lineage
        validation = validate_batch_lineage(batch_samples)
        assert validation['no_duplicate_file_ids_per_date'], "Lineage constraint violated"
        
    finally:
        Path(tmp_path).unlink()
