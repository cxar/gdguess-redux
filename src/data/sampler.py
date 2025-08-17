from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from torch.utils.data import Sampler
from typing import Iterator, List, Dict, Set


class LineageAwareBatchSampler(Sampler):
    """Batch sampler ensuring positives come from different file_id within same date_id."""
    
    def __init__(
        self,
        windows_df: pd.DataFrame,
        batch_size: int,
        samples_per_date: int = 4,
        drop_last: bool = True,
        shuffle: bool = True,
        seed: int = 42,
    ):
        """
        Args:
            windows_df: DataFrame with columns [date_id, file_id, window_idx, ...]
            batch_size: Total batch size (must be divisible by samples_per_date)
            samples_per_date: Number of samples per date_id in each batch
            drop_last: Whether to drop incomplete batches
            shuffle: Whether to shuffle data
            seed: Random seed for reproducibility
        """
        self.windows_df = windows_df.reset_index(drop=True)
        self.batch_size = batch_size
        self.samples_per_date = samples_per_date
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        
        if batch_size % samples_per_date != 0:
            raise ValueError(f"batch_size ({batch_size}) must be divisible by samples_per_date ({samples_per_date})")
        
        self.dates_per_batch = batch_size // samples_per_date
        
        # Build index structures
        self._build_indices()
        
        # Calculate number of batches
        self.num_batches = len(self.date_ids) // self.dates_per_batch
        if not self.drop_last and len(self.date_ids) % self.dates_per_batch > 0:
            self.num_batches += 1
    
    def _build_indices(self):
        """Build efficient lookup structures."""
        # Group by date_id and file_id
        self.date_to_files = defaultdict(set)
        self.date_file_to_indices = defaultdict(list)
        
        for idx, row in self.windows_df.iterrows():
            date_id = row['date_id']
            file_id = row['file_id']
            
            self.date_to_files[date_id].add(file_id)
            self.date_file_to_indices[(date_id, file_id)].append(idx)
        
        # Filter dates that have enough files for diverse sampling
        min_files_needed = max(2, self.samples_per_date)  # Need at least 2 files for diversity
        self.date_ids = [
            date_id for date_id in self.date_to_files.keys()
            if len(self.date_to_files[date_id]) >= min_files_needed
        ]
        
        if len(self.date_ids) < self.dates_per_batch:
            raise ValueError(
                f"Not enough valid date_ids ({len(self.date_ids)}) for batch requirements "
                f"({self.dates_per_batch} dates per batch). Each date needs at least "
                f"{min_files_needed} different file_ids."
            )
    
    def _sample_from_date(self, date_id: str, num_samples: int, rng: np.random.Generator) -> List[int]:
        """Sample windows from a date_id ensuring different file_ids."""
        available_files = list(self.date_to_files[date_id])
        
        if len(available_files) < num_samples:
            # If not enough files, sample with replacement from files
            # but still ensure we get different files when possible
            sampled_files = rng.choice(available_files, size=num_samples, replace=True)
        else:
            # Sample different files without replacement
            sampled_files = rng.choice(available_files, size=num_samples, replace=False)
        
        # Sample one window from each selected file
        indices = []
        for file_id in sampled_files:
            file_indices = self.date_file_to_indices[(date_id, file_id)]
            sampled_idx = rng.choice(file_indices)
            indices.append(sampled_idx)
        
        return indices
    
    def __iter__(self) -> Iterator[List[int]]:
        """Generate batches of indices."""
        rng = np.random.default_rng(self.seed)
        
        # Shuffle date order if requested
        date_order = self.date_ids.copy()
        if self.shuffle:
            rng.shuffle(date_order)
        
        # Generate batches
        for batch_start in range(0, len(date_order), self.dates_per_batch):
            batch_end = min(batch_start + self.dates_per_batch, len(date_order))
            batch_dates = date_order[batch_start:batch_end]
            
            # Skip incomplete batch if drop_last=True
            if self.drop_last and len(batch_dates) < self.dates_per_batch:
                break
            
            # Sample from each date in the batch
            batch_indices = []
            for date_id in batch_dates:
                date_indices = self._sample_from_date(date_id, self.samples_per_date, rng)
                batch_indices.extend(date_indices)
            
            # Shuffle within batch if requested
            if self.shuffle:
                rng.shuffle(batch_indices)
            
            yield batch_indices
    
    def __len__(self) -> int:
        return self.num_batches


class BalancedRandomSampler(Sampler):
    """Simple balanced sampler that ensures equal representation of date_ids over time."""
    
    def __init__(
        self,
        windows_df: pd.DataFrame,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 42,
        drop_last: bool = True,
    ):
        """
        Args:
            windows_df: DataFrame with date_id column
            batch_size: Batch size
            shuffle: Whether to shuffle
            seed: Random seed
            drop_last: Whether to drop incomplete batches
        """
        self.windows_df = windows_df.reset_index(drop=True)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        
        # Build date_id to indices mapping
        self.date_to_indices = defaultdict(list)
        for idx, row in self.windows_df.iterrows():
            self.date_to_indices[row['date_id']].append(idx)
        
        self.date_ids = list(self.date_to_indices.keys())
        
        # Calculate total samples and batches
        self.total_samples = len(self.windows_df)
        self.num_batches = self.total_samples // batch_size
        if not drop_last and self.total_samples % batch_size > 0:
            self.num_batches += 1
    
    def _create_balanced_epoch(self, rng: np.random.Generator) -> List[int]:
        """Create an epoch with balanced date representation."""
        # Calculate samples needed
        samples_needed = self.num_batches * self.batch_size if self.drop_last else self.total_samples
        
        # Create balanced sampling
        all_indices = []
        date_counters = {date_id: 0 for date_id in self.date_ids}
        
        while len(all_indices) < samples_needed:
            # Pick dates round-robin style
            for date_id in self.date_ids:
                if len(all_indices) >= samples_needed:
                    break
                
                date_indices = self.date_to_indices[date_id]
                if date_counters[date_id] < len(date_indices):
                    # Use next sample from this date
                    idx = date_indices[date_counters[date_id]]
                else:
                    # Wrap around if we've used all samples from this date
                    idx = rng.choice(date_indices)
                
                all_indices.append(idx)
                date_counters[date_id] += 1
        
        return all_indices[:samples_needed]
    
    def __iter__(self) -> Iterator[List[int]]:
        """Generate batches."""
        rng = np.random.default_rng(self.seed)
        
        # Create balanced epoch
        epoch_indices = self._create_balanced_epoch(rng)
        
        # Shuffle if requested
        if self.shuffle:
            rng.shuffle(epoch_indices)
        
        # Yield batches
        for i in range(0, len(epoch_indices), self.batch_size):
            batch = epoch_indices[i:i + self.batch_size]
            if self.drop_last and len(batch) < self.batch_size:
                break
            yield batch
    
    def __len__(self) -> int:
        return self.num_batches


def validate_batch_lineage(batch_data: List[Dict]) -> Dict[str, bool]:
    """Validate that batch satisfies lineage constraints."""
    results = {
        'no_duplicate_file_ids_per_date': True,
        'has_multiple_dates': True,
        'no_global_duplicate_file_ids': True,
    }
    
    # Group by date_id
    date_to_file_ids = defaultdict(set)
    all_file_ids = []
    
    for sample in batch_data:
        date_id = sample['date_id']
        file_id = sample['file_id']
        
        date_to_file_ids[date_id].add(file_id)
        all_file_ids.append(file_id)
    
    # Check no duplicate file_ids within each date
    for date_id, file_ids in date_to_file_ids.items():
        date_file_count = sum(1 for sample in batch_data if sample['date_id'] == date_id)
        if len(file_ids) != date_file_count:
            results['no_duplicate_file_ids_per_date'] = False
            break
    
    # Check multiple dates represented
    if len(date_to_file_ids) < 2:
        results['has_multiple_dates'] = False
    
    # Check no global duplicates across the batch
    if len(set(all_file_ids)) != len(all_file_ids):
        results['no_global_duplicate_file_ids'] = False
    
    return results


class SamplerStub:
    """Legacy stub - use LineageAwareBatchSampler or BalancedRandomSampler instead."""
    def __iter__(self):
        return iter(())

    def __len__(self):
        return 0