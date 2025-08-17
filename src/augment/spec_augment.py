from __future__ import annotations

import numpy as np
import torch


def time_mask(mel: torch.Tensor, max_time_masks: int = 2, max_time_width: int = 10, seed: int | None = None) -> torch.Tensor:
    """Apply time masking to mel spectrogram."""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    mel = mel.clone()
    _, time_steps = mel.shape
    
    num_masks = np.random.randint(0, max_time_masks + 1)
    
    for _ in range(num_masks):
        mask_width = np.random.randint(1, min(max_time_width, time_steps // 4) + 1)
        start_time = np.random.randint(0, max(1, time_steps - mask_width + 1))
        
        # Zero out the selected time range
        mel[:, start_time:start_time + mask_width] = 0.0
    
    return mel


def freq_mask(mel: torch.Tensor, max_freq_masks: int = 2, max_freq_width: int = 8, seed: int | None = None) -> torch.Tensor:
    """Apply frequency masking to mel spectrogram."""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    mel = mel.clone()
    freq_bins, _ = mel.shape
    
    num_masks = np.random.randint(0, max_freq_masks + 1)
    
    for _ in range(num_masks):
        mask_width = np.random.randint(1, min(max_freq_width, freq_bins // 4) + 1)
        start_freq = np.random.randint(0, max(1, freq_bins - mask_width + 1))
        
        # Zero out the selected frequency range
        mel[start_freq:start_freq + mask_width, :] = 0.0
    
    return mel


def spec_augment(mel: torch.Tensor, time_masks: int = 2, freq_masks: int = 2, 
                max_time_width: int = 10, max_freq_width: int = 8, seed: int | None = None) -> torch.Tensor:
    """Apply SpecAugment (time and frequency masking) to mel spectrogram.
    
    Args:
        mel: Mel spectrogram tensor of shape (freq_bins, time_steps) or (batch, freq_bins, time_steps)
        time_masks: Maximum number of time masks to apply
        freq_masks: Maximum number of frequency masks to apply
        max_time_width: Maximum width of time masks
        max_freq_width: Maximum width of frequency masks
        seed: Random seed for reproducibility
    
    Returns:
        Augmented mel spectrogram
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    original_shape = mel.shape
    
    # Handle batch dimension
    if mel.dim() == 3:
        batch_size = mel.shape[0]
        results = []
        for i in range(batch_size):
            # Use different seed for each item in batch
            item_seed = seed + i if seed is not None else None
            
            item_mel = mel[i]
            item_mel = time_mask(item_mel, time_masks, max_time_width, item_seed)
            item_mel = freq_mask(item_mel, freq_masks, max_freq_width, item_seed)
            results.append(item_mel)
        
        return torch.stack(results)
    
    elif mel.dim() == 2:
        # Single spectrogram
        mel = time_mask(mel, time_masks, max_time_width, seed)
        mel = freq_mask(mel, freq_masks, max_freq_width, seed)
        return mel
    
    else:
        raise ValueError(f"Expected mel spectrogram with 2 or 3 dimensions, got {mel.dim()}")


def adaptive_spec_augment(mel: torch.Tensor, severity: float = 0.5, seed: int | None = None) -> torch.Tensor:
    """Apply adaptive SpecAugment where mask sizes scale with input size.
    
    Args:
        mel: Mel spectrogram tensor
        severity: Augmentation severity (0.0 to 1.0)
        seed: Random seed
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    if mel.dim() == 3:
        _, freq_bins, time_steps = mel.shape
    else:
        freq_bins, time_steps = mel.shape
    
    # Scale mask parameters based on input size and severity
    max_time_width = max(1, int(time_steps * 0.1 * severity))
    max_freq_width = max(1, int(freq_bins * 0.1 * severity))
    
    time_masks = 1 if severity < 0.5 else 2
    freq_masks = 1 if severity < 0.5 else 2
    
    return spec_augment(mel, time_masks, freq_masks, max_time_width, max_freq_width, seed)