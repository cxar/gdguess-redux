from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy import signal


def design_phone_speaker_eq(sr: int = 24000) -> tuple[torch.Tensor, torch.Tensor]:
    """Design EQ filter to simulate phone speaker (emphasize mids, roll off bass/treble)."""
    nyquist = sr / 2
    
    # Simplified phone speaker EQ: single bandpass filter for mids
    # Bandpass 300Hz - 8kHz with mid boost
    low_freq = 300 / nyquist
    high_freq = 8000 / nyquist
    
    # Use a simple 2nd order bandpass
    b, a = signal.butter(2, [low_freq, high_freq], btype='band')
    
    return torch.from_numpy(b).float(), torch.from_numpy(a).float()


def design_laptop_mic_eq(sr: int = 24000) -> tuple[torch.Tensor, torch.Tensor]:
    """Design EQ filter to simulate laptop microphone (emphasize speech, roll off low end)."""
    nyquist = sr / 2
    
    # Simplified laptop mic: high-pass at 80Hz
    hp_freq = 80 / nyquist
    b, a = signal.butter(1, hp_freq, btype='high')
    
    return torch.from_numpy(b).float(), torch.from_numpy(a).float()


def apply_iir_filter(audio: torch.Tensor, b: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """Apply IIR filter using torch operations (simplified biquad-style)."""
    original_shape = audio.shape
    original_length = audio.shape[-1]
    
    # Apply as convolution with numerator coefficients only (FIR approximation)
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    
    b = b.unsqueeze(0)  # add batch dim
    
    # Use explicit padding calculation to ensure exact length preservation
    kernel_size = len(b[0])
    padding_left = kernel_size // 2
    padding_right = kernel_size - 1 - padding_left
    
    # Apply padding manually
    padded_audio = F.pad(audio, (padding_left, padding_right), mode='reflect')
    
    # Apply convolution
    result = F.conv1d(padded_audio.unsqueeze(1), b.unsqueeze(1), padding=0)
    result = result.squeeze(1)
    
    # Ensure exact original length
    result = result[..., :original_length]
    
    # Restore original shape
    if len(original_shape) == 1 and result.dim() == 2:
        result = result.squeeze(0)
    
    return result


def light_compression(audio: torch.Tensor, threshold: float = 0.7, ratio: float = 4.0) -> torch.Tensor:
    """Apply light compression to simulate device processing."""
    abs_audio = torch.abs(audio)
    # Simple soft-knee compression
    mask = abs_audio > threshold
    compressed = torch.where(
        mask,
        torch.sign(audio) * (threshold + (abs_audio - threshold) / ratio),
        audio
    )
    return compressed


def device_coloration(x: torch.Tensor, mode: str = "phone", sr: int = 24000, seed: int | None = None) -> torch.Tensor:
    """Apply device coloration (phone speaker or laptop mic simulation)."""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    if mode == "phone":
        b, a = design_phone_speaker_eq(sr)
    elif mode == "laptop":
        b, a = design_laptop_mic_eq(sr)
    else:
        # Random choice
        mode = np.random.choice(["phone", "laptop"])
        if mode == "phone":
            b, a = design_phone_speaker_eq(sr)
        else:
            b, a = design_laptop_mic_eq(sr)
    
    # Apply EQ
    result = apply_iir_filter(x, b, a)
    
    # Apply light compression with some randomness
    threshold = np.random.uniform(0.6, 0.8)
    ratio = np.random.uniform(3.0, 6.0)
    result = light_compression(result, threshold, ratio)
    
    # Normalize
    max_val = torch.max(torch.abs(result))
    if max_val > 1.0:
        result = result / max_val
    
    return result