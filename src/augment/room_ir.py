from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def generate_exponential_ir(sr: int, rt60: float, size: int = 8192) -> torch.Tensor:
    """Generate synthetic exponential decay room impulse response."""
    t = torch.arange(size, dtype=torch.float32) / sr
    # RT60 is time for 60dB decay; -60dB = factor of 0.001
    decay_factor = torch.pow(torch.tensor(0.001), t / rt60)
    # Add small initial spike and exponential decay
    ir = torch.zeros(size)
    ir[0] = 1.0  # direct sound
    ir[1:] = decay_factor[1:] * torch.randn(size - 1) * 0.1
    return ir


def convolve_ir(audio: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
    """Apply room impulse response via convolution."""
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)  # add batch dim
    if ir.dim() == 1:
        ir = ir.unsqueeze(0)  # add batch dim
    
    # Pad audio to avoid circular convolution artifacts
    padded_len = audio.shape[-1] + ir.shape[-1] - 1
    audio_padded = F.pad(audio, (0, padded_len - audio.shape[-1]))
    ir_padded = F.pad(ir, (0, padded_len - ir.shape[-1]))
    
    # FFT convolution
    audio_fft = torch.fft.rfft(audio_padded)
    ir_fft = torch.fft.rfft(ir_padded)
    conv_fft = audio_fft * ir_fft
    result = torch.fft.irfft(conv_fft, n=padded_len)
    
    # Trim back to original length
    return result[..., :audio.shape[-1]]


def apply_room_ir(x: torch.Tensor, sr: int = 24000, seed: int | None = None) -> torch.Tensor:
    """Apply random room impulse response to audio."""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    original_shape = x.shape
    
    # Random RT60 between small (0.2s) and medium (1.2s) rooms
    rt60 = np.random.uniform(0.2, 1.2)
    ir_size = min(int(0.1 * sr), 4096)  # max 100ms IR
    
    ir = generate_exponential_ir(sr, rt60, ir_size)
    result = convolve_ir(x, ir)
    
    # Normalize to prevent clipping
    max_val = torch.max(torch.abs(result))
    if max_val > 1.0:
        result = result / max_val
    
    # Restore original shape
    if len(original_shape) == 1 and result.dim() == 2:
        result = result.squeeze(0)
    
    return result