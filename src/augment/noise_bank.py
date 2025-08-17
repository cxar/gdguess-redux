from __future__ import annotations

import numpy as np
import torch


def generate_babble_noise(length: int, sr: int = 24000, seed: int | None = None) -> torch.Tensor:
    """Generate synthetic babble noise (multiple overlapping voices)."""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # Create multiple overlapping speech-like signals
    num_voices = np.random.randint(3, 8)
    babble = torch.zeros(length)
    
    for _ in range(num_voices):
        # Generate speech-like formant structure
        t = torch.arange(length).float() / sr
        
        # Fundamental frequency variations (80-250 Hz for mixed voices)
        f0 = np.random.uniform(80, 250)
        f0_variation = 0.1 * torch.sin(2 * np.pi * np.random.uniform(0.5, 3.0) * t)
        
        # Create formant-like structure with harmonics
        voice = torch.zeros(length)
        for harmonic in range(1, 6):
            freq = f0 * harmonic * (1 + f0_variation)
            amplitude = 1.0 / harmonic  # decreasing amplitude for higher harmonics
            voice += amplitude * torch.sin(2 * np.pi * freq * t)
        
        # Add some noise and envelope variations
        envelope = 0.5 + 0.5 * torch.sin(2 * np.pi * np.random.uniform(0.1, 2.0) * t)
        noise = 0.1 * torch.randn(length)
        voice = voice * envelope + noise
        
        # Random delay and amplitude for each voice
        delay = np.random.randint(0, min(length // 4, sr // 2))
        amplitude = np.random.uniform(0.3, 1.0)
        if delay < length:
            babble[delay:] += amplitude * voice[:length-delay]
    
    # Normalize
    babble = babble / torch.max(torch.abs(babble) + 1e-8)
    return babble


def generate_hvac_noise(length: int, sr: int = 24000, seed: int | None = None) -> torch.Tensor:
    """Generate HVAC noise (broadband with low-frequency emphasis)."""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # Generate pink noise (1/f characteristics)
    white_noise = torch.randn(length)
    
    # Apply 1/f filter to create pink noise
    freqs = torch.fft.rfftfreq(length, 1/sr)
    # Avoid division by zero
    freqs[0] = 1.0
    pink_filter = 1.0 / torch.sqrt(freqs)
    
    white_fft = torch.fft.rfft(white_noise)
    pink_fft = white_fft * pink_filter
    pink_noise = torch.fft.irfft(pink_fft, n=length)
    
    # Add periodic component (fan noise)
    t = torch.arange(length).float() / sr
    fan_freq = np.random.uniform(40, 120)  # typical fan frequencies
    periodic = 0.3 * torch.sin(2 * np.pi * fan_freq * t)
    
    hvac = pink_noise + periodic
    hvac = hvac / torch.max(torch.abs(hvac) + 1e-8)
    return hvac


def generate_traffic_noise(length: int, sr: int = 24000, seed: int | None = None) -> torch.Tensor:
    """Generate traffic noise (broadband with engine-like characteristics)."""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # Base broadband noise
    base_noise = torch.randn(length)
    
    # Add engine-like periodic components
    t = torch.arange(length).float() / sr
    num_vehicles = np.random.randint(2, 5)
    
    for _ in range(num_vehicles):
        # Engine fundamental (30-100 Hz)
        engine_freq = np.random.uniform(30, 100)
        engine_amplitude = np.random.uniform(0.2, 0.6)
        
        # Add harmonics
        engine_sound = torch.zeros(length)
        for harmonic in range(1, 4):
            freq = engine_freq * harmonic
            amp = engine_amplitude / harmonic
            phase = np.random.uniform(0, 2*np.pi)
            engine_sound += amp * torch.sin(2 * np.pi * freq * t + phase)
        
        base_noise += engine_sound
    
    # Apply some filtering to emphasize mid-range
    traffic = base_noise
    traffic = traffic / torch.max(torch.abs(traffic) + 1e-8)
    return traffic


def add_noise(x: torch.Tensor, snr_db: float = 10.0, kind: str = "babble", sr: int = 24000, seed: int | None = None) -> torch.Tensor:
    """Add background noise at specified SNR."""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    if x.dim() == 1:
        x = x.unsqueeze(0)  # add batch dim
    
    batch_size, length = x.shape
    
    # Generate noise based on type
    if kind == "babble":
        noise = generate_babble_noise(length, sr, seed)
    elif kind == "hvac":
        noise = generate_hvac_noise(length, sr, seed)
    elif kind == "traffic":
        noise = generate_traffic_noise(length, sr, seed)
    else:
        # Random choice
        kind = np.random.choice(["babble", "hvac", "traffic"])
        if kind == "babble":
            noise = generate_babble_noise(length, sr, seed)
        elif kind == "hvac":
            noise = generate_hvac_noise(length, sr, seed)
        else:
            noise = generate_traffic_noise(length, sr, seed)
    
    # Expand noise to batch size
    noise = noise.unsqueeze(0).expand(batch_size, -1)
    
    # Calculate signal and noise power
    signal_power = torch.mean(x ** 2, dim=-1, keepdim=True)
    noise_power = torch.mean(noise ** 2, dim=-1, keepdim=True)
    
    # Calculate noise scaling factor for desired SNR
    snr_linear = 10 ** (snr_db / 10.0)
    noise_scale = torch.sqrt(signal_power / (noise_power * snr_linear + 1e-8))
    
    # Add scaled noise
    noisy = x + noise_scale * noise
    
    # Normalize to prevent clipping
    max_val = torch.max(torch.abs(noisy))
    if max_val > 1.0:
        noisy = noisy / max_val
    
    return noisy.squeeze(0) if batch_size == 1 else noisy