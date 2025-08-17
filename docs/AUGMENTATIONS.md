# Augmentations

## Audio-level Augmentations

### Room Impulse Response (`room_ir.py`)
- **Purpose**: Simulate different room acoustics
- **Method**: Convolve with synthetic exponential decay IRs
- **Parameters**: RT60 ∈ [0.2s, 1.2s] (small to medium rooms)
- **When**: Applied to raw audio before feature extraction
- **Effect**: Adds reverb, simulates recording in different spaces

### Device Coloration (`device_color.py`)
- **Purpose**: Simulate phone speakers, laptop mics
- **Method**: EQ filtering + light compression
- **Types**: 
  - Phone: bandpass 300Hz-8kHz (emphasizes speech)
  - Laptop: high-pass at 80Hz (removes low rumble)
- **Parameters**: Random threshold ∈ [0.6, 0.8], ratio ∈ [3.0, 6.0]
- **Effect**: Mimics cheap device audio processing

### Background Noise (`noise_bank.py`)
- **Purpose**: Add realistic environmental noise
- **Types**:
  - **Babble**: 3-8 overlapping synthetic voices with formant structure
  - **HVAC**: Pink noise + periodic fan components (40-120Hz)
  - **Traffic**: Broadband + engine harmonics (30-100Hz fundamentals)
- **Parameters**: SNR ∈ [0, 20] dB
- **Effect**: Tests robustness to noisy conditions

## Feature-level Augmentations

### SpecAugment (`spec_augment.py`)
- **Purpose**: Mask parts of mel spectrogram
- **Method**: Random time/frequency masks
- **Parameters**: 
  - Time masks: max 2 masks, max 10 time steps each
  - Freq masks: max 2 masks, max 8 mel bins each
- **When**: Applied to mel spectrograms
- **Effect**: Forces model to use partial information

### MixStyle (`mixstyle.py`)
- **Purpose**: Domain adaptation by mixing feature statistics
- **Method**: Mix channel-wise mean/std between samples in batch
- **Parameters**:
  - Probability p = 0.5
  - Beta distribution α = 0.1
- **When**: Applied to intermediate features
- **Effect**: Reduces domain gap between different recording conditions

## Configuration

All augmentations are:
- **Deterministic**: Same seed → same output
- **Shape-preserving**: Input/output shapes match
- **Bounded**: Audio stays within [-1, 1], no clipping
- **Configurable**: Parameters exposed for tuning

Example usage:
```python
from src.augment import apply_room_ir, device_coloration, add_noise

# Audio augmentations
audio_aug = apply_room_ir(audio, seed=42)
audio_aug = device_coloration(audio_aug, mode="phone", seed=43)
audio_aug = add_noise(audio_aug, snr_db=10, kind="babble", seed=44)

# Feature augmentations  
mel_aug = spec_augment(mel_spec, seed=45)
features_aug = mixstyle(features, p=0.5, alpha=0.1, seed=46)
```
