import torch
import numpy as np
from pathlib import Path


def test_import_augment_modules():
    import src.augment.room_ir  # noqa: F401
    import src.augment.device_color  # noqa: F401
    import src.augment.noise_bank  # noqa: F401
    import src.augment.spec_augment  # noqa: F401
    import src.augment.mixstyle  # noqa: F401


def test_room_ir_determinism():
    from src.augment.room_ir import apply_room_ir
    
    # Create test audio
    audio = torch.randn(24000)  # 1 second at 24kHz
    
    # Apply with same seed should give identical results
    result1 = apply_room_ir(audio, seed=42)
    result2 = apply_room_ir(audio, seed=42)
    
    assert torch.allclose(result1, result2), "Room IR should be deterministic with same seed"
    assert result1.shape == audio.shape, "Output shape should match input"
    assert torch.max(torch.abs(result1)) <= 1.0, "Output should not clip"


def test_device_coloration_determinism():
    from src.augment.device_color import device_coloration
    
    audio = torch.randn(12000)  # 0.5 seconds
    
    # Test determinism
    result1 = device_coloration(audio, mode="phone", seed=42)
    result2 = device_coloration(audio, mode="phone", seed=42)
    
    assert torch.allclose(result1, result2, atol=1e-5), "Device coloration should be deterministic"
    assert result1.shape == audio.shape, "Output shape should match input"


def test_noise_augmentation():
    from src.augment.noise_bank import add_noise
    
    audio = torch.randn(12000)
    
    # Test different noise types
    for noise_type in ["babble", "hvac", "traffic"]:
        noisy = add_noise(audio, snr_db=10.0, kind=noise_type, seed=42)
        assert noisy.shape == audio.shape, f"Shape mismatch for {noise_type} noise"
        assert torch.max(torch.abs(noisy)) <= 1.0, f"Output should not clip for {noise_type}"
        
        # Test determinism
        noisy2 = add_noise(audio, snr_db=10.0, kind=noise_type, seed=42)
        assert torch.allclose(noisy, noisy2), f"{noise_type} noise should be deterministic"


def test_spec_augment():
    from src.augment.spec_augment import spec_augment
    
    # Create test mel spectrogram (80 mel bins, 100 time steps)
    mel = torch.randn(80, 100)
    
    # Apply SpecAugment
    augmented = spec_augment(mel, time_masks=2, freq_masks=2, seed=42)
    
    assert augmented.shape == mel.shape, "SpecAugment should preserve shape"
    
    # Test determinism
    augmented2 = spec_augment(mel, time_masks=2, freq_masks=2, seed=42)
    assert torch.allclose(augmented, augmented2), "SpecAugment should be deterministic"
    
    # Test batch processing
    mel_batch = torch.randn(4, 80, 100)
    augmented_batch = spec_augment(mel_batch, seed=42)
    assert augmented_batch.shape == mel_batch.shape, "Batch SpecAugment should preserve shape"


def test_mixstyle():
    from src.augment.mixstyle import mixstyle, MixStyle
    
    # Test functional interface
    feats = torch.randn(4, 64, 100)  # batch, channels, time
    mixed = mixstyle(feats, p=1.0, alpha=0.1, seed=42)  # p=1.0 to ensure it's applied
    
    assert mixed.shape == feats.shape, "MixStyle should preserve shape"
    
    # Test module interface
    mixstyle_layer = MixStyle(p=1.0, alpha=0.1)
    mixstyle_layer.train()
    mixed_module = mixstyle_layer(feats, seed=42)
    
    assert mixed_module.shape == feats.shape, "MixStyle module should preserve shape"
    
    # Test that it actually changes the features (with high probability)
    # Since p=1.0, it should always apply
    assert not torch.allclose(mixed, feats), "MixStyle should modify features when applied"


def test_augmentation_bounds():
    """Test that all augmentations respect audio bounds and don't cause NaNs."""
    from src.augment.room_ir import apply_room_ir
    from src.augment.device_color import device_coloration
    from src.augment.noise_bank import add_noise
    
    audio = torch.randn(12000) * 0.5  # Moderate amplitude
    
    # Test room IR
    ir_result = apply_room_ir(audio, seed=42)
    assert torch.isfinite(ir_result).all(), "Room IR should not produce NaNs/Infs"
    assert torch.max(torch.abs(ir_result)) <= 1.0, "Room IR should not clip"
    
    # Test device coloration
    device_result = device_coloration(audio, seed=42)
    assert torch.isfinite(device_result).all(), "Device coloration should not produce NaNs/Infs"
    assert torch.max(torch.abs(device_result)) <= 1.0, "Device coloration should not clip"
    
    # Test noise addition
    noise_result = add_noise(audio, snr_db=5.0, seed=42)
    assert torch.isfinite(noise_result).all(), "Noise addition should not produce NaNs/Infs"
    assert torch.max(torch.abs(noise_result)) <= 1.0, "Noise addition should not clip"
