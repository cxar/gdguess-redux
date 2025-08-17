from __future__ import annotations

import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset
import torchaudio.transforms as T
from typing import Dict, Any
import numpy as np

from .audio_std import load_audio, standardize_waveform


class AudioWindowDataset(Dataset):
    """Dataset for loading audio windows with mel feature extraction."""
    
    def __init__(
        self,
        windows_path: str | Path,
        audio_root: str | Path,
        target_sr: int = 24000,
        n_mels: int = 80,
        n_fft: int = 1024,
        hop_length: int = 512,
        win_length: int | None = None,
        f_min: float = 0.0,
        f_max: float | None = None,
        center: bool = True,
        pad_mode: str = "reflect",
        norm: str = "slaney",
        mel_scale: str = "htk",
        apply_augmentations: bool = False,
        augmentation_config: Dict[str, Any] | None = None,
    ):
        """
        Args:
            windows_path: Path to windows parquet file
            audio_root: Root directory containing audio files
            target_sr: Target sampling rate
            n_mels: Number of mel filter banks
            n_fft: FFT size
            hop_length: Hop length for STFT
            win_length: Window length for STFT (default: n_fft)
            f_min: Minimum frequency
            f_max: Maximum frequency (default: sr/2)
            center: Whether to center frames
            pad_mode: Padding mode for STFT
            norm: Mel scale normalization
            mel_scale: Mel scale type
            apply_augmentations: Whether to apply audio augmentations
            augmentation_config: Configuration for augmentations
        """
        self.windows_df = pd.read_parquet(windows_path)
        self.audio_root = Path(audio_root)
        self.target_sr = target_sr
        self.apply_augmentations = apply_augmentations
        self.augmentation_config = augmentation_config or {}
        
        # Mel spectrogram transform
        self.mel_transform = T.MelSpectrogram(
            sample_rate=target_sr,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length or n_fft,
            f_min=f_min,
            f_max=f_max or target_sr // 2,
            n_mels=n_mels,
            center=center,
            pad_mode=pad_mode,
            norm=norm,
            mel_scale=mel_scale,
        )
        
        # Log mel transform
        self.log_mel_transform = T.AmplitudeToDB(stype="power", top_db=80.0)
        
        # Build file path cache for efficiency
        self._build_file_cache()
    
    def _build_file_cache(self):
        """Build cache of file paths for faster lookup."""
        # Get unique file entries from windows
        file_info = self.windows_df.groupby('file_id').first()[['date_id', 'file_id']].reset_index(drop=True)
        
        # This would need to be joined with manifest to get relpath
        # For now, assume we can reconstruct paths from date_id structure
        self.file_cache = {}
        for _, row in file_info.iterrows():
            # Placeholder - in real implementation, join with manifest table
            # self.file_cache[row['file_id']] = self.audio_root / row['relpath']
            pass
    
    def _load_audio_window(self, file_id: str, offset_s: float, duration_s: float) -> torch.Tensor:
        """Load specific audio window from file."""
        # In real implementation, would use file_cache to get path
        # For now, this is a placeholder
        file_path = self.file_cache.get(file_id)
        if file_path is None:
            raise ValueError(f"File not found for file_id: {file_id}")
        
        # Load full audio file
        audio, sr = load_audio(str(file_path), target_sr=self.target_sr, mono=True)
        
        # Standardize audio
        audio, _ = standardize_waveform(audio, sr, target_sr=self.target_sr)
        
        # Extract window
        start_sample = int(offset_s * self.target_sr)
        end_sample = int((offset_s + duration_s) * self.target_sr)
        
        # Handle boundary conditions
        if start_sample >= len(audio):
            # Return silence if start is beyond audio
            return torch.zeros(int(duration_s * self.target_sr))
        
        window = audio[start_sample:end_sample]
        
        # Pad if window is shorter than expected
        expected_length = int(duration_s * self.target_sr)
        if len(window) < expected_length:
            padding = expected_length - len(window)
            window = torch.nn.functional.pad(window, (0, padding), mode='constant', value=0.0)
        
        return window
    
    def _apply_audio_augmentations(self, audio: torch.Tensor, seed: int | None = None) -> torch.Tensor:
        """Apply audio-level augmentations."""
        if not self.apply_augmentations:
            return audio
        
        # Import augmentations locally to avoid hard dependencies
        from ..augment.room_ir import apply_room_ir
        from ..augment.device_color import device_coloration  
        from ..augment.noise_bank import add_noise
        
        aug_config = self.augmentation_config
        
        # Room IR
        if aug_config.get('room_ir', {}).get('enabled', False):
            if np.random.random() < aug_config['room_ir'].get('p', 0.7):
                audio = apply_room_ir(audio, sr=self.target_sr, seed=seed)
        
        # Device coloration
        if aug_config.get('device_color', {}).get('enabled', False):
            if np.random.random() < aug_config['device_color'].get('p', 0.5):
                modes = aug_config['device_color'].get('modes', ['phone', 'laptop'])
                mode = np.random.choice(modes) if 'random' in modes else np.random.choice(modes)
                audio = device_coloration(audio, mode=mode, sr=self.target_sr, seed=seed)
        
        # Background noise
        if aug_config.get('noise', {}).get('enabled', False):
            if np.random.random() < aug_config['noise'].get('p', 0.6):
                snr_range = aug_config['noise'].get('snr_range', [5, 20])
                snr = np.random.uniform(*snr_range)
                noise_types = aug_config['noise'].get('types', ['babble', 'hvac', 'traffic'])
                noise_type = np.random.choice(noise_types)
                audio = add_noise(audio, snr_db=snr, kind=noise_type, sr=self.target_sr, seed=seed)
        
        return audio
    
    def _apply_spec_augmentations(self, mel: torch.Tensor, seed: int | None = None) -> torch.Tensor:
        """Apply spectrogram-level augmentations."""
        if not self.apply_augmentations:
            return mel
        
        from ..augment.spec_augment import spec_augment
        
        aug_config = self.augmentation_config
        
        # SpecAugment
        if aug_config.get('spec_augment', {}).get('enabled', False):
            if np.random.random() < aug_config['spec_augment'].get('p', 0.8):
                time_masks = aug_config['spec_augment'].get('time_masks', 2)
                freq_masks = aug_config['spec_augment'].get('freq_masks', 2)
                max_time_width = aug_config['spec_augment'].get('max_time_width', 10)
                max_freq_width = aug_config['spec_augment'].get('max_freq_width', 8)
                
                mel = spec_augment(
                    mel, 
                    time_masks=time_masks,
                    freq_masks=freq_masks,
                    max_time_width=max_time_width,
                    max_freq_width=max_freq_width,
                    seed=seed
                )
        
        return mel
    
    def __len__(self) -> int:
        return len(self.windows_df)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get audio window and extract features."""
        row = self.windows_df.iloc[idx]
        
        # Generate deterministic seed for augmentations based on index
        aug_seed = hash(f"{idx}_{row['file_id']}_{row['window_idx']}") % (2**32)
        np.random.seed(aug_seed)
        
        try:
            # Load audio window
            audio = self._load_audio_window(
                row['file_id'], 
                row['track_offset_s'], 
                row['win_s']
            )
            
            # Apply audio augmentations
            audio = self._apply_audio_augmentations(audio, seed=aug_seed)
            
            # Extract mel spectrogram
            mel = self.mel_transform(audio)
            log_mel = self.log_mel_transform(mel)
            
            # Apply spectrogram augmentations
            log_mel = self._apply_spec_augmentations(log_mel, seed=aug_seed + 1)
            
            return {
                'audio': audio,
                'mel': log_mel,
                'date_id': row['date_id'],
                'file_id': row['file_id'],
                'window_idx': int(row['window_idx']),
                'track_offset_s': float(row['track_offset_s']),
                'idx': idx,
            }
            
        except Exception as e:
            # Return dummy data if loading fails
            dummy_audio = torch.zeros(int(row['win_s'] * self.target_sr))
            dummy_mel = self.mel_transform(dummy_audio)
            dummy_log_mel = self.log_mel_transform(dummy_mel)
            
            return {
                'audio': dummy_audio,
                'mel': dummy_log_mel,
                'date_id': row['date_id'],
                'file_id': row['file_id'],
                'window_idx': int(row['window_idx']),
                'track_offset_s': float(row['track_offset_s']),
                'idx': idx,
                'error': str(e),
            }


class SimpleAudioDataset(Dataset):
    """Simplified dataset for testing without real audio files."""
    
    def __init__(self, windows_path: str | Path, target_sr: int = 24000, n_mels: int = 80):
        self.windows_df = pd.read_parquet(windows_path)
        self.target_sr = target_sr
        self.n_mels = n_mels
        
        # Simple mel transform for testing
        self.mel_transform = T.MelSpectrogram(
            sample_rate=target_sr,
            n_fft=1024,
            hop_length=512,
            n_mels=n_mels,
        )
        self.log_mel_transform = T.AmplitudeToDB(stype="power", top_db=80.0)
    
    def __len__(self) -> int:
        return len(self.windows_df)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Generate synthetic data for testing."""
        row = self.windows_df.iloc[idx]
        
        # Generate synthetic audio (sine wave with frequency based on date_id hash)
        duration_samples = int(row['win_s'] * self.target_sr)
        t = torch.arange(duration_samples).float() / self.target_sr
        
        # Frequency based on date_id for consistency
        date_hash = hash(row['date_id']) % 1000 + 200  # 200-1200 Hz
        audio = 0.1 * torch.sin(2 * np.pi * date_hash * t)
        
        # Add some noise for realism (seeded for determinism)
        torch.manual_seed(hash(f"{row['date_id']}_{row['file_id']}_{idx}") % (2**32))
        audio += 0.01 * torch.randn_like(audio)
        
        # Extract mel
        mel = self.mel_transform(audio)
        log_mel = self.log_mel_transform(mel)
        
        return {
            'audio': audio,
            'mel': log_mel,
            'date_id': row['date_id'],
            'file_id': row['file_id'],
            'window_idx': int(row['window_idx']),
            'track_offset_s': float(row['track_offset_s']),
            'idx': idx,
        }