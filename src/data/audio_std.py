from __future__ import annotations

import numpy as np
import soundfile as sf
import librosa


def load_audio(path: str, target_sr: int = 24000, mono: bool = True) -> tuple[np.ndarray, int]:
    y, sr = sf.read(path, always_2d=True)
    # to mono
    if mono:
        y = y.mean(axis=1)
    else:
        y = y.T  # channels first arrangement not necessary here
    if sr != target_sr:
        y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=target_sr, res_type="soxr_vhq")
        sr = target_sr
    y = np.asarray(y, dtype=np.float32)
    return y, sr


def standardize_waveform(y: np.ndarray, sr: int, target_sr: int = 24000, lufs_target: float = -23.0, peak_dbfs: float = -1.0) -> tuple[np.ndarray, int]:
    # resample if needed
    if sr != target_sr:
        y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=target_sr, res_type="soxr_vhq")
        sr = target_sr
    # mono
    if y.ndim > 1:
        y = y.mean(axis=-1)
    y = y.astype(np.float32)
    # Approximate LUFS via RMS scaling if pyloudnorm not available
    rms = float(np.sqrt(np.mean(np.square(y))) + 1e-12)
    # target RMS for -23 LUFS ~= 0.1 (heuristic). Map LUFS to linear: lin = 10^(LUFS/20) with reference
    target_rms = 10 ** (lufs_target / 20.0)
    if rms > 0:
        y = y * (target_rms / rms)
    # Peak limit to -1 dBFS
    peak_lin = 10 ** (peak_dbfs / 20.0)
    max_abs = float(np.max(np.abs(y)) + 1e-12)
    if max_abs > peak_lin:
        y = y * (peak_lin / max_abs)
    return y, sr
