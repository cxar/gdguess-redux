# Dataset

- Directory convention: `root/YYYY-MM-DD/**/*.{flac,wav,mp3}` (top-level folders are date_id)
- Determinism: pure preprocessing; deterministic UUID5 from relpath
- Preprocessing: resample to 24kHz, mono, loudness ≈ −23 LUFS (RMS proxy), peak clamp −1 dBFS
- Manifests: Parquet by default for `manifest` and `windows` tables
