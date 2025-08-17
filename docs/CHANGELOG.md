# Changelog

## M2
- Implemented complete augmentation suite: room IR, device coloration, noise bank, SpecAugment, MixStyle.
- All augmentations are deterministic, shape-preserving, and bounded.
- Added comprehensive tests for determinism and correctness.

## M1
- Manifest builder with ffprobe-based MP3 probing (avoids mpg123 crashes).
- Window table generation for 30s windows with 10s hop.
- Progress tracking and error logging.

## M0
- Scaffold repository, CI skeleton, docs policy.
