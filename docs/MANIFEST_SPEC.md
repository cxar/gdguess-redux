# Manifest Spec

Columns:
- date_id: str (YYYY-MM-DD)
- file_id: UUID5 (str; namespace URL, deterministic from relpath)
- relpath: str (path relative to root)
- sr: int (may be -1 if unknown from ffprobe)
- duration_s: float

Invariants:
- relpath unique; (date_id, file_id) unique per row
- duration_s > 0; sr > 0

Windows table:
- date_id: str
- file_id: str
- window_idx: int (0-based within file)
- track_offset_s: float
- win_s: float
- hop_s: float
