# Manifest Spec

Columns:
- date_id: str (YYYY-MM-DD)
- file_id: UUID4 (str)
- relpath: str (path relative to root)
- sr: int
- duration_s: float

Invariants:
- relpath unique per file_id
- duration_s > 0
