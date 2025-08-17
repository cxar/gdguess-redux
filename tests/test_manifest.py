import os
from pathlib import Path
import json
import pandas as pd
import numpy as np
import soundfile as sf


def write_sine(path: Path, sr: int, dur_s: float) -> None:
    t = np.linspace(0, dur_s, int(sr * dur_s), endpoint=False)
    y = 0.1 * np.sin(2 * np.pi * 440.0 * t)
    sf.write(path, y, sr)


def test_manifest_build(tmp_path: Path, monkeypatch):
    root = tmp_path / "data"
    (root / "2025-01-01").mkdir(parents=True)
    (root / "2025-01-02").mkdir(parents=True)
    write_sine(root / "2025-01-01" / "a.wav", 16000, 1.0)
    write_sine(root / "2025-01-02" / "b.wav", 22050, 2.1)

    out = tmp_path / "manifest.parquet"
    from src.data.build_manifest import main as build

    build(["--root", str(root), "--out", str(out)])

    df = pd.read_parquet(out)
    assert set(df.columns) == {"date_id", "file_id", "relpath", "sr", "duration_s"}
    assert len(df) == 2
    assert set(df["date_id"]) == {"2025-01-01", "2025-01-02"}
    # deterministic UUIDs: rerun and compare
    before = df[["relpath", "file_id"]].set_index("relpath").to_dict()["file_id"]
    build(["--root", str(root), "--out", str(out)])
    df2 = pd.read_parquet(out)
    after = df2[["relpath", "file_id"]].set_index("relpath").to_dict()["file_id"]
    assert before == after
