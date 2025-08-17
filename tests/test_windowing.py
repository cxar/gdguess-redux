from pathlib import Path
import pandas as pd


def test_window_offsets(tmp_path: Path):
    # Create a tiny manifest
    import pandas as pd

    man = pd.DataFrame([
        {"date_id": "2025-01-01", "file_id": "abc", "relpath": "2025-01-01/a.wav", "sr": 16000, "duration_s": 65.0},
    ])
    man_path = tmp_path / "manifest.parquet"
    man.to_parquet(man_path, index=False)

    out = tmp_path / "windows.parquet"
    from src.data.chunk_windows import main as chunk

    chunk(["--manifest", str(man_path), "--out", str(out), "--win_s", "30", "--hop_s", "10"])
    df = pd.read_parquet(out)
    offsets = list(df["track_offset_s"]) if not df.empty else []
    assert offsets == [0.0, 10.0, 20.0, 30.0]
    assert list(df["window_idx"]) == [0, 1, 2, 3]
