from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd


def compute_windows_row(duration_s: float, win_s: float, hop_s: float, drop_last: bool) -> list[tuple[int, float]]:
    if duration_s <= 0 or win_s <= 0 or hop_s <= 0:
        return []
    if drop_last:
        if duration_s < win_s:
            return []
        n = int(math.floor((duration_s - win_s) / hop_s)) + 1
    else:
        n = int(math.ceil(max(duration_s - 1e-9, 0.0) / hop_s))
    return [(i, float(i * hop_s)) for i in range(max(n, 0))]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Build window table from manifest")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--win_s", type=float, default=30.0)
    ap.add_argument("--hop_s", type=float, default=10.0)
    ap.add_argument("--drop_last", action="store_true", default=True)
    ap.add_argument("--format", default="parquet", choices=["parquet", "csv"])
    args = ap.parse_args(argv)

    man_path = Path(args.manifest)
    if man_path.suffix.lower() == ".csv":
        man = pd.read_csv(man_path)
    else:
        man = pd.read_parquet(man_path)

    rows = []
    for _, r in man.iterrows():
        wins = compute_windows_row(float(r["duration_s"]), float(args.win_s), float(args.hop_s), bool(args.drop_last))
        for idx, offset in wins:
            rows.append({
                "date_id": r["date_id"],
                "file_id": r["file_id"],
                "window_idx": idx,
                "track_offset_s": offset,
                "win_s": float(args.win_s),
                "hop_s": float(args.hop_s),
            })
    df = pd.DataFrame(rows, columns=["date_id", "file_id", "window_idx", "track_offset_s", "win_s", "hop_s"]) if rows else pd.DataFrame(columns=["date_id", "file_id", "window_idx", "track_offset_s", "win_s", "hop_s"]) 

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.format == "csv" or out.suffix.lower() == ".csv":
        df.to_csv(out, index=False)
    else:
        df.to_parquet(out, index=False)
    print(f"Wrote {len(df)} windows to {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
