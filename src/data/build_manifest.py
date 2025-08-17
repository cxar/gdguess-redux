from __future__ import annotations

import argparse
import concurrent.futures as cf
import re
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import soundfile as sf

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@dataclass(frozen=True)
class Probe:
    date_id: str
    relpath: str
    sr: int
    duration_s: float


def list_audio_files(root: Path, exts: tuple[str, ...]) -> Iterable[tuple[str, Path]]:
    for date_dir in root.iterdir():
        if not date_dir.is_dir():
            continue
        date_id = date_dir.name
        if not DATE_RE.match(date_id):
            continue
        for ext in exts:
            for p in date_dir.rglob(f"*.{ext}"):
                if p.is_file():
                    yield date_id, p


def _probe_ffprobe(path: Path) -> tuple[int | None, float | None]:
    import json
    import subprocess

    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=sample_rate,duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
        data = json.loads(out)
        streams = data.get("streams", [])
        if not streams:
            return None, None
        st = streams[0]
        sr = int(st.get("sample_rate")) if st.get("sample_rate") else None
        dur = float(st.get("duration")) if st.get("duration") else None
        return sr, dur
    except Exception:
        return None, None


def probe_file(root: Path, date_id: str, p: Path) -> Probe | None:
    relpath = str(p.relative_to(root))
    ext = p.suffix.lower()

    # MP3: use ffprobe only (avoid mpg123)
    if ext == ".mp3":
        sr, dur = _probe_ffprobe(p)
        if dur and dur > 0:
            return Probe(date_id=date_id, relpath=relpath, sr=int(sr or -1), duration_s=float(dur))
        return None

    # WAV/FLAC: try soundfile, fallback to ffprobe
    try:
        info = sf.info(str(p))
        duration_s = float(info.frames) / float(info.samplerate)
        return Probe(date_id=date_id, relpath=relpath, sr=int(info.samplerate), duration_s=duration_s)
    except Exception:
        pass

    # Fallback to ffprobe for non-MP3
    sr, dur = _probe_ffprobe(p)
    if dur and dur > 0:
        return Probe(date_id=date_id, relpath=relpath, sr=int(sr or -1), duration_s=float(dur))

    return None


def uuid5_for_relpath(root: Path, relpath: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{root.as_posix()}/{relpath}"))


def build_df(root: Path, files: list[tuple[str, Path]], num_workers: int, log_errors: str | None = None) -> pd.DataFrame:
    rows: list[Probe] = []
    skipped_files: list[Path] = []
    
    with cf.ThreadPoolExecutor(max_workers=num_workers) as ex:
        futs = [(date_id, p, ex.submit(probe_file, root, date_id, p)) for date_id, p in files]
        for date_id, p, fut in futs:
            pr = fut.result()
            if pr is not None and pr.duration_s > 0:
                rows.append(pr)
            else:
                skipped_files.append(p)
    
    # Log skipped files if requested
    if log_errors and skipped_files:
        with open(log_errors, "w") as f:
            for p in skipped_files:
                f.write(f"{p}\n")
        print(f"Logged {len(skipped_files)} skipped files to {log_errors}")
    
    if not rows:
        return pd.DataFrame(columns=["date_id", "file_id", "relpath", "sr", "duration_s"])  # empty
    df = pd.DataFrame([r.__dict__ for r in rows])
    df["file_id"] = [uuid5_for_relpath(root, rp) for rp in df["relpath"]]
    # enforce dtypes
    df = df[["date_id", "file_id", "relpath", "sr", "duration_s"]]
    df["date_id"] = df["date_id"].astype("string")
    df["file_id"] = df["file_id"].astype("string")
    df["relpath"] = df["relpath"].astype("string")
    # allow -1 when sr unknown from fallback
    df["sr"] = df["sr"].astype("int32")
    df["duration_s"] = df["duration_s"].astype("float64")
    return df


def write_table(df: pd.DataFrame, out: Path, fmt: str) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet" or out.suffix.lower() == ".parquet":
        df.to_parquet(out, index=False)
    elif fmt == "csv" or out.suffix.lower() == ".csv":
        df.to_csv(out, index=False)
    else:
        raise ValueError(f"Unsupported format: {fmt}")


def main(argv: list[str] | None = None) -> None:
    import os

    ap = argparse.ArgumentParser(description="Build audio manifest from date-organized tree")
    ap.add_argument("--root", required=True, help="Data root containing YYYY-MM-DD dirs")
    ap.add_argument("--out", required=True, help="Output path (.parquet or .csv)")
    ap.add_argument("--format", default="parquet", choices=["parquet", "csv"], help="Output format")
    ap.add_argument("--exts", default="flac,wav,mp3", help="Comma-separated extensions to include")
    ap.add_argument("--num-workers", type=int, default=max(1, (os.cpu_count() or 4)), help="Parallel workers")
    ap.add_argument("--skip-errors", action="store_true", default=True, help="Skip unreadable files instead of failing")
    ap.add_argument("--log-errors", type=str, default=None, help="Optional path to write skipped file paths")
    args = ap.parse_args(argv)

    root = Path(args.root).resolve()
    out = Path(args.out)
    exts = tuple(e.strip().lstrip(".") for e in args.exts.split(",") if e.strip())

    files = list(list_audio_files(root, exts))
    df = build_df(root, files, args.num_workers, args.log_errors)
    write_table(df, out, args.format)
    print(f"Wrote {len(df)} rows to {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
