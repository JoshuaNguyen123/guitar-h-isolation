from __future__ import annotations

import subprocess
from pathlib import Path

from src.pipeline.util import ensure_ffmpeg


def run_ffmpeg(args: list[str]) -> None:
    exe = ensure_ffmpeg()
    completed = subprocess.run(
        [exe, "-hide_banner", "-loglevel", "error", *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise RuntimeError(detail or "ffmpeg failed.")


def decode_to_wav(src: Path, dst: Path, sample_rate: int = 44100) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(
        [
            "-y",
            "-i",
            str(src),
            "-ac",
            "2",
            "-ar",
            str(sample_rate),
            str(dst),
        ]
    )
    return dst


def encode_ogg(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(
        [
            "-y",
            "-i",
            str(src),
            "-c:a",
            "libvorbis",
            "-q:a",
            "5",
            str(dst),
        ]
    )
    return dst
