from __future__ import annotations

import re
import shutil
from pathlib import Path


_INVALID_FOLDER = re.compile(r'[<>:"/\\|?*]')


def sanitize_folder_name(name: str) -> str:
    cleaned = _INVALID_FOLDER.sub("_", name).strip(" .")
    return cleaned or "Untitled"


def parse_filename_metadata(path: Path) -> tuple[str, str]:
    stem = path.stem.strip()
    for sep in (" - ", " – ", " — "):
        if sep in stem:
            left, right = stem.split(sep, 1)
            artist = left.strip()
            title = right.strip()
            if artist and title:
                return title, artist
    title = re.sub(r"[_]+", " ", stem).strip()
    return title or "Untitled", "Unknown"


def default_songs_root() -> Path:
    return Path.home() / "Documents" / "Clone Hero" / "Songs"


def default_output_dir(song_name: str) -> Path:
    return default_songs_root() / sanitize_folder_name(song_name)


def ensure_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise RuntimeError(
            "ffmpeg was not found on PATH. Install it with "
            "`winget install Gyan.FFmpeg`, then restart this app."
        )
    return exe
