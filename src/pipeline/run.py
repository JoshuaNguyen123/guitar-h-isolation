from __future__ import annotations

import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path

import soundfile as sf

from src.pipeline.fretmap import map_difficulties
from src.pipeline.hf import configure_fast_hf, model_is_cached
from src.pipeline.package import package_song
from src.pipeline.separate import isolate_guitar
from src.pipeline.tempo import estimate_tempo
from src.pipeline.transcribe import transcribe_guitar
from src.pipeline.types import SongMeta
from src.pipeline.util import ensure_ffmpeg, sanitize_folder_name

ProgressFn = Callable[[str, float], None]


def run_pipeline(
    input_path: Path,
    output_dir: Path,
    name: str,
    artist: str,
    album: str = "",
    genre: str = "",
    year: str = "",
    progress: ProgressFn | None = None,
) -> Path:
    def report(stage: str, fraction: float) -> None:
        if progress is not None:
            progress(stage, fraction)

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    if not input_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {input_path}")

    report("Checking tools", 0.0)
    configure_fast_hf()
    ensure_ffmpeg()
    output_dir = output_dir.parent / sanitize_folder_name(output_dir.name)

    work_dir = Path(tempfile.mkdtemp(prefix="guitar_h_"))
    try:
        if not model_is_cached():
            report("Downloading model", 0.04)
        report("Separate", 0.08)
        separation = isolate_guitar(input_path, work_dir)

        report("Transcribe", 0.62)
        notes = transcribe_guitar(separation.guitar_wav)

        report("Detecting tempo", 0.80)
        tempo = estimate_tempo(separation.backing_wav)

        report("Chart", 0.86)
        charts = map_difficulties(notes, tempo)
        duration_ms = int(round(sf.info(str(separation.backing_wav)).duration * 1000))
        meta = SongMeta(
            name=name.strip() or input_path.stem,
            artist=artist.strip() or "Unknown",
            album=album,
            genre=genre,
            year=year,
            duration_ms=duration_ms,
        )

        report("Package", 0.90)
        package_song(
            output_dir,
            backing_wav=separation.backing_wav,
            guitar_wav=separation.guitar_wav,
            meta=meta,
            tempo=tempo,
            charts=charts,
        )
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    report("Done", 1.0)
    return output_dir
