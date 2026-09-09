from __future__ import annotations

from pathlib import Path

from src.pipeline.audio import encode_ogg
from src.pipeline.chart_writer import write_chart
from src.pipeline.types import CHARTER_NAME, ChartData, SongMeta, TempoMap


def render_song_ini(meta: SongMeta) -> str:
    return (
        "[song]\n"
        f"name = {meta.name}\n"
        f"artist = {meta.artist}\n"
        f"album = {meta.album}\n"
        f"genre = {meta.genre}\n"
        f"year = {meta.year}\n"
        f"charter = {CHARTER_NAME}\n"
        "diff_guitar = 2\n"
        "preview_start_time = 0\n"
        f"song_length = {meta.duration_ms}\n"
    )


def write_song_ini(path: Path, meta: SongMeta) -> None:
    path.write_text(render_song_ini(meta), encoding="utf-8")


def package_song(
    output_dir: Path,
    *,
    backing_wav: Path,
    guitar_wav: Path,
    meta: SongMeta,
    tempo: TempoMap,
    charts: ChartData,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    encode_ogg(backing_wav, output_dir / "song.ogg")
    encode_ogg(guitar_wav, output_dir / "guitar.ogg")
    write_song_ini(output_dir / "song.ini", meta)
    write_chart(output_dir / "notes.chart", meta, tempo, charts)
    return output_dir
