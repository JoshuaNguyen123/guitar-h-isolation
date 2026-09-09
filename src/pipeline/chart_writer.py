from __future__ import annotations

from src.pipeline.types import (
    CHARTER_NAME,
    RESOLUTION,
    ChartData,
    ChartNote,
    SongMeta,
    TempoMap,
)

_DIFF_SECTIONS = (
    ("ExpertSingle", "expert"),
    ("HardSingle", "hard"),
    ("MediumSingle", "medium"),
    ("EasySingle", "easy"),
)


def _quote(value: str) -> str:
    escaped = value.replace('"', "'")
    return f'"{escaped}"'


def _year_field(year: str) -> str:
    year = year.strip()
    if not year:
        return ""
    if year.startswith(","):
        return year
    return f", {year}"


def render_chart(
    meta: SongMeta,
    tempo: TempoMap,
    charts: ChartData,
    music_stream: str = "song.ogg",
    guitar_stream: str = "guitar.ogg",
) -> str:
    resolution = tempo.resolution or RESOLUTION
    bpm_milli = int(round(tempo.bpm * 1000))
    lines = [
        "[Song]",
        "{",
        f"  Name = {_quote(meta.name)}",
        f"  Artist = {_quote(meta.artist)}",
        f"  Charter = {_quote(CHARTER_NAME)}",
        f"  Album = {_quote(meta.album)}",
        f"  Year = {_quote(_year_field(meta.year))}",
        "  Offset = 0",
        f"  Resolution = {resolution}",
        "  Difficulty = 0",
        "  PreviewStart = 0",
        "  PreviewEnd = 0",
        f"  Genre = {_quote(meta.genre)}",
        '  MediaType = "cd"',
        f"  MusicStream = {_quote(music_stream)}",
        f"  GuitarStream = {_quote(guitar_stream)}",
        "}",
        "[SyncTrack]",
        "{",
        "  0 = TS 4",
        f"  0 = B {bpm_milli}",
        "}",
        "[Events]",
        "{",
        "}",
    ]
    for section_name, attr in _DIFF_SECTIONS:
        notes: list[ChartNote] = getattr(charts, attr)
        lines.append(f"[{section_name}]")
        lines.append("{")
        for note in sorted(notes, key=lambda n: (n.tick, n.fret)):
            lines.append(f"  {note.tick} = N {note.fret} {note.sustain}")
        lines.append("}")
    return "\n".join(lines) + "\n"


def write_chart(
    path,
    meta: SongMeta,
    tempo: TempoMap,
    charts: ChartData,
) -> None:
    path.write_text(render_chart(meta, tempo, charts), encoding="utf-8")
