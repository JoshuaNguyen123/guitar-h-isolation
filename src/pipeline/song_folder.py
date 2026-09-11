"""Load Clone Hero song folders from this app (any version) or compatible charts.

Older Guitar H Isolation builds wrote the same four files the current build
writes. This module opens those folders without requiring a re-generate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from src.pipeline.types import RESOLUTION, ChartData, ChartNote, SongMeta, TempoMap

REQUIRED_FILES = ("song.ini", "notes.chart", "song.ogg")


@dataclass(frozen=True)
class SongFolderInfo:
    """Parsed view of an on-disk Clone Hero song folder."""

    path: Path
    meta: SongMeta
    tempo: TempoMap
    charts: ChartData
    music_stream: str = "song.ogg"
    guitar_stream: str = "guitar.ogg"
    missing_audio: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def note_counts(self) -> dict[str, int]:
        return {
            "expert": len(self.charts.expert),
            "hard": len(self.charts.hard),
            "medium": len(self.charts.medium),
            "easy": len(self.charts.easy),
        }

    def summary_lines(self) -> list[str]:
        counts = self.note_counts
        lines = [
            f"Opened song folder: {self.path}",
            f"  {self.meta.name} — {self.meta.artist}",
            f"  BPM {self.tempo.bpm:g} · resolution {self.tempo.resolution}",
            (
                "  Notes  Expert {expert} · Hard {hard} · Medium {medium} · Easy {easy}"
            ).format(**counts),
            f"  Streams  music={self.music_stream}  guitar={self.guitar_stream or '(none)'}",
        ]
        if self.missing_audio:
            lines.append(f"  Missing audio: {', '.join(self.missing_audio)}")
        for warning in self.warnings:
            lines.append(f"  Note: {warning}")
        return lines


def is_song_folder(path: Path) -> bool:
    """True when the folder looks like a Clone Hero / Guitar H Isolation song."""
    root = Path(path)
    if not root.is_dir():
        return False
    return all((root / name).is_file() for name in REQUIRED_FILES)


def load_song_folder(path: Path) -> SongFolderInfo:
    """Parse song.ini + notes.chart; tolerate older / sparse variants."""
    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(f"Song folder not found: {root}")
    missing = [name for name in REQUIRED_FILES if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "Not a Clone Hero song folder (missing "
            + ", ".join(missing)
            + f"): {root}"
        )

    ini_meta, ini_warnings = _parse_song_ini(root / "song.ini")
    chart_meta, tempo, charts, music_stream, guitar_stream, chart_warnings = (
        _parse_notes_chart(root / "notes.chart")
    )
    meta = _merge_meta(ini_meta, chart_meta)

    missing_audio: list[str] = []
    for name in ("song.ogg", "guitar.ogg"):
        if not (root / name).is_file():
            missing_audio.append(name)

    if music_stream and music_stream not in missing_audio and not (root / music_stream).is_file():
        chart_warnings.append(f"MusicStream file missing: {music_stream}")
    if guitar_stream and guitar_stream not in missing_audio and not (root / guitar_stream).is_file():
        missing_audio.append(guitar_stream)

    warnings = tuple(dict.fromkeys([*ini_warnings, *chart_warnings]))
    return SongFolderInfo(
        path=root.resolve(),
        meta=meta,
        tempo=tempo,
        charts=charts,
        music_stream=music_stream or "song.ogg",
        guitar_stream=guitar_stream or "",
        missing_audio=tuple(missing_audio),
        warnings=warnings,
    )


def _merge_meta(ini: SongMeta, chart: SongMeta) -> SongMeta:
    return SongMeta(
        name=(ini.name or chart.name or "Unknown").strip() or "Unknown",
        artist=(ini.artist or chart.artist or "Unknown").strip() or "Unknown",
        album=(ini.album or chart.album).strip(),
        genre=(ini.genre or chart.genre).strip(),
        year=_normalize_year(ini.year or chart.year),
        duration_ms=ini.duration_ms or chart.duration_ms,
    )


def _normalize_year(year: str) -> str:
    year = year.strip().strip(",").strip()
    return year


def _parse_song_ini(path: Path) -> tuple[SongMeta, list[str]]:
    text = path.read_text(encoding="utf-8-sig")
    warnings: list[str] = []
    fields: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("[") or line.startswith(";") or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        fields[key.strip().lower()] = value.strip()
    if "name" not in fields and "artist" not in fields:
        warnings.append("song.ini has no name/artist fields")
    duration = 0
    length = fields.get("song_length") or fields.get("songlength") or "0"
    try:
        duration = int(float(length))
    except ValueError:
        warnings.append(f"song.ini song_length was not a number: {length!r}")
    return (
        SongMeta(
            name=fields.get("name", ""),
            artist=fields.get("artist", ""),
            album=fields.get("album", ""),
            genre=fields.get("genre", ""),
            year=fields.get("year", ""),
            duration_ms=duration,
        ),
        warnings,
    )


_SECTION_RE = re.compile(r"^\[([^\]]+)\]\s*$")
_KV_RE = re.compile(r'^([A-Za-z0-9_]+)\s*=\s*(.*)$')
_NOTE_RE = re.compile(r"^(\d+)\s*=\s*N\s+(\d+)\s+(\d+)\s*$")
_BPM_RE = re.compile(r"^(\d+)\s*=\s*B\s+(\d+)\s*$")


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def _parse_notes_chart(
    path: Path,
) -> tuple[SongMeta, TempoMap, ChartData, str, str, list[str]]:
    text = path.read_text(encoding="utf-8-sig")
    warnings: list[str] = []
    section = ""
    song_fields: dict[str, str] = {}
    charts = ChartData()
    section_map = {
        "expertsingle": "expert",
        "hardsingle": "hard",
        "mediumsingle": "medium",
        "easysingle": "easy",
    }
    bpm = 120.0
    resolution = RESOLUTION
    found_bpm = False

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line in "{}":
            continue
        section_match = _SECTION_RE.match(line)
        if section_match:
            section = section_match.group(1).strip().lower()
            continue
        if section == "song":
            kv = _KV_RE.match(line)
            if not kv:
                continue
            key = kv.group(1)
            song_fields[key.lower()] = _unquote(kv.group(2))
            if key.lower() == "resolution":
                try:
                    resolution = int(float(song_fields["resolution"]))
                except ValueError:
                    warnings.append(f"Invalid Resolution: {kv.group(2)!r}")
            continue
        if section == "synctrack":
            bpm_match = _BPM_RE.match(line)
            if bpm_match:
                milli = int(bpm_match.group(2))
                bpm = milli / 1000.0
                found_bpm = True
            continue
        attr = section_map.get(section)
        if attr is None:
            continue
        note_match = _NOTE_RE.match(line)
        if not note_match:
            continue
        getattr(charts, attr).append(
            ChartNote(
                tick=int(note_match.group(1)),
                fret=int(note_match.group(2)),
                sustain=int(note_match.group(3)),
            )
        )

    if not found_bpm:
        warnings.append("notes.chart missing SyncTrack BPM; assuming 120")
    if not any(charts.expert or charts.hard or charts.medium or charts.easy):
        warnings.append("notes.chart has no guitar note events")

    meta = SongMeta(
        name=song_fields.get("name", ""),
        artist=song_fields.get("artist", ""),
        album=song_fields.get("album", ""),
        genre=song_fields.get("genre", ""),
        year=_normalize_year(song_fields.get("year", "")),
    )
    music_stream = song_fields.get("musicstream", "song.ogg") or "song.ogg"
    # Older charts may omit GuitarStream entirely.
    guitar_stream = song_fields.get("guitarstream", "") or ""
    return (
        meta,
        TempoMap(bpm=bpm, resolution=resolution),
        charts,
        music_stream,
        guitar_stream,
        warnings,
    )
