from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures"
SUITE_DIR = ROOT / "e2e_output" / "clip_suite"


@dataclass
class Clip:
    clip_id: str
    title: str
    artist: str
    source_name: str
    kind: str


CLIPS = [
    Clip("acoustic_chords", "Acoustic Chords", "Frederic Jacquot", "acoustic_chords.mp3", "solo acoustic"),
    Clip("electric_lick", "Electric Lick", "Include.aliment", "electric_lick.mp3", "solo electric"),
    Clip("acoustic_shuffle", "Acoustic Shuffle", "RiverCO", "acoustic_shuffle.mp3", "mixed band"),
    Clip("guitar_tune", "Guitar Tune", "AntumDeluge", "guitar_tune.mp3", "short melody"),
]


def parse_expert_notes(chart_text: str) -> list[tuple[int, int, int]]:
    section = re.search(r"\[ExpertSingle\]\s*\{(.*?)\n\}", chart_text, re.S)
    if not section:
        return []
    return [
        (int(tick), int(fret), int(sustain))
        for tick, fret, sustain in re.findall(r"(\d+) = N (\d+) (\d+)", section.group(1))
        if int(fret) <= 4
    ]


def clip_output_dir(clip: Clip) -> Path:
    return SUITE_DIR / clip.clip_id


def summarize_clip(clip: Clip) -> dict:
    out = clip_output_dir(clip)
    chart = out / "notes.chart"
    ini = out / "song.ini"
    song = out / "song.ogg"
    guitar = out / "guitar.ogg"
    notes = parse_expert_notes(chart.read_text(encoding="utf-8")) if chart.is_file() else []
    bpm = None
    if chart.is_file():
        match = re.search(r"0 = B (\d+)", chart.read_text(encoding="utf-8"))
        if match:
            bpm = int(match.group(1)) / 1000.0
    return {
        **asdict(clip),
        "ok": all(p.is_file() and p.stat().st_size > 0 for p in (chart, ini, song, guitar)) and len(notes) > 0,
        "expert_notes": len(notes),
        "unique_frets": sorted({n[1] for n in notes}),
        "bpm": bpm,
        "song_ogg_bytes": song.stat().st_size if song.is_file() else 0,
        "guitar_ogg_bytes": guitar.stat().st_size if guitar.is_file() else 0,
        "folder": str(out),
    }
