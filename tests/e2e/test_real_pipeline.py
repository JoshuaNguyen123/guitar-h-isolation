"""Real pipeline E2E: MP3 → stems → notes.chart Clone Hero package.

Playwright cannot drive this CustomTkinter desktop app. This test runs the
same isolate → transcribe → chart → package path the UI calls.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from src.pipeline.run import run_pipeline

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "guitar_chords.mp3"
OUT_DIR = Path(__file__).resolve().parents[2] / "e2e_output" / "Acoustic Chords"


def _parse_expert_notes(chart_text: str) -> list[tuple[int, int, int]]:
    section = re.search(r"\[ExpertSingle\]\s*\{(.*?)\n\}", chart_text, re.S)
    assert section, "notes.chart is missing [ExpertSingle]"
    notes = []
    for tick, fret, sustain in re.findall(r"(\d+) = N (\d+) (\d+)", section.group(1)):
        notes.append((int(tick), int(fret), int(sustain)))
    return notes


@pytest.mark.e2e
def test_real_mp3_produces_clone_hero_package() -> None:
    if not FIXTURE.is_file():
        pytest.skip(f"Missing fixture: {FIXTURE}")

    stages: list[tuple[str, float]] = []

    def progress(stage: str, fraction: float) -> None:
        print(f"[{fraction:0.2f}] {stage}", flush=True)
        stages.append((stage, fraction))

    output = run_pipeline(
        FIXTURE,
        OUT_DIR,
        name="Acoustic Chords",
        artist="Frederic Jacquot",
        progress=progress,
    )

    song = output / "song.ogg"
    guitar = output / "guitar.ogg"
    ini = output / "song.ini"
    chart = output / "notes.chart"
    for path in (song, guitar, ini, chart):
        assert path.is_file(), f"missing {path.name}"
    assert song.stat().st_size > 10_000, "song.ogg is too small"
    assert guitar.stat().st_size > 10_000, "guitar.ogg is too small"
    assert ini.stat().st_size > 50, "song.ini is empty"
    assert chart.stat().st_size > 200, "notes.chart is too small"

    ini_text = ini.read_text(encoding="utf-8")
    assert "name = Acoustic Chords" in ini_text
    assert "charter = Guitar H Isolation" in ini_text
    length = int(re.search(r"song_length = (\d+)", ini_text).group(1))
    assert 4000 <= length <= 20000

    chart_text = chart.read_text(encoding="utf-8")
    notes = _parse_expert_notes(chart_text)
    assert notes, "Expert chart has no notes"
    assert all(0 <= fret <= 4 for _tick, fret, _sus in notes)
    assert "[HardSingle]" in chart_text
    assert "MusicStream = \"song.ogg\"" in chart_text
    bpm = int(re.search(r"0 = B (\d+)", chart_text).group(1))
    assert 60000 <= bpm <= 220000

    report = {
        "output": str(output),
        "song_ogg_bytes": song.stat().st_size,
        "guitar_ogg_bytes": guitar.stat().st_size,
        "song_length_ms": length,
        "bpm": bpm / 1000.0,
        "expert_notes": len(notes),
        "unique_frets": sorted({fret for _t, fret, _s in notes}),
        "first_ticks": [n[0] for n in notes[:8]],
        "stages": stages,
    }
    report_path = output / "e2e_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    from tests.e2e.write_report import main as write_html_report

    write_html_report()
    print(json.dumps(report, indent=2), flush=True)
    assert any(stage == "Done" for stage, _frac in stages)
