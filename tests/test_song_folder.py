"""Backwards-compatible song folder open/view tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.pipeline.chart_writer import render_chart, write_chart
from src.pipeline.package import render_song_ini, write_song_ini
from src.pipeline.song_folder import is_song_folder, load_song_folder
from src.pipeline.types import ChartData, ChartNote, SongMeta, TempoMap


def _meta() -> SongMeta:
    return SongMeta(
        name="Legacy Lick",
        artist="Old Build",
        album="Demos",
        genre="rock",
        year="2024",
        duration_ms=12000,
    )


def _charts() -> ChartData:
    return ChartData(
        expert=[
            ChartNote(tick=192, fret=0, sustain=0),
            ChartNote(tick=384, fret=2, sustain=48),
        ],
        hard=[ChartNote(tick=192, fret=0, sustain=0)],
        medium=[ChartNote(tick=192, fret=0, sustain=0)],
        easy=[ChartNote(tick=192, fret=0, sustain=0)],
    )


def _write_stub_ogg(path: Path) -> None:
    # Loader only checks presence; Clone Hero needs real Vorbis later.
    path.write_bytes(b"OggS\x00stub")


def _write_current_folder(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    meta = _meta()
    write_song_ini(root / "song.ini", meta)
    write_chart(root / "notes.chart", meta, TempoMap(bpm=128.0), _charts())
    _write_stub_ogg(root / "song.ogg")
    _write_stub_ogg(root / "guitar.ogg")
    return root


def _write_legacy_sparse_folder(root: Path) -> Path:
    """Older / sparse package: BOM, Expert-only, no GuitarStream, missing guitar.ogg."""
    root.mkdir(parents=True, exist_ok=True)
    ini = (
        "\ufeff[song]\n"
        "name = Sparse Old Song\n"
        "artist = Vintage Charter\n"
        "charter = Guitar H Isolation\n"
        "diff_guitar = 2\n"
        "song_length = 9000\n"
    )
    (root / "song.ini").write_text(ini, encoding="utf-8")
    chart = (
        "[Song]\n"
        "{\n"
        '  Name = "Sparse Old Song"\n'
        '  Artist = "Vintage Charter"\n'
        '  Charter = "Guitar H Isolation"\n'
        "  Offset = 0\n"
        "  Resolution = 192\n"
        '  MusicStream = "song.ogg"\n'
        "}\n"
        "[SyncTrack]\n"
        "{\n"
        "  0 = TS 4\n"
        "  0 = B 100000\n"
        "}\n"
        "[Events]\n"
        "{\n"
        "}\n"
        "[ExpertSingle]\n"
        "{\n"
        "  0 = N 0 0\n"
        "  192 = N 3 0\n"
        "}\n"
    )
    (root / "notes.chart").write_text(chart, encoding="utf-8-sig")
    _write_stub_ogg(root / "song.ogg")
    return root


def test_is_song_folder_requires_core_files(tmp_path: Path):
    folder = tmp_path / "almost"
    folder.mkdir()
    (folder / "song.ini").write_text("[song]\nname = x\n", encoding="utf-8")
    assert not is_song_folder(folder)
    _write_current_folder(tmp_path / "full")
    assert is_song_folder(tmp_path / "full")


def test_load_round_trip_current_package(tmp_path: Path):
    folder = _write_current_folder(tmp_path / "current")
    info = load_song_folder(folder)
    assert info.meta.name == "Legacy Lick"
    assert info.meta.artist == "Old Build"
    assert info.meta.duration_ms == 12000
    assert abs(info.tempo.bpm - 128.0) < 1e-6
    assert info.note_counts["expert"] == 2
    assert info.note_counts["hard"] == 1
    assert info.music_stream == "song.ogg"
    assert info.guitar_stream == "guitar.ogg"
    assert info.missing_audio == ()
    assert any("Legacy Lick" in line for line in info.summary_lines())


def test_load_legacy_sparse_folder_without_guitar_ogg(tmp_path: Path):
    folder = _write_legacy_sparse_folder(tmp_path / "legacy")
    info = load_song_folder(folder)
    assert info.meta.name == "Sparse Old Song"
    assert info.meta.artist == "Vintage Charter"
    assert info.meta.duration_ms == 9000
    assert abs(info.tempo.bpm - 100.0) < 1e-6
    assert info.note_counts["expert"] == 2
    assert info.note_counts["hard"] == 0
    assert info.guitar_stream == ""
    assert "guitar.ogg" in info.missing_audio


def test_load_rejects_non_song_folder(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_song_folder(tmp_path / "missing")


def test_render_then_load_preserves_notes(tmp_path: Path):
    folder = tmp_path / "rendered"
    folder.mkdir()
    meta = _meta()
    charts = _charts()
    (folder / "song.ini").write_text(render_song_ini(meta), encoding="utf-8")
    (folder / "notes.chart").write_text(
        render_chart(meta, TempoMap(bpm=140.5), charts), encoding="utf-8"
    )
    _write_stub_ogg(folder / "song.ogg")
    _write_stub_ogg(folder / "guitar.ogg")
    info = load_song_folder(folder)
    assert [(n.tick, n.fret, n.sustain) for n in info.charts.expert] == [
        (192, 0, 0),
        (384, 2, 48),
    ]
    assert abs(info.tempo.bpm - 140.5) < 1e-6
