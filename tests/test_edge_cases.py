from __future__ import annotations

from pathlib import Path

import pytest

from src.pipeline.audio import decode_to_wav
from src.pipeline.run import run_pipeline
from src.pipeline.transcribe import transcribe_guitar
from src.pipeline.util import ensure_ffmpeg, parse_filename_metadata, sanitize_folder_name


def test_run_pipeline_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Audio file not found"):
        run_pipeline(tmp_path / "missing.mp3", tmp_path / "Out", "Song", "Artist")


def test_run_pipeline_does_not_create_output_before_work(tmp_path: Path) -> None:
    dest = tmp_path / "Should Stay Missing"
    with pytest.raises(FileNotFoundError):
        run_pipeline(tmp_path / "missing.mp3", dest, "Song", "Artist")
    assert not dest.exists()


def test_ensure_ffmpeg_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.pipeline.util.shutil.which", lambda _name: None)
    with pytest.raises(RuntimeError, match="ffmpeg was not found"):
        ensure_ffmpeg()


def test_transcribe_raises_when_no_notes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    wav = tmp_path / "empty.wav"
    wav.write_bytes(b"RIFF")

    monkeypatch.setattr(
        "basic_pitch.inference.predict",
        lambda *_args, **_kwargs: (None, None, []),
    )
    monkeypatch.setattr(
        "basic_pitch.build_icassp_2022_model_path",
        lambda _suffix: tmp_path / "nmp.onnx",
    )

    with pytest.raises(RuntimeError, match="No guitar notes"):
        transcribe_guitar(wav)


def test_sanitize_illegal_windows_name() -> None:
    assert sanitize_folder_name('A<>:"/\\|?*B') == "A_________B"
    assert sanitize_folder_name("   ") == "Untitled"
    assert sanitize_folder_name("ends.") == "ends"


def test_parse_filename_artist_title() -> None:
    title, artist = parse_filename_metadata(Path("Frederic Jacquot - Acoustic Chords.mp3"))
    assert title == "Acoustic Chords"
    assert artist == "Frederic Jacquot"


def test_parse_filename_without_separator() -> None:
    title, artist = parse_filename_metadata(Path("solo_guitar_riff.mp3"))
    assert title == "solo guitar riff"
    assert artist == "Unknown"


def test_decode_rejects_corrupt_audio(tmp_path: Path) -> None:
    junk = tmp_path / "corrupt.mp3"
    junk.write_bytes(b"this is not audio")
    with pytest.raises(RuntimeError, match="(?i)ffmpeg|invalid|error"):
        decode_to_wav(junk, tmp_path / "out.wav")


def test_legal_clone_hero_package_still_on_disk() -> None:
    folders = [
        Path.home() / "AppData" / "Local" / "guitar_h_isolation_e2e" / "AcousticChordsWalkthrough",
        Path.home() / "Documents" / "Clone Hero" / "Songs" / "Acoustic Chords",
    ]
    found = [path for path in folders if (path / "notes.chart").is_file()]
    if not found:
        pytest.skip("Legal walkthrough folder is not on this machine")
    folder = found[0]
    for name in ("song.ogg", "guitar.ogg", "song.ini", "notes.chart"):
        path = folder / name
        assert path.is_file(), f"missing {name}"
        assert path.stat().st_size > 0
    chart = (folder / "notes.chart").read_text(encoding="utf-8")
    for section in ("[ExpertSingle]", "[HardSingle]", "[MediumSingle]", "[EasySingle]"):
        assert section in chart
    assert " = N " in chart
    ini = (folder / "song.ini").read_text(encoding="utf-8")
    assert "name = Acoustic Chords" in ini
    assert (folder / "song.ogg").stat().st_size > 10_000
    assert (folder / "guitar.ogg").stat().st_size > 10_000


def test_install_and_run_are_one_step() -> None:
    root = Path(__file__).resolve().parents[1]
    install = (root / "install.bat").read_text(encoding="utf-8")
    run = (root / "run.bat").read_text(encoding="utf-8")
    assert "guitar_h_isolation_venv" in install
    assert "pip install --no-deps basic-pitch" in install
    assert "call" in run and "install.bat" in run
    assert "nopause" in run
