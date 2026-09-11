"""Confirm weak Basic Pitch notes against a guitar-oriented pyin track."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.pipeline.types import NoteEvent

WEAK_VELOCITY = 0.55
CENTS_TOLERANCE = 50.0
FMIN_HZ = 82.0
FMAX_HZ = 1318.5


def _hz_to_midi(hz: float) -> float:
    return 69.0 + 12.0 * np.log2(hz / 440.0)


def _pyin_track(guitar_wav: Path) -> tuple[np.ndarray, np.ndarray]:
    import librosa

    y, sr = librosa.load(str(guitar_wav), sr=22050, mono=True)
    if y.size == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    f0, voiced_flag, _prob = librosa.pyin(
        y,
        fmin=FMIN_HZ,
        fmax=FMAX_HZ,
        sr=sr,
    )
    times = librosa.times_like(f0, sr=sr)
    f0 = np.asarray(f0, dtype=np.float64)
    if voiced_flag is not None:
        f0 = np.where(voiced_flag, f0, np.nan)
    return times.astype(np.float64), f0


def _median_midi_near(times: np.ndarray, f0: np.ndarray, start_s: float, end_s: float) -> float | None:
    if times.size == 0:
        return None
    lo = start_s
    hi = max(start_s + 0.06, min(end_s, start_s + 0.12))
    mask = (times >= lo) & (times <= hi) & np.isfinite(f0) & (f0 > 0)
    if not np.any(mask):
        # widen slightly around onset
        mask = (np.abs(times - start_s) <= 0.08) & np.isfinite(f0) & (f0 > 0)
    if not np.any(mask):
        return None
    return float(np.median(_hz_to_midi(f0[mask])))


def confirm_with_pyin(
    notes: list[NoteEvent],
    guitar_wav: Path,
    *,
    weak_velocity: float = WEAK_VELOCITY,
    cents_tolerance: float = CENTS_TOLERANCE,
) -> list[NoteEvent]:
    """Keep strong notes; require pyin pitch agreement for weaker ones."""
    if not notes or not Path(guitar_wav).is_file():
        return notes
    times, f0 = _pyin_track(Path(guitar_wav))
    kept: list[NoteEvent] = []
    for note in notes:
        if note.velocity >= weak_velocity:
            kept.append(note)
            continue
        median_midi = _median_midi_near(times, f0, note.start_s, note.end_s)
        if median_midi is None:
            continue
        cents = abs(median_midi - note.midi_pitch) * 100.0
        if cents <= cents_tolerance:
            kept.append(note)
    return kept or notes
