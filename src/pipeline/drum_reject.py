"""Drop weak guitar notes that only exist because a drum hit leaked in."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.pipeline.types import NoteEvent

ALIGN_S = 0.04
STRONG_VELOCITY = 0.62


def detect_drum_onsets(drums_wav: Path) -> np.ndarray:
    import librosa

    y, sr = librosa.load(str(drums_wav), sr=22050, mono=True)
    if y.size == 0:
        return np.zeros(0, dtype=np.float64)
    onsets = librosa.onset.onset_detect(
        y=y,
        sr=sr,
        units="time",
        backtrack=True,
        delta=0.07,
    )
    # Kicks at t=0 are often missed by onset_detect.
    lead = y[: max(int(0.04 * sr), 1)]
    if float(np.max(np.abs(lead))) > 0.08:
        onsets = np.unique(np.concatenate(([0.0], np.asarray(onsets, dtype=np.float64))))
    return np.asarray(onsets, dtype=np.float64)


def reject_drum_aligned(
    notes: list[NoteEvent],
    drums_wav: Path | None,
    *,
    align_s: float = ALIGN_S,
    strong_velocity: float = STRONG_VELOCITY,
    drum_onsets: np.ndarray | None = None,
) -> list[NoteEvent]:
    """Remove low-confidence notes that sit on drum onsets; keep strong pitched hits."""
    if not notes or drums_wav is None or not Path(drums_wav).is_file():
        return notes
    onsets = (
        np.asarray(drum_onsets, dtype=np.float64)
        if drum_onsets is not None
        else detect_drum_onsets(Path(drums_wav))
    )
    if onsets.size == 0:
        return notes
    kept: list[NoteEvent] = []
    for note in notes:
        if note.velocity >= strong_velocity:
            kept.append(note)
            continue
        if np.any(np.abs(onsets - note.start_s) <= align_s):
            continue
        kept.append(note)
    return kept or notes
