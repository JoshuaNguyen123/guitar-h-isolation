"""Charter-like note thinning and bass-bleed rejection after Basic Pitch."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.pipeline.drum_reject import detect_drum_onsets
from src.pipeline.types import NoteEvent

MIN_CHARTER_DURATION_S = 0.07
MERGE_GAP_S = 0.05
CONFLICT_WINDOW_S = 0.05
BASS_MIDI_MAX = 42
BASS_ALIGN_S = 0.05
BASS_STRONG_VELOCITY = 0.85


def thin_charter_notes(
    notes: list[NoteEvent],
    *,
    min_duration_s: float = MIN_CHARTER_DURATION_S,
    merge_gap_s: float = MERGE_GAP_S,
    conflict_window_s: float = CONFLICT_WINDOW_S,
) -> list[NoteEvent]:
    """Drop tiny notes, merge same-pitch overlaps, drop quieter octave doubles."""
    if not notes:
        return notes
    lasting = [n for n in notes if (n.end_s - n.start_s) >= min_duration_s]
    lasting.sort(key=lambda n: (n.start_s, n.midi_pitch))
    merged: list[NoteEvent] = []
    for note in lasting:
        if (
            merged
            and note.midi_pitch == merged[-1].midi_pitch
            and note.start_s - merged[-1].end_s <= merge_gap_s
        ):
            prev = merged[-1]
            merged[-1] = NoteEvent(
                start_s=prev.start_s,
                end_s=max(prev.end_s, note.end_s),
                midi_pitch=prev.midi_pitch,
                velocity=max(prev.velocity, note.velocity),
            )
            continue
        merged.append(note)
    # Prefer louder pitch when two notes are an octave apart at the same onset.
    merged.sort(key=lambda n: -n.velocity)
    kept: list[NoteEvent] = []
    for note in merged:
        if any(
            abs(k.start_s - note.start_s) <= conflict_window_s
            and k.midi_pitch != note.midi_pitch
            and abs(k.midi_pitch - note.midi_pitch) % 12 == 0
            for k in kept
        ):
            continue
        kept.append(note)
    kept.sort(key=lambda n: (n.start_s, n.midi_pitch))
    return kept or notes


def reject_bass_aligned(
    notes: list[NoteEvent],
    bass_wav: Path | None,
    *,
    midi_max: int = BASS_MIDI_MAX,
    align_s: float = BASS_ALIGN_S,
    strong_velocity: float = BASS_STRONG_VELOCITY,
    bass_onsets: np.ndarray | None = None,
) -> list[NoteEvent]:
    """Drop low MIDI notes that sit on bass onsets (bass bleed into guitar)."""
    if not notes or bass_wav is None or not Path(bass_wav).is_file():
        return notes
    onsets = (
        np.asarray(bass_onsets, dtype=np.float64)
        if bass_onsets is not None
        else detect_drum_onsets(Path(bass_wav))
    )
    if onsets.size == 0:
        return notes
    kept: list[NoteEvent] = []
    for note in notes:
        if note.midi_pitch > midi_max or note.velocity >= strong_velocity:
            kept.append(note)
            continue
        if np.any(np.abs(onsets - note.start_s) <= align_s):
            continue
        kept.append(note)
    return kept or notes
