"""Tempo-relative note thinning after the evidence score.

Merges same-pitch overlaps and drops fragments shorter than half a 16th.
Keeps octave doubles (power chords).
"""

from __future__ import annotations

from src.pipeline.types import NoteEvent

MIN_CHARTER_DURATION_S = 0.04
MERGE_GAP_S = 0.05


def min_duration_for_tempo(bpm: float, *, floor_s: float = MIN_CHARTER_DURATION_S) -> float:
    """Half a 16th (a 32nd) at ``bpm``, never above ``floor_s``.

    Fast picking stays near 40 ms at normal rock tempi. Very high BPM can go
    slightly shorter. GuitarSet recall collapsed when this was a 70 ms+ floor.
    """
    if bpm <= 0:
        return floor_s
    thirty_second_s = (60.0 / bpm) / 8.0
    if thirty_second_s <= 0:
        return floor_s
    return min(floor_s, thirty_second_s)


def thin_charter_notes(
    notes: list[NoteEvent],
    *,
    min_duration_s: float = MIN_CHARTER_DURATION_S,
    merge_gap_s: float = MERGE_GAP_S,
) -> list[NoteEvent]:
    """Drop tiny notes and merge same-pitch overlaps. Keeps octave doubles."""
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
    merged.sort(key=lambda n: (n.start_s, n.midi_pitch))
    return merged or notes
