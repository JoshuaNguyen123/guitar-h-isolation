from __future__ import annotations

from collections import defaultdict

from src.pipeline.types import (
    RESOLUTION,
    ChartData,
    ChartNote,
    NoteEvent,
    TempoMap,
)

CHORD_WINDOW_S = 0.04
MAX_CHORD = 3
MIN_SUSTAIN_BEAT_FRACTION = 0.25

# Krumhansl-Schmuckler major-key profile
_MAJOR_PROFILE = (
    6.35,
    2.23,
    3.48,
    2.33,
    4.38,
    4.09,
    2.52,
    5.19,
    2.39,
    3.66,
    2.29,
    2.88,
)


def estimate_tonic(midi_pitches: list[int]) -> int:
    if not midi_pitches:
        return 0
    counts = [0.0] * 12
    for pitch in midi_pitches:
        counts[pitch % 12] += 1.0
    best_tonic = 0
    best_score = float("-inf")
    for tonic in range(12):
        score = sum(counts[(tonic + i) % 12] * _MAJOR_PROFILE[i] for i in range(12))
        if score > best_score:
            best_score = score
            best_tonic = tonic
    return best_tonic


def midi_to_fret(midi_pitch: int, tonic: int) -> int:
    pc = (midi_pitch - tonic) % 12
    if pc <= 1:
        return 0
    if pc <= 3:
        return 1
    if pc <= 6:
        return 2
    if pc <= 8:
        return 3
    return 4


def assign_chord_frets(midi_pitches: list[int], tonic: int, max_chord: int = MAX_CHORD) -> list[int]:
    unique = sorted(set(midi_pitches))
    if not unique:
        return []
    if len(unique) == 1:
        return [midi_to_fret(unique[0], tonic)]

    chosen = unique
    if len(chosen) > max_chord:
        # Keep lowest, highest, and evenly spaced inner pitches.
        if max_chord == 1:
            chosen = [unique[len(unique) // 2]]
        elif max_chord == 2:
            chosen = [unique[0], unique[-1]]
        else:
            indexes = [round(i * (len(unique) - 1) / (max_chord - 1)) for i in range(max_chord)]
            chosen = [unique[i] for i in dict.fromkeys(indexes)]

    low = chosen[0]
    high = chosen[-1]
    span = max(high - low, 1)
    frets: list[int] = []
    for pitch in chosen:
        raw = int(round((pitch - low) / span * min(4, max(1, len(chosen) - 1))))
        fret = max(0, min(4, raw))
        if fret in frets:
            for candidate in range(5):
                if candidate not in frets:
                    fret = candidate
                    break
        if fret not in frets:
            frets.append(fret)
    return sorted(frets)[:max_chord]


def snap_tick(tick: int, resolution: int = RESOLUTION) -> int:
    sixteenth = max(resolution // 4, 1)
    thirty_second = max(resolution // 8, 1)
    nearest_16 = int(round(tick / sixteenth) * sixteenth)
    if abs(tick - nearest_16) <= sixteenth / 3:
        return max(0, nearest_16)
    nearest_32 = int(round(tick / thirty_second) * thirty_second)
    return max(0, nearest_32)


def _group_onsets(notes: list[NoteEvent]) -> list[list[NoteEvent]]:
    ordered = sorted(notes, key=lambda n: (n.start_s, n.midi_pitch))
    groups: list[list[NoteEvent]] = []
    current: list[NoteEvent] = []
    for note in ordered:
        if not current or note.start_s - current[0].start_s <= CHORD_WINDOW_S:
            current.append(note)
        else:
            groups.append(current)
            current = [note]
    if current:
        groups.append(current)
    return groups


def _sustain_ticks(start_tick: int, end_tick: int, resolution: int) -> int:
    raw = max(0, end_tick - start_tick)
    minimum = int(resolution * MIN_SUSTAIN_BEAT_FRACTION)
    if raw < minimum:
        return 0
    return raw


def clamp_sustains(notes: list[ChartNote], resolution: int = RESOLUTION) -> list[ChartNote]:
    by_fret: dict[int, list[ChartNote]] = defaultdict(list)
    for note in notes:
        by_fret[note.fret].append(note)
    min_sustain = int(resolution * MIN_SUSTAIN_BEAT_FRACTION)
    for seq in by_fret.values():
        seq.sort(key=lambda n: n.tick)
        for current, nxt in zip(seq, seq[1:]):
            max_len = max(0, nxt.tick - current.tick - 1)
            if current.sustain > max_len:
                current.sustain = max_len if max_len >= min_sustain else 0
    return notes


def notes_to_expert(notes: list[NoteEvent], tempo: TempoMap) -> list[ChartNote]:
    tonic = estimate_tonic([n.midi_pitch for n in notes])
    chart: list[ChartNote] = []
    seen: set[tuple[int, int]] = set()
    for group in _group_onsets(notes):
        start_s = min(n.start_s for n in group)
        end_s = max(n.end_s for n in group)
        start_tick = snap_tick(tempo.time_to_tick(start_s), tempo.resolution)
        end_tick = tempo.time_to_tick(end_s)
        sustain = _sustain_ticks(start_tick, end_tick, tempo.resolution)
        frets = assign_chord_frets([n.midi_pitch for n in group], tonic)
        for fret in frets:
            key = (start_tick, fret)
            if key in seen:
                continue
            seen.add(key)
            chart.append(ChartNote(tick=start_tick, fret=fret, sustain=sustain))
    chart.sort(key=lambda n: (n.tick, n.fret))
    return clamp_sustains(chart, tempo.resolution)


def select_chord_frets(notes: list[ChartNote], max_chord: int) -> list[ChartNote]:
    if len(notes) <= max_chord:
        return notes
    ordered = sorted(notes, key=lambda n: n.fret)
    if max_chord <= 1:
        return [ordered[len(ordered) // 2]]
    if max_chord == 2:
        return [ordered[0], ordered[-1]]
    indexes = [round(i * (len(ordered) - 1) / (max_chord - 1)) for i in range(max_chord)]
    picked = []
    used = set()
    for i in indexes:
        if i not in used:
            picked.append(ordered[i])
            used.add(i)
    return picked


def thin_for_difficulty(
    notes: list[ChartNote],
    *,
    max_chord: int,
    min_spacing: int,
) -> list[ChartNote]:
    groups: dict[int, list[ChartNote]] = defaultdict(list)
    for note in notes:
        groups[note.tick].append(note)
    kept: list[ChartNote] = []
    last_tick = -(10**9)
    for tick in sorted(groups):
        if tick - last_tick < min_spacing:
            continue
        chord = select_chord_frets(groups[tick], max_chord)
        kept.extend(ChartNote(tick=n.tick, fret=n.fret, sustain=n.sustain) for n in chord)
        last_tick = tick
    kept.sort(key=lambda n: (n.tick, n.fret))
    return clamp_sustains(kept)


def map_difficulties(notes: list[NoteEvent], tempo: TempoMap) -> ChartData:
    expert = notes_to_expert(notes, tempo)
    res = tempo.resolution
    return ChartData(
        expert=expert,
        hard=thin_for_difficulty(expert, max_chord=2, min_spacing=res // 4),
        medium=thin_for_difficulty(expert, max_chord=2, min_spacing=res // 2),
        easy=thin_for_difficulty(expert, max_chord=1, min_spacing=res),
    )
