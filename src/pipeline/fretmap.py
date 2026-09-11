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
# Expert used to keep every Basic Pitch hit. A light spacing floor stops
# bleed/false-onset storms from becoming an unplayable highway.
EXPERT_MIN_SPACING_DIVISOR = 8  # 32nd notes at RESOLUTION=192 → 24 ticks
EXPERT_MAX_CHORD = MAX_CHORD

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


# Standard guitar tuning → Clone Hero lanes.
# High E shares orange with B so six strings fit five frets.
STANDARD_TUNING = (40, 45, 50, 55, 59, 64)  # E2 A2 D3 G3 B3 E4
STRING_TO_LANE = (0, 1, 2, 3, 4, 4)
MAX_PLAYABLE_FRET = 19
PREFERRED_FRET_CENTER = 5


def fingering_candidates(midi_pitch: int) -> list[tuple[int, int]]:
    found: list[tuple[int, int]] = []
    for string_i, open_midi in enumerate(STANDARD_TUNING):
        fret = midi_pitch - open_midi
        if 0 <= fret <= MAX_PLAYABLE_FRET:
            found.append((string_i, fret))
    return found


def pick_fingering(
    midi_pitch: int,
    last_fret: int | None = None,
    last_string: int | None = None,
    target_fret: int | None = None,
) -> tuple[int, int]:
    candidates = fingering_candidates(midi_pitch)
    if not candidates:
        if midi_pitch < STANDARD_TUNING[0]:
            return 0, 0
        return 5, min(MAX_PLAYABLE_FRET, midi_pitch - STANDARD_TUNING[5])

    def score(pair: tuple[int, int]) -> float:
        string_i, fret = pair
        cost = abs(fret - PREFERRED_FRET_CENTER) * 0.35
        if target_fret is not None:
            cost += abs(fret - target_fret) * 2.0
        if last_fret is not None:
            cost += abs(fret - last_fret) * 2.2
        if last_string is not None:
            cost += abs(string_i - last_string) * 0.45
        return cost

    return min(candidates, key=score)


def midi_to_lane(
    midi_pitch: int,
    last_fret: int | None = None,
    last_string: int | None = None,
    target_fret: int | None = None,
) -> tuple[int, int, int]:
    """Return (clone-hero fret 0-4, string index, guitar fret)."""
    string_i, fret = pick_fingering(
        midi_pitch,
        last_fret=last_fret,
        last_string=last_string,
        target_fret=target_fret,
    )
    return STRING_TO_LANE[string_i], string_i, fret


def _resolve_lane_collisions(lanes: list[int]) -> list[int]:
    used: set[int] = set()
    out: list[int] = []
    for lane in lanes:
        if lane not in used:
            used.add(lane)
            out.append(lane)
            continue
        placed = False
        for delta in (1, -1, 2, -2, 3, -3, 4, -4):
            cand = lane + delta
            if 0 <= cand <= 4 and cand not in used:
                used.add(cand)
                out.append(cand)
                placed = True
                break
        if not placed:
            continue
    return out


def _shared_target_fret(midi_pitches: list[int]) -> int:
    best = PREFERRED_FRET_CENTER
    best_score = float("inf")
    for target in range(0, 13):
        score = 0.0
        for pitch in midi_pitches:
            cands = fingering_candidates(pitch)
            if not cands:
                score += 24.0
                continue
            score += min(abs(fret - target) for _s, fret in cands)
        if score < best_score:
            best_score = score
            best = target
    return best


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


def assign_chord_frets(midi_pitches: list[int], tonic: int = 0, max_chord: int = MAX_CHORD) -> list[int]:
    """Map simultaneous MIDI pitches to distinct Clone Hero lanes via string fingering.

    ``tonic`` is unused; kept so existing callers/tests still type-check.
    """
    del tonic
    unique = sorted(set(midi_pitches))
    if not unique:
        return []
    if len(unique) > max_chord:
        if max_chord == 1:
            unique = [unique[len(unique) // 2]]
        elif max_chord == 2:
            unique = [unique[0], unique[-1]]
        else:
            indexes = [round(i * (len(unique) - 1) / (max_chord - 1)) for i in range(max_chord)]
            unique = [unique[i] for i in dict.fromkeys(indexes)]
    target = _shared_target_fret(unique)
    lanes = [midi_to_lane(pitch, target_fret=target)[0] for pitch in unique]
    resolved = _resolve_lane_collisions(lanes)
    return sorted(resolved)[:max_chord]


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
    chart: list[ChartNote] = []
    seen: set[tuple[int, int]] = set()
    last_fret: int | None = None
    last_string: int | None = None
    for group in _group_onsets(notes):
        start_s = min(n.start_s for n in group)
        end_s = max(n.end_s for n in group)
        start_tick = snap_tick(tempo.time_to_tick(start_s), tempo.resolution)
        end_tick = tempo.time_to_tick(end_s)
        sustain = _sustain_ticks(start_tick, end_tick, tempo.resolution)
        pitches = [n.midi_pitch for n in group]
        if len(pitches) == 1:
            lane, last_string, last_fret = midi_to_lane(
                pitches[0],
                last_fret=last_fret,
                last_string=last_string,
            )
            frets = [lane]
        else:
            frets = assign_chord_frets(pitches)
            _lane, last_string, last_fret = midi_to_lane(
                max(pitches),
                last_fret=last_fret,
                last_string=last_string,
            )
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


def prune_expert_density(notes: list[ChartNote], resolution: int = RESOLUTION) -> list[ChartNote]:
    """Keep Expert playable when transcription returns near-continuous false notes."""
    min_spacing = max(resolution // EXPERT_MIN_SPACING_DIVISOR, 1)
    return thin_for_difficulty(
        notes,
        max_chord=EXPERT_MAX_CHORD,
        min_spacing=min_spacing,
    )


def map_difficulties(notes: list[NoteEvent], tempo: TempoMap) -> ChartData:
    expert_raw = notes_to_expert(notes, tempo)
    expert = prune_expert_density(expert_raw, tempo.resolution)
    res = tempo.resolution
    return ChartData(
        expert=expert,
        hard=thin_for_difficulty(expert, max_chord=2, min_spacing=res // 4),
        medium=thin_for_difficulty(expert, max_chord=2, min_spacing=res // 2),
        easy=thin_for_difficulty(expert, max_chord=1, min_spacing=res),
    )
