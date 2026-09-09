from src.pipeline.fretmap import (
    assign_chord_frets,
    estimate_tonic,
    map_difficulties,
    midi_to_fret,
    notes_to_expert,
    snap_tick,
    thin_for_difficulty,
)
from src.pipeline.types import RESOLUTION, ChartNote, NoteEvent, TempoMap


def test_estimate_tonic_c_major():
    pitches = [60, 62, 64, 65, 67, 69, 71, 72]
    assert estimate_tonic(pitches) == 0


def test_estimate_tonic_empty_defaults_to_c():
    assert estimate_tonic([]) == 0


def test_midi_to_fret_relative_to_c():
    assert midi_to_fret(60, 0) == 0  # C
    assert midi_to_fret(62, 0) == 1  # D
    assert midi_to_fret(65, 0) == 2  # F
    assert midi_to_fret(67, 0) == 3  # G
    assert midi_to_fret(69, 0) == 4  # A


def test_chord_assigns_distinct_frets():
    frets = assign_chord_frets([60, 64, 67], tonic=0)
    assert len(frets) == 3
    assert frets == sorted(set(frets))
    assert all(0 <= fret <= 4 for fret in frets)


def test_chord_caps_at_three():
    frets = assign_chord_frets([60, 62, 64, 67, 71], tonic=0, max_chord=3)
    assert len(frets) <= 3
    assert len(set(frets)) == len(frets)


def test_snap_prefers_sixteenth_when_close():
    assert snap_tick(50, 192) == 48


def test_snap_falls_back_to_thirty_second():
    assert snap_tick(30, 192) == 24


def test_notes_to_expert_groups_chords_and_snaps():
    tempo = TempoMap(bpm=120.0, resolution=192)
    notes = [
        NoteEvent(0.5, 0.7, 60),
        NoteEvent(0.52, 0.7, 64),
        NoteEvent(0.53, 0.7, 67),
        NoteEvent(1.0, 1.1, 72),
    ]
    chart = notes_to_expert(notes, tempo)
    ticks = sorted({n.tick for n in chart})
    assert len(ticks) == 2
    first_chord = [n for n in chart if n.tick == ticks[0]]
    assert 1 <= len(first_chord) <= 3
    assert all(n.tick % 24 == 0 for n in chart)


def test_thin_easy_is_sparser_than_expert():
    expert = [
        ChartNote(tick=i * 24, fret=i % 5, sustain=0)
        for i in range(32)
    ]
    easy = thin_for_difficulty(expert, max_chord=1, min_spacing=RESOLUTION)
    hard = thin_for_difficulty(expert, max_chord=2, min_spacing=RESOLUTION // 4)
    assert len(easy) < len(hard) <= len(expert)
    assert all(len([n for n in easy if n.tick == t]) <= 1 for t in {n.tick for n in easy})


def test_map_difficulties_returns_all_tracks():
    tempo = TempoMap(bpm=100.0)
    notes = [NoteEvent(i * 0.25, i * 0.25 + 0.1, 60 + (i % 12)) for i in range(16)]
    data = map_difficulties(notes, tempo)
    assert data.expert
    assert len(data.easy) <= len(data.medium) <= len(data.hard) <= len(data.expert)
