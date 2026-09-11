"""Unit tests for accuracy metrics, stem cleanup, and Expert density."""

from __future__ import annotations

import numpy as np

import soundfile as sf

from src.pipeline.confirm import confirm_with_pyin
from src.pipeline.drum_reject import reject_drum_aligned
from src.pipeline.fretmap import map_difficulties, notes_to_expert, prune_expert_density
from src.pipeline.stem_clean import bleed_proxy_score, clean_guitar_stem, subtract_drum_bleed
from src.pipeline.transcribe import choose_sensitivity, filter_note_events
from src.pipeline.types import RESOLUTION, ChartNote, NoteEvent, TempoMap
from tests.eval.metrics import score_density, score_transcription
from tests.eval.synthesize import (
    EVAL_DIR,
    LABEL_DIR,
    MELODY,
    SR,
    _drum_hits,
    ground_truth_events,
    load_label_notes,
    render_guitar,
    write_eval_fixtures,
)


def test_score_transcription_perfect_match():
    truth = ground_truth_events()
    scores = score_transcription(truth, truth)
    assert scores.precision == 1.0
    assert scores.recall == 1.0
    assert scores.f_measure == 1.0


def test_score_transcription_penalizes_extra_notes():
    truth = ground_truth_events()
    noisy = list(truth) + [NoteEvent(0.1, 0.2, 40, 0.9)]
    scores = score_transcription(truth, noisy)
    assert scores.precision < 1.0
    assert scores.recall == 1.0


def test_velocity_filter_drops_bleed_ghosts():
    raw = [
        (0.4, 0.75, 52, 0.85),
        (0.0, 0.08, 48, 0.20),
        (0.5, 0.58, 55, 0.22),
        (0.85, 1.15, 55, 0.80),
    ]
    notes = filter_note_events(raw)
    assert len(notes) == 2
    assert {n.midi_pitch for n in notes} == {52, 55}


def test_expert_prune_reduces_32nd_spam():
    tempo = TempoMap(bpm=120.0)
    expert = [ChartNote(tick=i * 12, fret=i % 5, sustain=0) for i in range(40)]
    pruned = prune_expert_density(expert, tempo.resolution)
    assert len(pruned) < len(expert)
    ticks = sorted({n.tick for n in pruned})
    gaps = [b - a for a, b in zip(ticks, ticks[1:])]
    assert gaps
    assert min(gaps) >= tempo.resolution // 8


def test_map_difficulties_expert_not_denser_than_raw_expert():
    tempo = TempoMap(bpm=120.0)
    notes = [
        NoteEvent(i * 0.05, i * 0.05 + 0.04, 60 + (i % 5), 0.8)
        for i in range(40)
    ]
    raw = notes_to_expert(notes, tempo)
    charts = map_difficulties(notes, tempo)
    assert len(charts.expert) <= len(raw)
    assert len(charts.hard) <= len(charts.expert)
    assert len(charts.easy) <= len(charts.medium)


def test_clean_guitar_stem_reduces_drum_bleed_proxy():
    write_eval_fixtures()
    guitar = np.load(EVAL_DIR / "guitar_bleed_stem.npy")
    backing = np.load(EVAL_DIR / "backing_drums_stem.npy")
    before = bleed_proxy_score(guitar, backing, SR)
    after = bleed_proxy_score(clean_guitar_stem(guitar, SR), backing, SR)
    assert after < before
    assert (before - after) / before > 0.05


def test_density_scores_nps():
    expert = [ChartNote(tick=i * 48, fret=0, sustain=0) for i in range(10)]
    hard = expert[::2]
    scores = score_density(
        expert, hard, duration_s=5.0, tempo_bpm=120.0, resolution=RESOLUTION
    )
    assert scores.expert_notes == 10
    assert abs(scores.expert_nps - 2.0) < 1e-6


def test_render_distorted_differs_from_clean():
    clean = render_guitar(MELODY, distorted=False)
    dist = render_guitar(MELODY, distorted=True)
    assert clean.shape == dist.shape
    assert float(np.mean(np.abs(dist))) != float(np.mean(np.abs(clean)))


def test_drum_hits_have_energy_on_beats():
    drums = _drum_hits(SR * 2, SR, period_s=0.5)
    assert float(np.max(np.abs(drums))) > 0.1


def test_postfilter_improves_precision_on_bleed_ghosts():
    truth = ground_truth_events()
    noisy = list(truth)
    for beat in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
        noisy.append(NoteEvent(beat, beat + 0.08, 48, 0.22))
        noisy.append(NoteEvent(beat + 0.02, beat + 0.10, 55, 0.55))
    raw = [(n.start_s, n.end_s, n.midi_pitch, n.velocity) for n in noisy]
    legacy = [
        NoteEvent(n.start_s, n.end_s, n.midi_pitch, n.velocity)
        for n in noisy
        if 40 <= n.midi_pitch <= 88
    ]
    filtered = filter_note_events(raw)
    legacy_scores = score_transcription(truth, legacy)
    filtered_scores = score_transcription(truth, filtered)
    assert filtered_scores.precision > legacy_scores.precision
    assert filtered_scores.recall >= 0.99
    assert filtered_scores.f_measure > legacy_scores.f_measure


def test_subtract_drums_reduces_bleed_beyond_cleanup():
    write_eval_fixtures()
    guitar = np.load(EVAL_DIR / "guitar_bleed_stem.npy")
    drums = np.load(EVAL_DIR / "backing_drums_stem.npy")
    cleaned = clean_guitar_stem(guitar, SR)
    after_clean = bleed_proxy_score(cleaned, drums, SR)
    after_sub = bleed_proxy_score(subtract_drum_bleed(cleaned, drums, SR), drums, SR)
    assert after_sub < after_clean
    assert (after_clean - after_sub) / max(after_clean, 1e-9) > 0.05


def test_auto_sensitivity_is_strict_on_raw_bleed():
    write_eval_fixtures()
    guitar = np.load(EVAL_DIR / "guitar_bleed_stem.npy")
    drums = np.load(EVAL_DIR / "backing_drums_stem.npy")
    assert choose_sensitivity("auto", guitar, drums, SR) == "strict"
    assert choose_sensitivity("sensitive", guitar, drums, SR) == "sensitive"
    clean = np.load(EVAL_DIR / "guitar_clean_stem.npy")
    quiet = drums * 0.01
    assert choose_sensitivity("auto", clean, quiet, SR) in {"balanced", "sensitive"}


def test_drum_reject_drops_weak_beat_ghosts(tmp_path: Path):
    write_eval_fixtures()
    drums = EVAL_DIR / "drums_only.wav"
    notes = [
        NoteEvent(0.40, 0.75, 52, 0.85),
        NoteEvent(0.00, 0.08, 48, 0.40),
        NoteEvent(0.50, 0.58, 55, 0.40),
        NoteEvent(0.85, 1.15, 55, 0.80),
        NoteEvent(0.25, 0.35, 60, 0.40),
    ]
    kept = reject_drum_aligned(notes, drums)
    pitches = {(round(n.start_s, 2), n.midi_pitch) for n in kept}
    assert (0.4, 52) in pitches
    assert (0.85, 55) in pitches
    assert (0.0, 48) not in pitches
    assert (0.5, 55) not in pitches
    assert (0.25, 60) in pitches


def test_drum_reject_keeps_strong_notes_on_the_beat():
    write_eval_fixtures()
    notes = [NoteEvent(0.0, 0.2, 52, 0.9), NoteEvent(0.5, 0.7, 55, 0.85)]
    kept = reject_drum_aligned(notes, EVAL_DIR / "drums_only.wav")
    assert len(kept) == 2


def test_pyin_confirms_matching_weak_note_and_drops_wrong_pitch(tmp_path: Path):
    sr = 22050
    freq = 329.6276  # E4 / MIDI 64
    t = np.arange(int(sr * 1.2), dtype=np.float32) / sr
    tone = (0.35 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    wav = tmp_path / "e4.wav"
    sf.write(str(wav), tone, sr)
    notes = [
        NoteEvent(0.2, 0.7, 64, 0.40),
        NoteEvent(0.2, 0.7, 50, 0.40),
        NoteEvent(0.2, 0.7, 47, 0.90),
    ]
    kept = confirm_with_pyin(notes, wav)
    pitches = {n.midi_pitch for n in kept}
    assert 64 in pitches
    assert 50 not in pitches
    assert 47 in pitches


def test_band_mix_fixture_and_labels_exist():
    write_eval_fixtures()
    assert (EVAL_DIR / "band_mix.wav").is_file()
    for clip_id in ("electric_lick", "acoustic_chords", "acoustic_shuffle"):
        path = LABEL_DIR / f"{clip_id}.json"
        assert path.is_file(), path
        notes = load_label_notes(path)
        assert notes, clip_id


def test_sensitive_keeps_more_than_strict():
    raw = [
        (0.4, 0.8, 52, 0.85),
        (1.0, 1.3, 55, 0.30),
        (1.6, 1.9, 57, 0.28),
    ]
    strict = filter_note_events(raw, min_velocity=0.42)
    sensitive = filter_note_events(raw, min_velocity=0.25)
    assert len(sensitive) > len(strict)
