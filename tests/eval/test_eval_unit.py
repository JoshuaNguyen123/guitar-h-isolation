"""Unit tests for accuracy metrics, evidence scoring, and Expert density."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from src.pipeline.evidence import keep_scored, score_notes
from src.pipeline.filters import apply_evidence_filters
from src.pipeline.fretmap import map_difficulties, notes_to_expert, prune_expert_density
from src.pipeline.transcribe import (
    SENSITIVITY_PRESETS,
    choose_sensitivity,
    filter_note_events,
    resolve_preset,
)
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


def _fake_posterior(notes: list[NoteEvent], duration_s: float = 4.2) -> dict:
    n_frames = int(duration_s * 86.0) + 8
    mat = np.zeros((n_frames, 88), dtype=np.float32)
    for note in notes:
        idx = note.midi_pitch - 21
        if not 0 <= idx < 88:
            continue
        lo = int(round((note.start_s + 0.06) * 86.0))
        hi = int(round(min(note.end_s, note.start_s + 0.20) * 86.0))
        lo = max(0, lo)
        hi = min(n_frames, max(lo + 1, hi))
        value = 0.85 if (note.end_s - note.start_s) >= 0.20 else 0.04
        mat[lo:hi, idx] = value
    return {"note": mat}


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


def test_filter_note_events_keeps_quiet_notes():
    raw = [
        (0.4, 0.75, 52, 0.85),
        (0.0, 0.08, 48, 0.20),
        (0.85, 1.15, 55, 0.80),
    ]
    notes = filter_note_events(raw)
    assert len(notes) == 3


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


def test_auto_sensitivity_is_strict_on_high_bleed():
    write_eval_fixtures()
    guitar = np.load(EVAL_DIR / "guitar_bleed_stem.npy")
    drums = np.load(EVAL_DIR / "backing_drums_stem.npy")
    assert choose_sensitivity("auto", guitar, drums, SR) == "strict"
    assert choose_sensitivity("sensitive", guitar, drums, SR) == "sensitive"
    clean = np.load(EVAL_DIR / "guitar_clean_stem.npy")
    quiet = drums * 0.01
    assert choose_sensitivity("auto", clean, quiet, SR) in {"balanced", "sensitive"}


def test_band_mix_fixture_and_labels_exist():
    write_eval_fixtures()
    assert (EVAL_DIR / "band_mix.wav").is_file()
    for clip_id in ("electric_lick", "acoustic_chords", "acoustic_shuffle"):
        path = LABEL_DIR / f"{clip_id}.json"
        assert path.is_file(), path
        notes = load_label_notes(path)
        assert notes, clip_id


def test_evidence_drops_short_onbeat_ghosts_and_keeps_melody():
    write_eval_fixtures()
    truth = ground_truth_events()
    noisy = list(truth)
    for beat in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
        noisy.append(NoteEvent(beat, beat + 0.08, 48, 0.22))
        noisy.append(NoteEvent(beat + 0.02, beat + 0.10, 55, 0.55))
    kept = apply_evidence_filters(
        noisy,
        EVAL_DIR / "clean_melody.wav",
        drums_wav=EVAL_DIR / "drums_only.wav",
        model_output=_fake_posterior(noisy),
        mode="balanced",
        use_pyin=False,
    )
    scores = score_transcription(truth, kept)
    assert scores.recall >= 0.99
    assert scores.f_measure >= 0.80
    assert not any(n.midi_pitch == 48 and n.velocity < 0.3 for n in kept)


def test_evidence_keeps_sustained_onbeat_note():
    write_eval_fixtures()
    notes = [
        NoteEvent(0.0, 0.45, 52, 0.50),
        NoteEvent(0.0, 0.08, 70, 0.50),
    ]
    kept = apply_evidence_filters(
        notes,
        EVAL_DIR / "clean_melody.wav",
        drums_wav=EVAL_DIR / "drums_only.wav",
        model_output=_fake_posterior(notes),
        mode="balanced",
        use_pyin=False,
    )
    pitches = {n.midi_pitch for n in kept}
    assert 52 in pitches
    assert 70 not in pitches


def test_evidence_keeps_chord_voices(tmp_path: Path):
    sr = 22050
    t = np.arange(int(sr * 0.8), dtype=np.float32) / sr
    mix = (
        0.25 * np.sin(2 * np.pi * 196.0 * t)
        + 0.25 * np.sin(2 * np.pi * 246.94 * t)
    ).astype(np.float32)
    wav = tmp_path / "dyad.wav"
    sf.write(str(wav), mix, sr)
    notes = [
        NoteEvent(0.10, 0.55, 55, 0.45),
        NoteEvent(0.10, 0.55, 59, 0.45),
    ]
    kept = apply_evidence_filters(
        notes,
        wav,
        model_output=_fake_posterior(notes, duration_s=0.8),
        mode="balanced",
        use_pyin=False,
    )
    assert {n.midi_pitch for n in kept} >= {55, 59}


def test_strict_keep_bar_is_higher_than_sensitive():
    write_eval_fixtures()
    truth = ground_truth_events()
    noisy = list(truth) + [NoteEvent(0.12, 0.40, 49, 0.32)]
    scored = score_notes(
        noisy,
        _fake_posterior(noisy),
        EVAL_DIR / "clean_melody.wav",
        use_pyin=False,
    )
    strict = keep_scored(scored, SENSITIVITY_PRESETS["strict"].keep_threshold)
    sensitive = keep_scored(scored, SENSITIVITY_PRESETS["sensitive"].keep_threshold)
    assert len(sensitive) >= len(strict)
    assert resolve_preset("strict").keep_threshold > resolve_preset("balanced").keep_threshold
    assert resolve_preset("balanced").keep_threshold > resolve_preset("sensitive").keep_threshold


def test_thin_charter_notes_merges_and_keeps_octaves():
    from src.pipeline.refine import min_duration_for_tempo, thin_charter_notes

    notes = [
        NoteEvent(0.40, 0.42, 52, 0.40),
        NoteEvent(0.85, 1.20, 55, 0.80),
        NoteEvent(1.18, 1.50, 55, 0.70),
        NoteEvent(2.00, 2.30, 60, 0.50),
        NoteEvent(2.02, 2.25, 72, 0.90),
        NoteEvent(3.00, 3.30, 64, 0.80),
        NoteEvent(3.01, 3.28, 67, 0.75),
    ]
    thinned = thin_charter_notes(notes, min_duration_s=0.04)
    assert all((n.end_s - n.start_s) >= 0.04 for n in thinned)
    assert len([n for n in thinned if n.midi_pitch == 55]) == 1
    assert {n.midi_pitch for n in thinned if abs(n.start_s - 2.0) < 0.05} >= {60, 72}
    assert {n.midi_pitch for n in thinned if abs(n.start_s - 3.0) < 0.05} >= {64, 67}
    assert min_duration_for_tempo(120.0) == 0.04
