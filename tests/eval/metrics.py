"""Accuracy metrics for transcription, chart density, and isolation bleed."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from src.pipeline.types import ChartNote, NoteEvent


ONSET_TOLERANCE_S = 0.05


@dataclass(frozen=True)
class TranscriptionScores:
    precision: float
    recall: float
    f_measure: float
    ref_notes: int
    est_notes: int


@dataclass(frozen=True)
class DensityScores:
    duration_s: float
    expert_notes: int
    hard_notes: int
    expert_nps: float
    hard_nps: float
    expert_unique_ticks: int


def _intervals_and_pitches(notes: list[NoteEvent]) -> tuple[np.ndarray, np.ndarray]:
    if not notes:
        return np.zeros((0, 2), dtype=np.float64), np.zeros(0, dtype=np.float64)
    intervals = np.array([[n.start_s, n.end_s] for n in notes], dtype=np.float64)
    pitches = np.array([n.midi_pitch for n in notes], dtype=np.float64)
    return intervals, pitches


def score_transcription(
    reference: list[NoteEvent],
    estimated: list[NoteEvent],
    *,
    onset_tolerance: float = ONSET_TOLERANCE_S,
) -> TranscriptionScores:
    """Onset+pitch F-measure via mir_eval (50 ms default onset window)."""
    import mir_eval

    ref_int, ref_pitches = _intervals_and_pitches(reference)
    est_int, est_pitches = _intervals_and_pitches(estimated)
    if ref_int.size == 0 and est_int.size == 0:
        return TranscriptionScores(1.0, 1.0, 1.0, 0, 0)
    if ref_int.size == 0:
        return TranscriptionScores(0.0, 0.0, 0.0, 0, len(estimated))
    if est_int.size == 0:
        return TranscriptionScores(0.0, 0.0, 0.0, len(reference), 0)

    precision, recall, f_measure, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_int,
        ref_pitches,
        est_int,
        est_pitches,
        onset_tolerance=onset_tolerance,
        pitch_tolerance=50.0,  # cents
        offset_ratio=None,  # onset+pitch only; sustain mapping is heuristic
    )
    return TranscriptionScores(
        precision=float(precision),
        recall=float(recall),
        f_measure=float(f_measure),
        ref_notes=len(reference),
        est_notes=len(estimated),
    )


def score_density(
    expert: list[ChartNote],
    hard: list[ChartNote],
    *,
    duration_s: float,
    tempo_bpm: float,
    resolution: int,
) -> DensityScores:
    duration_s = max(float(duration_s), 1e-6)
    return DensityScores(
        duration_s=duration_s,
        expert_notes=len(expert),
        hard_notes=len(hard),
        expert_nps=len(expert) / duration_s,
        hard_nps=len(hard) / duration_s,
        expert_unique_ticks=len({n.tick for n in expert}),
    )


def scores_to_dict(scores: TranscriptionScores | DensityScores) -> dict:
    return asdict(scores)
