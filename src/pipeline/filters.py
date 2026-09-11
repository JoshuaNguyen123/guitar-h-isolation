"""Shipped post-Basic-Pitch path: evidence score, then tempo-relative thinning."""

from __future__ import annotations

from pathlib import Path

from src.pipeline.evidence import keep_scored, score_notes
from src.pipeline.refine import min_duration_for_tempo, thin_charter_notes
from src.pipeline.transcribe import resolve_preset
from src.pipeline.types import NoteEvent


def apply_evidence_filters(
    notes: list[NoteEvent],
    guitar_wav: Path,
    drums_wav: Path | None = None,
    bass_wav: Path | None = None,
    *,
    model_output: dict | None = None,
    mode: str = "balanced",
    bpm: float | None = None,
    use_pyin: bool = True,
) -> list[NoteEvent]:
    params = resolve_preset(mode)
    scored = score_notes(
        notes,
        model_output,
        guitar_wav,
        drums_wav=drums_wav,
        bass_wav=bass_wav,
        use_pyin=use_pyin,
    )
    kept = keep_scored(scored, params.keep_threshold)
    if bpm is None:
        return kept
    return thin_charter_notes(kept, min_duration_s=min_duration_for_tempo(bpm))
