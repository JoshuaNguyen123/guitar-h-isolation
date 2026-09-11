from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.pipeline.types import NoteEvent, TranscriptionResult

# Typical guitar range: E2–E6
MIN_GUITAR_HZ = 82.0
MAX_GUITAR_HZ = 1318.5
MIN_NOTE_DURATION_S = 0.04


@dataclass(frozen=True)
class SensitivityParams:
    """Basic Pitch thresholds and the evidence-score keep bar for one mode."""

    name: str
    onset_threshold: float
    frame_threshold: float
    keep_threshold: float


# Plan Phase 1: Balanced is stock Basic Pitch. Modes only move thresholds.
SENSITIVITY_PRESETS: dict[str, SensitivityParams] = {
    "strict": SensitivityParams("strict", 0.55, 0.35, 0.20),
    "balanced": SensitivityParams("balanced", 0.50, 0.30, 0.10),
    "sensitive": SensitivityParams("sensitive", 0.40, 0.25, 0.04),
}

ONSET_THRESHOLD = SENSITIVITY_PRESETS["balanced"].onset_threshold
FRAME_THRESHOLD = SENSITIVITY_PRESETS["balanced"].frame_threshold
KEEP_THRESHOLD = SENSITIVITY_PRESETS["balanced"].keep_threshold
# Older tests used this name for the Balanced velocity floor. Velocity is no
# longer a hard gate; keep the alias so docs/baselines can still print a number.
MIN_NOTE_VELOCITY = 0.0

BLEED_STRICT = 0.10
BLEED_SENSITIVE = 0.035


def resolve_preset(name: str | SensitivityParams | None) -> SensitivityParams:
    if isinstance(name, SensitivityParams):
        return name
    key = (name or "balanced").strip().lower()
    if key == "auto":
        key = "balanced"
    return SENSITIVITY_PRESETS.get(key, SENSITIVITY_PRESETS["balanced"])


def choose_sensitivity(
    mode: str,
    guitar: np.ndarray | None = None,
    backing: np.ndarray | None = None,
    sample_rate: int = 44100,
) -> str:
    """Manual modes win. Auto is Balanced unless isolation bleed is extreme."""
    key = (mode or "auto").strip().lower()
    if key in SENSITIVITY_PRESETS:
        return key
    if guitar is None or backing is None or sample_rate <= 0:
        return "balanced"
    from src.pipeline.stem_clean import bleed_proxy_score

    bleed = bleed_proxy_score(guitar, backing, sample_rate)
    if bleed >= BLEED_STRICT:
        return "strict"
    if bleed <= BLEED_SENSITIVE:
        return "sensitive"
    return "balanced"


def filter_note_events(raw_events, *, min_velocity: float = 0.0) -> list[NoteEvent]:
    """Convert Basic Pitch tuples. Velocity is not a hard keep/drop gate."""
    notes: list[NoteEvent] = []
    for event in raw_events:
        start_s = float(event[0])
        end_s = float(event[1])
        pitch = int(event[2])
        velocity = float(event[3]) if len(event) > 3 else 1.0
        if end_s - start_s < MIN_NOTE_DURATION_S:
            continue
        if pitch < 40 or pitch > 88:
            continue
        if velocity < min_velocity:
            continue
        notes.append(
            NoteEvent(
                start_s=start_s,
                end_s=end_s,
                midi_pitch=pitch,
                velocity=velocity,
            )
        )
    notes.sort(key=lambda n: (n.start_s, n.midi_pitch))
    return notes


def transcribe_guitar(
    guitar_wav: Path,
    sensitivity: str | SensitivityParams = "balanced",
) -> TranscriptionResult:
    from basic_pitch import FilenameSuffix, build_icassp_2022_model_path
    from basic_pitch.inference import predict

    params = resolve_preset(sensitivity)
    model_path = build_icassp_2022_model_path(FilenameSuffix.onnx)
    model_output, _midi, raw_events = predict(
        str(guitar_wav),
        model_or_model_path=model_path,
        onset_threshold=params.onset_threshold,
        frame_threshold=params.frame_threshold,
        minimum_frequency=MIN_GUITAR_HZ,
        maximum_frequency=MAX_GUITAR_HZ,
    )
    notes = filter_note_events(raw_events)
    if not notes:
        raise RuntimeError(
            "No guitar notes were detected. Try a clearer guitar recording, "
            "or a mix where the guitar is more present."
        )
    return TranscriptionResult(notes=notes, model_output=model_output)


def transcribe_pre_bc00cd4(guitar_wav: Path) -> list[NoteEvent]:
    """Pre-regression reference: stock 0.5/0.3, no evidence score, no thinning."""
    result = transcribe_guitar(guitar_wav, sensitivity="balanced")
    return result.notes
