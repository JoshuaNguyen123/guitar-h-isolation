from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.pipeline.types import NoteEvent

# Typical guitar range: E2–E6
MIN_GUITAR_HZ = 82.0
MAX_GUITAR_HZ = 1318.5
MIN_NOTE_DURATION_S = 0.04


@dataclass(frozen=True)
class SensitivityParams:
    name: str
    onset_threshold: float
    frame_threshold: float
    min_velocity: float


SENSITIVITY_PRESETS: dict[str, SensitivityParams] = {
    "strict": SensitivityParams("strict", 0.64, 0.42, 0.42),
    "balanced": SensitivityParams("balanced", 0.58, 0.38, 0.35),
    "sensitive": SensitivityParams("sensitive", 0.50, 0.30, 0.25),
}

# Back-compat names = balanced preset (see tests/eval/BASELINE.md).
ONSET_THRESHOLD = SENSITIVITY_PRESETS["balanced"].onset_threshold
FRAME_THRESHOLD = SENSITIVITY_PRESETS["balanced"].frame_threshold
MIN_NOTE_VELOCITY = SENSITIVITY_PRESETS["balanced"].min_velocity

BLEED_STRICT = 0.10
BLEED_SENSITIVE = 0.035
CREST_STRICT = 3.5
CREST_SENSITIVE = 7.5


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
    """Return strict / balanced / sensitive. Manual modes win; auto uses bleed + crest."""
    key = (mode or "auto").strip().lower()
    if key in SENSITIVITY_PRESETS:
        return key
    if guitar is None or backing is None or sample_rate <= 0:
        return "balanced"
    from src.pipeline.stem_clean import _to_channels_first, bleed_proxy_score

    bleed = bleed_proxy_score(guitar, backing, sample_rate)
    mono = np.mean(_to_channels_first(guitar), axis=0)
    peak = float(np.max(np.abs(mono))) + 1e-8
    rms = float(np.sqrt(np.mean(mono**2))) + 1e-8
    crest = peak / rms
    if bleed >= BLEED_STRICT or crest < CREST_STRICT:
        return "strict"
    if bleed <= BLEED_SENSITIVE and crest >= CREST_SENSITIVE:
        return "sensitive"
    return "balanced"


def filter_note_events(
    raw_events,
    *,
    min_velocity: float = MIN_NOTE_VELOCITY,
) -> list[NoteEvent]:
    """Convert Basic Pitch tuples into NoteEvents with bleed-oriented filtering."""
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
) -> list[NoteEvent]:
    from basic_pitch import FilenameSuffix, build_icassp_2022_model_path
    from basic_pitch.inference import predict

    params = resolve_preset(sensitivity)
    # Prefer ONNX even if TensorFlow is installed. TF is the default on
    # Python 3.11 and makes Transcribe look hung on long songs.
    model_path = build_icassp_2022_model_path(FilenameSuffix.onnx)
    _model_output, _midi, raw_events = predict(
        str(guitar_wav),
        model_or_model_path=model_path,
        onset_threshold=params.onset_threshold,
        frame_threshold=params.frame_threshold,
        minimum_frequency=MIN_GUITAR_HZ,
        maximum_frequency=MAX_GUITAR_HZ,
    )
    notes = filter_note_events(raw_events, min_velocity=params.min_velocity)
    if not notes:
        raise RuntimeError(
            "No guitar notes were detected. Try a clearer guitar recording, "
            "or a mix where the guitar is more present."
        )
    return notes
