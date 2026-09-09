from __future__ import annotations

from pathlib import Path

from src.pipeline.types import NoteEvent

# Typical guitar range: E2–E6
MIN_GUITAR_HZ = 82.0
MAX_GUITAR_HZ = 1318.5


def transcribe_guitar(guitar_wav: Path) -> list[NoteEvent]:
    from basic_pitch import FilenameSuffix, build_icassp_2022_model_path
    from basic_pitch.inference import predict

    # Prefer ONNX even if TensorFlow is installed. TF is the default on
    # Python 3.11 and makes Transcribe look hung on long songs.
    model_path = build_icassp_2022_model_path(FilenameSuffix.onnx)
    _model_output, _midi, raw_events = predict(
        str(guitar_wav),
        model_or_model_path=model_path,
        onset_threshold=0.5,
        frame_threshold=0.3,
        minimum_frequency=MIN_GUITAR_HZ,
        maximum_frequency=MAX_GUITAR_HZ,
    )
    notes: list[NoteEvent] = []
    for event in raw_events:
        start_s = float(event[0])
        end_s = float(event[1])
        pitch = int(event[2])
        velocity = float(event[3]) if len(event) > 3 else 1.0
        if end_s <= start_s:
            continue
        if pitch < 40 or pitch > 88:
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
    if not notes:
        raise RuntimeError(
            "No guitar notes were detected. Try a clearer guitar recording, "
            "or a mix where the guitar is more present."
        )
    return notes
