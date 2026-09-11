"""Draft then thin Basic Pitch notes into charter-like CC clip labels."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.transcribe import filter_note_events, resolve_preset
from tests.eval.synthesize import FIXTURE_CLIPS, LABEL_DIR

CLIPS = (
    ("electric_lick", FIXTURE_CLIPS / "electric_lick.mp3"),
    ("acoustic_chords", FIXTURE_CLIPS / "acoustic_chords.mp3"),
    ("acoustic_shuffle", FIXTURE_CLIPS / "acoustic_shuffle.mp3"),
)


def draft_one(clip_id: str, audio_path: Path) -> dict:
    from basic_pitch import FilenameSuffix, build_icassp_2022_model_path
    from basic_pitch.inference import predict

    params = resolve_preset("balanced")
    model_path = build_icassp_2022_model_path(FilenameSuffix.onnx)
    _mo, _midi, raw_events = predict(
        str(audio_path),
        model_or_model_path=model_path,
        onset_threshold=params.onset_threshold,
        frame_threshold=params.frame_threshold,
        minimum_frequency=82.0,
        maximum_frequency=1318.5,
    )
    notes = filter_note_events(raw_events)
    return {
        "clip_id": clip_id,
        "source": str(audio_path.relative_to(ROOT)).replace("\\", "/"),
        "curation": (
            "Basic Pitch Balanced draft (onset 0.50 / frame 0.30). "
            "Smoke fixture only — not independent human ground truth."
        ),
        "notes": [
            {
                "start_s": round(n.start_s, 4),
                "end_s": round(n.end_s, 4),
                "midi_pitch": n.midi_pitch,
                "velocity": round(n.velocity, 4),
            }
            for n in notes
        ],
    }


def main() -> int:
    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    for clip_id, path in CLIPS:
        if not path.is_file():
            print(f"skip missing {path}")
            continue
        payload = draft_one(clip_id, path)
        out = LABEL_DIR / f"{clip_id}.json"
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"{clip_id}: {len(payload['notes'])} notes -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
