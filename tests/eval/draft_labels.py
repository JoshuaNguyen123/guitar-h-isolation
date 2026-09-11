"""Draft then thin Basic Pitch notes into charter-like CC clip labels."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.eval.synthesize import FIXTURE_CLIPS, LABEL_DIR

CLIPS = (
    ("electric_lick", FIXTURE_CLIPS / "electric_lick.mp3"),
    ("acoustic_chords", FIXTURE_CLIPS / "acoustic_chords.mp3"),
    ("acoustic_shuffle", FIXTURE_CLIPS / "acoustic_shuffle.mp3"),
)


def _thin_draft(raw_events) -> list[dict]:
    notes: list[dict] = []
    for event in raw_events:
        start_s = float(event[0])
        end_s = float(event[1])
        pitch = int(event[2])
        velocity = float(event[3]) if len(event) > 3 else 1.0
        if end_s - start_s < 0.07:
            continue
        if pitch < 40 or pitch > 88:
            continue
        if velocity < 0.38:
            continue
        notes.append(
            {
                "start_s": round(start_s, 4),
                "end_s": round(end_s, 4),
                "midi_pitch": pitch,
                "velocity": round(velocity, 4),
            }
        )
    notes.sort(key=lambda n: (n["start_s"], n["midi_pitch"]))
    merged: list[dict] = []
    for note in notes:
        if (
            merged
            and note["midi_pitch"] == merged[-1]["midi_pitch"]
            and note["start_s"] - merged[-1]["end_s"] <= 0.05
        ):
            merged[-1]["end_s"] = max(merged[-1]["end_s"], note["end_s"])
            merged[-1]["velocity"] = max(merged[-1]["velocity"], note["velocity"])
            continue
        merged.append(note)
    return merged


def draft_one(clip_id: str, audio_path: Path) -> dict:
    from basic_pitch import FilenameSuffix, build_icassp_2022_model_path
    from basic_pitch.inference import predict

    model_path = build_icassp_2022_model_path(FilenameSuffix.onnx)
    _mo, _midi, raw_events = predict(
        str(audio_path),
        model_or_model_path=model_path,
        onset_threshold=0.5,
        frame_threshold=0.3,
        minimum_frequency=82.0,
        maximum_frequency=1318.5,
    )
    notes = _thin_draft(raw_events)
    return {
        "clip_id": clip_id,
        "source": str(audio_path.relative_to(ROOT)).replace("\\", "/"),
        "curation": (
            "Basic Pitch draft (onset 0.5 / frame 0.3) thinned to duration>=70ms, "
            "velocity>=0.38, guitar MIDI 40-88, merged same-pitch overlaps."
        ),
        "notes": notes,
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
