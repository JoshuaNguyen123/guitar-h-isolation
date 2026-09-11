"""GuitarSet JAMS parsing does not need the full Zenodo download."""

from __future__ import annotations

import json

from src.pipeline.types import NoteEvent
from tests.eval.guitarset import beats_from_jams, notes_from_jams


def test_notes_and_beats_from_jams(tmp_path):
    payload = {
        "annotations": [
            {
                "namespace": "note_midi",
                "data": [
                    {"time": 0.4, "duration": 0.3, "value": 52.2},
                    {"time": 0.8, "duration": 0.2, "value": 55.0},
                    {"time": 1.2, "duration": 0.1, "value": 20.0},
                ],
            },
            {
                "namespace": "beat",
                "data": [
                    {"time": 0.0, "duration": 0.0, "value": 1},
                    {"time": 0.5, "duration": 0.0, "value": 2},
                ],
            },
        ]
    }
    path = tmp_path / "demo.jams"
    path.write_text(json.dumps(payload), encoding="utf-8")
    notes = notes_from_jams(path)
    assert notes == [
        NoteEvent(0.4, 0.7, 52, 1.0),
        NoteEvent(0.8, 1.0, 55, 1.0),
    ]
    assert beats_from_jams(path) == [0.0, 0.5]
