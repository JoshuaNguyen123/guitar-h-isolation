"""Synthetic guitar fixtures with known MIDI ground truth.

Generates short CC0 synthetic WAVs: clean plucks, distorted plucks, and
distorted plucks with drum-like bleed. Used by the accuracy harness.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from src.pipeline.types import NoteEvent


SR = 44100
EVAL_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "eval"
LABEL_DIR = EVAL_DIR / "labels"
FIXTURE_CLIPS = Path(__file__).resolve().parents[1] / "fixtures"


@dataclass(frozen=True)
class SynthNote:
    start_s: float
    duration_s: float
    midi_pitch: int
    velocity: float = 0.85


# Sparse major-ish lick (~4 s) — easy for humans and Basic Pitch.
MELODY: list[SynthNote] = [
    SynthNote(0.40, 0.35, 52),  # E3
    SynthNote(0.85, 0.30, 55),  # G3
    SynthNote(1.25, 0.30, 57),  # A3
    SynthNote(1.70, 0.40, 59),  # B3
    SynthNote(2.25, 0.35, 60),  # C4
    SynthNote(2.70, 0.30, 64),  # E4
    SynthNote(3.15, 0.45, 67),  # G4
]


def midi_to_hz(midi: int) -> float:
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


def _pluck(freq: float, duration_s: float, sr: int, velocity: float) -> np.ndarray:
    n = int(duration_s * sr)
    t = np.arange(n, dtype=np.float32) / sr
    # Harmonic stack with exponential decay — closer to a guitar than a sine.
    wave = np.zeros(n, dtype=np.float32)
    for harm, amp in ((1, 1.0), (2, 0.45), (3, 0.22), (4, 0.12), (5, 0.06)):
        wave += (amp * np.sin(2 * np.pi * freq * harm * t)).astype(np.float32)
    env = np.exp(-t * 3.8).astype(np.float32)
    attack = np.minimum(t / 0.008, 1.0).astype(np.float32)
    return (wave * env * attack * velocity * 0.22).astype(np.float32)


def _distort(mono: np.ndarray, drive: float = 8.0) -> np.ndarray:
    driven = np.tanh(mono * drive).astype(np.float32)
    peak = float(np.max(np.abs(driven))) + 1e-8
    return (driven / peak * 0.85).astype(np.float32)


def _bass_hits(n_samples: int, sr: int, period_s: float = 1.0) -> np.ndarray:
    """Low E/A pulses sitting under the guitar — a mixed-band fixture."""
    out = np.zeros(n_samples, dtype=np.float32)
    for i, start in enumerate(range(0, n_samples, int(period_s * sr))):
        midi = 40 if i % 2 == 0 else 45
        tone = _pluck(midi_to_hz(midi), 0.42, sr, 0.7)
        end = min(start + tone.size, n_samples)
        if start >= n_samples:
            break
        out[start:end] += tone[: end - start] * 0.85
    peak = float(np.max(np.abs(out))) + 1e-8
    return (out / peak * 0.7).astype(np.float32)


def _drum_hits(n_samples: int, sr: int, period_s: float = 0.5) -> np.ndarray:
    """Broadband kick/snare-ish impulses — the “beat bleed” failure mode."""
    out = np.zeros(n_samples, dtype=np.float32)
    t = np.arange(int(0.08 * sr), dtype=np.float32) / sr
    kick = (np.sin(2 * np.pi * 70 * t) * np.exp(-t * 28)).astype(np.float32)
    snare = (np.random.default_rng(0).standard_normal(t.size).astype(np.float32) * np.exp(-t * 40))
    period = int(period_s * sr)
    for i, start in enumerate(range(0, n_samples - t.size, period)):
        hit = kick if i % 2 == 0 else snare * 0.7
        out[start : start + t.size] += hit * 0.9
    return out


def render_guitar(notes: list[SynthNote], *, distorted: bool, sr: int = SR) -> np.ndarray:
    end_s = max(n.start_s + n.duration_s for n in notes) + 0.6
    n = int(end_s * sr)
    mono = np.zeros(n, dtype=np.float32)
    for note in notes:
        start = int(note.start_s * sr)
        tone = _pluck(midi_to_hz(note.midi_pitch), note.duration_s, sr, note.velocity)
        end = min(start + tone.size, n)
        mono[start:end] += tone[: end - start]
    if distorted:
        mono = _distort(mono)
    peak = float(np.max(np.abs(mono))) + 1e-8
    return (mono / peak * 0.9).astype(np.float32)


def ground_truth_events(notes: list[SynthNote] | None = None) -> list[NoteEvent]:
    notes = notes or MELODY
    return [
        NoteEvent(
            start_s=n.start_s,
            end_s=n.start_s + n.duration_s,
            midi_pitch=n.midi_pitch,
            velocity=n.velocity,
        )
        for n in notes
    ]


def write_eval_fixtures(out_dir: Path | None = None) -> dict[str, Path]:
    out_dir = Path(out_dir or EVAL_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    clean = render_guitar(MELODY, distorted=False)
    dist = render_guitar(MELODY, distorted=True)
    bleed = dist.copy()
    drums = _drum_hits(bleed.size, SR, period_s=0.5)
    # Stronger bleed: models often hear kick/snare as pitched notes on distorted stems.
    bleed = bleed + drums * 0.95
    peak = float(np.max(np.abs(bleed))) + 1e-8
    bleed = (bleed / peak * 0.9).astype(np.float32)

    # Stereo stems shaped like Demucs output for bleed-proxy tests.
    backing = np.stack([drums, drums], axis=0)
    guitar_bleed = np.stack([bleed, bleed], axis=0)
    guitar_clean = np.stack([clean, clean], axis=0)

    specs = {
        "clean_melody.wav": clean,
        "distorted_melody.wav": dist,
        "distorted_bleed_melody.wav": bleed,
    }
    for name, mono in specs.items():
        path = out_dir / name
        sf.write(str(path), mono, SR)
        paths[name] = path

    bass = _bass_hits(dist.size, SR, period_s=1.0)
    n_band = min(dist.size, drums.size, bass.size)
    band = dist[:n_band] + drums[:n_band] * 0.65 + bass[:n_band] * 0.42
    peak = float(np.max(np.abs(band))) + 1e-8
    band = (band / peak * 0.9).astype(np.float32)

    np.save(out_dir / "guitar_bleed_stem.npy", guitar_bleed)
    np.save(out_dir / "guitar_clean_stem.npy", guitar_clean)
    np.save(out_dir / "backing_drums_stem.npy", backing)
    paths["guitar_bleed_stem.npy"] = out_dir / "guitar_bleed_stem.npy"
    paths["backing_drums_stem.npy"] = out_dir / "backing_drums_stem.npy"

    extras = {
        "band_mix.wav": band,
        "drums_only.wav": drums,
        "bass_only.wav": bass,
    }
    for name, mono in extras.items():
        path = out_dir / name
        sf.write(str(path), mono, SR)
        paths[name] = path

    gt = {
        "sample_rate": SR,
        "notes": [asdict(n) for n in MELODY],
        "description": (
            "Synthetic 7-note guitar lick. clean_melody is dry harmonics; "
            "distorted_melody is tanh overdrive; distorted_bleed_melody adds "
            "kick/snare impulses every 0.5 s (beat bleed). band_mix.wav adds "
            "drums + bass under the same distorted lick."
        ),
        "license": "CC0 — generated by Guitar H Isolation test suite",
    }
    gt_path = out_dir / "ground_truth.json"
    gt_path.write_text(json.dumps(gt, indent=2), encoding="utf-8")
    paths["ground_truth.json"] = gt_path
    return paths


def load_ground_truth(path: Path | None = None) -> list[NoteEvent]:
    path = path or (EVAL_DIR / "ground_truth.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        NoteEvent(
            start_s=float(n["start_s"]),
            end_s=float(n["start_s"]) + float(n["duration_s"]),
            midi_pitch=int(n["midi_pitch"]),
            velocity=float(n.get("velocity", 1.0)),
        )
        for n in data["notes"]
    ]


def note_dict_to_event(n: dict) -> NoteEvent:
    start = float(n.get("start_s", n.get("start", 0.0)))
    if "end_s" in n:
        end = float(n["end_s"])
    else:
        end = start + float(n.get("duration_s", 0.1))
    return NoteEvent(
        start_s=start,
        end_s=end,
        midi_pitch=int(n["midi_pitch"]),
        velocity=float(n.get("velocity", 1.0)),
    )


def load_label_notes(path: Path) -> list[NoteEvent]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    notes = [note_dict_to_event(n) for n in data.get("notes", [])]
    notes.sort(key=lambda n: (n.start_s, n.midi_pitch))
    return notes


if __name__ == "__main__":
    written = write_eval_fixtures()
    for name, path in written.items():
        print(f"{name}: {path}")
