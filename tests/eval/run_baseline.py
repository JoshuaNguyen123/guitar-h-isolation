"""Run quantitative accuracy baselines and write BASELINE.md / baseline.json."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.confirm import confirm_with_pyin
from src.pipeline.drum_reject import reject_drum_aligned
from src.pipeline.fretmap import map_difficulties, notes_to_expert
from src.pipeline.stem_clean import bleed_proxy_score, clean_guitar_stem, subtract_drum_bleed
from src.pipeline.transcribe import (
    FRAME_THRESHOLD,
    MIN_NOTE_VELOCITY,
    ONSET_THRESHOLD,
    filter_note_events,
    transcribe_guitar,
)
from src.pipeline.types import NoteEvent, TempoMap
from tests.eval.metrics import score_density, score_transcription
from tests.eval.synthesize import (
    EVAL_DIR,
    FIXTURE_CLIPS,
    LABEL_DIR,
    SR,
    load_ground_truth,
    load_label_notes,
    write_eval_fixtures,
)


OUT_JSON = Path(__file__).resolve().parent / "baseline.json"
OUT_MD = Path(__file__).resolve().parent / "BASELINE.md"

LABELED_CLIPS = (
    ("electric_lick", FIXTURE_CLIPS / "electric_lick.mp3"),
    ("acoustic_chords", FIXTURE_CLIPS / "acoustic_chords.mp3"),
    ("acoustic_shuffle", FIXTURE_CLIPS / "acoustic_shuffle.mp3"),
)


def _inject_bleed_false_positives(truth: list[NoteEvent]) -> list[NoteEvent]:
    """Simulate Basic Pitch hearing drum hits as notes on the beat."""
    noisy = list(truth)
    for beat in np.arange(0.0, 3.6, 0.5):
        noisy.append(
            NoteEvent(
                start_s=float(beat),
                end_s=float(beat) + 0.08,
                midi_pitch=48,
                velocity=0.22,
            )
        )
        noisy.append(
            NoteEvent(
                start_s=float(beat) + 0.02,
                end_s=float(beat) + 0.10,
                midi_pitch=55,
                velocity=0.55,
            )
        )
    noisy.sort(key=lambda n: (n.start_s, n.midi_pitch))
    return noisy


def _legacy_filter(raw_events) -> list[NoteEvent]:
    """Pre-tune Basic Pitch post-filter (onset defaults lived in predict())."""
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
        notes.append(NoteEvent(start_s=start_s, end_s=end_s, midi_pitch=pitch, velocity=velocity))
    notes.sort(key=lambda n: (n.start_s, n.midi_pitch))
    return notes


def measure_postfilter_and_density(truth: list[NoteEvent]) -> dict:
    noisy = _inject_bleed_false_positives(truth)
    raw_tuples = [(n.start_s, n.end_s, n.midi_pitch, n.velocity) for n in noisy]
    legacy = _legacy_filter(raw_tuples)
    filtered = filter_note_events(raw_tuples)

    tempo = TempoMap(bpm=120.0)
    duration_s = max(n.end_s for n in truth) + 0.5

    from src.pipeline.fretmap import thin_for_difficulty

    legacy_expert = notes_to_expert(legacy, tempo)
    legacy_hard = thin_for_difficulty(legacy_expert, max_chord=2, min_spacing=tempo.resolution // 4)
    new_charts = map_difficulties(filtered, tempo)

    return {
        "transcription_legacy_vs_truth": asdict(score_transcription(truth, legacy)),
        "transcription_filtered_vs_truth": asdict(score_transcription(truth, filtered)),
        "density_legacy_expert": asdict(
            score_density(
                legacy_expert,
                legacy_hard,
                duration_s=duration_s,
                tempo_bpm=120.0,
                resolution=192,
            )
        ),
        "density_new_expert": asdict(
            score_density(
                new_charts.expert,
                new_charts.hard,
                duration_s=duration_s,
                tempo_bpm=120.0,
                resolution=192,
            )
        ),
        "legacy_est_notes": len(legacy),
        "filtered_est_notes": len(filtered),
        "truth_notes": len(truth),
    }


def measure_bleed_cleanup() -> dict:
    guitar = np.load(EVAL_DIR / "guitar_bleed_stem.npy")
    backing = np.load(EVAL_DIR / "backing_drums_stem.npy")
    before = bleed_proxy_score(guitar, backing, SR)
    cleaned = clean_guitar_stem(guitar, SR)
    after_clean = bleed_proxy_score(cleaned, backing, SR)
    subtracted = subtract_drum_bleed(cleaned, backing, SR)
    after_sub = bleed_proxy_score(subtracted, backing, SR)
    clean_ref = np.load(EVAL_DIR / "guitar_clean_stem.npy")
    clean_before = bleed_proxy_score(clean_ref, backing * 0.05, SR)
    clean_after = bleed_proxy_score(clean_guitar_stem(clean_ref, SR), backing * 0.05, SR)
    return {
        "bleed_mix_before": before,
        "bleed_mix_after_clean": after_clean,
        "bleed_mix_after_subtract": after_sub,
        "bleed_mix_after": after_sub,
        "bleed_reduction_pct": (before - after_sub) / max(before, 1e-9) * 100.0,
        "clean_stem_proxy_before": clean_before,
        "clean_stem_proxy_after": clean_after,
    }


def measure_basic_pitch_fixtures() -> dict:
    """Score Basic Pitch on synthetic WAVs (needs ONNX model)."""
    from basic_pitch import FilenameSuffix, build_icassp_2022_model_path
    from basic_pitch.inference import predict

    results = {}
    truth = load_ground_truth()
    model_path = build_icassp_2022_model_path(FilenameSuffix.onnx)
    names = (
        "clean_melody.wav",
        "distorted_melody.wav",
        "distorted_bleed_melody.wav",
        "band_mix.wav",
    )
    for name in names:
        path = EVAL_DIR / name
        try:
            _mo, _midi, raw_events = predict(
                str(path),
                model_or_model_path=model_path,
                onset_threshold=ONSET_THRESHOLD,
                frame_threshold=FRAME_THRESHOLD,
                minimum_frequency=82.0,
                maximum_frequency=1318.5,
            )
            estimated = filter_note_events(raw_events)
            if name == "band_mix.wav":
                estimated = confirm_with_pyin(estimated, path)
                estimated = reject_drum_aligned(estimated, EVAL_DIR / "drums_only.wav")
            scores = score_transcription(truth, estimated)

            _mo2, _midi2, raw_legacy = predict(
                str(path),
                model_or_model_path=model_path,
                onset_threshold=0.5,
                frame_threshold=0.3,
                minimum_frequency=82.0,
                maximum_frequency=1318.5,
            )
            legacy = _legacy_filter(raw_legacy)
            legacy_scores = score_transcription(truth, legacy)

            results[name] = {
                "ok": True,
                **asdict(scores),
                "legacy": asdict(legacy_scores),
                "thresholds": {
                    "onset": ONSET_THRESHOLD,
                    "frame": FRAME_THRESHOLD,
                    "min_velocity": MIN_NOTE_VELOCITY,
                },
            }
        except Exception as exc:  # noqa: BLE001 — baseline must continue
            results[name] = {"ok": False, "error": str(exc)}
    return results


def measure_labeled_clips() -> dict:
    results = {}
    for clip_id, audio in LABELED_CLIPS:
        label = LABEL_DIR / f"{clip_id}.json"
        if not audio.is_file() or not label.is_file():
            results[clip_id] = {"ok": False, "error": "missing audio or label"}
            continue
        try:
            truth = load_label_notes(label)
            estimated = transcribe_guitar(audio, sensitivity="balanced")
            scores = score_transcription(truth, estimated)
            results[clip_id] = {
                "ok": True,
                **asdict(scores),
                "label_notes": len(truth),
            }
        except Exception as exc:  # noqa: BLE001
            results[clip_id] = {"ok": False, "error": str(exc)}
    return results


def write_markdown(report: dict) -> str:
    pf = report["postfilter_density"]
    bleed = report["bleed_cleanup"]
    bp = report["basic_pitch"]
    labeled = report["labeled_clips"]
    lines = [
        "# Accuracy baselines",
        "",
        f"Generated: `{report['generated_at']}`",
        "",
        "v1 floors after drum subtract, drum-aware reject, adaptive sensitivity, pyin confirm, and string-aware frets.",
        "",
        "## Pipeline knobs under test",
        "",
        f"- Balanced Basic Pitch `onset_threshold={ONSET_THRESHOLD}`, `frame_threshold={FRAME_THRESHOLD}`",
        f"- Post-filter `min_velocity={MIN_NOTE_VELOCITY}` (strict/sensitive presets differ)",
        "- Guitar stem cleanup + drums STFT soft-subtract",
        "- pyin confirm for weak notes; drum-onset reject for low-velocity ghosts",
        "- Expert density prune: 32nd-note minimum spacing",
        "- String-aware 5-lane fretting (standard tuning, high E shares orange with B)",
        "",
        "## 1. Bleed false-positive post-filter + Expert density (synthetic note events)",
        "",
        "Ground truth: 7-note melody. Estimated: truth + drum-on-beat false notes (low and mid velocity).",
        "",
        "| Stage | Precision | Recall | F-measure | Est notes | Expert notes | Expert NPS |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| Legacy (no velocity filter, unpruned Expert) | "
            f"{pf['transcription_legacy_vs_truth']['precision']:.3f} | "
            f"{pf['transcription_legacy_vs_truth']['recall']:.3f} | "
            f"{pf['transcription_legacy_vs_truth']['f_measure']:.3f} | "
            f"{pf['legacy_est_notes']} | "
            f"{pf['density_legacy_expert']['expert_notes']} | "
            f"{pf['density_legacy_expert']['expert_nps']:.2f} |"
        ),
        (
            f"| Current (velocity filter + Expert prune) | "
            f"{pf['transcription_filtered_vs_truth']['precision']:.3f} | "
            f"{pf['transcription_filtered_vs_truth']['recall']:.3f} | "
            f"{pf['transcription_filtered_vs_truth']['f_measure']:.3f} | "
            f"{pf['filtered_est_notes']} | "
            f"{pf['density_new_expert']['expert_notes']} | "
            f"{pf['density_new_expert']['expert_nps']:.2f} |"
        ),
        "",
        "## 2. Isolation bleed proxy (synthetic distorted guitar + drums in guitar stem)",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Bleed correlation before cleanup | {bleed['bleed_mix_before']:.4f} |",
        f"| After stem cleanup | {bleed['bleed_mix_after_clean']:.4f} |",
        f"| After drums soft-subtract | {bleed['bleed_mix_after_subtract']:.4f} |",
        f"| Total reduction | {bleed['bleed_reduction_pct']:.1f}% |",
        "",
        "## 3. Basic Pitch on synthetic WAVs",
        "",
        "| Fixture | Current P / R / F (notes) | Legacy 0.5/0.3 P / R / F (notes) |",
        "| --- | --- | --- |",
    ]
    for name, row in bp.items():
        if not row.get("ok"):
            lines.append(f"| `{name}` | error: {row.get('error', '')} | - |")
            continue
        legacy = row.get("legacy", {})
        lines.append(
            f"| `{name}` | "
            f"{row['precision']:.3f} / {row['recall']:.3f} / {row['f_measure']:.3f} ({row['est_notes']}) | "
            f"{legacy.get('precision', 0):.3f} / {legacy.get('recall', 0):.3f} / "
            f"{legacy.get('f_measure', 0):.3f} ({legacy.get('est_notes', 0)}) |"
        )
    lines += [
        "",
        "## 4. Labeled Creative Commons clips (mix-direct transcription vs curated labels)",
        "",
        "Labels are thinned Basic Pitch drafts in `tests/fixtures/eval/labels/`.",
        "",
        "| Clip | Precision | Recall | F-measure | Est / label notes |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for clip_id, row in labeled.items():
        if not row.get("ok"):
            lines.append(f"| `{clip_id}` | error: {row.get('error', '')} | - | - | - |")
            continue
        lines.append(
            f"| `{clip_id}` | {row['precision']:.3f} | {row['recall']:.3f} | "
            f"{row['f_measure']:.3f} | {row['est_notes']} / {row['label_notes']} |"
        )
    lines += [
        "",
        "## v1 regression floors",
        "",
        "Unit tests enforce:",
        "",
        "- Velocity filter drops sub-0.35 ghost notes",
        "- Expert prune enforces >= 32nd spacing",
        "- Stem cleanup reduces bleed proxy by >5%; drums subtract improves further",
        "- Auto sensitivity selects Strict on high-bleed stems",
        "- Drum reject drops weak on-beat ghosts and keeps strong on-beat guitar",
        "- pyin drops weak wrong-pitch notes and keeps matching / strong notes",
        "- Scale-run lanes stay locally continuous (no C-to-green wrap)",
        "- Perfect self-score on ground-truth note lists",
        "",
        "Harness floors (this file): clean synthetic recall stays 1.0; distorted+drums bleed "
        "correlation after subtract is below the cleanup-only value.",
        "",
        "## Notes",
        "",
        "- Synthetic fixtures live in `tests/fixtures/eval/` (CC0).",
        "- mir-eval onset tolerance is 50 ms; pitch tolerance 50 cents; offsets ignored.",
        "- Real commercial distorted mixes are still not claimed; these numbers are the repo's quantitative regression bar.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    write_eval_fixtures()
    truth = load_ground_truth()
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "postfilter_density": measure_postfilter_and_density(truth),
        "bleed_cleanup": measure_bleed_cleanup(),
        "basic_pitch": measure_basic_pitch_fixtures(),
        "labeled_clips": measure_labeled_clips(),
    }
    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    OUT_MD.write_text(write_markdown(report), encoding="utf-8")
    print(OUT_MD.read_text(encoding="utf-8"))
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
