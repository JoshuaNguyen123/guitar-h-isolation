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

from src.pipeline.filters import apply_evidence_filters
from src.pipeline.fretmap import map_difficulties, notes_to_expert
from src.pipeline.stem_clean import bleed_proxy_score, clean_guitar_stem, subtract_drum_bleed
from src.pipeline.transcribe import (
    FRAME_THRESHOLD,
    ONSET_THRESHOLD,
    choose_sensitivity,
    filter_note_events,
    resolve_preset,
    transcribe_guitar,
    transcribe_pre_bc00cd4,
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

MODE_NAMES = ("strict", "balanced", "sensitive")


def _inject_bleed_false_positives(truth: list[NoteEvent]) -> list[NoteEvent]:
    noisy = list(truth)
    for beat in np.arange(0.0, 3.6, 0.5):
        noisy.append(NoteEvent(float(beat), float(beat) + 0.08, 48, 0.22))
        noisy.append(NoteEvent(float(beat) + 0.02, float(beat) + 0.10, 55, 0.55))
    noisy.sort(key=lambda n: (n.start_s, n.midi_pitch))
    return noisy


def _fake_posterior(notes: list[NoteEvent], duration_s: float = 4.2) -> dict:
    n_frames = int(duration_s * 86.0) + 8
    mat = np.zeros((n_frames, 88), dtype=np.float32)
    for note in notes:
        idx = note.midi_pitch - 21
        if not 0 <= idx < 88:
            continue
        lo = int(round((note.start_s + 0.06) * 86.0))
        hi = int(round(min(note.end_s, note.start_s + 0.20) * 86.0))
        lo = max(0, lo)
        hi = min(n_frames, max(lo + 1, hi))
        mat[lo:hi, idx] = 0.85 if (note.end_s - note.start_s) >= 0.20 else 0.04
    return {"note": mat}


def _apply_full_chain(
    notes: list[NoteEvent],
    *,
    guitar_wav: Path,
    drums_wav: Path,
    mode: str = "balanced",
    model_output: dict | None = None,
    bpm: float = 120.0,
) -> list[NoteEvent]:
    return apply_evidence_filters(
        notes,
        guitar_wav,
        drums_wav=drums_wav,
        model_output=model_output if model_output is not None else _fake_posterior(notes),
        mode=mode,
        bpm=bpm,
        use_pyin=False,
    )


def measure_postfilter_and_density(truth: list[NoteEvent]) -> dict:
    noisy = _inject_bleed_false_positives(truth)
    raw_tuples = [(n.start_s, n.end_s, n.midi_pitch, n.velocity) for n in noisy]
    legacy = [
        NoteEvent(n.start_s, n.end_s, n.midi_pitch, n.velocity)
        for n in noisy
        if 40 <= n.midi_pitch <= 88
    ]
    filtered = filter_note_events(raw_tuples)
    guitar_wav = EVAL_DIR / "clean_melody.wav"
    drums_wav = EVAL_DIR / "drums_only.wav"
    full = _apply_full_chain(filtered, guitar_wav=guitar_wav, drums_wav=drums_wav)
    tempo = TempoMap(bpm=120.0)
    duration_s = max(n.end_s for n in truth) + 0.5
    from src.pipeline.fretmap import thin_for_difficulty

    legacy_expert = notes_to_expert(legacy, tempo)
    legacy_hard = thin_for_difficulty(legacy_expert, max_chord=2, min_spacing=tempo.resolution // 4)
    filter_charts = map_difficulties(filtered, tempo)
    full_charts = map_difficulties(full, tempo)
    return {
        "transcription_legacy_vs_truth": asdict(score_transcription(truth, legacy)),
        "transcription_filtered_vs_truth": asdict(score_transcription(truth, filtered)),
        "transcription_full_chain_vs_truth": asdict(score_transcription(truth, full)),
        "density_legacy_expert": asdict(
            score_density(
                legacy_expert, legacy_hard, duration_s=duration_s, tempo_bpm=120.0, resolution=192
            )
        ),
        "density_filter_expert": asdict(
            score_density(
                filter_charts.expert,
                filter_charts.hard,
                duration_s=duration_s,
                tempo_bpm=120.0,
                resolution=192,
            )
        ),
        "density_full_chain_expert": asdict(
            score_density(
                full_charts.expert,
                full_charts.hard,
                duration_s=duration_s,
                tempo_bpm=120.0,
                resolution=192,
            )
        ),
        "density_new_expert": asdict(
            score_density(
                full_charts.expert,
                full_charts.hard,
                duration_s=duration_s,
                tempo_bpm=120.0,
                resolution=192,
            )
        ),
        "legacy_est_notes": len(legacy),
        "filtered_est_notes": len(filtered),
        "full_chain_est_notes": len(full),
        "truth_notes": len(truth),
        "stage_counts_balanced": {
            "after_raw": len(filtered),
            "after_evidence": len(full),
        },
    }


def measure_bleed_cleanup() -> dict:
    guitar = np.load(EVAL_DIR / "guitar_bleed_stem.npy")
    backing = np.load(EVAL_DIR / "backing_drums_stem.npy")
    before = bleed_proxy_score(guitar, backing, SR)
    cleaned = clean_guitar_stem(guitar, SR)
    after_clean = bleed_proxy_score(cleaned, backing, SR)
    subtracted = subtract_drum_bleed(cleaned, backing, SR)
    after_sub = bleed_proxy_score(subtracted, backing, SR)
    return {
        "bleed_mix_before": before,
        "bleed_mix_after_clean": after_clean,
        "bleed_mix_after_subtract": after_sub,
        "bleed_mix_after": after_sub,
        "bleed_reduction_pct": (before - after_sub) / max(before, 1e-9) * 100.0,
        "note": "Helpers remain for diagnostics; shipped Generate uses the raw Demucs stem.",
    }


def measure_basic_pitch_fixtures() -> dict:
    results = {}
    truth = load_ground_truth()
    drums_wav = EVAL_DIR / "drums_only.wav"
    names = (
        "clean_melody.wav",
        "distorted_melody.wav",
        "distorted_bleed_melody.wav",
        "band_mix.wav",
    )
    for name in names:
        path = EVAL_DIR / name
        try:
            shipped = transcribe_guitar(path, sensitivity="balanced")
            estimated = shipped.notes
            if name in {"band_mix.wav", "distorted_bleed_melody.wav"}:
                estimated = apply_evidence_filters(
                    shipped.notes,
                    path,
                    drums_wav=drums_wav,
                    model_output=shipped.model_output,
                    mode="balanced",
                    bpm=120.0,
                )
            scores = score_transcription(truth, estimated)
            legacy = transcribe_pre_bc00cd4(path)
            legacy_scores = score_transcription(truth, legacy)
            results[name] = {
                "ok": True,
                **asdict(scores),
                "legacy": asdict(legacy_scores),
                "thresholds": {"onset": ONSET_THRESHOLD, "frame": FRAME_THRESHOLD},
            }
        except Exception as exc:  # noqa: BLE001
            results[name] = {"ok": False, "error": str(exc)}
    return results


def measure_mode_matrix() -> dict:
    truth = load_ground_truth()
    drums_wav = EVAL_DIR / "drums_only.wav"
    guitar_clean = EVAL_DIR / "clean_melody.wav"
    out: dict = {"fixtures": {}, "bleed_injection": {}, "auto_selection": {}, "presets": {}}
    for mode in MODE_NAMES:
        params = resolve_preset(mode)
        out["presets"][mode] = {
            "onset_threshold": params.onset_threshold,
            "frame_threshold": params.frame_threshold,
            "keep_threshold": params.keep_threshold,
        }
    noisy = _inject_bleed_false_positives(truth)
    for mode in MODE_NAMES:
        full = _apply_full_chain(noisy, guitar_wav=guitar_clean, drums_wav=drums_wav, mode=mode)
        scores = score_transcription(truth, full)
        out["bleed_injection"][mode] = {**asdict(scores), "est_notes": len(full)}
    for name in ("clean_melody.wav", "band_mix.wav"):
        path = EVAL_DIR / name
        out["fixtures"][name] = {}
        for mode in MODE_NAMES:
            try:
                shipped = transcribe_guitar(path, sensitivity=mode)
                estimated = shipped.notes
                if name == "band_mix.wav":
                    estimated = apply_evidence_filters(
                        shipped.notes,
                        path,
                        drums_wav=drums_wav,
                        model_output=shipped.model_output,
                        mode=mode,
                        bpm=120.0,
                    )
                scores = score_transcription(truth, estimated)
                out["fixtures"][name][mode] = {"ok": True, **asdict(scores)}
            except Exception as exc:  # noqa: BLE001
                out["fixtures"][name][mode] = {"ok": False, "error": str(exc)}
    guitar_bleed = np.load(EVAL_DIR / "guitar_bleed_stem.npy")
    drums = np.load(EVAL_DIR / "backing_drums_stem.npy")
    clean = np.load(EVAL_DIR / "guitar_clean_stem.npy")
    quiet = drums * 0.01
    out["auto_selection"] = {
        "bleed_stem": choose_sensitivity("auto", guitar_bleed, drums, SR),
        "clean_stem": choose_sensitivity("auto", clean, quiet, SR),
    }
    return out


def measure_labeled_clips() -> dict:
    results = {}
    for clip_id, audio in LABELED_CLIPS:
        label = LABEL_DIR / f"{clip_id}.json"
        if not audio.is_file() or not label.is_file():
            results[clip_id] = {"ok": False, "error": "missing audio or label"}
            continue
        try:
            truth = load_label_notes(label)
            estimated = transcribe_guitar(audio, sensitivity="balanced").notes
            scores = score_transcription(truth, estimated)
            results[clip_id] = {"ok": True, **asdict(scores), "label_notes": len(truth)}
        except Exception as exc:  # noqa: BLE001
            results[clip_id] = {"ok": False, "error": str(exc)}
    return results


def write_markdown(report: dict) -> str:
    pf = report["postfilter_density"]
    bleed = report["bleed_cleanup"]
    bp = report["basic_pitch"]
    labeled = report["labeled_clips"]
    modes = report.get("mode_matrix", {})
    gs = report.get("guitarset") or {}
    lines = [
        "# Accuracy baselines",
        "",
        f"Generated: `{report['generated_at']}`",
        "",
        "Shipped path: raw Demucs guitar stem → Basic Pitch → evidence score → tempo-relative thin → Expert prune.",
        "The pre-`bc00cd4` reference is stock Basic Pitch 0.5 / 0.3 with no post-filters.",
        "",
        "## Pipeline knobs under test",
        "",
        f"- Balanced Basic Pitch `onset_threshold={ONSET_THRESHOLD}`, `frame_threshold={FRAME_THRESHOLD}`",
        "- Velocity is an evidence input, not a hard gate",
        "- Evidence score: velocity + posterior sustain − drum/bass dominance when sustain is low",
        "- Tempo-relative thinning (half a 16th, keeps octave doubles)",
        "- Expert density prune: 32nd-note minimum spacing",
        "",
        "## 1. Bleed false-positive post-filter + Expert density (synthetic note events)",
        "",
        "| Stage | Precision | Recall | F-measure | Est notes | Expert notes | Expert NPS |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| Legacy (no evidence score, unpruned Expert) | "
            f"{pf['transcription_legacy_vs_truth']['precision']:.3f} | "
            f"{pf['transcription_legacy_vs_truth']['recall']:.3f} | "
            f"{pf['transcription_legacy_vs_truth']['f_measure']:.3f} | "
            f"{pf['legacy_est_notes']} | "
            f"{pf['density_legacy_expert']['expert_notes']} | "
            f"{pf['density_legacy_expert']['expert_nps']:.2f} |"
        ),
        (
            f"| Duration/pitch filter only + Expert prune | "
            f"{pf['transcription_filtered_vs_truth']['precision']:.3f} | "
            f"{pf['transcription_filtered_vs_truth']['recall']:.3f} | "
            f"{pf['transcription_filtered_vs_truth']['f_measure']:.3f} | "
            f"{pf['filtered_est_notes']} | "
            f"{pf['density_filter_expert']['expert_notes']} | "
            f"{pf['density_filter_expert']['expert_nps']:.2f} |"
        ),
        (
            f"| Full chain (evidence score → thin → Expert) | "
            f"{pf['transcription_full_chain_vs_truth']['precision']:.3f} | "
            f"{pf['transcription_full_chain_vs_truth']['recall']:.3f} | "
            f"{pf['transcription_full_chain_vs_truth']['f_measure']:.3f} | "
            f"{pf['full_chain_est_notes']} | "
            f"{pf['density_full_chain_expert']['expert_notes']} | "
            f"{pf['density_full_chain_expert']['expert_nps']:.2f} |"
        ),
        "",
        "## 2. Isolation bleed proxy (diagnostic helpers only — not on Generate)",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Bleed correlation before cleanup | {bleed['bleed_mix_before']:.4f} |",
        f"| After stem cleanup | {bleed['bleed_mix_after_clean']:.4f} |",
        f"| After drums soft-subtract | {bleed['bleed_mix_after_subtract']:.4f} |",
        f"| Total reduction | {bleed['bleed_reduction_pct']:.1f}% |",
        "",
        "## 3. Basic Pitch on synthetic WAVs (Balanced vs pre-bc00cd4)",
        "",
        "| Fixture | Shipped P / R / F (notes) | Pre-bc00cd4 0.5/0.3 P / R / F (notes) |",
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
        "## 4. Sensitivity modes",
        "",
        "| Mode | onset / frame / keep |",
        "| --- | --- |",
    ]
    for mode in MODE_NAMES:
        p = (modes.get("presets") or {}).get(mode, {})
        lines.append(
            f"| `{mode}` | {p.get('onset_threshold')} / {p.get('frame_threshold')} / "
            f"{p.get('keep_threshold')} |"
        )
    lines += [
        "",
        "### Bleed injection (evidence chain)",
        "",
        "| Mode | Precision | Recall | F-measure | Est notes |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for mode in MODE_NAMES:
        row = (modes.get("bleed_injection") or {}).get(mode, {})
        if not row:
            lines.append(f"| `{mode}` | - | - | - | - |")
            continue
        lines.append(
            f"| `{mode}` | {row['precision']:.3f} | {row['recall']:.3f} | "
            f"{row['f_measure']:.3f} | {row['est_notes']} |"
        )
    lines += [
        "",
        "### Synthetic fixtures",
        "",
        "| Fixture | Mode | Precision | Recall | F-measure | Est notes |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for name, by_mode in (modes.get("fixtures") or {}).items():
        for mode in MODE_NAMES:
            row = by_mode.get(mode, {})
            if not row.get("ok"):
                lines.append(f"| `{name}` | `{mode}` | error | - | - | - |")
                continue
            lines.append(
                f"| `{name}` | `{mode}` | {row['precision']:.3f} | {row['recall']:.3f} | "
                f"{row['f_measure']:.3f} | {row['est_notes']} |"
            )
    auto = modes.get("auto_selection") or {}
    lines += [
        "",
        "### Auto selection",
        "",
        f"- High-bleed stem -> `{auto.get('bleed_stem', '?')}`",
        f"- Clean stem -> `{auto.get('clean_stem', '?')}`",
        "",
        "## 5. Labeled Creative Commons clips (smoke only)",
        "",
        "Not independent human GT. Real-audio accuracy is GuitarSet (`python -m tests.eval.run_guitarset`).",
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
    if gs:
        lines += [
            "",
            "## 6. GuitarSet (independent labels)",
            "",
            f"Tracks: {gs.get('n_tracks', 0)}. "
            f"Solo shipped F `{gs.get('mean_solo_f_shipped', 0):.3f}` vs pre-bc00cd4 `{gs.get('mean_solo_f_reference', 0):.3f}`. "
            f"Band shipped F `{gs.get('mean_band_f_shipped', 0):.3f}` vs pre-bc00cd4 `{gs.get('mean_band_f_reference', 0):.3f}`. "
            + (
                f"Band Strict P `{gs.get('mean_band_precision_strict', 0):.3f}` vs "
                f"Balanced P `{gs.get('mean_band_precision_balanced', 0):.3f}`."
                if gs.get("mean_band_precision_strict")
                else ""
            ),
            "",
        ]
    lines += [
        "",
        "## Regression floors",
        "",
        "- Evidence score keeps sustained on-beat guitar and drops short broadband ghosts",
        "- Chord voices are not pyin-vetoed",
        "- Strict keep bar > Balanced > Sensitive",
        "- Auto selects Strict on high-bleed stems (no crest rule)",
        "- Charter thin merges same-pitch overlaps and keeps octave doubles",
        "- Generate transcribes the raw Demucs stem",
        "",
        "## Notes",
        "",
        "- Synthetic fixtures live in `tests/fixtures/eval/` (CC0).",
        "- mir-eval onset tolerance is 50 ms; pitch tolerance 50 cents; offsets ignored.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    write_eval_fixtures()
    truth = load_ground_truth()
    guitarset = {}
    gs_path = Path(__file__).resolve().parent / "guitarset_baseline.json"
    if gs_path.is_file():
        try:
            guitarset = json.loads(gs_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            guitarset = {}
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "postfilter_density": measure_postfilter_and_density(truth),
        "bleed_cleanup": measure_bleed_cleanup(),
        "basic_pitch": measure_basic_pitch_fixtures(),
        "mode_matrix": measure_mode_matrix(),
        "labeled_clips": measure_labeled_clips(),
        "guitarset": guitarset,
    }
    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = write_markdown(report)
    OUT_MD.write_text(md, encoding="utf-8")
    try:
        print(md)
    except UnicodeEncodeError:
        print(md.encode("ascii", errors="replace").decode("ascii"))
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
