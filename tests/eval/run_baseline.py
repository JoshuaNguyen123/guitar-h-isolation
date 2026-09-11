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
from src.pipeline.drum_reject import detect_drum_onsets, reject_drum_aligned
from src.pipeline.fretmap import map_difficulties, notes_to_expert
from src.pipeline.refine import reject_bass_aligned, thin_charter_notes
from src.pipeline.stem_clean import bleed_proxy_score, clean_guitar_stem, subtract_drum_bleed
from src.pipeline.transcribe import (
    FRAME_THRESHOLD,
    MIN_NOTE_VELOCITY,
    ONSET_THRESHOLD,
    choose_sensitivity,
    filter_note_events,
    resolve_preset,
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

MODE_NAMES = ("strict", "balanced", "sensitive")


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


def _inject_mode_bleed_false_positives(truth: list[NoteEvent]) -> list[NoteEvent]:
    """Bleed ghosts including near-strong on-beat notes that separate modes."""
    noisy = _inject_bleed_false_positives(truth)
    for beat in np.arange(0.0, 3.6, 0.5):
        noisy.append(
            NoteEvent(
                start_s=float(beat) + 0.01,
                end_s=float(beat) + 0.09,
                midi_pitch=50,
                velocity=0.71,
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


def _apply_full_chain(
    notes: list[NoteEvent],
    *,
    guitar_wav: Path,
    drums_wav: Path,
    mode: str = "balanced",
    bass_wav: Path | None = None,
) -> list[NoteEvent]:
    """Match shipped path: velocity already applied -> pyin -> drum/bass reject -> thin."""
    params = resolve_preset(mode)
    drum_onsets = detect_drum_onsets(drums_wav)
    confirmed = confirm_with_pyin(
        notes,
        guitar_wav,
        weak_velocity=params.weak_velocity,
        drum_aligned_confirm_velocity=params.drum_aligned_confirm_velocity,
        drum_onsets=drum_onsets,
    )
    rejected = reject_drum_aligned(
        confirmed,
        drums_wav,
        strong_velocity=params.drum_reject_strong_velocity,
        drum_onsets=drum_onsets,
    )
    rejected = reject_bass_aligned(rejected, bass_wav)
    return thin_charter_notes(rejected, min_duration_s=params.min_charter_duration_s)


def _stage_counts(
    raw_tuples,
    *,
    guitar_wav: Path,
    drums_wav: Path,
    mode: str = "balanced",
    bass_wav: Path | None = None,
) -> dict:
    """Count notes kept after each post-Basic-Pitch stage (diagnosis)."""
    params = resolve_preset(mode)
    filtered = filter_note_events(raw_tuples, min_velocity=params.min_velocity)
    drum_onsets = detect_drum_onsets(drums_wav)
    confirmed = confirm_with_pyin(
        filtered,
        guitar_wav,
        weak_velocity=params.weak_velocity,
        drum_aligned_confirm_velocity=params.drum_aligned_confirm_velocity,
        drum_onsets=drum_onsets,
    )
    rejected = reject_drum_aligned(
        confirmed,
        drums_wav,
        strong_velocity=params.drum_reject_strong_velocity,
        drum_onsets=drum_onsets,
    )
    after_bass = reject_bass_aligned(rejected, bass_wav)
    thinned = thin_charter_notes(after_bass, min_duration_s=params.min_charter_duration_s)
    return {
        "after_velocity": len(filtered),
        "after_pyin": len(confirmed),
        "after_drum_reject": len(rejected),
        "after_bass_reject": len(after_bass),
        "after_thin": len(thinned),
    }


def measure_postfilter_and_density(truth: list[NoteEvent]) -> dict:
    noisy = _inject_bleed_false_positives(truth)
    raw_tuples = [(n.start_s, n.end_s, n.midi_pitch, n.velocity) for n in noisy]
    legacy = _legacy_filter(raw_tuples)
    filtered = filter_note_events(raw_tuples)
    guitar_wav = EVAL_DIR / "clean_melody.wav"
    drums_wav = EVAL_DIR / "drums_only.wav"
    full = _apply_full_chain(filtered, guitar_wav=guitar_wav, drums_wav=drums_wav, mode="balanced")

    tempo = TempoMap(bpm=120.0)
    duration_s = max(n.end_s for n in truth) + 0.5

    from src.pipeline.fretmap import thin_for_difficulty

    legacy_expert = notes_to_expert(legacy, tempo)
    legacy_hard = thin_for_difficulty(legacy_expert, max_chord=2, min_spacing=tempo.resolution // 4)
    filter_charts = map_difficulties(filtered, tempo)
    full_charts = map_difficulties(full, tempo)
    stages = _stage_counts(
        raw_tuples, guitar_wav=guitar_wav, drums_wav=drums_wav, mode="balanced"
    )

    return {
        "transcription_legacy_vs_truth": asdict(score_transcription(truth, legacy)),
        "transcription_filtered_vs_truth": asdict(score_transcription(truth, filtered)),
        "transcription_full_chain_vs_truth": asdict(score_transcription(truth, full)),
        "density_legacy_expert": asdict(
            score_density(
                legacy_expert,
                legacy_hard,
                duration_s=duration_s,
                tempo_bpm=120.0,
                resolution=192,
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
        # Back-compat alias used by older readers: "new" = full shipped chain.
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
        "stage_counts_balanced": stages,
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
    drums_wav = EVAL_DIR / "drums_only.wav"
    bass_wav = EVAL_DIR / "bass_only.wav"
    names = (
        "clean_melody.wav",
        "distorted_melody.wav",
        "distorted_bleed_melody.wav",
        "band_mix.wav",
    )
    balanced = resolve_preset("balanced")
    for name in names:
        path = EVAL_DIR / name
        try:
            _mo, _midi, raw_events = predict(
                str(path),
                model_or_model_path=model_path,
                onset_threshold=balanced.onset_threshold,
                frame_threshold=balanced.frame_threshold,
                minimum_frequency=82.0,
                maximum_frequency=1318.5,
            )
            estimated = filter_note_events(raw_events, min_velocity=balanced.min_velocity)
            if name in {"band_mix.wav", "distorted_bleed_melody.wav"}:
                estimated = _apply_full_chain(
                    estimated,
                    guitar_wav=path,
                    drums_wav=drums_wav,
                    mode="balanced",
                    bass_wav=bass_wav if name == "band_mix.wav" else None,
                )
            elif name == "clean_melody.wav":
                # Still run pyin on clean so confirm regressions show up; no drums needed.
                estimated = confirm_with_pyin(
                    estimated,
                    path,
                    weak_velocity=balanced.weak_velocity,
                    drum_aligned_confirm_velocity=balanced.drum_aligned_confirm_velocity,
                )
                estimated = thin_charter_notes(
                    estimated, min_duration_s=balanced.min_charter_duration_s
                )
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
                    "onset": balanced.onset_threshold,
                    "frame": balanced.frame_threshold,
                    "min_velocity": balanced.min_velocity,
                },
            }
            if name == "band_mix.wav":
                results[name]["stage_counts"] = _stage_counts(
                    [(e[0], e[1], e[2], e[3] if len(e) > 3 else 1.0) for e in raw_events],
                    guitar_wav=path,
                    drums_wav=drums_wav,
                    mode="balanced",
                    bass_wav=bass_wav,
                )
        except Exception as exc:  # noqa: BLE001 — baseline must continue
            results[name] = {"ok": False, "error": str(exc)}
    return results


def measure_mode_matrix() -> dict:
    """Per-mode P/R/F on synthetic fixtures + bleed injection + Auto picks."""
    from basic_pitch import FilenameSuffix, build_icassp_2022_model_path
    from basic_pitch.inference import predict

    truth = load_ground_truth()
    model_path = build_icassp_2022_model_path(FilenameSuffix.onnx)
    drums_wav = EVAL_DIR / "drums_only.wav"
    bass_wav = EVAL_DIR / "bass_only.wav"
    fixtures = ("clean_melody.wav", "band_mix.wav")
    out: dict = {"fixtures": {}, "bleed_injection": {}, "auto_selection": {}, "presets": {}}

    for mode in MODE_NAMES:
        params = resolve_preset(mode)
        out["presets"][mode] = {
            "onset_threshold": params.onset_threshold,
            "frame_threshold": params.frame_threshold,
            "min_velocity": params.min_velocity,
            "weak_velocity": params.weak_velocity,
            "drum_aligned_confirm_velocity": params.drum_aligned_confirm_velocity,
            "drum_reject_strong_velocity": params.drum_reject_strong_velocity,
            "min_charter_duration_s": params.min_charter_duration_s,
        }

    noisy = _inject_mode_bleed_false_positives(truth)
    raw_tuples = [(n.start_s, n.end_s, n.midi_pitch, n.velocity) for n in noisy]
    guitar_clean = EVAL_DIR / "clean_melody.wav"
    for mode in MODE_NAMES:
        params = resolve_preset(mode)
        filtered = filter_note_events(raw_tuples, min_velocity=params.min_velocity)
        full = _apply_full_chain(
            filtered, guitar_wav=guitar_clean, drums_wav=drums_wav, mode=mode
        )
        scores = score_transcription(truth, full)
        out["bleed_injection"][mode] = {
            **asdict(scores),
            "est_notes": len(full),
        }

    for name in fixtures:
        path = EVAL_DIR / name
        out["fixtures"][name] = {}
        for mode in MODE_NAMES:
            params = resolve_preset(mode)
            try:
                _mo, _midi, raw_events = predict(
                    str(path),
                    model_or_model_path=model_path,
                    onset_threshold=params.onset_threshold,
                    frame_threshold=params.frame_threshold,
                    minimum_frequency=82.0,
                    maximum_frequency=1318.5,
                )
                estimated = filter_note_events(raw_events, min_velocity=params.min_velocity)
                if name == "band_mix.wav":
                    estimated = _apply_full_chain(
                        estimated,
                        guitar_wav=path,
                        drums_wav=drums_wav,
                        mode=mode,
                        bass_wav=bass_wav,
                    )
                else:
                    estimated = confirm_with_pyin(
                        estimated,
                        path,
                        weak_velocity=params.weak_velocity,
                        drum_aligned_confirm_velocity=params.drum_aligned_confirm_velocity,
                    )
                    estimated = thin_charter_notes(
                        estimated, min_duration_s=params.min_charter_duration_s
                    )
                scores = score_transcription(truth, estimated)
                out["fixtures"][name][mode] = {
                    "ok": True,
                    **asdict(scores),
                }
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
    balanced = resolve_preset("balanced")
    for clip_id, audio in LABELED_CLIPS:
        label = LABEL_DIR / f"{clip_id}.json"
        if not audio.is_file() or not label.is_file():
            results[clip_id] = {"ok": False, "error": "missing audio or label"}
            continue
        try:
            truth = load_label_notes(label)
            estimated = transcribe_guitar(audio, sensitivity="balanced")
            estimated = thin_charter_notes(
                estimated, min_duration_s=balanced.min_charter_duration_s
            )
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
    modes = report.get("mode_matrix", {})
    lines = [
        "# Accuracy baselines",
        "",
        f"Generated: `{report['generated_at']}`",
        "",
        "Floors after drum/bass reject, charter thinning, mode-scoped pyin, adaptive sensitivity, "
        "and string-aware frets. Post-filter baselines score the **full shipped chain** "
        "(velocity -> pyin -> drum/bass reject -> charter thin -> Expert prune).",
        "",
        "## Pipeline knobs under test",
        "",
        f"- Balanced Basic Pitch `onset_threshold={ONSET_THRESHOLD}`, `frame_threshold={FRAME_THRESHOLD}`",
        f"- Balanced post-filter `min_velocity={MIN_NOTE_VELOCITY}` (strict/sensitive differ)",
        "- Guitar stem cleanup + drums STFT soft-subtract",
        "- Mode-scoped pyin confirm; drum-aligned mid-velocity ghosts require pyin unless very strong",
        "- Bass-onset reject for low MIDI bleed; charter thinning (duration/merge/conflict)",
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
            f"| Velocity filter only + Expert prune | "
            f"{pf['transcription_filtered_vs_truth']['precision']:.3f} | "
            f"{pf['transcription_filtered_vs_truth']['recall']:.3f} | "
            f"{pf['transcription_filtered_vs_truth']['f_measure']:.3f} | "
            f"{pf['filtered_est_notes']} | "
            f"{pf['density_filter_expert']['expert_notes']} | "
            f"{pf['density_filter_expert']['expert_nps']:.2f} |"
        ),
        (
            f"| Full chain (velocity -> pyin -> drum reject -> Expert) | "
            f"{pf['transcription_full_chain_vs_truth']['precision']:.3f} | "
            f"{pf['transcription_full_chain_vs_truth']['recall']:.3f} | "
            f"{pf['transcription_full_chain_vs_truth']['f_measure']:.3f} | "
            f"{pf['full_chain_est_notes']} | "
            f"{pf['density_full_chain_expert']['expert_notes']} | "
            f"{pf['density_full_chain_expert']['expert_nps']:.2f} |"
        ),
        "",
    ]
    stages = pf.get("stage_counts_balanced") or {}
    if stages:
        lines += [
            "Balanced stage note counts on bleed injection: "
            f"velocity={stages.get('after_velocity')}, "
            f"pyin={stages.get('after_pyin')}, "
            f"drum_reject={stages.get('after_drum_reject')}.",
            "",
        ]
    lines += [
        "## 2. Isolation bleed proxy (synthetic distorted guitar + drums in guitar stem)",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Bleed correlation before cleanup | {bleed['bleed_mix_before']:.4f} |",
        f"| After stem cleanup | {bleed['bleed_mix_after_clean']:.4f} |",
        f"| After drums soft-subtract | {bleed['bleed_mix_after_subtract']:.4f} |",
        f"| Total reduction | {bleed['bleed_reduction_pct']:.1f}% |",
        "",
        "## 3. Basic Pitch on synthetic WAVs (Balanced)",
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
        "## 4. Sensitivity modes (Strict / Balanced / Sensitive)",
        "",
        "Contracts: Strict favors precision on bleed; Sensitive favors recall/note count on clean; "
        "Balanced stays between them; Auto picks from stem bleed/crest.",
        "",
    ]
    presets = modes.get("presets") or {}
    if presets:
        lines += [
            "| Mode | onset / frame / min_vel | weak / drum-confirm / drum-reject |",
            "| --- | --- | --- |",
        ]
        for mode in MODE_NAMES:
            p = presets.get(mode, {})
            lines.append(
                f"| `{mode}` | {p.get('onset_threshold')} / {p.get('frame_threshold')} / "
                f"{p.get('min_velocity')} | {p.get('weak_velocity')} / "
                f"{p.get('drum_aligned_confirm_velocity')} / {p.get('drum_reject_strong_velocity')} |"
            )
        lines.append("")

    lines += [
        "### Bleed injection (full chain)",
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
        f"- Clean high-crest stem -> `{auto.get('clean_stem', '?')}`",
        "",
        "## 5. Labeled Creative Commons clips (mix-direct Balanced + charter thin vs curated labels)",
        "",
        "Labels are Balanced Basic Pitch drafts with the same charter thinning in `tests/fixtures/eval/labels/`. "
        "They are a regression lock for the charter-thin path (not independent human GT).",
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
        "## Regression floors",
        "",
        "Unit tests enforce:",
        "",
        "- Velocity filter drops sub-0.35 ghost notes (Balanced)",
        "- Full chain drops mid-velocity drum-aligned ghosts; keeps strong on-beat guitar",
        "- Expert prune enforces >= 32nd spacing",
        "- Stem cleanup reduces bleed proxy by >5%; drums subtract improves further",
        "- Auto sensitivity selects Strict on high-bleed stems",
        "- Mode note-count ordering: Strict <= Balanced <= Sensitive on shared raw events",
        "- Strict precision >= Balanced precision + 0.05 on bleed injection (full chain)",
        "- Bass reject drops low MIDI bleed on bass onsets",
        "- Charter thinning merges same-pitch overlaps and drops quieter octave doubles",
        "- Scale-run lanes stay locally continuous (no C-to-green wrap)",
        "- Perfect self-score on ground-truth note lists",
        "",
        "Harness floors (this file): clean synthetic recall stays 1.0 under Balanced/Sensitive; "
        "section-1 full-chain mid-velocity bleed-injection F >= 0.80 under Balanced; "
        "band_mix Balanced F = 1.0 after bass reject; "
        "CC charter-thin labels self-score F = 1.0 under Balanced+thin; "
        "section-4 mode matrix adds near-strong ghosts to separate Strict/Balanced/Sensitive; "
        "distorted+drums bleed correlation after subtract is below the cleanup-only value.",
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
        "mode_matrix": measure_mode_matrix(),
        "labeled_clips": measure_labeled_clips(),
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
