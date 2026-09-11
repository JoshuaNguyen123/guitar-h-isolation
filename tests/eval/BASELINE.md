# Accuracy baselines

Generated: `2026-09-11T00:43:03.642134+00:00`

v1 floors after drum subtract, drum-aware reject, adaptive sensitivity, pyin confirm, and string-aware frets.

## Pipeline knobs under test

- Balanced Basic Pitch `onset_threshold=0.58`, `frame_threshold=0.38`
- Post-filter `min_velocity=0.35` (strict/sensitive presets differ)
- Guitar stem cleanup + drums STFT soft-subtract
- pyin confirm for weak notes; drum-onset reject for low-velocity ghosts
- Expert density prune: 32nd-note minimum spacing
- String-aware 5-lane fretting (standard tuning, high E shares orange with B)

## 1. Bleed false-positive post-filter + Expert density (synthetic note events)

Ground truth: 7-note melody. Estimated: truth + drum-on-beat false notes (low and mid velocity).

| Stage | Precision | Recall | F-measure | Est notes | Expert notes | Expert NPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Legacy (no velocity filter, unpruned Expert) | 0.304 | 1.000 | 0.467 | 23 | 23 | 5.61 |
| Current (velocity filter + Expert prune) | 0.467 | 1.000 | 0.636 | 15 | 15 | 3.66 |

## 2. Isolation bleed proxy (synthetic distorted guitar + drums in guitar stem)

| Metric | Value |
| --- | ---: |
| Bleed correlation before cleanup | 0.2783 |
| After stem cleanup | 0.0848 |
| After drums soft-subtract | 0.0153 |
| Total reduction | 94.5% |

## 3. Basic Pitch on synthetic WAVs

| Fixture | Current P / R / F (notes) | Legacy 0.5/0.3 P / R / F (notes) |
| --- | --- | --- |
| `clean_melody.wav` | 1.000 / 1.000 / 1.000 (7) | 1.000 / 1.000 / 1.000 (7) |
| `distorted_melody.wav` | 1.000 / 1.000 / 1.000 (7) | 1.000 / 1.000 / 1.000 (7) |
| `distorted_bleed_melody.wav` | 1.000 / 1.000 / 1.000 (7) | 1.000 / 1.000 / 1.000 (7) |
| `band_mix.wav` | 0.700 / 1.000 / 0.824 (10) | 0.500 / 1.000 / 0.667 (14) |

## 4. Labeled Creative Commons clips (mix-direct transcription vs curated labels)

Labels are thinned Basic Pitch drafts in `tests/fixtures/eval/labels/`.

| Clip | Precision | Recall | F-measure | Est / label notes |
| --- | ---: | ---: | ---: | ---: |
| `electric_lick` | 0.579 | 0.647 | 0.611 | 19 / 17 |
| `acoustic_chords` | 0.875 | 0.820 | 0.847 | 104 / 111 |
| `acoustic_shuffle` | 0.741 | 0.768 | 0.754 | 116 / 112 |

## v1 regression floors

Unit tests enforce:

- Velocity filter drops sub-0.35 ghost notes
- Expert prune enforces >= 32nd spacing
- Stem cleanup reduces bleed proxy by >5%; drums subtract improves further
- Auto sensitivity selects Strict on high-bleed stems
- Drum reject drops weak on-beat ghosts and keeps strong on-beat guitar
- pyin drops weak wrong-pitch notes and keeps matching / strong notes
- Scale-run lanes stay locally continuous (no C-to-green wrap)
- Perfect self-score on ground-truth note lists

Harness floors (this file): clean synthetic recall stays 1.0; distorted+drums bleed correlation after subtract is below the cleanup-only value.

## Notes

- Synthetic fixtures live in `tests/fixtures/eval/` (CC0).
- mir-eval onset tolerance is 50 ms; pitch tolerance 50 cents; offsets ignored.
- Real commercial distorted mixes are still not claimed; these numbers are the repo's quantitative regression bar.
