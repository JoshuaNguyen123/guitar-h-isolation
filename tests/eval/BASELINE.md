# Accuracy baselines

Generated: `2026-09-11T22:09:51.594125+00:00`

Shipped path: raw Demucs guitar stem → Basic Pitch → evidence score → tempo-relative thin → Expert prune.
The pre-`bc00cd4` reference is stock Basic Pitch 0.5 / 0.3 with no post-filters.

## Pipeline knobs under test

- Balanced Basic Pitch `onset_threshold=0.5`, `frame_threshold=0.3`
- Velocity is an evidence input, not a hard gate
- Evidence score: velocity + posterior sustain − drum/bass dominance when sustain is low
- Tempo-relative thinning (half a 16th, keeps octave doubles)
- Expert density prune: 32nd-note minimum spacing

## 1. Bleed false-positive post-filter + Expert density (synthetic note events)

| Stage | Precision | Recall | F-measure | Est notes | Expert notes | Expert NPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Legacy (no evidence score, unpruned Expert) | 0.304 | 1.000 | 0.467 | 23 | 23 | 5.61 |
| Duration/pitch filter only + Expert prune | 0.304 | 1.000 | 0.467 | 23 | 23 | 5.61 |
| Full chain (evidence score → thin → Expert) | 1.000 | 1.000 | 1.000 | 7 | 7 | 1.71 |

## 2. Isolation bleed proxy (diagnostic helpers only — not on Generate)

| Metric | Value |
| --- | ---: |
| Bleed correlation before cleanup | 0.2783 |
| After stem cleanup | 0.0848 |
| After drums soft-subtract | 0.0153 |
| Total reduction | 94.5% |

## 3. Basic Pitch on synthetic WAVs (Balanced vs pre-bc00cd4)

| Fixture | Shipped P / R / F (notes) | Pre-bc00cd4 0.5/0.3 P / R / F (notes) |
| --- | --- | --- |
| `clean_melody.wav` | 1.000 / 1.000 / 1.000 (7) | 1.000 / 1.000 / 1.000 (7) |
| `distorted_melody.wav` | 1.000 / 1.000 / 1.000 (7) | 1.000 / 1.000 / 1.000 (7) |
| `distorted_bleed_melody.wav` | 1.000 / 1.000 / 1.000 (7) | 1.000 / 1.000 / 1.000 (7) |
| `band_mix.wav` | 0.538 / 1.000 / 0.700 (13) | 0.500 / 1.000 / 0.667 (14) |

## 4. Sensitivity modes

| Mode | onset / frame / keep |
| --- | --- |
| `strict` | 0.55 / 0.35 / 0.2 |
| `balanced` | 0.5 / 0.3 / 0.1 |
| `sensitive` | 0.4 / 0.25 / 0.04 |

### Bleed injection (evidence chain)

| Mode | Precision | Recall | F-measure | Est notes |
| --- | ---: | ---: | ---: | ---: |
| `strict` | 1.000 | 1.000 | 1.000 | 7 |
| `balanced` | 1.000 | 1.000 | 1.000 | 7 |
| `sensitive` | 1.000 | 1.000 | 1.000 | 7 |

### Synthetic fixtures

| Fixture | Mode | Precision | Recall | F-measure | Est notes |
| --- | --- | ---: | ---: | ---: | ---: |
| `clean_melody.wav` | `strict` | 1.000 | 1.000 | 1.000 | 7 |
| `clean_melody.wav` | `balanced` | 1.000 | 1.000 | 1.000 | 7 |
| `clean_melody.wav` | `sensitive` | 1.000 | 1.000 | 1.000 | 7 |
| `band_mix.wav` | `strict` | 0.538 | 1.000 | 0.700 | 13 |
| `band_mix.wav` | `balanced` | 0.538 | 1.000 | 0.700 | 13 |
| `band_mix.wav` | `sensitive` | 0.538 | 1.000 | 0.700 | 13 |

### Auto selection

- High-bleed stem -> `strict`
- Clean stem -> `sensitive`

## 5. Labeled Creative Commons clips (smoke only)

Not independent human GT. Real-audio accuracy is GuitarSet (`python -m tests.eval.run_guitarset`).

| Clip | Precision | Recall | F-measure | Est / label notes |
| --- | ---: | ---: | ---: | ---: |
| `electric_lick` | 0.333 | 0.917 | 0.489 | 33 / 12 |
| `acoustic_chords` | 0.546 | 0.934 | 0.689 | 130 / 76 |
| `acoustic_shuffle` | 0.433 | 0.802 | 0.563 | 150 / 81 |

## 6. GuitarSet (independent labels)

Generated: `2026-09-11T22:19:25.647418+00:00`

Tracks: 6. Solo shipped F `0.733` vs pre-bc00cd4 `0.733`. Band shipped F `0.553` vs pre-bc00cd4 `0.551`. Band Strict P `0.653` vs Balanced P `0.619`.

## Regression floors

- Evidence score keeps sustained on-beat guitar and drops short broadband ghosts
- Chord voices are not pyin-vetoed
- Strict keep bar > Balanced > Sensitive
- Auto selects Strict on high-bleed stems (no crest rule)
- Charter thin merges same-pitch overlaps and keeps octave doubles
- Generate transcribes the raw Demucs stem

## Notes

- Synthetic fixtures live in `tests/fixtures/eval/` (CC0).
- mir-eval onset tolerance is 50 ms; pitch tolerance 50 cents; offsets ignored.
