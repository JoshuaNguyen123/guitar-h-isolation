# Accuracy baselines

Generated: `2026-09-11T19:32:39.005400+00:00`

Floors after drum/bass reject, charter thinning, mode-scoped pyin, adaptive sensitivity, and string-aware frets. Post-filter baselines score the **full shipped chain** (velocity -> pyin -> drum/bass reject -> charter thin -> Expert prune).

## Pipeline knobs under test

- Balanced Basic Pitch `onset_threshold=0.58`, `frame_threshold=0.38`
- Balanced post-filter `min_velocity=0.35` (strict/sensitive differ)
- Guitar stem cleanup + drums STFT soft-subtract
- Mode-scoped pyin confirm; drum-aligned mid-velocity ghosts require pyin unless very strong
- Bass-onset reject for low MIDI bleed; charter thinning (duration/merge/conflict)
- Expert density prune: 32nd-note minimum spacing
- String-aware 5-lane fretting (standard tuning, high E shares orange with B)

## 1. Bleed false-positive post-filter + Expert density (synthetic note events)

Ground truth: 7-note melody. Estimated: truth + drum-on-beat false notes (low and mid velocity).

| Stage | Precision | Recall | F-measure | Est notes | Expert notes | Expert NPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Legacy (no velocity filter, unpruned Expert) | 0.304 | 1.000 | 0.467 | 23 | 23 | 5.61 |
| Velocity filter only + Expert prune | 0.467 | 1.000 | 0.636 | 15 | 15 | 3.66 |
| Full chain (velocity -> pyin -> drum reject -> Expert) | 1.000 | 1.000 | 1.000 | 7 | 7 | 1.71 |

Balanced stage note counts on bleed injection: velocity=15, pyin=8, drum_reject=7.

## 2. Isolation bleed proxy (synthetic distorted guitar + drums in guitar stem)

| Metric | Value |
| --- | ---: |
| Bleed correlation before cleanup | 0.2783 |
| After stem cleanup | 0.0848 |
| After drums soft-subtract | 0.0153 |
| Total reduction | 94.5% |

## 3. Basic Pitch on synthetic WAVs (Balanced)

| Fixture | Current P / R / F (notes) | Legacy 0.5/0.3 P / R / F (notes) |
| --- | --- | --- |
| `clean_melody.wav` | 1.000 / 1.000 / 1.000 (7) | 1.000 / 1.000 / 1.000 (7) |
| `distorted_melody.wav` | 1.000 / 1.000 / 1.000 (7) | 1.000 / 1.000 / 1.000 (7) |
| `distorted_bleed_melody.wav` | 1.000 / 1.000 / 1.000 (7) | 1.000 / 1.000 / 1.000 (7) |
| `band_mix.wav` | 1.000 / 1.000 / 1.000 (7) | 0.500 / 1.000 / 0.667 (14) |

## 4. Sensitivity modes (Strict / Balanced / Sensitive)

Contracts: Strict favors precision on bleed; Sensitive favors recall/note count on clean; Balanced stays between them; Auto picks from stem bleed/crest.

| Mode | onset / frame / min_vel | weak / drum-confirm / drum-reject |
| --- | --- | --- |
| `strict` | 0.66 / 0.44 / 0.45 | 0.48 / 0.78 / 0.72 |
| `balanced` | 0.58 / 0.38 / 0.35 | 0.55 / 0.7 / 0.62 |
| `sensitive` | 0.48 / 0.28 / 0.22 | 0.62 / 0.58 / 0.52 |

### Bleed injection (full chain)

| Mode | Precision | Recall | F-measure | Est notes |
| --- | ---: | ---: | ---: | ---: |
| `strict` | 1.000 | 1.000 | 1.000 | 7 |
| `balanced` | 0.467 | 1.000 | 0.636 | 15 |
| `sensitive` | 0.438 | 1.000 | 0.609 | 16 |

### Synthetic fixtures

| Fixture | Mode | Precision | Recall | F-measure | Est notes |
| --- | --- | ---: | ---: | ---: | ---: |
| `clean_melody.wav` | `strict` | 1.000 | 1.000 | 1.000 | 7 |
| `clean_melody.wav` | `balanced` | 1.000 | 1.000 | 1.000 | 7 |
| `clean_melody.wav` | `sensitive` | 1.000 | 1.000 | 1.000 | 7 |
| `band_mix.wav` | `strict` | 1.000 | 1.000 | 1.000 | 7 |
| `band_mix.wav` | `balanced` | 1.000 | 1.000 | 1.000 | 7 |
| `band_mix.wav` | `sensitive` | 0.778 | 1.000 | 0.875 | 9 |

### Auto selection

- High-bleed stem -> `strict`
- Clean high-crest stem -> `balanced`

## 5. Labeled Creative Commons clips (mix-direct Balanced + charter thin vs curated labels)

Labels are Balanced Basic Pitch drafts with the same charter thinning in `tests/fixtures/eval/labels/`. They are a regression lock for the charter-thin path (not independent human GT).

| Clip | Precision | Recall | F-measure | Est / label notes |
| --- | ---: | ---: | ---: | ---: |
| `electric_lick` | 1.000 | 1.000 | 1.000 | 12 / 12 |
| `acoustic_chords` | 1.000 | 1.000 | 1.000 | 76 / 76 |
| `acoustic_shuffle` | 1.000 | 1.000 | 1.000 | 81 / 81 |

## Regression floors

Unit tests enforce:

- Velocity filter drops sub-0.35 ghost notes (Balanced)
- Full chain drops mid-velocity drum-aligned ghosts; keeps strong on-beat guitar
- Expert prune enforces >= 32nd spacing
- Stem cleanup reduces bleed proxy by >5%; drums subtract improves further
- Auto sensitivity selects Strict on high-bleed stems
- Mode note-count ordering: Strict <= Balanced <= Sensitive on shared raw events
- Strict precision >= Balanced precision + 0.05 on bleed injection (full chain)
- Bass reject drops low MIDI bleed on bass onsets
- Charter thinning merges same-pitch overlaps and drops quieter octave doubles
- Scale-run lanes stay locally continuous (no C-to-green wrap)
- Perfect self-score on ground-truth note lists

Harness floors (this file): clean synthetic recall stays 1.0 under Balanced/Sensitive; section-1 full-chain mid-velocity bleed-injection F >= 0.80 under Balanced; section-4 mode matrix adds near-strong ghosts to separate Strict/Balanced/Sensitive; distorted+drums bleed correlation after subtract is below the cleanup-only value.

## Notes

- Synthetic fixtures live in `tests/fixtures/eval/` (CC0).
- mir-eval onset tolerance is 50 ms; pitch tolerance 50 cents; offsets ignored.
- Real commercial distorted mixes are still not claimed; these numbers are the repo's quantitative regression bar.
