# Tests

## Unit

`test_fretmap.py`, `test_chart_writer.py`, and `tests/eval/test_eval_unit.py` do not download Demucs. The eval unit tests do need `numpy`, `scipy`, `soundfile`, `mir-eval`, and `librosa`.

```powershell
python -m pytest tests/test_fretmap.py tests/test_chart_writer.py tests/eval/test_eval_unit.py -q
```

## Accuracy baselines

```powershell
python -m tests.eval.run_baseline
```

Writes `tests/eval/BASELINE.md` and `tests/eval/baseline.json`. Synthetic fixtures are under `tests/fixtures/eval/`. Labeled CC clips are `tests/fixtures/eval/labels/*.json` (thinned Basic Pitch drafts of the public fixtures).

The harness scores the **full shipped post-filter chain** (velocity -> pyin -> drum/bass reject -> charter thin -> Expert) and publishes a **Strict / Balanced / Sensitive** comparison matrix. Mode contracts: Strict favors precision on bleed; Sensitive keeps more notes; Balanced is the middle; Auto picks from stem bleed/crest. CC clip labels match Balanced+thin (regression lock, not human GT).

## End-to-end

`tests/e2e/test_real_pipeline.py` runs the full isolate → transcribe → package path on `fixtures/guitar_chords.mp3`. It is slow and writes `e2e_output/` (gitignored).

The fixture is [Guitare accoustique accords](https://commons.wikimedia.org/wiki/File:Guitare_accoustique_accords.ogg) by Frédéric Jacquot, licensed [CC BY-SA 2.5](https://creativecommons.org/licenses/by-sa/2.5/). See `fixtures/ATTRIBUTION.txt`.
