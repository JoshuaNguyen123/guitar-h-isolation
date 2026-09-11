# Tests

## Unit

`test_fretmap.py`, `test_chart_writer.py`, and `tests/eval/test_eval_unit.py` do not download Demucs. The eval unit tests do need `numpy`, `scipy`, `soundfile`, `mir-eval`, and `librosa`.

```powershell
python -m pytest tests/test_fretmap.py tests/test_chart_writer.py tests/eval/test_eval_unit.py tests/eval/test_guitarset_unit.py -q
```

## Accuracy baselines

```powershell
python -m tests.eval.run_baseline
```

Writes `tests/eval/BASELINE.md` and `tests/eval/baseline.json`. Synthetic fixtures are under `tests/fixtures/eval/`. Labeled CC clips are `tests/fixtures/eval/labels/*.json` (Basic Pitch drafts of the public fixtures; smoke only).

The harness scores the **full shipped chain** (evidence score → tempo-relative thin → Expert) against a pre-`bc00cd4` reference (Basic Pitch 0.5 / 0.3, no filters).

Independent real-audio eval (GuitarSet, not in default CI):

```powershell
python -m tests.eval.run_guitarset --download-only
python -m tests.eval.run_guitarset
```

A/B a local stem against pre-bc00cd4 (0.5 / 0.3, no filters):

```powershell
python -m tests.eval.ab_compare path\to\guitar.wav --drums path\to\drums.wav
```

## End-to-end

`tests/e2e/test_real_pipeline.py` runs the full isolate → transcribe → package path on `fixtures/guitar_chords.mp3`. It is slow and writes `e2e_output/` (gitignored).

The fixture is [Guitare accoustique accords](https://commons.wikimedia.org/wiki/File:Guitare_accoustique_accords.ogg) by Frédéric Jacquot, licensed [CC BY-SA 2.5](https://creativecommons.org/licenses/by-sa/2.5/). See `fixtures/ATTRIBUTION.txt`.
