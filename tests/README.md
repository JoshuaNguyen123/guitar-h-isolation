# Tests

## Unit

`test_fretmap.py` and `test_chart_writer.py` do not download models.

```powershell
python -m pytest tests/test_fretmap.py tests/test_chart_writer.py -q
```

## End-to-end

`tests/e2e/test_real_pipeline.py` runs the full isolate → transcribe → package path on `fixtures/guitar_chords.mp3`. It is slow and writes `e2e_output/` (gitignored).

The fixture is [Guitare accoustique accords](https://commons.wikimedia.org/wiki/File:Guitare_accoustique_accords.ogg) by Frédéric Jacquot, licensed [CC BY-SA 2.5](https://creativecommons.org/licenses/by-sa/2.5/). See `fixtures/ATTRIBUTION.txt`.
