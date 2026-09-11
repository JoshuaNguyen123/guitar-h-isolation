# Guitar H Isolation

Turn a song on your computer into a custom Clone Hero guitar chart.

You pick an MP3, click Generate, and the app makes a folder Clone Hero can play. It pulls out the guitar and builds the colored notes (green, red, yellow, blue, orange).

The chart is a first draft. It is playable, but it will not look as clean as a chart someone made by hand.

This app is **free**. There is no account, no subscription, and no paid tier. You run it on your computer.

This project is not affiliated with Guitar Hero, Harmonix, or Clone Hero.

Only use songs you are allowed to use.

---

## Download the app (easiest way)

You do not need Git.

1. Open this page: [https://github.com/JoshuaNguyen123/guitar-h-isolation](https://github.com/JoshuaNguyen123/guitar-h-isolation)
2. Click the green **Code** button.
3. Click **Download ZIP**.
4. Open your **Downloads** folder.
5. Right-click `guitar-h-isolation-main.zip` and click **Extract All**.
6. Click **Extract**.
7. Open the new folder named `guitar-h-isolation-main`.

Leave this folder open. You will come back to it.

---

## One-time setup (one step)

You need **Python 3** on this computer first. If you do not have it:

1. Open [https://www.python.org/downloads/](https://www.python.org/downloads/)
2. Click the big yellow **Download Python** button.
3. Run the installer.
4. At the bottom of the first screen, turn on **Add python.exe to PATH**.
5. Click **Install Now**.

Then, in the `guitar-h-isolation-main` folder, **double-click `run.bat`**.

The first launch installs the app and tries to install FFmpeg if it is missing. That can take several minutes. After that, the Guitar H Isolation window opens. Later launches just open the app.

If the window never opens, double-click `install.bat`, wait until it says **Install OK**, then double-click `run.bat` again.

---

## How to make a Clone Hero song

1. In the app, click **Browse** next to **Audio file**.
2. Pick a song file on your computer. MP3 is the usual choice. WAV, FLAC, OGG, and M4A also work.
3. Check **Song name** and **Artist**. Change them if they look wrong.
4. Leave **Note sensitivity** on **Auto** unless a chart feels wrong.

   - **Auto** picks Strict on bleed-heavy stems, Sensitive on very clean stems, otherwise Balanced
   - **Strict** highest precision (use this if Expert tracks the drums)
   - **Balanced** stock Basic Pitch (0.5 / 0.3) plus the evidence score
   - **Sensitive** highest recall on quiet or clean guitar

5. Leave **Output folder** alone unless you know Clone Hero is in portable mode.

   The default is:

   `Documents\Clone Hero\Songs\` and then your song name

6. Click **Generate**.
7. Wait. The first song is the slowest. It can take several minutes. You will see:

   - Separate (pulls out the guitar)
   - Transcribe (finds the notes)
   - Chart (builds the colored track)
   - Package (saves the folder)

8. When it says **Done**, click **Open in Explorer**.

To look at a song folder this app (or an older version) already made, click **Open song folder**, pick that folder, and the app shows the name, BPM, and note counts.

You should see four files:

- `song.ogg` (the band without the isolated guitar)
- `guitar.ogg` (just the guitar)
- `song.ini` (the song name and artist)
- `notes.chart` (the notes you play)

---

## How to play it in Clone Hero

1. Make sure the folder you just opened is **inside** Clone Hero's Songs folder.

   - Normal install: `Documents\Clone Hero\Songs\Your Song Name`
   - Portable install: the game folder, then `PlayerData\Songs\Your Song Name`

   The four files must stay together in that song folder.

2. Open Clone Hero.
3. Click **Settings**.
4. Click **General**.
5. Click **Scan Songs** and wait.
6. Go back and open **Quickplay**.
7. Find your song and start on **Easy** or **Medium**.

If the song does not show up, open `badsongs.txt` in your Clone Hero folder. That file explains what is missing.

---

## What to expect

- Songs with a clear guitar part work best.
- Busy songs, heavy distortion, or many guitars at once will miss notes or add extras. The app transcribes the raw guitar stem, then keeps notes that still look like guitar (velocity + sustain) and only penalizes drum/bass bleed when a note does not ring. Use **Strict** if Expert still tracks the drums.
- This version only makes a guitar chart. It does not chart drums, bass, or vocals.
- You can clean up the chart later in Moonscraper if you want to share it.
- Accuracy regression numbers: [tests/eval/BASELINE.md](tests/eval/BASELINE.md).


## How the app actually works

The full write-up is in [ARCHITECTURE.md](ARCHITECTURE.md), including the [architecture diagram](architecture-diagram.svg). [ARCHITECTURE.doc](ARCHITECTURE.doc) is a Word copy of that page. If they ever disagree, trust `ARCHITECTURE.md`.

Short version:

1. FFmpeg turns your file into a WAV.
2. Demucs (`htdemucs_6s`) splits the mix into six stems. The first run downloads that model from Hugging Face (that is the slow “chunk” download). Later songs reuse the cached file.
3. `song.ogg` is the band **without** guitar. `guitar.ogg` is the raw isolated guitar. Clone Hero plays both so the guitar is not doubled.
4. Basic Pitch guesses notes from that raw stem. Each candidate gets one evidence score. Drum hits are not treated as proof the note is fake. That is pitch detection, not real guitar tab.
5. Notes map onto five Clone Hero lanes from standard-tuning fingerings, then thin for Hard, Medium, and Easy. Expert also has a 32nd-note density floor.
6. Tempo is one BPM number for the whole song.

The chart is a machine draft. It is not official Guitar Hero or Clone Hero content.

---

## For testers and developers

Fast checks (no song processing):

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
python -m pytest tests/test_fretmap.py tests/test_chart_writer.py tests/eval/test_eval_unit.py tests/eval/test_guitarset_unit.py -q
```

Accuracy baselines (writes `tests/eval/BASELINE.md`; Basic Pitch section needs the ONNX model):

```powershell
python -m tests.eval.run_baseline
```

Independent GuitarSet scores vs the pre-`bc00cd4` 0.5/0.3 reference (slow, not in default CI):

```powershell
python -m tests.eval.run_guitarset --download-only
python -m tests.eval.run_guitarset --all-modes --limit 6
```

Full run on one public guitar clip (slow):

```powershell
python -m pytest tests/e2e/test_real_pipeline.py -s
```

Four public clips, then a Playwright check (slow):

```powershell
python tests\e2e\run_clip_suite.py
python -m pytest tests/e2e/test_playwright_clip_suite.py -q
```

---

## Credits

- [Demucs](https://github.com/facebookresearch/demucs) and [demucs-onnx](https://github.com/stemsplit/demucs-onnx) pull out the guitar.
- [Basic Pitch](https://github.com/spotify/basic-pitch) finds the notes.
- Test clips are listed in `tests/fixtures/CLIPS.md`.

## License

MIT. See [LICENSE](LICENSE).
