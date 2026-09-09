# Guitar H Isolation

A Windows desktop app that takes a song file and builds a custom Clone Hero track from it.

You pick an MP3, click Generate, and get a song folder you can drop into Clone Hero. The folder has isolated guitar audio plus a 5-fret guitar chart.

The chart is a first draft. It is playable, but it is not as polished as a chart someone made by hand. If you want to share it, clean it up in [Moonscraper](https://github.com/fireFox1918/Moonscraper-Chart-Editor) or Editor on Fire.

This project is not affiliated with Guitar Hero, Harmonix, or Clone Hero.

## What the output folder contains

| File | What it is |
|------|------------|
| `song.ogg` | The rest of the mix (no isolated guitar) |
| `guitar.ogg` | Just the guitar |
| `song.ini` | Title, artist, and other Clone Hero metadata |
| `notes.chart` | Green / red / yellow / blue / orange notes for Expert, Hard, Medium, and Easy |

Clone Hero plays `song.ogg` and `guitar.ogg` together, so you hear the full band without the guitar being doubled.

## What you need

- Windows
- Python 3.11 or newer
- FFmpeg on your PATH

Install FFmpeg with:

```powershell
winget install Gyan.FFmpeg
```

Or with Chocolatey:

```powershell
choco install ffmpeg
```

Then close and reopen the terminal. Confirm it worked:

```powershell
ffmpeg -version
```

## Install the app

From the project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If `pip` gets stuck installing TensorFlow (this happens a lot on OneDrive), stop it and run:

```powershell
pip install customtkinter demucs-onnx librosa soundfile numpy onnxruntime huggingface-hub pretty-midi resampy scipy scikit-learn soxr tqdm mir-eval
pip install --no-deps basic-pitch
```

The app uses the ONNX Basic Pitch model, so TensorFlow is not required.

## Download

Get the source from GitHub, then follow Install the app below.

```powershell
git clone https://github.com/JoshuaNguyen123/guitar-h-isolation.git
cd guitar-h-isolation
```

You can also use **Code**, then **Download ZIP** on the GitHub page.

## How to use the app

1. Open a terminal in the project folder.
2. Activate the venv if it is not already active:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

3. Start the app:

   ```powershell
   python -m src.app.main
   ```

   After the venv exists, you can also double-click `run.bat`.

4. Click **Browse** and pick your song. MP3 is the usual choice. WAV, FLAC, OGG, and M4A work too.
5. Check **Song name** and **Artist**. The app fills these from the filename when it can (for example `Artist - Title.mp3`).
6. Check **Output folder**. The default is:

   `Documents\Clone Hero\Songs\<Song Name>`

   That is the usual Clone Hero Songs folder on Windows. Change it if your game is in portable mode.
7. Click **Generate**.
8. Wait for the four stages to finish: Separate, Transcribe, Chart, Package.

   The first song is slower because the Demucs model downloads into a local cache. A song can take several minutes on CPU.
9. When it says Done, click **Open folder** or **Copy path**.

You should see `song.ogg`, `guitar.ogg`, `song.ini`, and `notes.chart` in that folder.

## How to play it in Clone Hero

1. Make sure the generated folder is inside your Clone Hero Songs directory.

   - Typical install: `Documents\Clone Hero\Songs\`
   - Portable install: `<game folder>\PlayerData\Songs\`

2. Open Clone Hero.
3. Go to **Settings**, then **General**, then **Scan Songs**.
4. Open **Quickplay** and find the song.

If it does not show up, check Clone Hero's `badsongs.txt` for a missing file or a bad chart.

## How to test

Install the extra test tools once:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
```

### Fast tests (run these first)

These check fret mapping and that `notes.chart` / `song.ini` text is valid. They do not download models and should finish in about a second.

```powershell
python -m pytest tests/test_fretmap.py tests/test_chart_writer.py -q
```

You want to see all tests passed.

### Full pipeline test (slow, real song)

This runs the same path the app uses on a short public guitar clip in `tests/fixtures/`. It needs FFmpeg and will download models on the first run. Expect several minutes.

```powershell
python -m pytest tests/e2e/test_real_pipeline.py -s
```

If it passes, look in `e2e_output\Acoustic Chords\` for a real Clone Hero folder.

### Several real clips (slow)

This runs the same path on four public clips: solo acoustic, solo electric, mixed band, and a short melody.

```powershell
python tests\e2e\run_clip_suite.py
python -m pytest tests/e2e/test_playwright_clip_suite.py -q
```

Results land in `e2e_output\clip_suite\`. Playwright opens `index.html` and checks that every clip passed with at least one Expert note.

### Single-clip report (optional)

```powershell
python -m pytest tests/e2e/test_playwright_report.py -q
```

## What to expect

- Clear electric or acoustic guitar parts work best.
- Busy mixes, heavy distortion, or stacked guitars will miss notes or add extras.
- The app can split drums, bass, and vocals internally to build the backing track. Version 1 only charts guitar.
- Only use audio you have the right to process.

## Project layout

```
src/app/          Desktop window
src/pipeline/     Isolate guitar, find notes, write the Clone Hero folder
tests/            Fast unit tests and optional real-song tests
tests/fixtures/   Short CC-licensed guitar clip used by the full pipeline test
```

## Credits

- [Demucs](https://github.com/facebookresearch/demucs) and [demucs-onnx](https://github.com/stemsplit/demucs-onnx) isolate the guitar.
- [Basic Pitch](https://github.com/spotify/basic-pitch) finds the notes.
- Test clips are listed in `tests/fixtures/CLIPS.md`.

## License

MIT. See [LICENSE](LICENSE).
