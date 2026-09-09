# Guitar H Isolation

Turn a song on your computer into a custom Clone Hero guitar chart.

You pick an MP3, click Generate, and the app makes a folder Clone Hero can play. It pulls out the guitar and builds the colored notes (green, red, yellow, blue, orange).

The chart is a first draft. It is playable, but it will not look as clean as a chart someone made by hand.

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

## One-time setup (do this once)

You need two free programs: **Python** and **FFmpeg**. Then you install this app.

### A. Install Python

1. Open [https://www.python.org/downloads/](https://www.python.org/downloads/)
2. Click the big yellow **Download Python** button.
3. Run the installer.
4. At the bottom of the first screen, turn on **Add python.exe to PATH**.
5. Click **Install Now**.
6. When it finishes, click **Close**.

### B. Install FFmpeg

1. Press the **Windows** key, type `PowerShell`, and press Enter.
2. Copy this line, paste it, and press Enter:

```powershell
winget install Gyan.FFmpeg
```

3. If Windows asks to finish the install, click **Yes**.
4. Close PowerShell.
5. Open a **new** PowerShell window.
6. Type this and press Enter:

```powershell
ffmpeg -version
```

You should see text that starts with `ffmpeg version`. If you see an error, restart the computer and try that last command again.

### C. Install Guitar H Isolation

1. In the File Explorer window from the download step, click the address bar at the top.
2. Type `powershell` and press Enter. A black or blue window opens in that folder.
3. Copy these three lines, paste them, and press Enter:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

4. Wait. The last line can take several minutes. You will see a lot of text. That is normal.

**If PowerShell says scripts are disabled**, run this once, then run the three lines again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

**If `pip` gets stuck for a long time** (especially if the folder is on OneDrive), press Ctrl+C, then paste these two lines instead:

```powershell
pip install customtkinter demucs-onnx librosa soundfile numpy onnxruntime huggingface-hub pretty-midi resampy scipy scikit-learn soxr tqdm mir-eval
pip install --no-deps basic-pitch
```

When the prompt comes back and there is no red error, setup is done. You can close PowerShell.

---

## How to start the app each time

1. Open the `guitar-h-isolation-main` folder.
2. Double-click `run.bat`.

If a window titled **Guitar H Isolation** opens, you are ready.

If `run.bat` closes right away:

1. In that same folder, click the address bar, type `powershell`, and press Enter.
2. Paste these two lines and press Enter:

```powershell
.\.venv\Scripts\Activate.ps1
python -m src.app.main
```

---

## How to make a Clone Hero song

1. In the app, click **Browse** next to **Audio file**.
2. Pick a song file on your computer. MP3 is the usual choice. WAV, FLAC, OGG, and M4A also work.
3. Check **Song name** and **Artist**. Change them if they look wrong.
4. Leave **Output folder** alone unless you know Clone Hero is in portable mode.

   The default is:

   `Documents\Clone Hero\Songs\` and then your song name

5. Click **Generate**.
6. Wait. The first song is the slowest. It can take several minutes. You will see:

   - Separate (pulls out the guitar)
   - Transcribe (finds the notes)
   - Chart (builds the colored track)
   - Package (saves the folder)

7. When it says **Done**, click **Open folder**.

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
- Busy songs, heavy distortion, or many guitars at once will miss notes or add extras.
- This version only makes a guitar chart. It does not chart drums, bass, or vocals.
- You can clean up the chart later in Moonscraper if you want to share it.

---

## For testers and developers

Fast checks (no song processing):

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
python -m pytest tests/test_fretmap.py tests/test_chart_writer.py -q
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
