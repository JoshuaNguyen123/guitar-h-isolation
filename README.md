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

## How the app actually works

The full write-up is in [ARCHITECTURE.md](ARCHITECTURE.md), including the [architecture diagram](architecture-diagram.svg). The same document is also [ARCHITECTURE.doc](ARCHITECTURE.doc) if you want to open it in Word.

Short version:

1. FFmpeg turns your file into a WAV.
2. Demucs (`htdemucs_6s`) splits the mix into six stems. The first run downloads that model from Hugging Face (that is the slow “chunk” download). Later songs reuse the cached file.
3. `song.ogg` is the band **without** guitar. `guitar.ogg` is the isolated guitar. Clone Hero plays both so the guitar is not doubled.
4. Basic Pitch guesses notes from the guitar stem. That is pitch detection, not real guitar tab.
5. Those notes are squashed onto five Clone Hero frets, then thinned for Hard, Medium, and Easy.
6. Tempo is one BPM number for the whole song.

The chart is a machine draft. It is not official Guitar Hero or Clone Hero content.

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
