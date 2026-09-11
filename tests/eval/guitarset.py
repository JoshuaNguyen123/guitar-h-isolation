"""GuitarSet v1.1.0 download, JAMS ground truth, and band-mix rendering.

Dataset: Xi et al., Zenodo 3371780. Annotations are MIT / CC-BY 4.0.
Audio is cached under ``.cache/guitarset/`` (gitignored) and is not redistributed.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import soundfile as sf

from src.pipeline.types import NoteEvent
from tests.eval.synthesize import SR, _bass_hits, _drum_hits, midi_to_hz

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / ".cache" / "guitarset"
ZENODO = "https://zenodo.org/record/3371780/files"

REMOTES = {
    "annotations": {
        "filename": "annotation.zip",
        "url": f"{ZENODO}/annotation.zip?download=1",
        "md5": "b39b78e63d3446f2e54ddb7a54df9b10",
        "extract_dir": "annotation",
    },
    "audio_mic": {
        "filename": "audio_mono-mic.zip",
        "url": f"{ZENODO}/audio_mono-mic.zip?download=1",
        "md5": "275966d6610ac34999b58426beb119c3",
        "extract_dir": "audio_mono-mic",
    },
}

# Rock / Funk, several players, comp + solo, both tempi families.
PREFERRED_TRACKS = (
    "00_Rock1-90-C#_comp",
    "00_Rock1-90-C#_solo",
    "00_Rock2-142-D_comp",
    "00_Rock3-148-C_solo",
    "02_Rock1-130-A_comp",
    "03_Funk1-114-Ab_comp",
    "03_Funk1-114-Ab_solo",
    "03_Funk2-119-G_solo",
    "04_Funk3-98-A_comp",
    "01_Rock2-85-F#_solo",
    "05_SS2-107-Ab_comp",
    "02_Funk3-112-C#_solo",
)

MIN_SUBSET = 8


def annotation_dir() -> Path:
    return CACHE_DIR / "annotation"


def mic_dir() -> Path:
    return CACHE_DIR / "audio_mono-mic"


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urlopen(url, timeout=120) as resp, tmp.open("wb") as out:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    tmp.replace(dest)


def ensure_remote(name: str) -> Path:
    meta = REMOTES[name]
    extracted = CACHE_DIR / meta["extract_dir"]
    if extracted.is_dir() and any(extracted.rglob("*")):
        return extracted
    archive = CACHE_DIR / meta["filename"]
    if not archive.is_file() or _md5(archive) != meta["md5"]:
        print(f"Downloading GuitarSet {meta['filename']}...")
        _download(meta["url"], archive)
        if _md5(archive) != meta["md5"]:
            raise RuntimeError(f"Checksum mismatch for {archive}")
    extracted.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(extracted)
    return extracted


def dataset_ready() -> bool:
    return annotation_dir().is_dir() and mic_dir().is_dir()


def ensure_guitarset() -> Path:
    ensure_remote("annotations")
    ensure_remote("audio_mic")
    return CACHE_DIR


def _find_file(root: Path, stem: str, suffixes: tuple[str, ...]) -> Path | None:
    matches: list[Path] = []
    for path in root.rglob("*"):
        if path.suffix.lower() not in suffixes:
            continue
        if path.stem == stem or path.stem.startswith(stem):
            matches.append(path)
    if not matches:
        return None
    exact = [path for path in matches if path.stem == stem]
    return (exact or matches)[0]


def list_track_ids() -> list[str]:
    if not annotation_dir().is_dir():
        return []
    ids = sorted({path.stem for path in annotation_dir().rglob("*.jams")})
    preferred = [tid for tid in PREFERRED_TRACKS if tid in ids]
    if len(preferred) >= MIN_SUBSET:
        return preferred
    extras = [tid for tid in ids if _is_eval_style(tid) and tid not in preferred]
    return (preferred + extras)[:12]


def _is_eval_style(track_id: str) -> bool:
    upper = track_id.upper()
    return any(tag in upper for tag in ("ROCK", "FUNK", "_SS"))


def jams_path(track_id: str) -> Path | None:
    return _find_file(annotation_dir(), track_id, (".jams",))


def mic_path(track_id: str) -> Path | None:
    return _find_file(mic_dir(), track_id, (".wav", ".flac"))


def notes_from_jams(path: Path) -> list[NoteEvent]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    notes: list[NoteEvent] = []
    for ann in data.get("annotations", []):
        if ann.get("namespace") != "note_midi":
            continue
        for obs in ann.get("data", []):
            start = float(obs["time"])
            duration = float(obs["duration"])
            midi = int(round(float(obs["value"])))
            if midi < 40 or midi > 88 or duration <= 0:
                continue
            notes.append(
                NoteEvent(
                    start_s=start,
                    end_s=start + duration,
                    midi_pitch=midi,
                    velocity=1.0,
                )
            )
    notes.sort(key=lambda n: (n.start_s, n.midi_pitch))
    return notes


def beats_from_jams(path: Path) -> list[float]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    beats: list[float] = []
    for ann in data.get("annotations", []):
        if ann.get("namespace") != "beat":
            continue
        for obs in ann.get("data", []):
            beats.append(float(obs["time"]))
    return sorted(beats)


def _hits_at_times(
    n_samples: int,
    sr: int,
    times: list[float],
    *,
    kind: str,
) -> np.ndarray:
    out = np.zeros(n_samples, dtype=np.float32)
    if not times:
        if kind == "drums":
            return _drum_hits(n_samples, sr, period_s=0.5)
        return _bass_hits(n_samples, sr, period_s=1.0)
    rng = np.random.default_rng(1)
    for i, start_s in enumerate(times):
        start = int(start_s * sr)
        if start >= n_samples:
            continue
        if kind == "drums":
            n = int(0.08 * sr)
            t = np.arange(n, dtype=np.float32) / sr
            if i % 2 == 0:
                hit = (np.sin(2 * np.pi * 70 * t) * np.exp(-t * 28)).astype(np.float32)
            else:
                hit = (rng.standard_normal(n).astype(np.float32) * np.exp(-t * 40) * 0.7)
            hit = hit * 0.9
        else:
            midi = 40 if i % 2 == 0 else 45
            freq = midi_to_hz(midi)
            n = int(0.42 * sr)
            t = np.arange(n, dtype=np.float32) / sr
            hit = np.zeros(n, dtype=np.float32)
            for harm, amp in ((1, 1.0), (2, 0.4)):
                hit += (amp * np.sin(2 * np.pi * freq * harm * t)).astype(np.float32)
            hit *= np.exp(-t * 3.8).astype(np.float32) * 0.35
        end = min(start + hit.size, n_samples)
        out[start:end] += hit[: end - start]
    peak = float(np.max(np.abs(out))) + 1e-8
    return (out / peak * (0.7 if kind == "drums" else 0.55)).astype(np.float32)


def render_band_mix(mic_wav: Path, jams: Path, out_wav: Path) -> tuple[Path, Path]:
    guitar, sr = sf.read(str(mic_wav), always_2d=False)
    guitar = np.asarray(guitar, dtype=np.float32)
    if guitar.ndim > 1:
        guitar = np.mean(guitar, axis=1)
    if sr != SR:
        import librosa

        guitar = librosa.resample(guitar, orig_sr=sr, target_sr=SR).astype(np.float32)
        sr = SR
    beats = beats_from_jams(jams)
    drums = _hits_at_times(guitar.size, sr, beats, kind="drums")
    bass = _hits_at_times(guitar.size, sr, beats[::2] if beats else [], kind="bass")
    mix = guitar + drums * 0.75 + bass * 0.40
    peak = float(np.max(np.abs(mix))) + 1e-8
    mix = (mix / peak * 0.9).astype(np.float32)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    drums_wav = out_wav.with_name(out_wav.stem + "_drums.wav")
    sf.write(str(out_wav), mix, sr)
    sf.write(str(drums_wav), drums, sr)
    return out_wav, drums_wav


def iter_eval_items() -> list[dict]:
    items = []
    for track_id in list_track_ids():
        jams = jams_path(track_id)
        mic = mic_path(track_id)
        if jams is None or mic is None:
            continue
        truth = notes_from_jams(jams)
        if len(truth) < 8:
            continue
        items.append(
            {
                "track_id": track_id,
                "jams": jams,
                "mic": mic,
                "truth": truth,
            }
        )
    return items
