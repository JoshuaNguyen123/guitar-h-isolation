from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np

from src.pipeline.types import TempoMap


def estimate_tempo(audio_path: Path) -> TempoMap:
    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    if y.size == 0:
        return TempoMap(bpm=120.0)
    tempo, _beats = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(np.atleast_1d(tempo)[0])
    if not np.isfinite(bpm) or bpm <= 0:
        bpm = 120.0
    while bpm < 70.0:
        bpm *= 2.0
    while bpm > 200.0:
        bpm /= 2.0
    return TempoMap(bpm=round(bpm, 3))
