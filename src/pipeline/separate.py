from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from src.pipeline.audio import decode_to_wav


BACKING_STEMS = ("drums", "bass", "vocals", "piano", "other")


@dataclass
class SeparationResult:
    guitar_wav: Path
    backing_wav: Path
    sample_rate: int


def _import_separate():
    try:
        from demucs_onnx import separate
    except ImportError:
        from demucs_onnx.inference import separate  # type: ignore
    return separate


def isolate_guitar(input_path: Path, work_dir: Path) -> SeparationResult:
    work_dir.mkdir(parents=True, exist_ok=True)
    mix_wav = work_dir / "mix.wav"
    decode_to_wav(input_path, mix_wav)

    separate = _import_separate()
    stems = separate(
        str(mix_wav),
        output_dir=None,
        model="htdemucs_6s",
        progress=True,
    )
    if "guitar" not in stems:
        available = ", ".join(sorted(stems)) or "(none)"
        raise RuntimeError(f"Demucs did not return a guitar stem. Got: {available}")

    guitar = np.asarray(stems["guitar"], dtype=np.float32)
    backing = None
    for name in BACKING_STEMS:
        if name not in stems:
            continue
        stem = np.asarray(stems[name], dtype=np.float32)
        backing = stem.copy() if backing is None else backing + stem
    if backing is None:
        raise RuntimeError("Demucs did not return backing stems to rebuild the mix.")

    peak = max(float(np.max(np.abs(guitar))), float(np.max(np.abs(backing))), 1e-6)
    if peak > 1.0:
        guitar = guitar / peak
        backing = backing / peak

    sample_rate = int(sf.info(str(mix_wav)).samplerate)
    guitar_wav = work_dir / "guitar.wav"
    backing_wav = work_dir / "backing.wav"
    sf.write(str(guitar_wav), guitar.T, sample_rate)
    sf.write(str(backing_wav), backing.T, sample_rate)
    return SeparationResult(
        guitar_wav=guitar_wav,
        backing_wav=backing_wav,
        sample_rate=sample_rate,
    )
