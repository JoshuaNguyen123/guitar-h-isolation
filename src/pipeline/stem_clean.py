"""Post-Demucs guitar stem cleanup to reduce drum/other bleed."""

from __future__ import annotations

import numpy as np


# Keep energy in the playable guitar band; cut sub-bass kicks and hiss.
HP_HZ = 80.0
LP_HZ = 5000.0
GATE_PERCENTILE = 18.0
GATE_FLOOR = 0.08
HARMONICITY_FRAME_S = 0.046
HARMONICITY_HOP_S = 0.023
BROADBAND_PENALTY = 0.55


def _to_channels_first(audio: np.ndarray) -> np.ndarray:
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 1:
        return arr[np.newaxis, :]
    if arr.shape[0] <= 8 and arr.shape[0] < arr.shape[-1]:
        return arr
    return arr.T


def _butter_sos(sr: int, low_hz: float, high_hz: float):
    from scipy.signal import butter

    nyq = 0.5 * sr
    low = max(low_hz / nyq, 1e-5)
    high = min(high_hz / nyq, 0.999)
    if not (0.0 < low < high < 1.0):
        raise ValueError(f"Invalid band for sr={sr}: {low_hz}-{high_hz} Hz")
    return butter(2, [low, high], btype="band", output="sos")


def _band_limit(audio: np.ndarray, sr: int) -> np.ndarray:
    from scipy.signal import sosfiltfilt

    sos = _butter_sos(sr, HP_HZ, LP_HZ)
    cleaned = np.empty_like(audio)
    for ch in range(audio.shape[0]):
        cleaned[ch] = sosfiltfilt(sos, audio[ch]).astype(np.float32)
    return cleaned


def _frame_matrix(mono: np.ndarray, frame: int, hop: int) -> np.ndarray:
    if mono.size < frame:
        pad = np.zeros(frame, dtype=np.float32)
        pad[: mono.size] = mono
        return pad[np.newaxis, :]
    n = 1 + (mono.size - frame) // hop
    # stride tricks: shape (n, frame)
    out = np.lib.stride_tricks.as_strided(
        mono,
        shape=(n, frame),
        strides=(mono.strides[0] * hop, mono.strides[0]),
        writeable=False,
    )
    return np.asarray(out, dtype=np.float32)


def _frame_rms(frames: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(frames**2, axis=1) + 1e-12).astype(np.float32)


def _harmonicity(frames: np.ndarray, sr: int) -> np.ndarray:
    """Pitched-vs-broadband score in [0, 1] via normalized autocorrelation peak."""
    frame = frames.shape[1]
    centered = frames - np.mean(frames, axis=1, keepdims=True)
    energy = np.sum(centered**2, axis=1) + 1e-12
    min_lag = max(int(sr / 1200.0), 2)
    max_lag = min(int(sr / 80.0), frame - 2)
    if max_lag <= min_lag:
        return np.ones(frames.shape[0], dtype=np.float32)
    best = np.zeros(frames.shape[0], dtype=np.float32)
    for lag in range(min_lag, max_lag):
        corr = np.sum(centered[:, :-lag] * centered[:, lag:], axis=1) / energy
        best = np.maximum(best, corr.astype(np.float32))
    return np.clip(best, 0.0, 1.0)


def _apply_frame_gains(audio: np.ndarray, gains: np.ndarray, hop: int) -> np.ndarray:
    channels, n = audio.shape
    out = np.zeros_like(audio)
    weights = np.zeros(n, dtype=np.float32)
    frame = hop * 2
    for i, gain in enumerate(gains):
        start = i * hop
        end = min(start + frame, n)
        if start >= n:
            break
        out[:, start:end] += audio[:, start:end] * float(gain)
        weights[start:end] += 1.0
    weights = np.maximum(weights, 1e-6)
    out /= weights
    return out.astype(np.float32)


def clean_guitar_stem(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Reduce kick/cymbal bleed and out-of-band noise in a Demucs guitar stem.

    Expects Demucs-style ``(channels, samples)`` float audio. Returns the same layout.
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    channels_first = _to_channels_first(audio)
    if channels_first.size == 0:
        return channels_first

    banded = _band_limit(channels_first, sample_rate)
    mono = np.mean(banded, axis=0).astype(np.float32)
    frame = max(int(round(HARMONICITY_FRAME_S * sample_rate)), 64)
    hop = max(int(round(HARMONICITY_HOP_S * sample_rate)), 32)
    frames = _frame_matrix(mono, frame, hop)
    rms = _frame_rms(frames)
    harm = _harmonicity(frames, sample_rate)

    gate_thresh = float(np.percentile(rms, GATE_PERCENTILE)) if rms.size else 0.0
    gate_thresh = max(gate_thresh, 1e-6)
    gate = GATE_FLOOR + (1.0 - GATE_FLOOR) * np.minimum(1.0, rms / gate_thresh)
    pitch_weight = (1.0 - BROADBAND_PENALTY) + BROADBAND_PENALTY * harm
    gains = (gate * pitch_weight).astype(np.float32)

    cleaned = _apply_frame_gains(banded, gains, hop)
    peak = float(np.max(np.abs(cleaned)))
    src_peak = float(np.max(np.abs(channels_first)))
    if peak > 1e-8 and src_peak > 1e-8:
        cleaned *= min(1.0, src_peak / peak)
    return cleaned.astype(np.float32)


DRUM_SUB_ALPHA = 0.85
DRUM_SUB_FLOOR = 0.18
DRUM_SUB_N_FFT = 2048
DRUM_SUB_HOP = 512


def subtract_drum_bleed(
    guitar: np.ndarray,
    drums: np.ndarray,
    sample_rate: int,
    *,
    alpha: float = DRUM_SUB_ALPHA,
) -> np.ndarray:
    """STFT soft-mask: attenuate guitar bins where drums dominate."""
    import librosa

    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    g = _to_channels_first(guitar)
    d = _to_channels_first(drums)
    if g.size == 0 or d.size == 0:
        return g
    n = min(g.shape[1], d.shape[1])
    g = g[:, :n]
    d = d[:, :n]
    g_mono = np.mean(g, axis=0)
    d_mono = np.mean(d, axis=0)
    G = librosa.stft(g_mono, n_fft=DRUM_SUB_N_FFT, hop_length=DRUM_SUB_HOP)
    D = librosa.stft(d_mono, n_fft=DRUM_SUB_N_FFT, hop_length=DRUM_SUB_HOP)
    mask = np.clip(
        1.0 - alpha * (np.abs(D) / (np.abs(G) + 1e-8)),
        DRUM_SUB_FLOOR,
        1.0,
    )
    out = np.empty_like(g)
    for ch in range(g.shape[0]):
        spec = librosa.stft(g[ch], n_fft=DRUM_SUB_N_FFT, hop_length=DRUM_SUB_HOP)
        rec = librosa.istft(spec * mask, hop_length=DRUM_SUB_HOP, length=n)
        out[ch] = rec.astype(np.float32)
    peak = float(np.max(np.abs(out)))
    src_peak = float(np.max(np.abs(g)))
    if peak > 1e-8 and src_peak > 1e-8:
        out *= min(1.0, src_peak / peak)
    return out.astype(np.float32)


def bleed_proxy_score(guitar: np.ndarray, backing: np.ndarray, sample_rate: int) -> float:
    """Higher = more guitar/backing correlation (worse isolation bleed). Range roughly 0–1."""
    g = np.mean(_to_channels_first(guitar), axis=0)
    b = np.mean(_to_channels_first(backing), axis=0)
    n = min(g.size, b.size)
    if n < max(sample_rate // 10, 1):
        return 0.0
    g = g[:n] - float(np.mean(g[:n]))
    b = b[:n] - float(np.mean(b[:n]))
    denom = float(np.linalg.norm(g) * np.linalg.norm(b)) + 1e-12
    return float(abs(np.dot(g, b)) / denom)
