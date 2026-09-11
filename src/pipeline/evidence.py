"""One evidence score per Basic Pitch note. Drum alignment is never a veto."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.pipeline.types import NoteEvent, ScoredNote

ANALYSIS_SR = 22050
ANNOTATIONS_FPS = 86.0
MIDI_OFFSET = 21
NOTE_BINS = 88
SUSTAIN_LO_S = 0.06
SUSTAIN_HI_S = 0.20
ONSET_WIN_S = 0.04
BASS_MIDI_MAX = 45
WEIGHT_VELOCITY = 0.35
WEIGHT_SUSTAIN = 0.55
WEIGHT_DRUM = 0.35
WEIGHT_BASS = 0.25
WEIGHT_PYIN = 0.08
PYIN_CENTS = 50.0


def detect_drum_onsets(drums_wav: Path) -> np.ndarray:
    import librosa

    y, sr = librosa.load(str(drums_wav), sr=ANALYSIS_SR, mono=True)
    if y.size == 0:
        return np.zeros(0, dtype=np.float64)
    onsets = librosa.onset.onset_detect(
        y=y,
        sr=sr,
        units="time",
        backtrack=True,
        delta=0.07,
    )
    lead = y[: max(int(0.04 * sr), 1)]
    if float(np.max(np.abs(lead))) > 0.08:
        onsets = np.unique(np.concatenate(([0.0], np.asarray(onsets, dtype=np.float64))))
    return np.asarray(onsets, dtype=np.float64)


def _load_mono(path: Path | None) -> tuple[np.ndarray, int]:
    if path is None or not Path(path).is_file():
        return np.zeros(0, dtype=np.float32), ANALYSIS_SR
    import librosa

    y, sr = librosa.load(str(path), sr=ANALYSIS_SR, mono=True)
    return np.asarray(y, dtype=np.float32), int(sr)


def _midi_to_hz(midi: float) -> float:
    return 440.0 * (2.0 ** ((float(midi) - 69.0) / 12.0))


def _rms(y: np.ndarray, sr: int, start_s: float, end_s: float) -> float:
    if y.size == 0 or sr <= 0 or end_s <= start_s:
        return 0.0
    i0 = max(0, int(round(start_s * sr)))
    i1 = min(y.size, max(i0 + 1, int(round(end_s * sr))))
    seg = y[i0:i1]
    if seg.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(seg, dtype=np.float64)) + 1e-12))


def _band_rms(y: np.ndarray, sr: int, start_s: float, end_s: float, midi: int) -> float:
    if y.size == 0 or sr <= 0 or end_s <= start_s:
        return 0.0
    i0 = max(0, int(round(start_s * sr)))
    i1 = min(y.size, max(i0 + 32, int(round(end_s * sr))))
    seg = np.asarray(y[i0:i1], dtype=np.float64)
    if seg.size < 32:
        return 0.0
    spec = np.abs(np.fft.rfft(seg * np.hanning(seg.size)))
    freqs = np.fft.rfftfreq(seg.size, 1.0 / sr)
    freq = _midi_to_hz(midi)
    lo = freq * (2.0 ** (-80.0 / 1200.0))
    hi = freq * (2.0 ** (80.0 / 1200.0))
    mask = (freqs >= lo) & (freqs <= hi)
    if not np.any(mask):
        return 0.0
    return float(np.sqrt(np.mean(np.square(spec[mask])) + 1e-12))


def _note_matrix(model_output: dict | None) -> np.ndarray | None:
    if not model_output:
        return None
    raw = model_output.get("note")
    if raw is None:
        return None
    mat = np.asarray(raw, dtype=np.float64)
    if mat.ndim != 2:
        return None
    if mat.shape[1] != NOTE_BINS and mat.shape[0] == NOTE_BINS:
        mat = mat.T
    return mat


def posterior_sustain(model_output: dict | None, note: NoteEvent) -> float | None:
    mat = _note_matrix(model_output)
    if mat is None:
        return None
    idx = int(note.midi_pitch) - MIDI_OFFSET
    if idx < 0 or idx >= mat.shape[1]:
        return 0.0
    duration = note.end_s - note.start_s
    if duration < SUSTAIN_HI_S:
        lo = int(round(note.start_s * ANNOTATIONS_FPS))
        hi = int(round(note.end_s * ANNOTATIONS_FPS))
    else:
        lo = int(round((note.start_s + SUSTAIN_LO_S) * ANNOTATIONS_FPS))
        hi = int(round(min(note.end_s, note.start_s + SUSTAIN_HI_S) * ANNOTATIONS_FPS))
    lo = max(0, lo)
    hi = min(mat.shape[0], max(lo + 1, hi))
    if hi <= lo:
        return 0.0
    return float(np.clip(np.mean(mat[lo:hi, idx]), 0.0, 1.0))


def audio_sustain(guitar: np.ndarray, sr: int, note: NoteEvent) -> float:
    """Fallback when Basic Pitch's note posterior is missing (unit tests)."""
    sus_hi = min(note.end_s, note.start_s + SUSTAIN_HI_S)
    sus_lo = min(note.start_s + SUSTAIN_LO_S, sus_hi)
    if sus_hi - sus_lo < 0.04:
        return 0.0
    sustain = _band_rms(guitar, sr, sus_lo, sus_hi, note.midi_pitch)
    onset = _band_rms(guitar, sr, note.start_s, note.start_s + 0.045, note.midi_pitch)
    peak = float(np.max(np.abs(guitar))) + 1e-8 if guitar.size else 1e-8
    if sustain < 0.02 * peak:
        return 0.0
    return float(np.clip(sustain / (onset + 1e-8), 0.0, 1.0))


def _dominance(num: float, den: float) -> float:
    return float(np.clip(num / (den + 1e-8), 0.0, 4.0))


def _pyin_votes(notes: list[NoteEvent], guitar_wav: Path) -> dict[int, float]:
    """Weak positive votes only. Missing or wrong pyin never subtracts."""
    votes = {i: 0.0 for i in range(len(notes))}
    if not notes or not Path(guitar_wav).is_file():
        return votes
    import librosa

    y, sr = librosa.load(str(guitar_wav), sr=ANALYSIS_SR, mono=True)
    if y.size == 0:
        return votes
    f0, voiced_flag, _prob = librosa.pyin(y, fmin=82.0, fmax=1318.5, sr=sr)
    times = librosa.times_like(f0, sr=sr)
    f0 = np.asarray(f0, dtype=np.float64)
    if voiced_flag is not None:
        f0 = np.where(voiced_flag, f0, np.nan)
    for i, note in enumerate(notes):
        lo = note.start_s
        hi = max(note.start_s + 0.06, min(note.end_s, note.start_s + 0.12))
        mask = (times >= lo) & (times <= hi) & np.isfinite(f0) & (f0 > 0)
        if not np.any(mask):
            continue
        median_midi = 69.0 + 12.0 * np.log2(float(np.median(f0[mask])) / 440.0)
        cents = abs(median_midi - note.midi_pitch) * 100.0
        if cents <= PYIN_CENTS:
            votes[i] = 1.0
    return votes


def score_notes(
    notes: list[NoteEvent],
    model_output: dict | None,
    guitar_wav: Path,
    drums_wav: Path | None = None,
    bass_wav: Path | None = None,
    sample_rate: int = ANALYSIS_SR,
    use_pyin: bool = True,
) -> list[ScoredNote]:
    del sample_rate
    guitar, g_sr = _load_mono(Path(guitar_wav) if guitar_wav else None)
    drums, d_sr = _load_mono(Path(drums_wav) if drums_wav else None)
    bass, b_sr = _load_mono(Path(bass_wav) if bass_wav else None)
    votes = (
        _pyin_votes(notes, Path(guitar_wav))
        if use_pyin and guitar_wav
        else {i: 0.0 for i in range(len(notes))}
    )
    scored: list[ScoredNote] = []
    for i, note in enumerate(notes):
        sustain = posterior_sustain(model_output, note)
        if sustain is None:
            sustain = audio_sustain(guitar, g_sr, note)
        onset_lo = note.start_s - 0.02
        onset_hi = note.start_s + ONSET_WIN_S
        g_on = _band_rms(guitar, g_sr, onset_lo, onset_hi, note.midi_pitch)
        d_on = _rms(drums, d_sr, onset_lo, onset_hi)
        drum_dom = _dominance(d_on, g_on) if drums.size else 0.0
        bass_dom = 0.0
        if bass.size and note.midi_pitch <= BASS_MIDI_MAX:
            b_on = _band_rms(bass, b_sr, onset_lo, onset_hi, note.midi_pitch)
            bass_dom = _dominance(b_on, g_on)
        # Drums/bass only penalize when the note does not sustain.
        bleed = max(0.0, 0.30 - sustain)
        score = (
            WEIGHT_VELOCITY * note.velocity
            + WEIGHT_SUSTAIN * sustain
            - WEIGHT_DRUM * drum_dom * bleed
            - WEIGHT_BASS * bass_dom * bleed
            + WEIGHT_PYIN * votes[i]
        )
        duration = note.end_s - note.start_s
        if duration < 0.11 and sustain < 0.15:
            score -= 0.20
        scored.append(
            ScoredNote(
                note=note,
                velocity=note.velocity,
                sustain=float(sustain),
                drum_dominance=drum_dom,
                bass_dominance=bass_dom,
                pyin_vote=votes[i],
                score=float(score),
            )
        )
    return scored


def keep_scored(scored: list[ScoredNote], threshold: float) -> list[NoteEvent]:
    kept = [row.note for row in scored if row.score >= threshold]
    if kept:
        return kept
    if not scored:
        return []
    best = max(scored, key=lambda row: row.score)
    if best.sustain >= 0.30 or best.score >= threshold - 0.08:
        return [best.note]
    return [row.note for row in scored]
