"""
Read-only WAV metadata, checksums, and Welch PSD band-energy analysis.

No filtering, no entropy math here -- this module supports
notebooks/02_audio_inspection.py (metadata table, site-inference table).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
from scipy.signal import welch

try:
    import soundfile as sf

    _HAVE_SOUNDFILE = True
except ImportError:  # pragma: no cover
    import wave

    _HAVE_SOUNDFILE = False


@dataclass
class WavMetadata:
    path: str
    sample_rate: int
    n_samples: int
    duration_s: float
    n_channels: int
    bit_depth: int
    peak_amplitude: float
    rms_amplitude: float
    sha256: str


def read_wav(path: str) -> Tuple[np.ndarray, int]:
    """
    Read a WAV file, returning (samples, sample_rate). If multi-channel,
    channels are averaged to mono. Samples are returned as float64 in the
    normalized range produced by the underlying reader.
    """
    if _HAVE_SOUNDFILE:
        data, sr = sf.read(path, always_2d=False)
        data = np.asarray(data, dtype=np.float64)
        if data.ndim > 1:
            data = data.mean(axis=1)
        return data, int(sr)
    else:  # pragma: no cover -- fallback stdlib path
        with wave.open(path, "rb") as w:
            sr = w.getframerate()
            n = w.getnframes()
            sampwidth = w.getsampwidth()
            nchan = w.getnchannels()
            raw = w.readframes(n)
        dtype_map = {1: np.uint8, 2: np.int16, 4: np.int32}
        dtype = dtype_map.get(sampwidth)
        if dtype is None:
            raise ValueError(f"Unsupported sample width: {sampwidth} bytes")
        arr = np.frombuffer(raw, dtype=dtype).astype(np.float64)
        if nchan > 1:
            arr = arr.reshape(-1, nchan).mean(axis=1)
        if dtype != np.uint8:
            arr = arr / np.iinfo(dtype).max
        else:
            arr = (arr - 128) / 128.0
        return arr, int(sr)


def sha256sum(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def get_wav_metadata(path: str) -> WavMetadata:
    """Extract read-only metadata for a WAV file (no filtering/entropy)."""
    data, sr = read_wav(path)
    n = data.shape[0]
    duration = n / sr if sr else 0.0

    if _HAVE_SOUNDFILE:
        info = sf.info(path)
        n_channels = info.channels
        bit_depth = {
            "PCM_16": 16,
            "PCM_24": 24,
            "PCM_32": 32,
            "PCM_U8": 8,
            "FLOAT": 32,
            "DOUBLE": 64,
        }.get(info.subtype, -1)
    else:  # pragma: no cover
        with wave.open(path, "rb") as w:
            n_channels = w.getnchannels()
            bit_depth = w.getsampwidth() * 8

    return WavMetadata(
        path=str(path),
        sample_rate=sr,
        n_samples=n,
        duration_s=duration,
        n_channels=n_channels,
        bit_depth=bit_depth,
        peak_amplitude=float(np.max(np.abs(data))) if n else 0.0,
        rms_amplitude=float(np.sqrt(np.mean(data**2))) if n else 0.0,
        sha256=sha256sum(path),
    )


def welch_psd(x: np.ndarray, sr: float, nperseg: int = 4096) -> Tuple[np.ndarray, np.ndarray]:
    """Welch power spectral density estimate. Returns (freqs, psd)."""
    nperseg = min(nperseg, len(x))
    freqs, psd = welch(x, fs=sr, nperseg=nperseg)
    return freqs, psd


def band_energy_fraction(
    x: np.ndarray, sr: float, low_hz: float, high_hz: float, nperseg: int = 4096
) -> float:
    """
    Fraction of total PSD energy falling within [low_hz, high_hz], used for
    the site-inference table in notebooks/02_audio_inspection.py (Section 2
    of the build spec: sample-rate + PSD band-energy based site guesses,
    explicitly NOT confirmed metadata).
    """
    freqs, psd = welch_psd(x, sr, nperseg=nperseg)
    _trapezoid = getattr(np, "trapezoid", None) or np.trapz  # numpy>=2.0 renamed trapz
    total = _trapezoid(psd, freqs)
    if total <= 0:
        return 0.0
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    band = _trapezoid(psd[mask], freqs[mask]) if mask.any() else 0.0
    return float(band / total)
