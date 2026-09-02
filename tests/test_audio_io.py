import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.audio_io import get_wav_metadata, read_wav, sha256sum

sf = pytest.importorskip("soundfile")


def test_metadata_matches_synthetic_wav(tmp_path):
    sr = 48000
    duration_s = 2.0
    amp = 0.5
    t = np.arange(int(sr * duration_s)) / sr
    x = (amp * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    wav_path = tmp_path / "synthetic.wav"
    sf.write(str(wav_path), x, sr, subtype="PCM_16")

    meta = get_wav_metadata(str(wav_path))
    assert meta.sample_rate == sr
    assert meta.duration_s == pytest.approx(duration_s, abs=0.01)
    assert meta.n_channels == 1
    assert meta.peak_amplitude == pytest.approx(amp, abs=0.02)
    assert meta.sha256 == sha256sum(str(wav_path))


def test_read_wav_roundtrip(tmp_path):
    sr = 22050
    x = (0.3 * np.random.default_rng(0).standard_normal(sr)).astype(np.float32)
    wav_path = tmp_path / "noise.wav"
    sf.write(str(wav_path), x, sr, subtype="PCM_16")

    data, read_sr = read_wav(str(wav_path))
    assert read_sr == sr
    assert len(data) == len(x)
