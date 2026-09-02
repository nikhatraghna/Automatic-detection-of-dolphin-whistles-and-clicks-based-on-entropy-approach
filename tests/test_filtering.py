import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.filtering import bandpass_filter, highpass_filter


def _pure_tone(freq_hz, sr, duration_s=1.0, amp=1.0):
    t = np.arange(int(sr * duration_s)) / sr
    return amp * np.sin(2 * np.pi * freq_hz * t)


def _rms(x):
    return float(np.sqrt(np.mean(x**2)))


def test_bandpass_retains_tone_inside_passband():
    sr = 96000
    tone = _pure_tone(5000, sr)  # inside 3500-8000 Hz ETS whistle band
    filtered = bandpass_filter(tone, sr, 3500, 8000)
    ratio = _rms(filtered) / _rms(tone)
    assert ratio > 0.9  # most amplitude retained


def test_bandpass_attenuates_tone_outside_passband():
    sr = 96000
    tone = _pure_tone(20000, sr)  # well outside 3500-8000 Hz
    filtered = bandpass_filter(tone, sr, 3500, 8000)
    ratio = _rms(filtered) / _rms(tone)
    assert ratio < 0.1  # strongly attenuated


def test_highpass_retains_tone_above_cutoff():
    sr = 64000
    tone = _pure_tone(15000, sr)  # above 6000 Hz cutoff
    filtered = highpass_filter(tone, sr, 6000)
    ratio = _rms(filtered) / _rms(tone)
    assert ratio > 0.9


def test_highpass_attenuates_tone_below_cutoff():
    sr = 64000
    tone = _pure_tone(1000, sr)  # below 6000 Hz cutoff
    filtered = highpass_filter(tone, sr, 6000)
    ratio = _rms(filtered) / _rms(tone)
    assert ratio < 0.1


def test_invalid_band_raises():
    with pytest.raises(ValueError):
        bandpass_filter(np.zeros(1000), 1000, 100, 600)  # 600Hz > Nyquist(500)
