import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.detrending import moving_window_detrend


def test_linear_trend_removed_within_tolerance():
    """A synthetic signal with a known injected linear trend should have
    that trend removed (mean of each window ~0) within a documented tolerance."""
    rng = np.random.default_rng(5)
    n = 600  # e.g. 600 "samples" at 1 Hz => 10 windows of 60s each
    t = np.arange(n, dtype=float)
    trend = 0.01 * t  # slow linear drift
    noise = 0.05 * rng.standard_normal(n)
    x = trend + noise

    detrended_mean = moving_window_detrend(x, sr=1.0, window_seconds=60.0, method="mean_subtract")
    detrended_linear = moving_window_detrend(x, sr=1.0, window_seconds=60.0, method="linear")

    # Simple global check: overall mean near 0 and much smaller than original trend range.
    assert abs(np.mean(detrended_mean)) < 0.5
    assert abs(np.mean(detrended_linear)) < 0.5
    # Linear detrend should remove the drift better than mean-subtract on a
    # ramp within each window (since ramp has non-zero slope even after
    # mean-subtraction).
    assert np.std(detrended_linear) <= np.std(detrended_mean) + 1e-6


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        moving_window_detrend(np.arange(10.0), sr=1.0, window_seconds=5.0, method="bogus")


def test_output_same_length_as_input():
    x = np.random.default_rng(0).standard_normal(123)
    out = moving_window_detrend(x, sr=2.0, window_seconds=10.0)
    assert out.shape == x.shape
