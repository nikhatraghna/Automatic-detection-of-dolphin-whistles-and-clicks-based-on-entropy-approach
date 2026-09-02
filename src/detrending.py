"""
Long-duration H detrending -- Section 3.3.

FACT (Section 3.3): "A moving de-trending window (~1 min) was utilized to
detrend the estimated H."

ASSUMPTION (NOT SPECIFIED by the paper): the exact detrending algorithm
(mean-subtraction? linear/polynomial fit per window? high-pass filter?) is
not given. We implement mean-subtraction with a 1-minute moving window as
the primary documented assumption, with an alternative ("linear" per-window
detrend) exposed via the `method` parameter for a sensitivity comparison,
per the build spec.
"""

from __future__ import annotations

import numpy as np

# ASSUMPTION: moving detrending window length. FACT only says "~1 min"; we
# take this literally as 60.0 seconds.
DEFAULT_DETREND_WINDOW_S = 60.0


def _moving_window_indices(n: int, win_n: int) -> list:
    """Non-overlapping window index ranges covering [0, n), last window
    truncated if n is not a multiple of win_n."""
    return [(s, min(s + win_n, n)) for s in range(0, n, win_n)]


def moving_window_detrend(
    x: np.ndarray,
    sr: float,
    window_seconds: float = DEFAULT_DETREND_WINDOW_S,
    method: str = "mean_subtract",
) -> np.ndarray:
    """
    Detrend a (typically long-duration) H(t) or SE(t) time series using a
    moving window, per Section 3.3.

    Parameters
    ----------
    x : np.ndarray
        The series to detrend (e.g. H values sampled at the sliding-window
        cadence, treated here as "samples" at effective rate `sr`).
    sr : float
        Effective sampling rate of `x` in Hz (e.g. 1/step_seconds for a
        sliding-window entropy series).
    window_seconds : float
        Detrending window length. ASSUMPTION: paper states "~1 min"; we
        default to 60.0 s but expose this as swappable.
    method : {"mean_subtract", "linear"}
        - "mean_subtract" (ASSUMPTION, primary/documented default):
          subtract the local window mean from every sample in that window.
        - "linear": subtract a per-window linear (degree-1 polynomial) fit,
          exposed as an alternative for sensitivity comparison, per the
          build spec's request for a swappable detrending method.

    Returns
    -------
    np.ndarray
        Detrended series, same length as `x`.
    """
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    win_n = max(1, int(round(window_seconds * sr)))
    out = np.empty_like(x)

    for start, end in _moving_window_indices(n, win_n):
        seg = x[start:end]
        if method == "mean_subtract":
            out[start:end] = seg - seg.mean()
        elif method == "linear":
            if end - start < 2:
                out[start:end] = seg - seg.mean()
            else:
                t = np.arange(end - start, dtype=float)
                coeffs = np.polyfit(t, seg, deg=1)
                trend = np.polyval(coeffs, t)
                out[start:end] = seg - trend
        else:
            raise ValueError(f"Unknown detrend method: {method!r}")

    return out
