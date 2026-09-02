"""
Preprocessing filters -- Section 3.1 of the paper ("Data processing").

FACT (Section 3.1):
  - ETS: bandpass 3,500-8,000 Hz for whistles; bandpass 32,000-48,000 Hz
    for clicks.
  - HB: highpass filter at 6,000 Hz (whistles and clicks overlap there,
    6,000-30,000 Hz).

ASSUMPTION (NOT SPECIFIED by the paper):
  - Filter type/order: we use a Butterworth filter, order 4, zero-phase
    (scipy.signal.butter + filtfilt). The paper gives no basis to prefer
    this over any other reasonable choice (e.g. FIR, elliptic, different
    order) -- documented here as an explicit assumption, exposed as a
    swappable parameter (`order`).
  - Edge handling for filtering: we use scipy.signal.filtfilt's default
    padding behaviour (odd-reflection padding, padlen chosen automatically
    from the filter order), also NOT specified by the paper.

Spectrogram parameters (visualization only, PAMGuide/Merchant 2015 tool;
never affects the H/SE math): FFT size 1024 points, 0.05 s time segment,
Hanning window, 90% overlap (Section 3.1). These two parameters are
mutually inconsistent at every sample rate used in this study (a 0.05 s
segment always contains more samples than 1024 at 50/64/96 kHz, and a
standard STFT requires nfft >= nperseg). We resolve this by using
nperseg = nfft = 1024 literally and treat "0.05 s" as an approximate,
unreconciled figure -- see `spectrogram_params()` below. This affects
plotting only.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.signal import butter, filtfilt

# ASSUMPTION: Butterworth filter, order 4 (paper does not specify filter
# type or order for the bandpass/highpass steps in Section 3.1).
DEFAULT_FILTER_ORDER = 4

# FACT (Section 3.1): ETS bandpass bands.
ETS_WHISTLE_BAND_HZ = (3500.0, 8000.0)
ETS_CLICK_BAND_HZ = (32000.0, 48000.0)

# FACT (Section 3.1): HB highpass cutoff.
HB_HIGHPASS_HZ = 6000.0

# FACT (Section 3.1, PAMGuide/Merchant 2015): spectrogram parameters.
# ASSUMPTION (reconciliation): nfft = nperseg = 1024 taken literally; the
# "0.05 s" segment-length figure is not arithmetically consistent with
# 1024 points at 50/64/96 kHz and is treated as approximate. Visualization
# only.
SPECTROGRAM_NFFT = 1024
SPECTROGRAM_WINDOW = "hann"
SPECTROGRAM_OVERLAP_FRAC = 0.90
SPECTROGRAM_NOMINAL_SEGMENT_S = 0.05  # stated by paper; not reconciled, see above


def bandpass_filter(
    x: np.ndarray, sr: float, low_hz: float, high_hz: float, order: int = DEFAULT_FILTER_ORDER
) -> np.ndarray:
    """
    Zero-phase Butterworth bandpass filter (ASSUMPTION: filter type/order
    not specified by the paper; see module docstring).

    Parameters
    ----------
    x : np.ndarray
        Input signal.
    sr : float
        Sampling rate in Hz.
    low_hz, high_hz : float
        Passband edges in Hz.
    order : int
        Butterworth filter order (default 4, ASSUMPTION).

    Returns
    -------
    np.ndarray
        Filtered signal, same length as `x` (filtfilt default edge
        handling -- ASSUMPTION, not specified by the paper).
    """
    nyq = sr / 2.0
    if not (0 < low_hz < high_hz < nyq):
        raise ValueError(
            f"Invalid band ({low_hz}, {high_hz}) Hz for sampling rate {sr} Hz "
            f"(Nyquist = {nyq} Hz)."
        )
    b, a = butter(order, [low_hz / nyq, high_hz / nyq], btype="bandpass")
    return filtfilt(b, a, x)  # default padding/edge handling, ASSUMPTION


def highpass_filter(
    x: np.ndarray, sr: float, cutoff_hz: float, order: int = DEFAULT_FILTER_ORDER
) -> np.ndarray:
    """
    Zero-phase Butterworth highpass filter (ASSUMPTION: filter type/order
    not specified by the paper; see module docstring).
    """
    nyq = sr / 2.0
    if not (0 < cutoff_hz < nyq):
        raise ValueError(
            f"Invalid cutoff {cutoff_hz} Hz for sampling rate {sr} Hz "
            f"(Nyquist = {nyq} Hz)."
        )
    b, a = butter(order, cutoff_hz / nyq, btype="highpass")
    return filtfilt(b, a, x)


def spectrogram_params() -> dict:
    """
    Return the (partially irreconcilable, see module docstring) PAMGuide
    spectrogram parameters stated in Section 3.1, for use by plotting code
    only. Never used for H/SE computation.
    """
    return {
        "nfft": SPECTROGRAM_NFFT,
        "nperseg": SPECTROGRAM_NFFT,  # ASSUMPTION: literal nfft=nperseg=1024
        "window": SPECTROGRAM_WINDOW,
        "overlap_frac": SPECTROGRAM_OVERLAP_FRAC,
        "nominal_segment_s_unreconciled": SPECTROGRAM_NOMINAL_SEGMENT_S,
    }
