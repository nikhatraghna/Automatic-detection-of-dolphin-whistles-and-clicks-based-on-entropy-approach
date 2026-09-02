"""
Threshold-crossing density counting and confusion-matrix metrics.

FACT (Section 3.4): thresholds H < 0.5 (whistles), SE < 0.6 (clicks);
Eqs. (13)-(14) for accuracy and error rate.

FACT (Section 3.3): whistle "density" binned per 1 minute; click "density"
binned per 30 seconds ("The whistle and click densities were evaluated by
quantifying the number of clicks per every 30-second file" for HB; "The
whistle density based on manual detection was obtained by quantifying the
number of whistles for 1 min" for ETS).

This module builds the FULL pipeline (density counting + confusion matrix +
derived metrics) so it can be tested end-to-end on synthetic data. See
notebooks/07_long_duration_pipeline_BLOCKED.py and
notebooks/08_confusion_matrix_BLOCKED.py: real Table 1 numbers are NOT
reproducible with the 8 short WAV files available (see docs section "What
is NOT reproducible" / Section 3 of the build spec) and are never fabricated
here.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

WHISTLE_DENSITY_BIN_S = 60.0  # FACT: Section 3.3, "quantifying...whistles for 1 min"
CLICK_DENSITY_BIN_S = 30.0  # FACT: Section 3.3, "per every 30-second file"


def count_threshold_crossings(
    values: np.ndarray, times: np.ndarray, threshold: float, bin_seconds: float
) -> "tuple[np.ndarray, np.ndarray]":
    """
    Count the number of threshold-crossing detections (values < threshold)
    falling into fixed time bins, producing a density time series.

    Parameters
    ----------
    values : np.ndarray
        The entropy time series (H or SE), one value per sliding-window
        estimate.
    times : np.ndarray
        Center time (seconds) of each entropy estimate, same length as
        `values`.
    threshold : float
        Detection threshold; a sample counts as a detection iff
        values[i] < threshold (per the paper's H < 0.5 / SE < 0.6
        convention).
    bin_seconds : float
        Width of each density bin, in seconds (60 s for whistles, 30 s for
        clicks per Section 3.3 -- see WHISTLE_DENSITY_BIN_S /
        CLICK_DENSITY_BIN_S).

    Returns
    -------
    (bin_centers, counts) : tuple of np.ndarray
        `bin_centers` are the center times (seconds) of each bin;
        `counts` are the number of threshold-crossing detections in each
        bin.
    """
    values = np.asarray(values, dtype=float)
    times = np.asarray(times, dtype=float)
    if values.shape != times.shape:
        raise ValueError("values and times must have the same shape.")
    if values.size == 0:
        return np.array([]), np.array([], dtype=int)

    is_detection = values < threshold
    t_min, t_max = times.min(), times.max()
    n_bins = max(1, int(np.ceil((t_max - t_min) / bin_seconds)) + 1)
    bin_edges = t_min + np.arange(n_bins + 1) * bin_seconds
    bin_idx = np.clip(
        np.digitize(times, bin_edges[1:-1], right=False), 0, n_bins - 1
    )

    counts = np.zeros(n_bins, dtype=int)
    for b, det in zip(bin_idx, is_detection):
        if det:
            counts[b] += 1

    bin_centers = bin_edges[:-1] + bin_seconds / 2.0
    return bin_centers, counts


def build_confusion_matrix(manual_binary: np.ndarray, detected_binary: np.ndarray) -> Dict[str, int]:
    """
    Build a confusion matrix, per Section 3.4: "we used the manual detection
    of the whistles and clicks as predicted values ... and detection of the
    indices as actual values".

    Parameters
    ----------
    manual_binary : array-like of {0, 1}
        Manual annotation (presence/absence), treated as PREDICTED values,
        matching the paper's own convention (Section 3.4).
    detected_binary : array-like of {0, 1}
        Automated threshold-based detection, treated as ACTUAL values.

    Returns
    -------
    dict with keys TP, FP, TN, FN (int counts).
    """
    manual = np.asarray(manual_binary).astype(bool)
    detected = np.asarray(detected_binary).astype(bool)
    if manual.shape != detected.shape:
        raise ValueError("manual_binary and detected_binary must have the same shape.")

    # Paper's convention: predicted = manual, actual = automated detection.
    tp = int(np.sum(manual & detected))
    fp = int(np.sum(~manual & detected))
    fn = int(np.sum(manual & ~detected))
    tn = int(np.sum(~manual & ~detected))
    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn}


def accuracy_metrics(confusion_dict: Dict[str, int]) -> Dict[str, float]:
    """
    Derive accuracy/error-rate (Eqs. 13-14) plus the standard derived
    metrics precision, recall (sensitivity), and specificity from a
    confusion-matrix dict (paper only names accuracy/error rate explicitly;
    the rest are standard derived metrics added per the build spec).

        Accuracy   = (TP+TN) / (TP+FP+TN+FN)          Eq. (13)
        Error rate = (FN+FP) / (TP+FP+TN+FN)           Eq. (14)
        Precision  = TP / (TP+FP)
        Recall     = TP / (TP+FN)
        Specificity= TN / (TN+FP)
    """
    tp, fp, tn, fn = (
        confusion_dict["TP"],
        confusion_dict["FP"],
        confusion_dict["TN"],
        confusion_dict["FN"],
    )
    total = tp + fp + tn + fn
    if total == 0:
        raise ValueError("Confusion matrix is empty (all counts zero).")

    accuracy = (tp + tn) / total
    error_rate = (fn + fp) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

    return {
        "accuracy": accuracy,
        "error_rate": error_rate,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
    }
