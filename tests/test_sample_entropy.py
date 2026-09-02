import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.sample_entropy import sample_entropy, sliding_sample_entropy


def _brute_force_sample_entropy(y, d=2, r=None, r_factor=0.5):
    """
    Independent, deliberately-naive (pure Python double loop) re-implementation
    of Eqs. (5)-(12), used ONLY as a cross-check in this one test -- never as
    the shipped implementation (per reproduction Rule 1: "may only appear in
    an optional, clearly-labeled cross-validation cell").
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    if r is None:
        r = r_factor * np.std(y, ddof=0)

    def embed(dim):
        return [tuple(y[i : i + dim]) for i in range(n - dim + 1)]

    def corr_sum(dim):
        vecs = embed(dim)
        N = len(vecs)
        row_means = []
        for i in range(N):
            cnt = 0
            for j in range(N):
                if i == j:
                    continue
                dist = max(abs(a - b) for a, b in zip(vecs[i], vecs[j]))
                if dist <= r:
                    cnt += 1
            row_means.append(cnt / (N - 1))
        return sum(row_means) / N

    Cm = corr_sum(d)
    Dm = corr_sum(d + 1)
    if Cm == 0 or Dm == 0:
        return float("inf")
    return -np.log(Dm / Cm)


def test_hand_checked_against_independent_brute_force():
    """
    Worked/hand-checkable example: a short synthetic series where our
    vectorized implementation is cross-checked against an independent,
    deliberately-naive brute-force re-implementation of the same equations
    (used only for this one validation test).
    """
    rng = np.random.default_rng(7)
    y = rng.standard_normal(30)
    got = sample_entropy(y, d=2, r_factor=0.5)
    expected = _brute_force_sample_entropy(y, d=2, r_factor=0.5)
    assert got == pytest.approx(expected, rel=1e-9, abs=1e-9)


def test_synthetic_ordering_click_lowers_se():
    """
    Synthetic ordering check: an impulsive/click-like transient embedded in
    an otherwise quiet/regular background should show LOW SE in a window
    containing the transient and HIGHER SE in a window of quiet background,
    mirroring the paper's central claim that clicks depress SE.
    """
    rng = np.random.default_rng(3)
    n = 400
    quiet = 0.01 * rng.standard_normal(n)  # near-silent, low-variance background

    click_like = quiet.copy()
    # Sharp impulsive spikes at regular intervals -- a repeating, low-SE pattern.
    for i in range(0, n, 20):
        click_like[i] = 5.0

    se_quiet = sample_entropy(quiet, d=2, r_factor=0.5)
    se_click = sample_entropy(click_like, d=2, r_factor=0.5)

    assert se_click < se_quiet


def test_edge_case_all_identical_values():
    """All-identical-values edge case: SD=0 -> r cannot be derived, must raise clearly."""
    x = np.full(20, 3.0)
    with pytest.raises(ValueError):
        sample_entropy(x, d=2)


def test_edge_case_n_too_small_for_embedding():
    with pytest.raises(ValueError):
        sample_entropy([1.0, 2.0], d=2)  # need >= d+2 = 4 samples


def test_explicit_r_overrides_r_factor():
    rng = np.random.default_rng(11)
    y = rng.standard_normal(50)
    se_default = sample_entropy(y, d=2, r_factor=0.5)
    se_explicit = sample_entropy(y, d=2, r=0.5 * np.std(y, ddof=0))
    assert se_default == pytest.approx(se_explicit)


def test_sliding_sample_entropy_shapes_and_mandatory_args():
    rng = np.random.default_rng(4)
    fs = 1000.0
    y = rng.standard_normal(int(fs * 2))
    times, se = sliding_sample_entropy(y, fs=fs, window_seconds=0.2, step_seconds=0.05, d=2)
    assert len(times) == len(se)
    assert len(times) > 0
