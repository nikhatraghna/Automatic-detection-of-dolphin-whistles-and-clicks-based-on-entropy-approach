import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.permutation_entropy import (
    ordinal_pattern,
    extract_ordinal_patterns,
    permutation_entropy,
    sliding_permutation_entropy,
)


def test_worked_example_from_paper():
    """
    Section 2.1: series {7, 3, 4, 5, 2, 9, ...}, m=4, tau=1.
    First embedding vector (7,3,4,5) -> ordinal pattern (3,0,1,2).
    Second embedding vector (3,4,5,2) -> ordinal pattern (1,2,3,0).
    Non-negotiable, per the build spec: this is the authors' own
    hand-worked example.
    """
    series = [7, 3, 4, 5, 2, 9]
    patterns = extract_ordinal_patterns(series, m=4, tau=1)
    assert patterns[0] == (3, 0, 1, 2)
    assert patterns[1] == (1, 2, 3, 0)


def test_ordinal_pattern_direct():
    """
    Direct check of ordinal_pattern() on the paper's own first embedding
    vector (7,3,4,5): x_{s-3}=7, x_{s-2}=3, x_{s-1}=4, x_s=5. Descending
    value order is 7(lag3), 5(lag0), 4(lag1), 3(lag2) -> pattern (3,0,1,2),
    exactly as stated in Section 2.1.
    """
    assert ordinal_pattern([7, 3, 4, 5]) == (3, 0, 1, 2)


def test_ordinal_pattern_second_vector_direct():
    """Second worked-example vector (3,4,5,2) -> pattern (1,2,3,0)."""
    assert ordinal_pattern([3, 4, 5, 2]) == (1, 2, 3, 0)


def test_synthetic_ordering_ramp_lt_tone_lt_noisy_tone_lt_noise():
    """
    Sanity-ordering check (Rule 5b): H(ramp) < H(pure tone) < H(noisy tone) < H(white noise).
    A monotonic ramp is maximally ordinal-predictable (H~0); white noise is
    maximally disordered (H~1); a clean periodic tone sits low; adding noise
    to the tone raises H toward (but below) pure noise.
    """
    rng = np.random.default_rng(42)
    n = 5000

    ramp = np.arange(n, dtype=float)

    t = np.arange(n) / 1000.0
    tone = np.sin(2 * np.pi * 5 * t)

    noisy_tone = tone + 0.5 * rng.standard_normal(n)

    noise = rng.standard_normal(n)

    m, tau = 4, 1
    H_ramp = permutation_entropy(ramp, m=m, tau=tau)
    H_tone = permutation_entropy(tone, m=m, tau=tau)
    H_noisy_tone = permutation_entropy(noisy_tone, m=m, tau=tau)
    H_noise = permutation_entropy(noise, m=m, tau=tau)

    assert H_ramp < H_tone < H_noisy_tone < H_noise


def test_constant_signal_edge_case():
    """Constant signal: only one ordinal pattern ever observed -> H = 0, no crash."""
    x = np.ones(50)
    H = permutation_entropy(x, m=4, tau=1)
    assert H == pytest.approx(0.0, abs=1e-12)


def test_too_short_input_raises():
    with pytest.raises(ValueError):
        permutation_entropy([1, 2], m=6, tau=1)  # need > (6-1)*1 = 5 samples


def test_normalization_bounds():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(2000)
    H = permutation_entropy(x, m=5, tau=1, normalize=True)
    assert 0.0 <= H <= 1.0 + 1e-9


def test_normalize_false_matches_true_times_ln_factorial():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(3000)
    m = 4
    H_raw = permutation_entropy(x, m=m, tau=1, normalize=False)
    H_norm = permutation_entropy(x, m=m, tau=1, normalize=True)
    assert H_norm == pytest.approx(H_raw / math.log(math.factorial(m)))


def test_sliding_permutation_entropy_shapes_and_mandatory_args():
    rng = np.random.default_rng(2)
    fs = 1000.0
    x = rng.standard_normal(int(fs * 5))  # 5 seconds
    times, H = sliding_permutation_entropy(
        x, fs=fs, window_seconds=0.5, step_seconds=0.1, m=4, tau=1
    )
    assert len(times) == len(H)
    assert len(times) > 0
    assert np.all((H >= 0) & (H <= 1 + 1e-9))
