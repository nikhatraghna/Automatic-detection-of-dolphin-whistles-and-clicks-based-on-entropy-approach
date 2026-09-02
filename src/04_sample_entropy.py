# %% [markdown]
# # 04 — Sample Entropy (SE): Validation and Application
#
# Reproduces: Section 2.2, Eqs. (5)-(12). Known deviation: sliding window
# length is NOT SPECIFIED and is swept explicitly below rather than
# hard-coded. See docs/paper_parameters.md.

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.audio_io import read_wav
from src.sample_entropy import (
    DEFAULT_D,
    DEFAULT_R_FACTOR,
    sample_entropy,
    sliding_sample_entropy,
)

RAW_DIR = REPO_ROOT / "data" / "raw"
FIG_DIR = REPO_ROOT / "results" / "figures"
TABLE_DIR = REPO_ROOT / "results" / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

CLICK_FILES = ["Click1.wav", "Click2.wav", "Click3.wav", "Click4.wav"]

# NOT SPECIFIED by the paper -- swept explicitly rather than hard-coded.
# ASSUMPTION (performance-driven, documented): sample entropy is O(N^2)
# per window (full pairwise max-norm distance matrix, twice per window --
# see src/sample_entropy.py). At real click-recording sample rates
# (64-96 kHz), a 0.1s window is 6,400-9,600 samples and costs several
# seconds PER WINDOW; with ~100+ windows per file at the paper's ~0.0125s
# step, that is 10s of minutes PER (file, window-size) combination -- not a
# hang, just genuinely expensive brute-force computation. We therefore
# default this sweep to sub-20ms-per-window window sizes (still spanning a
# plausible range for detecting brief click transients). To reproduce the
# original, much larger sweep (e.g. up to 0.1s), see the commented-out line
# below and budget accordingly (tens of minutes to hours; consider
# increasing STEP_S too, since it directly multiplies the number of
# O(N^2) window evaluations).
WINDOW_SWEEP_S = [0.005, 0.01, 0.02]
# WINDOW_SWEEP_S = [0.05, 0.1, 0.2]  # SLOW: budget tens of minutes+ per file, see note above

STEP_S = 0.0125  # FACT-derived: midpoint of the Discussion's "~0.011-0.014 s" range


def _brute_force_sample_entropy(y, d=2, r_factor=0.5):
    """Independent, deliberately-naive brute-force cross-check, used ONLY
    in this validation cell -- never as the shipped implementation."""
    y = np.asarray(y, dtype=float)
    n = len(y)
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


def validate_hand_checked_example():
    rng = np.random.default_rng(7)
    y = rng.standard_normal(30)
    got = sample_entropy(y, d=DEFAULT_D, r_factor=DEFAULT_R_FACTOR)
    expected = _brute_force_sample_entropy(y, d=DEFAULT_D, r_factor=DEFAULT_R_FACTOR)
    assert abs(got - expected) < 1e-9, (got, expected)
    print(f"[PASS] Cross-check against independent brute-force implementation: {got:.6f} == {expected:.6f}")


def validate_synthetic_ordering():
    """A click-like transient in an otherwise quiet background should show
    low SE at the transient, high SE in quiet background."""
    rng = np.random.default_rng(3)
    n = 400
    quiet = 0.01 * rng.standard_normal(n)
    click_like = quiet.copy()
    for i in range(0, n, 20):
        click_like[i] = 5.0

    se_quiet = sample_entropy(quiet, d=DEFAULT_D, r_factor=DEFAULT_R_FACTOR)
    se_click = sample_entropy(click_like, d=DEFAULT_D, r_factor=DEFAULT_R_FACTOR)
    assert se_click < se_quiet, (se_click, se_quiet)
    print(f"[PASS] Synthetic ordering: SE(click-like)={se_click:.4f} < SE(quiet)={se_quiet:.4f}")


def run_window_sweep_on_click_files():
    available = [f for f in CLICK_FILES if (RAW_DIR / f).exists()]
    if not available:
        print(
            "No click WAV files available in data/raw/ -- skipping "
            "window-length sensitivity sweep. See data/raw/SOURCE.md."
        )
        return

    rows = []
    for fname in available:
        data, sr = read_wav(str(RAW_DIR / fname))
        fig, axes = plt.subplots(len(WINDOW_SWEEP_S), 1, figsize=(8, 2.2 * len(WINDOW_SWEEP_S)), sharex=True)
        for ax, win_s in zip(np.atleast_1d(axes), WINDOW_SWEEP_S):
            win_n = int(round(win_s * sr))
            n_windows_est = max(0, (len(data) - win_n) // int(round(STEP_S * sr)) + 1)
            t_start = time.time()
            print(
                f"  {fname}: window={win_s}s ({win_n} samples) -> "
                f"~{n_windows_est} windows, starting...",
                flush=True,
            )
            try:
                times, SE = sliding_sample_entropy(
                    data, fs=sr, window_seconds=win_s, step_seconds=STEP_S, d=DEFAULT_D, r_factor=DEFAULT_R_FACTOR
                )
            except ValueError as e:
                print(f"  [skip] {fname} window={win_s}s: {e}")
                continue
            print(f"    done in {time.time() - t_start:.1f}s", flush=True)
            finite = np.isfinite(SE)
            ax.plot(times[finite], SE[finite], marker="o", markersize=2, linewidth=0.7)
            ax.axhline(0.6, color="red", linestyle="--", linewidth=0.8, label="threshold 0.6")
            ax.set_ylabel(f"SE\n(win={win_s}s)")
            rows.append(
                {
                    "file": fname,
                    "window_s": win_s,
                    "step_s": STEP_S,
                    "n_estimates": len(SE),
                    "SE_mean_finite": float(np.mean(SE[finite])) if finite.any() else float("nan"),
                    "frac_below_0.6": float(np.mean(SE[finite] < 0.6)) if finite.any() else float("nan"),
                }
            )
        fig.suptitle(f"Sliding SE window-length sweep: {fname} (unfiltered)")
        fig.supxlabel("Time [s]")
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"SE_window_sweep_{fname.replace('.wav', '')}.png", dpi=120)
        plt.close(fig)

    df = pd.DataFrame(rows)
    df.to_csv(TABLE_DIR / "SE_window_sweep_summary.csv", index=False)
    print(df.to_string(index=False))


def main():
    validate_hand_checked_example()
    validate_synthetic_ordering()
    run_window_sweep_on_click_files()


if __name__ == "__main__":
    main()
