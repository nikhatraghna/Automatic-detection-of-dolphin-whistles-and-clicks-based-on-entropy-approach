# %% [markdown]
# # 03 — Permutation Entropy (H): Validation and Application
#
# Reproduces: Section 2.1, Eqs. (1)-(4). Known deviation: embedding delay
# tau=1 is an ASSUMPTION (not re-confirmed for m=6 in the paper); sliding
# window length is NOT SPECIFIED and is swept explicitly below rather than
# hard-coded. See docs/paper_parameters.md.

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.audio_io import read_wav
from src.permutation_entropy import (
    DEFAULT_M,
    DEFAULT_TAU,
    extract_ordinal_patterns,
    permutation_entropy,
    sliding_permutation_entropy,
)

RAW_DIR = REPO_ROOT / "data" / "raw"
FIG_DIR = REPO_ROOT / "results" / "figures"
TABLE_DIR = REPO_ROOT / "results" / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

WHISTLE_FILES = ["Whistle1.wav", "Whistle2.wav", "Whistle3.wav", "Whistle4.wav"]

# NOT SPECIFIED by the paper -- swept explicitly rather than hard-coded,
# per Rule 2 of the build spec.
WINDOW_SWEEP_S = [0.05, 0.1, 0.2, 0.5]
STEP_S = 0.18  # FACT: Section 3.4, ETS whistle experiment step size


def validate_worked_example():
    """Assert exact match with the paper's own hand-worked example
    (Section 2.1) -- non-negotiable, per the build spec."""
    series = [7, 3, 4, 5, 2, 9]
    patterns = extract_ordinal_patterns(series, m=4, tau=1)
    assert patterns[0] == (3, 0, 1, 2), patterns[0]
    assert patterns[1] == (1, 2, 3, 0), patterns[1]
    print("[PASS] Worked example (Section 2.1) reproduced exactly:")
    print(f"       (7,3,4,5) -> {patterns[0]}  (expected (3,0,1,2))")
    print(f"       (3,4,5,2) -> {patterns[1]}  (expected (1,2,3,0))")


def validate_synthetic_ordering():
    """Sanity-ordering check: H(ramp) < H(tone) < H(noisy tone) < H(noise)."""
    rng = np.random.default_rng(42)
    n = 5000
    ramp = np.arange(n, dtype=float)
    t = np.arange(n) / 1000.0
    tone = np.sin(2 * np.pi * 5 * t)
    noisy_tone = tone + 0.5 * rng.standard_normal(n)
    noise = rng.standard_normal(n)

    H_ramp = permutation_entropy(ramp, m=4, tau=1)
    H_tone = permutation_entropy(tone, m=4, tau=1)
    H_noisy = permutation_entropy(noisy_tone, m=4, tau=1)
    H_noise = permutation_entropy(noise, m=4, tau=1)

    assert H_ramp < H_tone < H_noisy < H_noise, (H_ramp, H_tone, H_noisy, H_noise)
    print(
        f"[PASS] Synthetic ordering: H(ramp)={H_ramp:.4f} < "
        f"H(tone)={H_tone:.4f} < H(noisy tone)={H_noisy:.4f} < "
        f"H(noise)={H_noise:.4f}"
    )


def run_window_sweep_on_whistle_files():
    available = [f for f in WHISTLE_FILES if (RAW_DIR / f).exists()]
    if not available:
        print(
            "No whistle WAV files available in data/raw/ -- skipping "
            "window-length sensitivity sweep. See data/raw/SOURCE.md."
        )
        return

    rows = []
    for fname in available:
        data, sr = read_wav(str(RAW_DIR / fname))
        fig, axes = plt.subplots(len(WINDOW_SWEEP_S), 1, figsize=(8, 2.2 * len(WINDOW_SWEEP_S)), sharex=True)
        for ax, win_s in zip(np.atleast_1d(axes), WINDOW_SWEEP_S):
            try:
                times, H = sliding_permutation_entropy(
                    data, fs=sr, window_seconds=win_s, step_seconds=STEP_S, m=DEFAULT_M, tau=DEFAULT_TAU
                )
            except ValueError as e:
                print(f"  [skip] {fname} window={win_s}s: {e}")
                continue
            ax.plot(times, H, marker="o", markersize=3, linewidth=0.8)
            ax.axhline(0.5, color="red", linestyle="--", linewidth=0.8, label="threshold 0.5")
            ax.set_ylabel(f"H\n(win={win_s}s)")
            ax.set_ylim(0, 1)
            rows.append(
                {
                    "file": fname,
                    "window_s": win_s,
                    "step_s": STEP_S,
                    "n_estimates": len(H),
                    "H_mean": float(np.mean(H)),
                    "H_min": float(np.min(H)),
                    "H_max": float(np.max(H)),
                    "frac_below_0.5": float(np.mean(H < 0.5)),
                }
            )
        axes[0].legend(loc="upper right", fontsize=7) if hasattr(axes, "__len__") else None
        fig.suptitle(f"Sliding H window-length sweep: {fname} (unfiltered)")
        fig.supxlabel("Time [s]")
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"H_window_sweep_{fname.replace('.wav', '')}.png", dpi=120)
        plt.close(fig)

    df = pd.DataFrame(rows)
    df.to_csv(TABLE_DIR / "H_window_sweep_summary.csv", index=False)
    print(df.to_string(index=False))

    if not df.empty:
        # Section 2 of the build spec: on these short, continuously-vocalizing
        # clips, H is expected to stay relatively flat rather than oscillate
        # the way the paper's Fig. 1/2/6/7 curves do (which require longer
        # recordings with quiet stretches). Report the observed spread as a
        # qualitative check of this expectation -- NOT a claim that the paper's
        # detection dynamic has been reproduced.
        spread = df.groupby("file")["H_mean"].std()
        print(
            "\nNote (Section 2 of build spec): these are short, "
            "continuously-vocalizing clips, not long recordings with quiet "
            "stretches. H is not expected to show the on/off oscillation seen "
            "in the paper's Figs. 1/2/6/7 on this dataset -- see "
            "docs/reproduction_audit.md."
        )


def main():
    validate_worked_example()
    validate_synthetic_ordering()
    run_window_sweep_on_whistle_files()


if __name__ == "__main__":
    main()
