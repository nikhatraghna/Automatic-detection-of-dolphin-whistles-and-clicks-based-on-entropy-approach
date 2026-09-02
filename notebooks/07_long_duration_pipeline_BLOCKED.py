# %% [markdown]
# # 07 — Long-Duration Pipeline (BLOCKED: requires real ETS/HB long recordings)
#
# Reproduces: Section 3.3 (detrending, whistle/click density counting).
#
# **This notebook's real-data cells are BLOCKED.** The paper's long-duration
# analysis (Figs. 6, 7) requires 60-minute-long ETS/HB recordings that are
# NOT part of the 8-file public dataset (see Section 3 of the build spec,
# "What is NOT reproducible", and docs/author_correspondence.md). The full
# pipeline (detrend -> threshold -> count -> density) is built and tested
# below against clearly-labeled SYNTHETIC placeholder data of the right
# shape, so it is ready to run on real data the moment it's obtained. No
# numbers resembling the paper's real results are ever printed here.

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.detection import (
    count_threshold_crossings,
    WHISTLE_DENSITY_BIN_S,
    CLICK_DENSITY_BIN_S,
)
from src.detrending import moving_window_detrend

FIG_DIR = REPO_ROOT / "results" / "figures"
TABLE_DIR = REPO_ROOT / "results" / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

REAL_DATA_AVAILABLE = False  # flip to True only once real ETS/HB long recordings exist locally


def make_synthetic_long_duration_H(
    duration_s: float = 4 * 60 * 60,  # 4 hours, matching the paper's ETS experiment
    step_s: float = 0.18,  # FACT: Section 3.4, ETS whistle step size
    n_injected_whistles: int = 40,
    whistle_duration_s: float = 2.0,
    seed: int = 0,
):
    """
    Fabricate a multi-hour synthetic H(t) array with KNOWN injected
    "whistle events" (H dips below 0.5) at known times, for testing the
    detrend -> threshold -> count pipeline end-to-end. This is
    SYNTHETIC PLACEHOLDER DATA, explicitly not real ocean recordings --
    never presented as a paper result.

    Returns
    -------
    times : np.ndarray
    H : np.ndarray
    injected_whistle_times : np.ndarray
        Ground-truth center times of injected whistle events (for testing
        recovery by the pipeline).
    """
    rng = np.random.default_rng(seed)
    n = int(duration_s / step_s)
    times = np.arange(n) * step_s

    # Ambient-noise-like baseline (paper: H typically 0.7-0.8+ in ambient noise)
    # plus a slow synthetic drift (to exercise the detrending step) and noise.
    baseline = 0.75 + 0.05 * np.sin(2 * np.pi * times / (30 * 60))  # slow drift, ~30-min period
    H = baseline + 0.03 * rng.standard_normal(n)

    injected_whistle_times = rng.choice(times[times < duration_s - whistle_duration_s], size=n_injected_whistles, replace=False)
    injected_whistle_times.sort()
    for wt in injected_whistle_times:
        mask = (times >= wt) & (times < wt + whistle_duration_s)
        H[mask] = np.clip(0.25 + 0.05 * rng.standard_normal(mask.sum()), 0.0, 1.0)

    H = np.clip(H, 0.0, 1.0)
    return times, H, injected_whistle_times


def test_pipeline_on_synthetic_data():
    """
    Full pipeline test: detrend -> threshold -> count, on synthetic data
    with known ground truth. This is the pytest-equivalent check embedded
    here for a runnable end-to-end demonstration; see also
    tests/test_detrending.py and tests/test_detection.py-equivalent checks
    in tests/ for the unit-level versions.
    """
    times, H_raw, injected_times = make_synthetic_long_duration_H()
    effective_sr = 1.0 / 0.18  # step_s

    H_detrended = moving_window_detrend(H_raw, sr=effective_sr, window_seconds=60.0, method="mean_subtract")

    # NOTE: after mean-subtraction detrending, H is centered around 0, not
    # 0.5 -- so a "whistle" (locally low H) now corresponds to a local dip
    # BELOW the local mean, not literally H<0.5. For the synthetic
    # end-to-end test, we detect against the RAW (un-detrended) threshold
    # (H_raw < 0.5), matching the paper's literal H<0.5 convention (Section
    # 3.3: "considering only those values higher than a certain threshold
    # of H (H < 0.5)" is applied to H AFTER detrending in the paper's own
    # description, but the paper's detrending is mean-subtraction over
    # already-normalized H, and the 0.5 threshold is stated as an absolute
    # cutoff throughout -- we flag this as an internal ambiguity in the
    # paper's own pipeline description, not silently resolved).
    bin_centers, counts = count_threshold_crossings(H_raw, times, threshold=0.5, bin_seconds=WHISTLE_DENSITY_BIN_S)

    # Ground truth: how many 1-min bins should contain at least one injected whistle
    gt_bin_has_whistle = np.zeros_like(bin_centers, dtype=bool)
    for wt in injected_times:
        idx = np.argmin(np.abs(bin_centers - wt))
        gt_bin_has_whistle[idx] = True
    detected_bin_has_whistle = counts > 0

    # Recovery rate on synthetic data (informational only -- NOT the
    # paper's accuracy figure, see docs/reproduction_audit.md)
    recall_on_synthetic = np.mean(detected_bin_has_whistle[gt_bin_has_whistle]) if gt_bin_has_whistle.any() else float("nan")
    print(
        f"[synthetic-only check] {len(injected_times)} injected whistle events; "
        f"pipeline recovered a detection in {recall_on_synthetic:.1%} of the "
        f"corresponding 1-min bins. (This is a pipeline SANITY CHECK on "
        f"fabricated data, NOT a paper result.)"
    )

    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(times / 3600, H_raw, linewidth=0.4)
    axes[0].axhline(0.5, color="red", linestyle="--", linewidth=0.8)
    axes[0].set_ylabel("H (raw, synthetic)")
    axes[0].set_title("SYNTHETIC PLACEHOLDER DATA -- pipeline sanity check only, not real ETS data")

    axes[1].plot(times / 3600, H_detrended, linewidth=0.4, color="tab:orange")
    axes[1].set_ylabel("H (detrended, synthetic)")

    axes[2].plot(bin_centers / 3600, counts, marker="o", markersize=3, color="tab:green")
    axes[2].set_ylabel("Whistle density\n(per 1-min bin)")
    axes[2].set_xlabel("Time [hours]")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "07_long_duration_pipeline_SYNTHETIC.png", dpi=120)
    plt.close(fig)
    print(f"Saved synthetic pipeline demo plot to {FIG_DIR / '07_long_duration_pipeline_SYNTHETIC.png'}")


def real_data_section():
    if not REAL_DATA_AVAILABLE:
        print(
            "\nBLOCKED: requires long-duration ETS/HB recordings, see "
            "docs/author_correspondence.md. The 4-hour ETS whistle "
            "recording and its manual annotation vector (needed for a real "
            "run of this pipeline) are not part of the public 8-file "
            "dataset and are not available in this environment."
        )
        return
    raise NotImplementedError(
        "Real-data long-duration pipeline not implemented: load real ETS/HB "
        "recordings here once obtained from the corresponding authors."
    )


def main():
    print(f"Whistle density bin width (FACT, Section 3.3): {WHISTLE_DENSITY_BIN_S} s")
    print(f"Click density bin width (FACT, Section 3.3): {CLICK_DENSITY_BIN_S} s")
    test_pipeline_on_synthetic_data()
    real_data_section()


if __name__ == "__main__":
    main()
