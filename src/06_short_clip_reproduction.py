# %% [markdown]
# # 06 — Short-Clip Qualitative Reproduction
#
# Attempts the closest achievable approximation of Figs. 1/3's plotting
# style (spectrogram + H or SE trace stacked) on the 8 available files.
#
# **This is explicitly a short-clip qualitative check, NOT equivalent to
# Fig. 1 or Fig. 3 of the paper** (see Section 3 of the build spec,
# "What is NOT reproducible"). The paper's figures show detection dynamics
# over recordings containing quiet stretches between vocalizations; the 8
# available files are short, continuously-vocalizing clips and empirically
# do not show the same on/off oscillation (see docs/reproduction_audit.md).

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram as scipy_spectrogram

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.audio_io import read_wav
from src.filtering import spectrogram_params
from src.permutation_entropy import sliding_permutation_entropy
from src.sample_entropy import sliding_sample_entropy

RAW_DIR = REPO_ROOT / "data" / "raw"
FIG_DIR = REPO_ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

WHISTLE_FILES = ["Whistle1.wav", "Whistle2.wav", "Whistle3.wav", "Whistle4.wav"]
CLICK_FILES = ["Click1.wav", "Click2.wav", "Click3.wav", "Click4.wav"]

# PERFORMANCE FIX (see docs/reproduction_audit.md, notebooks/04_sample_entropy.py):
# a single 0.1s window is fine for H, but SE's O(N^2)-ish neighbor search
# makes 0.1s (~9,600 samples) take ~20s per window on real audio, hanging
# this notebook. Use separate windows: H keeps 0.1s, SE uses the smaller
# window established in Stage 4.
WINDOW_S_H = 0.1  # NOT SPECIFIED by the paper; representative choice for this plot
WINDOW_S_SE = 0.01  # 960 samples at 96kHz; matches Stage 4's fix, ~1-8s/file


def plot_spectrogram_plus_entropy(fname, entropy_kind):
    path = RAW_DIR / fname
    if not path.exists():
        print(f"  [skip] {fname} not found in data/raw/")
        return
    data, sr = read_wav(str(path))

    params = spectrogram_params()
    noverlap = int(params["nperseg"] * params["overlap_frac"])
    f, tt, Sxx = scipy_spectrogram(
        data, fs=sr, window=params["window"], nperseg=params["nperseg"],
        noverlap=noverlap, nfft=params["nfft"],
    )

    if entropy_kind == "H":
        step_s = 0.18
        times, vals = sliding_permutation_entropy(data, fs=sr, window_seconds=WINDOW_S_H, step_seconds=step_s)
        thresh, ylabel = 0.5, "H"
    else:
        step_s = 0.0125
        times, vals = sliding_sample_entropy(data, fs=sr, window_seconds=WINDOW_S_SE, step_seconds=step_s)
        thresh, ylabel = 0.6, "SE"

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    ax1.pcolormesh(tt, f, 10 * np.log10(Sxx + 1e-20), shading="auto")
    ax1.set_ylabel("Frequency [Hz]")
    ax1.set_title(
        f"{fname} — short-clip qualitative check (NOT equivalent to paper's "
        f"Fig. 1/3, see module docstring)"
    )

    finite = np.isfinite(vals)
    ax2.plot(times[finite], vals[finite], marker="o", markersize=2, linewidth=0.7)
    ax2.axhline(thresh, color="red", linestyle="--", linewidth=0.8, label=f"threshold {thresh}")
    ax2.set_ylabel(ylabel)
    ax2.set_xlabel("Time [s]")
    ax2.legend(fontsize=7)

    fig.tight_layout()
    out_path = FIG_DIR / f"shortclip_{entropy_kind}_{fname.replace('.wav', '')}.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Saved {out_path}")


def main():
    any_found = any((RAW_DIR / f).exists() for f in WHISTLE_FILES + CLICK_FILES)
    if not any_found:
        print("No WAV files available in data/raw/ -- nothing to plot. See data/raw/SOURCE.md.")
        return

    print("Whistle files (H):")
    for f in WHISTLE_FILES:
        plot_spectrogram_plus_entropy(f, "H")

    print("Click files (SE):")
    for f in CLICK_FILES:
        plot_spectrogram_plus_entropy(f, "SE")


if __name__ == "__main__":
    main()
