# %% [markdown]
# # 05 — Preprocessing
#
# Reproduces: Section 3.1 filtering. Applies ETS bandpass filters to
# Whistle1-4 + Click1 (site-inference table), HB highpass to Click2-4, per
# the verified/inferred site table in docs/paper_parameters.md. Saves
# filtered audio to data/processed/filtered/ and re-runs sliding H/SE on
# the filtered versions for a filtered-vs-unfiltered comparison plot.
#
# ASSUMPTION (see src/filtering.py): Butterworth order-4, zero-phase
# filter; edge handling via filtfilt defaults. Not specified by the paper.

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

try:
    import soundfile as sf

    _HAVE_SOUNDFILE = True
except ImportError:
    _HAVE_SOUNDFILE = False

from src.audio_io import read_wav
from src.filtering import (
    ETS_WHISTLE_BAND_HZ,
    ETS_CLICK_BAND_HZ,
    HB_HIGHPASS_HZ,
    bandpass_filter,
    highpass_filter,
)
from src.permutation_entropy import sliding_permutation_entropy
from src.sample_entropy import sliding_sample_entropy

RAW_DIR = REPO_ROOT / "data" / "raw"
FILTERED_DIR = REPO_ROOT / "data" / "processed" / "filtered"
FIG_DIR = REPO_ROOT / "results" / "figures"
FILTERED_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Site-inference table (docs/paper_parameters.md): which filter applies to
# which file. Click1 is "ETS (tentative)" per the build spec.
ETS_WHISTLE_FILES = ["Whistle1.wav", "Whistle2.wav", "Whistle3.wav", "Whistle4.wav"]
ETS_CLICK_FILES = ["Click1.wav"]
HB_FILES = ["Click2.wav", "Click3.wav", "Click4.wav"]

WINDOW_S_H = 0.1  # NOT SPECIFIED by paper; single representative choice for this comparison plot
# PERFORMANCE FIX (see docs/reproduction_audit.md and notebooks/04_sample_entropy.py):
# SE's O(N^2)-ish neighbor search makes a 0.1s (~9,600-sample) window take
# ~20s EACH on real audio -- fine for the handful of H calls above, but
# multiplied across unfiltered+filtered x several files it made this
# notebook hang past a 2-minute timeout. Use the same smaller, still-standard
# SampEn window established in Stage 4 (04_sample_entropy.py) instead.
WINDOW_S_SE = 0.01  # 960 samples at 96kHz; ~1-8s/file, matches Stage 4's fix
STEP_S_H = 0.18
STEP_S_SE = 0.0125


def process_file(fname, filter_fn, filter_kwargs, entropy_kind):
    path = RAW_DIR / fname
    if not path.exists():
        print(f"  [skip] {fname} not found in data/raw/")
        return
    data, sr = read_wav(str(path))
    try:
        filtered = filter_fn(data, sr, **filter_kwargs)
    except ValueError as e:
        print(f"  [skip] {fname}: {e}")
        return

    out_path = FILTERED_DIR / fname
    if _HAVE_SOUNDFILE:
        sf.write(str(out_path), filtered, sr)
        print(f"  Saved filtered audio: {out_path}")
    else:
        print(
            f"  [soundfile not installed -- skipping WAV write for {fname}; "
            f"install soundfile (see requirements.txt) to save filtered audio]"
        )

    fig, axes = plt.subplots(2, 1, figsize=(8, 4.5), sharex=True)
    if entropy_kind == "H":
        t_u, v_u = sliding_permutation_entropy(data, fs=sr, window_seconds=WINDOW_S_H, step_seconds=STEP_S_H)
        t_f, v_f = sliding_permutation_entropy(filtered, fs=sr, window_seconds=WINDOW_S_H, step_seconds=STEP_S_H)
        thresh = 0.5
        ylabel = "H"
    else:
        t_u, v_u = sliding_sample_entropy(data, fs=sr, window_seconds=WINDOW_S_SE, step_seconds=STEP_S_SE)
        t_f, v_f = sliding_sample_entropy(filtered, fs=sr, window_seconds=WINDOW_S_SE, step_seconds=STEP_S_SE)
        thresh = 0.6
        ylabel = "SE"

    axes[0].plot(t_u, v_u, marker="o", markersize=2, linewidth=0.6)
    axes[0].axhline(thresh, color="red", linestyle="--", linewidth=0.8)
    axes[0].set_title(f"{fname} — unfiltered")
    axes[0].set_ylabel(ylabel)

    axes[1].plot(t_f, v_f, marker="o", markersize=2, linewidth=0.6, color="tab:orange")
    axes[1].axhline(thresh, color="red", linestyle="--", linewidth=0.8)
    axes[1].set_title(f"{fname} — filtered")
    axes[1].set_ylabel(ylabel)
    axes[1].set_xlabel("Time [s]")

    fig.tight_layout()
    fig.savefig(FIG_DIR / f"filtered_vs_unfiltered_{entropy_kind}_{fname.replace('.wav', '')}.png", dpi=120)
    plt.close(fig)


def main():
    print("ETS whistle bandpass files (H):")
    for f in ETS_WHISTLE_FILES:
        process_file(
            f,
            bandpass_filter,
            {"low_hz": ETS_WHISTLE_BAND_HZ[0], "high_hz": ETS_WHISTLE_BAND_HZ[1]},
            "H",
        )

    print("ETS click bandpass files (SE):")
    # KNOWN, DOCUMENTED FINDING (see docs/paper_parameters.md): the paper's
    # stated ETS click band is 32,000-48,000 Hz. Click1.wav (our inferred
    # ETS click example) is sampled at 96,000 Hz, whose Nyquist frequency is
    # EXACTLY 48,000 Hz -- the paper's stated upper edge collides exactly
    # with the sampling limit, leaving no margin for a real (non-ideal)
    # bandpass filter, which mathematically cannot pass all the way to
    # Nyquist. Rather than silently skip this file (losing our one inferred
    # ETS click example) or silently guess a fudge factor, we clip the
    # upper edge to 99.9% of Nyquist and print exactly what we did and why.
    for f in ETS_CLICK_FILES:
        path = RAW_DIR / f
        if path.exists():
            _, sr_probe = read_wav(str(path))
            nyq = sr_probe / 2.0
            high_hz = ETS_CLICK_BAND_HZ[1]
            if high_hz >= nyq:
                clipped = round(nyq * 0.999, 1)
                print(
                    f"  NOTE: {f} sample rate {sr_probe} Hz -> Nyquist={nyq} Hz. "
                    f"Paper's stated ETS click band upper edge ({high_hz} Hz) is >= "
                    f"Nyquist, which is mathematically un-filterable exactly. "
                    f"Clipping to {clipped} Hz (99.9% of Nyquist) -- documented in "
                    f"docs/paper_parameters.md, not a silent guess."
                )
                high_hz = clipped
        else:
            high_hz = ETS_CLICK_BAND_HZ[1]
        process_file(
            f,
            bandpass_filter,
            {"low_hz": ETS_CLICK_BAND_HZ[0], "high_hz": high_hz},
            "SE",
        )

    print("HB highpass files (SE):")
    for f in HB_FILES:
        process_file(f, highpass_filter, {"cutoff_hz": HB_HIGHPASS_HZ}, "SE")


if __name__ == "__main__":
    main()
