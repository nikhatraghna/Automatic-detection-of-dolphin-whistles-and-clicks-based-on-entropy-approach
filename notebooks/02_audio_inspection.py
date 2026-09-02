# %% [markdown]
# # 02 — Audio Inspection
#
# Reproduces: Section 2 of the build spec ("Available dataset — exactly
# what exists, use only this") and Section 3.1's spectrogram description
# (visualization only). No entropy math here.
#
# Known deviation: the "0.05 s" PAMGuide spectrogram segment length does
# not arithmetically reconcile with an FFT size of 1024 points at any of
# this study's sample rates (50/64/96 kHz) — see
# `src/filtering.py::spectrogram_params` and `docs/paper_parameters.md`.
# We use nfft = nperseg = 1024 literally for plotting; this never affects
# the H/SE computation elsewhere in this repo.

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import spectrogram as scipy_spectrogram

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.audio_io import get_wav_metadata, read_wav, band_energy_fraction
from src.filtering import (
    ETS_WHISTLE_BAND_HZ,
    ETS_CLICK_BAND_HZ,
    spectrogram_params,
)

RAW_DIR = REPO_ROOT / "data" / "raw"
FIG_DIR = REPO_ROOT / "results" / "figures"
TABLE_DIR = REPO_ROOT / "results" / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_FILES = [
    "Whistle1.wav",
    "Whistle2.wav",
    "Whistle3.wav",
    "Whistle4.wav",
    "Click1.wav",
    "Click2.wav",
    "Click3.wav",
    "Click4.wav",
]


def main():
    available = [f for f in EXPECTED_FILES if (RAW_DIR / f).exists()]
    missing = [f for f in EXPECTED_FILES if f not in available]

    if missing:
        print(
            "The following expected files are missing from data/raw/ "
            f"(see data/raw/SOURCE.md for the download link):\n  "
            + "\n  ".join(missing)
        )
    if not available:
        print("No WAV files available -- nothing to inspect. Exiting.")
        return

    rows = []
    for fname in available:
        path = RAW_DIR / fname
        meta = get_wav_metadata(str(path))
        data, sr = read_wav(str(path))

        # Site-inference heuristic (Section 2 of the build spec): NOT
        # confirmed metadata, based on sample rate + PSD band-energy.
        if sr == 96000:
            ets_whistle_frac = band_energy_fraction(data, sr, *ETS_WHISTLE_BAND_HZ)
            ets_click_frac = band_energy_fraction(data, sr, *ETS_CLICK_BAND_HZ)
            site_guess = "ETS (96 kHz; PSD-band heuristic)"
        elif sr == 64000:
            ets_whistle_frac = np.nan
            ets_click_frac = np.nan
            site_guess = "HB (64 kHz; ETS click band above Nyquist at this rate)"
        else:
            ets_whistle_frac = np.nan
            ets_click_frac = np.nan
            site_guess = f"UNKNOWN (unexpected sample rate {sr} Hz)"

        rows.append(
            {
                "file": fname,
                "sample_rate_hz": meta.sample_rate,
                "duration_s": round(meta.duration_s, 3),
                "n_samples": meta.n_samples,
                "n_channels": meta.n_channels,
                "bit_depth": meta.bit_depth,
                "peak_amplitude": round(meta.peak_amplitude, 4),
                "rms_amplitude": round(meta.rms_amplitude, 4),
                "sha256": meta.sha256,
                "site_guess_NOT_confirmed": site_guess,
                "ets_whistle_band_psd_frac": ets_whistle_frac,
                "ets_click_band_psd_frac": ets_click_frac,
            }
        )

        # --- waveform plot ---
        t = np.arange(len(data)) / sr
        plt.figure(figsize=(8, 2.5))
        plt.plot(t, data, linewidth=0.5)
        plt.title(f"Waveform: {fname}")
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"waveform_{fname.replace('.wav', '')}.png", dpi=120)
        plt.close()

        # --- spectrogram plot (visualization only; see module docstring) ---
        params = spectrogram_params()
        noverlap = int(params["nperseg"] * params["overlap_frac"])
        f, tt, Sxx = scipy_spectrogram(
            data,
            fs=sr,
            window=params["window"],
            nperseg=params["nperseg"],
            noverlap=noverlap,
            nfft=params["nfft"],
        )
        plt.figure(figsize=(8, 3.5))
        plt.pcolormesh(tt, f, 10 * np.log10(Sxx + 1e-20), shading="auto")
        plt.title(
            f"Spectrogram: {fname}  (nfft=nperseg={params['nfft']}, "
            f"'0.05s' segment figure unreconciled -- see docstring)"
        )
        plt.xlabel("Time [s]")
        plt.ylabel("Frequency [Hz]")
        plt.colorbar(label="dB")
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"spectrogram_{fname.replace('.wav', '')}.png", dpi=120)
        plt.close()

    df = pd.DataFrame(rows)
    df.to_csv(TABLE_DIR / "audio_inspection_metadata.csv", index=False)
    print(df.to_string(index=False))
    print(f"\nSaved metadata table to {TABLE_DIR / 'audio_inspection_metadata.csv'}")
    print(f"Saved waveform/spectrogram figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
