# %% [markdown]
# # 09 — Computational Benchmark
#
# Reproduces: Section 3.5 ("Computational speed of H and SE"), for
# comparison reference only.
#
# **Important caveat (Section 1.5 of the build spec, restated here
# explicitly):** the paper's own numbers were measured on an Intel Core
# i9-7920X @ 2.9 GHz (turbo 4.3 GHz) with 128 GB RAM, using R 3.4.1 for H
# and MATLAB (version unstated) for SE, on a real 60-minute recording at
# 96 kHz (H ~120s, SE ~420s). Our benchmark is **Python-only, on different,
# unspecified hardware** (see results/tables/environment_info.txt for what
# this environment actually is), applied to a **synthetically extended
# clip** (tiled/concatenated from the available short WAV files, or
# synthetic noise if no WAV files are available) to reach a comparable
# duration. This is comparable to the paper **in kind** (same task, same
# nominal duration) but is explicitly **NOT a like-for-like speed claim**
# against the paper's cross-language, cross-hardware figures.

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.audio_io import read_wav
from src.benchmarking import benchmark_function, collect_environment_info
from src.permutation_entropy import sliding_permutation_entropy
from src.sample_entropy import sliding_sample_entropy

RAW_DIR = REPO_ROOT / "data" / "raw"
TABLE_DIR = REPO_ROOT / "results" / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

# Paper's reference numbers (Section 3.5), reference-only.
PAPER_SR_HZ = 96000
PAPER_DURATION_MIN = 60
PAPER_H_SECONDS = 120
PAPER_SE_SECONDS = 420

# Benchmark target duration. NOT the paper's full 60 minutes by default --
# that is computationally expensive for SE (O(N^2) per window, thousands of
# windows at the paper's ~0.0125s step) in a shared sandbox; the target is
# configurable and clearly reported alongside the result rather than
# silently substituted for the paper's number. Increase BENCHMARK_DURATION_S
# (and/or restore STEP_S_SE to the paper's ~0.0125s) for a fuller benchmark
# on your own hardware.
BENCHMARK_DURATION_S = 5.0  # short default for tractable out-of-the-box runs
BENCHMARK_SR_HZ = 96000

WINDOW_S_H = 0.1
STEP_S_H = 0.18
WINDOW_S_SE = 0.05
# ASSUMPTION (benchmark-only convenience, does NOT affect detection
# results elsewhere in this repo): a coarser step than the paper's
# ~0.011-0.014s Discussion-section figure, to keep the O(N^2)-per-window
# SE benchmark tractable by default. Restore to ~0.0125s for a
# paper-comparable step size once running on adequate hardware/duration.
STEP_S_SE = 0.05


def build_benchmark_signal(duration_s: float, sr: int) -> np.ndarray:
    """
    Build a signal of the requested duration for benchmarking. If real WAV
    files are available, tile/concatenate one to reach the target duration
    (explicitly labeled as a SYNTHETIC STAND-IN, not a real long-duration
    ocean recording, per Section 1.5 / "What IS reproducible" of the build
    spec). Otherwise, fall back to synthetic white noise at the target
    sample rate.
    """
    candidates = sorted(RAW_DIR.glob("*.wav"))
    n_target = int(duration_s * sr)

    if candidates:
        data, file_sr = read_wav(str(candidates[0]))
        if file_sr != sr:
            # Simple resample-by-repetition is out of scope; just use the
            # file's own sample rate for the benchmark instead of silently
            # mismatching rates.
            sr = file_sr
            n_target = int(duration_s * sr)
        reps = int(np.ceil(n_target / len(data)))
        tiled = np.tile(data, reps)[:n_target]
        print(
            f"Benchmark signal: {candidates[0].name} tiled x{reps} to "
            f"{duration_s:.0f}s at {sr} Hz "
            f"(SYNTHETIC STAND-IN, not a real long-duration recording)."
        )
        return tiled, sr

    rng = np.random.default_rng(0)
    print(
        f"No WAV files found in data/raw/ -- using synthetic white noise "
        f"as the benchmark signal ({duration_s:.0f}s at {sr} Hz)."
    )
    return rng.standard_normal(n_target), sr


def main():
    env_info = collect_environment_info(out_path=str(TABLE_DIR / "environment_info.txt"))
    print("Environment info (saved to results/tables/environment_info.txt):")
    for k, v in env_info.items():
        print(f"  {k}: {v}")

    signal, sr = build_benchmark_signal(BENCHMARK_DURATION_S, BENCHMARK_SR_HZ)
    actual_duration_min = len(signal) / sr / 60.0

    print(
        f"\nBenchmarking on a {actual_duration_min:.2f}-minute signal at "
        f"{sr} Hz (paper's reference test file: {PAPER_DURATION_MIN}-minute "
        f"recording at {PAPER_SR_HZ} Hz)."
    )

    h_bench = benchmark_function(
        sliding_permutation_entropy, signal, fs=sr, window_seconds=WINDOW_S_H, step_seconds=STEP_S_H, n_repeats=1
    )
    print(f"H (permutation entropy): {h_bench['mean_s']:.2f} s (n_repeats={h_bench['n_repeats']})")

    # SE is O(N^2) per window -- keep the benchmark tractable by using a
    # coarser step (fewer windows) unless a shorter signal is used; this is
    # noted explicitly rather than silently truncated.
    se_bench = benchmark_function(
        sliding_sample_entropy, signal, fs=sr, window_seconds=WINDOW_S_SE, step_seconds=STEP_S_SE, n_repeats=1
    )
    print(f"SE (sample entropy): {se_bench['mean_s']:.2f} s (n_repeats={se_bench['n_repeats']})")

    results = pd.DataFrame(
        [
            {
                "metric": "H",
                "our_duration_min": round(actual_duration_min, 3),
                "our_sample_rate_hz": sr,
                "our_runtime_s": round(h_bench["mean_s"], 3),
                "paper_duration_min": PAPER_DURATION_MIN,
                "paper_sample_rate_hz": PAPER_SR_HZ,
                "paper_runtime_s": PAPER_H_SECONDS,
                "paper_language": "R 3.4.1",
                "comparable_in_kind_only": True,
            },
            {
                "metric": "SE",
                "our_duration_min": round(actual_duration_min, 3),
                "our_sample_rate_hz": sr,
                "our_runtime_s": round(se_bench["mean_s"], 3),
                "paper_duration_min": PAPER_DURATION_MIN,
                "paper_sample_rate_hz": PAPER_SR_HZ,
                "paper_runtime_s": PAPER_SE_SECONDS,
                "paper_language": "MATLAB (version unstated)",
                "comparable_in_kind_only": True,
            },
        ]
    )
    results.to_csv(TABLE_DIR / "computational_benchmark.csv", index=False)
    print(f"\nSaved benchmark results to {TABLE_DIR / 'computational_benchmark.csv'}")
    print(
        "\nReminder: these numbers are Python-only, on this environment's "
        "hardware, on a synthetically-tiled/noise signal -- comparable to "
        "the paper's Section 3.5 figures IN KIND (same task, same nominal "
        "duration) but NOT a like-for-like cross-language/cross-hardware "
        "speed claim."
    )


if __name__ == "__main__":
    main()
