# Paper Parameters Ledger

Living document. One row per parameter referenced anywhere in this
reproduction, from Siddagangaiah et al. (2020), *Ecological Indicators*
117, 106559.

Status legend:
- **FACT** — stated explicitly in the paper text.
- **INFERRED** — not stated verbatim, but follows unambiguously from stated
  facts (e.g. arithmetic derivable from other FACTs).
- **ASSUMPTION** — not specified by the paper; we chose a value and
  document the reasoning as a code comment (`# ASSUMPTION: <reasoning>`)
  wherever it's used, per Rule 2 of the build spec.
- **NOT SPECIFIED** — no value given anywhere in the paper; exposed as a
  mandatory, named parameter in code (never silently defaulted).

| Parameter | Value | Status | Source (section / eq.) |
|---|---|---|---|
| Permutation entropy embedding dimension, m | 6 | FACT | §2.1 |
| Permutation entropy embedding delay, tau | 1 | ASSUMPTION | §2.1 (Bandt & Pompe general recommendation, not re-confirmed for m=6) |
| Ordinal-pattern tie-breaking convention | smaller-lag (more recent) sample wins ties | ASSUMPTION | not specified; `src/permutation_entropy.py::ordinal_pattern` |
| Whistle detection threshold | H < 0.5 | FACT | §2.1, §3.4, throughout |
| Ambient-noise H (qualitative) | ~0.7-0.8+ | FACT | §2.1 |
| Whistle-occurrence H (qualitative) | ~0-0.5 | FACT | §2.1 |
| Statistical reliability requirement | N >> m! (m!=720 for m=6) | FACT (cited from Staniek et al. 2007) | §2.1 |
| Sliding-window length for H | — | NOT SPECIFIED | mandatory `window_seconds` arg, `sliding_permutation_entropy` |
| Step size, H, ETS 4-hr whistle experiment | 0.18 s | FACT | §3.4 ("80,000 steps in 4 hours") |
| Step size, H, Discussion section | ~0.11-0.18 s (unreconciled range; experiment not specified) | FACT (with noted inconsistency) | §5 Discussion |
| Original H reference implementation | R `statcomp` v0.0.1.1000, `global_complexity()` | FACT (NOT used as our implementation, per Rule 1) | §2.1 |
| Sample entropy embedding dimension, d | 2 | FACT | §2.2 |
| Sample entropy tolerance, r | 0.5 * SD(segment) | FACT | §2.2 |
| Click detection threshold | SE < 0.6 | FACT | §2.2, §3.4, throughout |
| Sliding-window length for SE | — | NOT SPECIFIED | mandatory `window_seconds` arg, `sliding_sample_entropy` |
| Step size, SE, Discussion section | ~0.011-0.014 s (does not arithmetically reconcile with 8,656-step/4-hr HB figure) | FACT (with noted inconsistency) | §5 Discussion |
| SE vectorization strategy | scipy.spatial.distance for max-norm distances only, entropy math is from-scratch | ASSUMPTION (engineering choice per build spec Rule 6) | `src/sample_entropy.py` |
| Original SE reference implementation | MATLAB, Supplementary Material S6.2 | FACT (NOT used as our implementation, per Rule 1) | §2.2 |
| Spectrogram FFT size | 1024 points | FACT | §3.1 |
| Spectrogram time segment | 0.05 s | FACT (mutually inconsistent with 1024 pts at 50/64/96 kHz) | §3.1 |
| Spectrogram window | Hanning | FACT | §3.1 |
| Spectrogram overlap | 90% | FACT | §3.1 |
| Spectrogram nfft/nperseg reconciliation | nfft = nperseg = 1024 (literal), "0.05 s" treated as approximate, unreconciled | ASSUMPTION | `src/filtering.py::spectrogram_params` (visualization only, never affects H/SE) |
| ETS whistle bandpass | 3,500-8,000 Hz | FACT | §3.1 |
| ETS click bandpass | 32,000-48,000 Hz | FACT | §3.1 |
| HB highpass | 6,000 Hz | FACT | §3.1 |
| Filter type / order | Butterworth, order 4, zero-phase (filtfilt) | ASSUMPTION | not specified; `src/filtering.py` |
| Filter edge handling | scipy.signal.filtfilt default padding | ASSUMPTION | not specified; `src/filtering.py` |
| Long-duration H detrending window | ~1 min | FACT (exact algorithm not specified) | §3.3 |
| Long-duration H detrending algorithm | mean-subtraction, 1-min moving window (default); linear per-window fit exposed as alternative | ASSUMPTION | §3.3; `src/detrending.py` |
| Whistle density bin width | 1 minute | FACT | §3.3 |
| Click density bin width | 30 seconds | FACT | §3.3 |
| Confusion matrix convention | predicted = manual annotation; actual = automated threshold detection | FACT | §3.4 |
| Accuracy formula | (TP+TN)/(TP+FP+TN+FN) | FACT, Eq. (13) | §3.4 |
| Error rate formula | (FN+FP)/(TP+FP+TN+FN) | FACT, Eq. (14) | §3.4 |
| ETS whistle confusion matrix, n=80,000 | TN=76524, FP=2593, FN=85, TP=798, accuracy≈0.965-0.966 | FACT (reference only — NOT reproducible, see docs/reproduction_audit.md) | Table 1a |
| HB click confusion matrix, n=8,656 | TN=7236, FP=80, FN=112, TP=1228, accuracy≈0.978 (paper rounds to 0.97) | FACT (reference only — NOT reproducible) | Table 1b |
| ETS whistle-density Pearson r | r > 0.95, p < 0.001 | FACT (reference only — NOT reproducible) | §4 |
| HB whistle/click-density Pearson r | r=0.89 (whistle), r=0.97 (click), both p<0.001 | FACT (reference only — NOT reproducible) | §4 |
| Benchmark hardware | Intel Core i9-7920X @ 2.9 GHz (turbo 4.3 GHz), 128 GB RAM | FACT (reference only) | §3.5 |
| Benchmark languages | H in R 3.4.1; SE in MATLAB (version unstated) | FACT (reference only) | §3.5 |
| Benchmark test file | 60-min recording at 96 kHz | FACT (reference only) | §3.5 |
| Benchmark runtimes | H ≈ 120 s; SE ≈ 420 s | FACT (reference only, cross-language/cross-hardware — not a like-for-like target) | §3.5 |

## Dataset (8 available WAV files — verified by direct inspection)

| File | Sample rate | Duration | Likely site (inferred, NOT confirmed metadata) |
|---|---|---|---|
| Whistle1.wav | 96,000 Hz | 3.0 s | ETS |
| Whistle2.wav | 96,000 Hz | 3.4 s | ETS |
| Whistle3.wav | 96,000 Hz | 5.0 s | ETS |
| Whistle4.wav | 96,000 Hz | 5.0 s | ETS |
| Click1.wav | 96,000 Hz | 2.0 s | ETS (tentative) |
| Click2.wav | 64,000 Hz | 3.0 s | HB |
| Click3.wav | 64,000 Hz | 1.5 s | HB |
| Click4.wav | 64,000 Hz | 0.1 s | HB |

These 8 files are short, isolated, continuously-vocalizing clips — NOT
excerpts of longer recordings with alternating noise/vocalization periods.
See docs/reproduction_audit.md and notebooks/06_short_clip_reproduction.py
for how this limits what can be demonstrated.
