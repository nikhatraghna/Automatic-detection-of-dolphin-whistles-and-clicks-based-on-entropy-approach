# Reproduction Audit

Honest, self-critical status document. Regenerate/update this at each
milestone. **Do not claim full reproduction while H, SE, or their detection
accuracy remain unimplemented or untested** — these are the paper's entire
contribution, not peripheral.

## What is implemented

- [x] `src/permutation_entropy.py` — from-scratch H, validated against the
  paper's own worked example (exact match, non-negotiable check) and a
  synthetic-signal sanity ordering (ramp < tone < noisy tone < noise).
- [x] `src/sample_entropy.py` — from-scratch SE (max-norm distance
  vectorized via scipy.spatial.distance only), validated against an
  independent brute-force re-implementation on a small series, and a
  synthetic click-vs-quiet ordering check.
- [x] `src/filtering.py` — Butterworth bandpass/highpass (ASSUMPTION: type
  and order), validated with pure-tone passband/stopband attenuation
  checks.
- [x] `src/detrending.py` — 1-min moving-window detrend (mean-subtract
  default, linear alternative), validated on a synthetic linear-trend
  signal.
- [x] `src/detection.py` — threshold-crossing density counting,
  confusion-matrix construction, and accuracy/error-rate/precision/
  recall/specificity, validated against a hand-computed confusion matrix.
- [x] `src/audio_io.py` — WAV metadata, checksums, Welch PSD band-energy
  fraction, validated against a synthetically-generated WAV file.
- [x] `src/benchmarking.py` — timing harness and environment-info
  collector.
- [x] `tests/` — pytest suite covering all of the above (hand-computable
  checks, synthetic sanity-ordering checks, and edge cases per module).


## Comparison table: paper's reported numbers vs. this reproduction

| Result | Paper's number | This reproduction |
|---|---|---|
| H worked example (Section 2.1) | (3,0,1,2), (1,2,3,0) | **Reproduced exactly** |
| H qualitative whistle-detection behavior | H dips below 0.5 during whistles | computed on real data ( 8 WAV files; see below) |
| SE qualitative click-detection behavior | SE dips below 0.6 during clicks |  computed on real data (the 8 WAV files) |
| ETS whistle confusion matrix (n=80,000), accuracy ≈ 0.966 | 76524/2593/85/798 | **Not computed** — requires 4-hr ETS long-duration recording + manual annotation, not available (see docs/author_correspondence.md) |
| HB click confusion matrix (n=8,656), accuracy ≈ 0.978 | 7236/80/112/1228 | **Not computed** — requires 4-hr HB long-duration recording + manual annotation, not available |
| ETS whistle-density Pearson r > 0.95 | — | **Not computed** — same blocker |
| HB whistle/click-density Pearson r = 0.89 / 0.97 | — | **Not computed** — same blocker |
| Computational benchmark (60-min @ 96kHz): H≈120s, SE≈420s | R/MATLAB, i9-7920X | **Not computed** — will be measured on a synthetically-extended clip; comparable in kind only, per §1.5 of the build spec |

## What is BLOCKED and why

- **Table 1's exact confusion matrices and accuracy/error-rate figures**
  (96.6%, 97.8%) require the original 4-hour ETS and 4-hour HB recordings
  plus their manual annotation vectors. These are available only from the
  corresponding authors on request (see `docs/author_correspondence.md`).
  Not fabricated here.
- **The Pearson correlation figures** (r>0.95, r=0.89, r=0.97) — same
  blocker.
- **Figs. 1, 2, 6, 7's qualitative on/off detection dynamic** — the 8
  available clips are too short and continuously-vocalizing to show this
  dynamic (see `docs/paper_parameters.md`, dataset table).
- **Exact reproduction of Figs. 3, 4, 5** (SE on ETS/HB long recordings) —
  possible only in reduced/short-clip form on Click1-4.wav.

## Verdict

**This is NOT a full reproduction of the paper's results.** The two core
algorithms (H, SE) have been implemented from scratch and validated
against the paper's own worked example and synthetic sanity checks — this
is the strongest claim that can honestly be made at this stage. The
paper's headline empirical claims (>95% detection accuracy, Pearson
correlations on long-duration recordings) require data that is not
available in this environment and have not been reproduced, qualitatively
or otherwise, against real long-duration recordings. Once the 8 WAV files
are downloaded (see `data/raw/SOURCE.md`) and/or the long-duration data is
obtained from the corresponding authors, this audit should be re-run and
updated.
