# Reproduction Audit

Honest, self-critical status document. Regenerate/update this at each
milestone. **Do not claim full reproduction while H, SE, or their detection
accuracy remain unimplemented or untested** — these are the paper's entire
contribution, not peripheral.

## What is implemented (code exists and has been run)

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

## What has actually been RUN (not just planned)

As of this audit, every notebook (`02` through `10`) has been executed
end-to-end in this environment and completed without error. This sandbox
has no network access, so two things could not be installed/downloaded
here:

- `pytest` — the formal suite in `tests/` is written (mirrors the manual
  checks below in proper pytest form, including edge cases) but has not
  been run via `pytest` itself in this environment. Install it and run
  `pytest tests/ -v` to execute it.
- `soundfile` — `notebooks/05_preprocessing.py` degrades gracefully when
  it's missing (skips writing filtered WAV output, still runs everything
  else); `src/audio_io.py` has a stdlib `wave`-based fallback for reading.
  Install `soundfile` (in `requirements.txt`) for full WAV I/O support.
- The 8 real WAV files themselves are not present (`data/raw/` is empty;
  see `data/raw/SOURCE.md`) — every notebook that needs them detects their
  absence and skips the corresponding real-data steps with an explicit
  message, rather than failing or fabricating output.

Algorithm correctness was instead validated with manual verification
scripts (equivalent in content to the `tests/` files) run directly in this
environment. All checks reported here **passed**:

- Permutation entropy worked example (Section 2.1): exact match, both
  embedding vectors, both derived ordinal patterns.
- Permutation entropy synthetic ordering: `H(ramp) < H(tone) < H(noisy
  tone) < H(noise)` — confirmed on synthetic data with a fixed random
  seed.
- Permutation entropy constant-signal edge case: `H = 0`, no crash.
- Permutation entropy too-short-input: raises `ValueError` as expected.
- Sample entropy cross-check against an independent brute-force
  re-implementation: exact numerical agreement (to floating-point
  precision) on a 30-sample synthetic series.
- Sample entropy synthetic click-vs-quiet ordering: click-embedded signal
  produces lower SE than quiet background, as hypothesized by the paper.
- Sample entropy edge cases (constant signal, too-short series): raise
  `ValueError` as expected.
- Filtering: pure tones inside the ETS/HB passbands retain >90% RMS
  amplitude after filtering; tones well outside the passbands are
  attenuated to <10% RMS amplitude.
- Detrending: a synthetic linear-trend signal has its per-window mean
  reduced to ~0 by both the mean-subtract and linear methods.
- Detection: a hand-computed 8-sample confusion matrix matches exactly
  (`notebooks/08`); derived accuracy/precision/recall/specificity match
  hand calculation; a larger (n=10,000) synthetic confusion matrix also
  matches an independent recomputation.
- Long-duration pipeline (`notebooks/07`): detrend -> threshold -> count
  run end-to-end on a fabricated 4-hour synthetic H(t) series with 40
  known injected "whistle events"; the pipeline recovered a detection in
  100% of the corresponding 1-minute bins on this synthetic run (a
  pipeline sanity check, not a paper result).
- Computational benchmark (`notebooks/09`): H and SE sliding-window
  computations both run end-to-end and complete in bounded time on a
  short synthetic/tiled signal; environment info captured to
  `results/tables/environment_info.txt`.

## Comparison table: paper's reported numbers vs. this reproduction

| Result | Paper's number | This reproduction |
|---|---|---|
| H worked example (Section 2.1) | (3,0,1,2), (1,2,3,0) | **Reproduced exactly** |
| H qualitative whistle-detection behavior | H dips below 0.5 during whistles | Not computed on real data (requires the 8 WAV files; see below) |
| SE qualitative click-detection behavior | SE dips below 0.6 during clicks | Not computed on real data (requires the 8 WAV files) |
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
