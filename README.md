# dolphin-entropy-reproduction

A rigorous, from-scratch reproduction of:

> Siddagangaiah, S., Chen, C.-F., Hu, W.-C., Akamatsu, T., McElligott, M.,
> Lammers, M.O., Pieretti, N. (2020). "Automatic detection of dolphin
> whistles and clicks based on entropy approach." *Ecological Indicators*,
> 117, 106559. https://doi.org/10.1016/j.ecolind.2020.106559 (open access,
> CC BY 4.0)

## Status

**Not a full reproduction.** See `docs/reproduction_audit.md` for the
current, honest status: what's implemented, what's actually been run, and
what remains blocked pending data that is not publicly available (the
original long-duration recordings and manual annotations — see
`docs/author_correspondence.md`).

## Quick start

```bash
pip install -r requirements.txt

# 1. Download the 8 WAV files (see data/raw/SOURCE.md) into data/raw/
# 2. Run the notebooks in order:
python notebooks/02_audio_inspection.py
python notebooks/03_permutation_entropy.py
python notebooks/04_sample_entropy.py
python notebooks/05_preprocessing.py
python notebooks/06_short_clip_reproduction.py
python notebooks/07_long_duration_pipeline_BLOCKED.py
python notebooks/08_confusion_matrix_BLOCKED.py
python notebooks/09_computational_benchmark.py
python notebooks/10_final_report_generation.py

# Run the test suite
pytest tests/ -v
```

## Repository layout

```
dolphin-entropy-reproduction/
├── docs/                 paper_parameters.md, reproduction_audit.md,
│                          author_correspondence.md, final_report.md
├── data/raw/              8 WAV files (not included — see SOURCE.md)
├── data/processed/        filtered audio + entropy time series (generated)
├── src/                   from-scratch algorithm implementations
├── notebooks/              orchestration scripts (import from src/)
├── tests/                 pytest suite
└── results/               figures/, tables/, metrics/ (generated)
```

## Core algorithms — implementation notes

Both permutation entropy (`src/permutation_entropy.py`) and sample entropy
(`src/sample_entropy.py`) are implemented directly from the equations in
Sections 2.1-2.2 of the paper, using only numpy (and, for SE, scipy purely
as a vectorized max-norm *distance* primitive — never for the entropy
formula itself). No existing entropy/complexity library (statcomp,
EntropyHub, nolds, antropy, pyEntropy, ...) is used as the core
implementation.

Every parameter the paper does not specify explicitly (sliding-window
length, filter type/order, detrending algorithm, ordinal-pattern
tie-breaking) is exposed as a named, documented, swappable parameter — see
`docs/paper_parameters.md` for the full ledger of FACT / ASSUMPTION / NOT
SPECIFIED status per parameter.

## What is NOT reproducible with the publicly available data

The 8 WAV files on Mendeley Data (DOI 10.17632/zv6ygrptr4.1) are short,
isolated, continuously-vocalizing clips — not excerpts of longer
recordings with alternating noise/vocalization periods. They cannot
reproduce:

- Table 1's exact confusion matrices / accuracy figures (96.6%, 97.8%),
- the reported Pearson correlations (r>0.95, r=0.89, r=0.97),
- the on/off detection dynamic shown in Figs. 1, 2, 6, 7.

These require the original long-duration ETS/HB recordings and manual
annotations, available only from the corresponding authors on request
(draft request in `docs/author_correspondence.md`). The full pipeline code
for these experiments is built and unit-tested against synthetic
placeholder data (`notebooks/07`, `notebooks/08`), but no fabricated
numbers resembling the paper's are ever printed — blocked steps raise/print
an explicit message instead.

