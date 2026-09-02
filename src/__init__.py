"""
dolphin-entropy-reproduction/src

Reproduction of:
Siddagangaiah, S., Chen, C.-F., Hu, W.-C., Akamatsu, T., McElligott, M.,
Lammers, M.O., Pieretti, N. (2020). "Automatic detection of dolphin
whistles and clicks based on entropy approach." Ecological Indicators,
117, 106559. https://doi.org/10.1016/j.ecolind.2020.106559

This package contains from-scratch implementations of the paper's two
core algorithms (permutation entropy, sample entropy) plus supporting
preprocessing / detection / benchmarking utilities. See docs/paper_parameters.md
for a full ledger of every parameter used, its status (FACT / INFERRED /
ASSUMPTION / NOT SPECIFIED) and its source in the paper.
"""
