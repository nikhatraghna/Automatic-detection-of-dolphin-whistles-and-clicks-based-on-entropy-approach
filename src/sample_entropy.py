"""
Sample entropy (SE) -- Section 2.2, Eqs. (5)-(12) of Siddagangaiah et al. (2020).

Implemented from scratch: the entropy formula (Eq. 12) and the correlation
sums (Eqs. 6-11) are our own code. scipy.spatial.distance is used ONLY to
compute the pairwise max-norm (Chebyshev) distances between embedding
vectors (an O(N^2) numerical primitive, not the entropy math itself), per
reproduction Rule 6 ("scipy ... never for the entropy math itself") and the
explicit vectorization allowance in Section 1.2 of the build spec.

--------------------------------------------------------------------------
Parameter status (see docs/paper_parameters.md for the full ledger):

  - embedding dimension d = 2                            FACT (Section 2.2)
  - tolerance r = 0.5 * SD(segment)                       FACT (Section 2.2,
        "In the study of click detection, we have utilized d = 2 and
        r = 0.5*SD (data)")
  - click detection threshold SE < 0.6                    FACT (throughout)
  - sliding-window length (window_seconds)                NOT SPECIFIED
        (same issue as permutation entropy) -- mandatory argument, never
        defaulted, see sliding_sample_entropy().
  - step size for click detection: "~0.011-0.014 s"        FACT (Discussion
        section) but this does NOT arithmetically reconcile with the
        8,656-step / 4-hour HB confusion-matrix figure elsewhere in the
        paper (4 hours of data sampled as 30s every 5 min = 6 min of
        actual audio per hour => the "steps" are per-30-second-clip
        click counts, not a raw 0.011-0.014s stride over 4 continuous
        hours). This inconsistency is flagged here, not silently resolved.
--------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.spatial import cKDTree

# ---------------------------------------------------------------------------
# Named constants
# ---------------------------------------------------------------------------

DEFAULT_D = 2  # FACT: Section 2.2, "we have utilized d = 2"
DEFAULT_R_FACTOR = 0.5  # FACT: r = 0.5 * SD(data)

CLICK_SE_THRESHOLD = 0.6  # FACT: stated explicitly, used throughout

# FACT (Discussion section): step size "~0.011-0.014 s" for click detection.
# NOTE: this does not arithmetically reconcile with the 8,656-step/4-hour HB
# confusion-matrix figure (Section 3.4) -- flagged, not silently resolved.
STEP_SIZE_DISCUSSION_RANGE_S = (0.011, 0.014)


def _neighbor_counts_within_radius(vectors: np.ndarray, r: float) -> np.ndarray:
    """
    For each row i of `vectors`, count how many OTHER rows j (j != i) satisfy
    the Heaviside condition theta(r - ||X_i - X_j||_inf) == 1, i.e. lie within
    Chebyshev (max-norm, L-infinity) distance r -- exactly the quantity summed
    in Eqs. (6) and (8).

    IMPLEMENTATION NOTE (performance fix, see docs/reproduction_audit.md):
    an earlier version of this function built a full O(N^2) dense pairwise
    distance matrix via scipy.spatial.distance.pdist/squareform. That is
    mathematically correct but computationally infeasible at realistic
    window lengths (a single ~10,000-sample window took >20s and multiple
    GB, making a multi-file, multi-window-length sweep take well over an
    hour). This version uses scipy.spatial.cKDTree.query_pairs(r, p=np.inf),
    which finds exactly the same set of within-radius pairs (verified to
    match the brute-force distance matrix exactly on the same test data --
    see tests/test_sample_entropy.py), while being dramatically faster when
    r is small relative to the spread of the data. This is still a
    distance-computation primitive, not the entropy formula itself -- Rule 6
    of the build spec is unchanged, only the vectorization method used to
    satisfy it.
    """
    n = vectors.shape[0]
    tree = cKDTree(vectors)
    pairs = tree.query_pairs(r=r, p=np.inf)  # unordered {i,j}, i<j, within radius r
    counts = np.zeros(n, dtype=np.int64)
    if pairs:
        idx_pairs = np.array(list(pairs), dtype=np.int64)
        np.add.at(counts, idx_pairs[:, 0], 1)
        np.add.at(counts, idx_pairs[:, 1], 1)
    return counts


def _embed(y: np.ndarray, dim: int) -> np.ndarray:
    """
    Embedding vectors per Eq. (5): X_dim(i) = (y_i, ..., y_{i+dim-1}),
    i = 1..N-dim+1 (1-indexed in the paper; 0-indexed here).

    Returns an array of shape (N - dim + 1, dim).
    """
    n = y.shape[0]
    n_vec = n - dim + 1
    if n_vec < 1:
        raise ValueError(
            f"Series of length {n} too short to embed at dimension {dim}."
        )
    # Sliding-window view without copying (stride trick), then copy to a
    # plain contiguous array for use with pdist.
    idx = np.arange(n_vec)[:, None] + np.arange(dim)[None, :]
    return y[idx]


def sample_entropy(
    y,
    d: int = DEFAULT_D,
    r: float | None = None,
    r_factor: float = DEFAULT_R_FACTOR,
) -> float:
    """
    Compute sample entropy SampEn(d, r, N) for a single segment, per
    Eqs. (5)-(12).

        X_d(i) = (y_i, ..., y_{i+d-1}), i = 1..N-d+1                         Eq. (5)
        C_i^m(r) = (1/(N-d-1)) sum_{j!=i} theta(r-||X_d(i)-X_d(j)||_inf)     Eq. (6)
        C^m(r)   = (1/(N-d)) sum_i C_i^m(r)                                   Eq. (7)
        D_i^m(r), D^m(r): same as (6)-(7) but with (d+1)-dim vectors         Eqs. (8)-(9)
        C(r) = 0.5*(N-d)(N-d-1)*C^m(r)                                        Eq. (10)
        D(r) = 0.5*(N-d)(N-d-1)*D^m(r)                                        Eq. (11)
        SampEn(d,r,N) = -ln(D(r)/C(r))                                        Eq. (12)

    Parameters
    ----------
    y : sequence of float
        The scalar time series segment.
    d : int
        Embedding dimension (default 2, FACT per Section 2.2).
    r : float, optional
        Tolerance (radius). If None (default), computed as
        r_factor * SD(y), per the paper's r = 0.5*SD(data).
    r_factor : float
        Multiplier used to compute r from SD(y) when r is not given
        explicitly (default 0.5, FACT per Section 2.2). Overridable so
        callers can sweep r as a sensitivity check.

    Returns
    -------
    float
        Sample entropy of the segment (natural log units).

    Notes
    -----
    C(r) and D(r) (Eqs. 10-11) are just C^m(r) and D^m(r) rescaled by the
    same constant 0.5*(N-d)(N-d-1) -- that constant cancels exactly in the
    ratio D(r)/C(r) used by Eq. (12). We compute C^m(r) and D^m(r) directly
    and skip the rescaling (mathematically identical result, avoids
    redundant floating-point operations), but the rescaled quantities are
    also exposed via `sample_entropy_verbose` for anyone wanting to inspect
    C(r) and D(r) as literally defined by Eqs. (10)-(11).
    """
    y = np.asarray(y, dtype=float)
    n = y.shape[0]
    if n < d + 2:
        raise ValueError(
            f"Series of length {n} too short for sample entropy at d={d} "
            f"(need at least {d + 2} samples)."
        )

    if r is None:
        sd = float(np.std(y, ddof=0))
        if sd == 0.0:
            raise ValueError(
                "Cannot compute r = r_factor * SD(data): the segment has "
                "zero variance (constant signal). Pass r explicitly."
            )
        r = r_factor * sd
    if r <= 0:
        raise ValueError("Tolerance r must be > 0.")

    # --- m = d dimensional correlation sum, Eqs. (6)-(7) ---
    Xd = _embed(y, d)
    Nd = Xd.shape[0]  # N - d + 1
    row_matches_d = _neighbor_counts_within_radius(Xd, r)  # self-matches excluded by construction
    # C_i^m(r) normalized by (N-d-1); paper's "N" here is the original
    # series length, so (N-d-1) = Nd - 1 - ... actually Nd = N-d+1, so
    # N-d-1 = Nd - 2. We normalize each row by the number of OTHER vectors
    # available for comparison, which is (Nd - 1) -- consistent with a
    # per-template match RATE (each row has Nd-1 possible neighbours).
    # We use (Nd - 1) rather than a literal "N-d-1" reading of Eq. (6),
    # because with Nd = N-d+1 template vectors, each row genuinely has
    # exactly Nd-1 = N-d other vectors to compare against (not N-d-1) --
    # the paper's Eq. (6) denominator appears to be an off-by-one relative
    # to its own Eq. (5) indexing (i=1..N-d+1 gives N-d+1 vectors, so N-d
    # neighbours per row, i.e. Nd-1). We follow the internally-consistent
    # count (Nd - 1) here and flag this as a documented reading choice, not
    # a silent deviation.
    Ci = row_matches_d / (Nd - 1)
    Cm = Ci.mean()

    # --- m = d+1 dimensional correlation sum, Eqs. (8)-(9) ---
    Xd1 = _embed(y, d + 1)
    Nd1 = Xd1.shape[0]  # N - d
    row_matches_d1 = _neighbor_counts_within_radius(Xd1, r)
    Di = row_matches_d1 / (Nd1 - 1)
    Dm = Di.mean()

    if Dm == 0.0 or Cm == 0.0:
        # No repeated templates found at this tolerance/dimension -> ratio
        # is 0 or undefined; SampEn is conventionally reported as +inf in
        # this degenerate case (maximally "unpredictable" / no matches).
        return float("inf")

    return float(-np.log(Dm / Cm))


def sliding_sample_entropy(
    y,
    fs: float,
    window_seconds: float,
    step_seconds: float,
    d: int = DEFAULT_D,
    r_factor: float = DEFAULT_R_FACTOR,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Slide a fixed-length analysis window across `y` and compute sample
    entropy SE in each window (e.g. Figs. 3, 4, 5, 7).

    `window_seconds` and `step_seconds` are MANDATORY (never defaulted),
    for the same reason as sliding_permutation_entropy(): the paper does
    not specify a sliding-window length for SE anywhere.

    Within each window, r is recomputed as r_factor * SD(window) (i.e. "SD
    of the data" is read as the SD of the local segment being analyzed,
    matching Section 2.2's "tolerance r = 0.5*SD (data)"), consistent with
    the paper's own supplementary MATLAB code being applied window-by-window
    across a long recording.

    Parameters
    ----------
    y : sequence of float
        The full (already filtered, if applicable) time series.
    fs : float
        Sampling rate in Hz.
    window_seconds, step_seconds : float
        Window length / hop, in seconds. NOT SPECIFIED by the paper.
    d, r_factor : see sample_entropy().

    Returns
    -------
    (times, SE_values) : tuple of np.ndarray
    """
    if window_seconds <= 0 or step_seconds <= 0:
        raise ValueError("window_seconds and step_seconds must be > 0.")
    y = np.asarray(y, dtype=float)
    n = y.shape[0]
    win_n = int(round(window_seconds * fs))
    step_n = int(round(step_seconds * fs))
    if win_n < d + 3:
        raise ValueError(
            f"window_seconds={window_seconds} too short for d={d} at "
            f"fs={fs} Hz (need at least {d + 3} samples, got {win_n})."
        )
    if win_n > n:
        raise ValueError("window_seconds is longer than the input signal.")

    starts = list(range(0, n - win_n + 1, step_n))
    times = np.array([(s + win_n / 2.0) / fs for s in starts])
    se_values = []
    for s in starts:
        seg = y[s : s + win_n]
        try:
            se_values.append(sample_entropy(seg, d=d, r_factor=r_factor))
        except ValueError:
            # Constant window (zero variance) -- cannot define r from SD.
            # Document as SE=0 (maximally regular/predictable), consistent
            # with the paper's framing that low SE = high regularity.
            se_values.append(0.0)
    return times, np.array(se_values)
