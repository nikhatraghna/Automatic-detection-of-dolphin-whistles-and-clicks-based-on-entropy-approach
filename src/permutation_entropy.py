"""
Permutation entropy (H) -- Section 2.1, Eqs. (1)-(4) of Siddagangaiah et al. (2020).

Implemented entirely from scratch using only numpy for array bookkeeping.
No entropy/complexity library (statcomp, EntropyHub, nolds, antropy, pyEntropy,
etc.) is used as the core implementation, per the reproduction spec Rule 1.

--------------------------------------------------------------------------
Worked example given verbatim in the paper (Section 2.1), used below as the
primary correctness test (see tests/test_permutation_entropy.py):

    series = {7, 3, 4, 5, 2, 9, ...}, m = 4, tau = 1

    First embedding vector  (7, 3, 4, 5) -> ordinal pattern (3, 0, 1, 2)
    Second embedding vector (3, 4, 5, 2) -> ordinal pattern (1, 2, 3, 0)

Our implementation must reproduce these two permutations exactly.
--------------------------------------------------------------------------

Parameter status (see docs/paper_parameters.md for the full ledger):

  - embedding dimension m = 6                          FACT   (Section 2.1)
  - embedding delay tau = 1                             ASSUMPTION
        The paper only states tau=1 as Bandt & Pompe's general
        recommendation when introducing the method; it is not explicitly
        re-stated for the m=6 analysis used later in the paper. We adopt
        tau=1 as a documented default, not a confirmed fact.
  - whistle detection threshold H < 0.5                 FACT   (throughout)
  - tie-breaking convention for the ordinal pattern      ASSUMPTION
        (equal-valued samples): NOT specified by the paper. We use a
        stable sort that gives priority to the smaller-lag (more recent)
        sample -- i.e. ties are broken in favour of the sample closest to
        the "now" end of the embedding window. This is implemented via
        numpy's default stable ('quicksort' is NOT stable, so we
        explicitly request kind='stable') sort on (-value, lag) pairs.
  - sliding-window length (window_seconds)               NOT SPECIFIED
        Never defaulted silently -- see sliding_permutation_entropy(),
        which requires window_seconds and step_seconds as mandatory
        arguments (no default values).
  - step size between successive H estimates             FACT (0.18 s,
        Section 3.4, ETS 4-hour whistle experiment) but the Discussion
        section separately states "~0.11-0.18 s" without specifying which
        experiment that range applies to -- this inconsistency is noted
        wherever step size is used, it is not silently resolved.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Named constants (never silently hard-coded elsewhere in the codebase)
# ---------------------------------------------------------------------------

DEFAULT_M = 6  # FACT: Section 2.1, "for embedding dimension m = 6"
# ASSUMPTION: tau is not re-confirmed for m=6 in the paper; Bandt & Pompe's
# general recommendation (cited by the paper when introducing the method) is
# tau = 1. See module docstring.
DEFAULT_TAU = 1

WHISTLE_H_THRESHOLD = 0.5  # FACT: stated explicitly, used throughout

# FACT (Section 3.4, ETS whistle long-duration experiment): step size 0.18 s.
# NOTE: the Discussion section separately gives a step-size range of
# "~0.11-0.18 s" without stating which experiment(s) it applies to. This
# inconsistency is intentionally NOT resolved here -- callers must choose
# explicitly, see sliding_permutation_entropy().
STEP_SIZE_ETS_WHISTLE_S = 0.18
STEP_SIZE_DISCUSSION_RANGE_S = (0.11, 0.18)


def ordinal_pattern(vector: Sequence[float]) -> Tuple[int, ...]:
    """
    Compute the ordinal pattern (permutation) of an embedding vector, per
    Eq. (2). The paper defines the pattern in terms of LAGS relative to
    "now" (s), not raw array positions: for the embedding vector
    (x_{s-(m-1)tau}, ..., x_{s-tau}, x_s) (Eq. 1, chronological order,
    oldest to newest), the ordinal pattern pi = (r_0, ..., r_{m-1}) is the
    permutation of (0, ..., m-1) such that

        x_{s-r_0*tau} >= x_{s-r_1*tau} >= ... >= x_{s-r_{m-1}*tau}      Eq. (2)

    i.e. r_0 is the LAG (0 = "now", m-1 = oldest) of the largest sample,
    r_1 the lag of the second-largest, and so on.

    Given `vector` passed in chronological order (index 0 = oldest = lag
    m-1; index m-1 = newest/"now" = lag 0), lag and array position relate
    by: lag = (m-1) - position. We verify this convention exactly
    reproduces the paper's own worked example (Section 2.1): for
    vector=(7,3,4,5) (m=4), sorting by descending value gives array
    positions [0,3,2,1] (values 7,5,4,3); converting position->lag via
    lag=(m-1)-pos gives lags [3,0,1,2] -- matching the paper's stated
    result (3,0,1,2) exactly (see tests/test_permutation_entropy.py).

    Tie-breaking (ASSUMPTION, not specified by the paper): when two samples
    are exactly equal, priority (i.e. the smaller r_j / being treated as
    "larger") is given to the smaller-lag (more recent) sample -- the one
    closer to "now". Implemented by sorting on (-value, lag) so that among
    equal values the smaller lag sorts first.

    Parameters
    ----------
    vector : sequence of float, length m
        Embedding vector in chronological order (oldest -> newest), as
        produced by Eq. (1).

    Returns
    -------
    tuple of int, length m
        The ordinal pattern (r_0, ..., r_{m-1}), expressed as LAGS
        (0 = most recent), matching the paper's own convention and worked
        example.
    """
    v = np.asarray(vector, dtype=float)
    m = v.shape[0]
    positions = np.arange(m)  # 0 = oldest ... m-1 = newest ("now")
    lags = (m - 1) - positions  # position -> lag: m-1=oldest maps to lag m-1; newest(pos m-1) -> lag 0
    # Sort key: primarily by -value (descending value, largest first), then
    # by ascending lag (smaller lag / more recent sample gets priority on
    # ties) -- documented ASSUMPTION above.
    order = sorted(range(m), key=lambda i: (-v[i], lags[i]))
    return tuple(int(lags[i]) for i in order)


def extract_ordinal_patterns(
    x: Sequence[float], m: int = DEFAULT_M, tau: int = DEFAULT_TAU
) -> List[Tuple[int, ...]]:
    """
    Slide an m-dimensional, delay-tau embedding across the whole input
    segment `x` with step 1 (independent of tau, per the paper's
    description of Eq. (1)-(2)) and return the ordinal pattern for each
    embedding position.

    Embedding vectors are formed per Eq. (1):
        [x(s-(m-1)*tau), ..., x(s-tau), x(s)]   for s = (m-1)*tau, ..., N-1
    (0-indexed here), i.e. chronological order, oldest to newest.

    Parameters
    ----------
    x : sequence of float
        The scalar time series.
    m : int
        Embedding dimension (default 6, per the paper).
    n_tau : int
        Embedding delay (default 1, ASSUMPTION -- see module docstring).

    Returns
    -------
    list of tuple of int
        One ordinal pattern per valid embedding position, in order.
    """
    if m < 2:
        raise ValueError("Embedding dimension m must be >= 2.")
    if tau < 1:
        raise ValueError("Embedding delay tau must be >= 1.")
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    span = (m - 1) * tau
    if n <= span:
        raise ValueError(
            f"Input segment too short for m={m}, tau={tau}: "
            f"need > {span} samples, got {n}."
        )
    patterns: List[Tuple[int, ...]] = []
    for s in range(span, n):
        # chronological order, oldest -> newest: x[s-span], ..., x[s-tau], x[s]
        vec = x[s - span : s + 1 : tau]
        patterns.append(ordinal_pattern(vec))
    return patterns


def permutation_entropy(
    x: Sequence[float],
    m: int = DEFAULT_M,
    tau: int = DEFAULT_TAU,
    normalize: bool = True,
) -> float:
    """
    Compute the (optionally normalized) permutation entropy H(m) of a single
    segment, per Eqs. (3)-(4):

        H(m) = - sum_j p_j * ln(p_j)                          Eq. (3)
        H_norm = H(m) / ln(m!)   with 0 <= H_norm <= 1         Eq. (4)

    Parameters
    ----------
    x : sequence of float
        The scalar time series segment.
    m : int
        Embedding dimension (default 6).
    tau : int
        Embedding delay (default 1, ASSUMPTION).
    normalize : bool
        If True (default), return H(m) / ln(m!) as in Eq. (4). If False,
        return the raw Shannon entropy H(m) from Eq. (3) (natural log,
        nats).

    Returns
    -------
    float
        Permutation entropy (normalized to [0, 1] if normalize=True).

    Notes
    -----
    Reliability requirement (Staniek et al. 2007, cited by the paper):
    N >> m!. For m=6, m! = 720, so this function does not itself enforce a
    minimum length beyond what extract_ordinal_patterns requires
    ((m-1)*tau + 1 samples) -- callers computing single "trustworthy" H
    values should ensure their window is several thousand samples, per the
    paper's own caveat. This function will happily compute H on shorter
    windows (needed for the sliding-window analysis in Section 3.4), but
    the statistical-reliability caveat applies to interpretation, not to
    whether the function runs.
    """
    patterns = extract_ordinal_patterns(x, m=m, tau=tau)
    n_patterns = len(patterns)
    if n_patterns == 0:
        raise ValueError("No ordinal patterns could be extracted.")

    # Tally relative frequencies P = {p_1, ..., p_D} of each distinct
    # observed ordinal pattern (Section 2.1, paragraph after Eq. (2)).
    counts: dict = {}
    for p in patterns:
        counts[p] = counts.get(p, 0) + 1

    probs = np.array([c / n_patterns for c in counts.values()], dtype=float)

    # Edge case: a single distinct pattern (e.g. constant signal) has
    # probability 1 for that pattern -> H = 0 exactly (0 * ln(0) is defined
    # as 0 by convention, and here there's no 0-probability term at all
    # since we only sum over OBSERVED patterns).
    H = float(-np.sum(probs * np.log(probs)))

    if not normalize:
        return H

    max_H = math.log(math.factorial(m))  # ln(m!), Eq. (4) denominator
    if max_H == 0:
        # m=1 edge case (not used by this study, m is fixed at 6, but guard
        # against div-by-zero for completeness / test robustness).
        return 0.0
    return H / max_H


def sliding_permutation_entropy(
    x: Sequence[float],
    fs: float,
    window_seconds: float,
    step_seconds: float,
    m: int = DEFAULT_M,
    tau: int = DEFAULT_TAU,
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Slide a fixed-length analysis window across `x` and compute normalized
    permutation entropy H in each window, per the sliding-window analyses
    described throughout Section 3 (e.g. Figs. 1, 2, 6, 7).

    `window_seconds` and `step_seconds` are MANDATORY, deliberately never
    given default values: the paper does not specify the sliding-window
    length anywhere (NOT SPECIFIED, see module docstring and
    docs/paper_parameters.md), so silently defaulting it here would violate
    reproduction Rule 2. The step size for whistle detection is FACT-cited
    at 0.18 s for the ETS 4-hour experiment (Section 3.4) but the
    Discussion section separately gives an unreconciled "~0.11-0.18 s"
    range -- pass whichever value you intend and see the module-level
    constants STEP_SIZE_ETS_WHISTLE_S / STEP_SIZE_DISCUSSION_RANGE_S.

    Parameters
    ----------
    x : sequence of float
        The full (already filtered, if applicable) time series.
    fs : float
        Sampling rate in Hz.
    window_seconds : float
        Length, in seconds, of each analysis window fed to permutation_entropy.
        NOT SPECIFIED by the paper -- must be chosen explicitly by the
        caller and treated as a sensitivity-tested parameter.
    step_seconds : float
        Hop, in seconds, between successive window start times.
    m, tau, normalize : see permutation_entropy().

    Returns
    -------
    (times, H_values) : tuple of np.ndarray
        `times` gives the center time (seconds) of each window;
        `H_values` gives the corresponding permutation entropy.
    """
    if window_seconds <= 0 or step_seconds <= 0:
        raise ValueError("window_seconds and step_seconds must be > 0.")
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    win_n = int(round(window_seconds * fs))
    step_n = int(round(step_seconds * fs))
    if win_n < (m - 1) * tau + 2:
        raise ValueError(
            f"window_seconds={window_seconds} too short for m={m}, tau={tau} "
            f"at fs={fs} Hz (need at least {(m - 1) * tau + 2} samples, got {win_n})."
        )
    if win_n > n:
        raise ValueError("window_seconds is longer than the input signal.")

    starts = list(range(0, n - win_n + 1, step_n))
    times = np.array([(s + win_n / 2.0) / fs for s in starts])
    H_values = np.array(
        [
            permutation_entropy(x[s : s + win_n], m=m, tau=tau, normalize=normalize)
            for s in starts
        ]
    )
    return times, H_values
