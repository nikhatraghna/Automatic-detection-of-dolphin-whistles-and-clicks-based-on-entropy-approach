# %% [markdown]
# # 08 — Confusion Matrix / Accuracy Assessment (BLOCKED: requires real manual annotations)
#
# Reproduces: Section 3.4, Eqs. (13)-(14) (accuracy, error rate) plus
# standard derived metrics (precision, recall, specificity).
#
# **This notebook's real-data cells are BLOCKED.** Table 1's exact
# confusion matrices (96.6% ETS whistle accuracy, 97.8% HB click accuracy)
# require the original 4-hour ETS/HB recordings and their manual
# whistle/click annotation vectors, which are not part of the public
# 8-file dataset (see docs/author_correspondence.md). `build_confusion_matrix`
# and `accuracy_metrics` (src/detection.py) are tested below against
# SYNTHETIC binary vectors with a KNOWN, hand-computed expected confusion
# matrix (exact match asserted). The real Table 1 reproduction is left as
# an explicit blocked step -- no fabricated numbers resembling the paper's
# are ever printed.

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.detection import build_confusion_matrix, accuracy_metrics

TABLE_DIR = REPO_ROOT / "results" / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

REAL_DATA_AVAILABLE = False  # flip to True only once real ETS/HB manual annotations exist locally

# Paper's own reported Table 1 numbers -- reproduced here as REFERENCE TEXT
# ONLY (for comparison against a REAL run, never fabricated as our own
# output). See docs/paper_parameters.md.
PAPER_TABLE1_ETS_WHISTLE = {"TN": 76524, "FP": 2593, "FN": 85, "TP": 798, "n": 80000}
PAPER_TABLE1_HB_CLICK = {"TN": 7236, "FP": 80, "FN": 112, "TP": 1228, "n": 8656}


def test_confusion_matrix_on_hand_computed_synthetic_example():
    """
    A small, fully hand-computable synthetic example with a KNOWN expected
    confusion matrix, asserted to match exactly.

    manual (predicted, per paper's convention):   [1,1,0,0,1,0,1,0]
    detected (actual, per paper's convention):    [1,0,0,1,1,0,1,1]

    By inspection:
      idx0: manual=1,detected=1 -> TP
      idx1: manual=1,detected=0 -> FN
      idx2: manual=0,detected=0 -> TN
      idx3: manual=0,detected=1 -> FP
      idx4: manual=1,detected=1 -> TP
      idx5: manual=0,detected=0 -> TN
      idx6: manual=1,detected=1 -> TP
      idx7: manual=0,detected=1 -> FP
    Expected: TP=3, FP=2, TN=2, FN=1
    """
    manual = [1, 1, 0, 0, 1, 0, 1, 0]
    detected = [1, 0, 0, 1, 1, 0, 1, 1]
    expected = {"TP": 3, "FP": 2, "TN": 2, "FN": 1}

    cm = build_confusion_matrix(manual, detected)
    assert cm == expected, (cm, expected)
    print(f"[PASS] Hand-computed confusion matrix reproduced exactly: {cm}")

    metrics = accuracy_metrics(cm)
    expected_accuracy = (3 + 2) / 8
    expected_error_rate = (1 + 2) / 8
    assert abs(metrics["accuracy"] - expected_accuracy) < 1e-12
    assert abs(metrics["error_rate"] - expected_error_rate) < 1e-12
    print(
        f"[PASS] Accuracy (Eq. 13) = {metrics['accuracy']:.4f} "
        f"(expected {expected_accuracy:.4f}); "
        f"Error rate (Eq. 14) = {metrics['error_rate']:.4f} "
        f"(expected {expected_error_rate:.4f})"
    )
    print(
        f"       Derived metrics -- precision={metrics['precision']:.3f}, "
        f"recall={metrics['recall']:.3f}, specificity={metrics['specificity']:.3f}"
    )


def test_confusion_matrix_on_larger_synthetic_vectors():
    """
    Larger synthetic manual/detected vectors (still with a
    programmatically-verifiable, i.e. independently recomputed via
    np.sum, expected confusion matrix) as an additional sanity check
    before the pipeline is trusted on anything resembling real data.
    """
    rng = np.random.default_rng(0)
    n = 10000
    manual = rng.integers(0, 2, size=n)
    # detected = manual with some flips, to get a non-trivial confusion matrix
    flips = rng.random(n) < 0.1
    detected = np.where(flips, 1 - manual, manual)

    cm = build_confusion_matrix(manual, detected)

    # Independently recomputed expectation via plain numpy boolean ops.
    m = manual.astype(bool)
    d = detected.astype(bool)
    expected = {
        "TP": int(np.sum(m & d)),
        "FP": int(np.sum(~m & d)),
        "TN": int(np.sum(~m & ~d)),
        "FN": int(np.sum(m & ~d)),
    }
    assert cm == expected, (cm, expected)
    print(f"[PASS] Larger synthetic confusion matrix (n={n}) matches independent recomputation: {cm}")


def print_paper_reference_numbers():
    print("\nPaper's own reported Table 1 numbers (REFERENCE ONLY, not reproduced here):")
    for label, d in [("ETS whistles", PAPER_TABLE1_ETS_WHISTLE), ("HB clicks", PAPER_TABLE1_HB_CLICK)]:
        acc = (d["TP"] + d["TN"]) / d["n"]
        print(f"  {label}: TP={d['TP']}, FP={d['FP']}, TN={d['TN']}, FN={d['FN']}, n={d['n']}, accuracy={acc:.3f}")


def real_data_section():
    if not REAL_DATA_AVAILABLE:
        print(
            "\nBLOCKED: requires the original 4-hour ETS/HB recordings and "
            "their manual whistle/click annotation vectors, see "
            "docs/author_correspondence.md. No confusion matrix is computed "
            "or printed for real data in this environment."
        )
        return
    raise NotImplementedError(
        "Real-data confusion matrix not implemented: load real manual "
        "annotations and automated detections here once obtained."
    )


def main():
    test_confusion_matrix_on_hand_computed_synthetic_example()
    test_confusion_matrix_on_larger_synthetic_vectors()
    print_paper_reference_numbers()
    real_data_section()


if __name__ == "__main__":
    main()
