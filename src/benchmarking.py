"""
Computational benchmarking utilities -- supports notebooks/09_computational_benchmark.py
(Section 3.5 of the paper, "Computational speed of H and SE").

The paper's own numbers (Intel Core i9-7920X, R for H, MATLAB for SE,
60-minute file at 96 kHz: H ~120s, SE ~420s) are cross-language,
cross-hardware figures. Our benchmark is Python-only, on different
hardware -- comparable IN KIND (same task, same duration) but NOT a
like-for-like speed claim against the paper. This caveat must be restated
in the benchmark notebook's markdown, per Section 1.5 of the build spec.
"""

from __future__ import annotations

import platform
import statistics
import time
from pathlib import Path
from typing import Callable, Dict


def benchmark_function(func: Callable, *args, n_repeats: int = 3, **kwargs) -> Dict[str, float]:
    """
    Time `func(*args, **kwargs)` over `n_repeats` runs.

    Returns
    -------
    dict with keys: mean_s, std_s, min_s, max_s, n_repeats, result_preview
    """
    times = []
    result = None
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return {
        "mean_s": statistics.mean(times),
        "std_s": statistics.stdev(times) if len(times) > 1 else 0.0,
        "min_s": min(times),
        "max_s": max(times),
        "n_repeats": n_repeats,
    }


def collect_environment_info(out_path: str | None = None) -> Dict[str, str]:
    """
    Collect Python version, OS, CPU model (best effort), and installed
    package versions relevant to this reproduction. Optionally writes the
    info to `out_path` (e.g. results/tables/environment_info.txt).
    """
    info = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "machine": platform.machine(),
    }

    for pkg in ("numpy", "scipy", "pandas", "matplotlib", "soundfile"):
        try:
            mod = __import__(pkg)
            info[f"{pkg}_version"] = getattr(mod, "__version__", "unknown")
        except ImportError:
            info[f"{pkg}_version"] = "not installed"

    if out_path:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            for k, v in info.items():
                f.write(f"{k}: {v}\n")

    return info
