"""Detrended fluctuation analysis (DFA) and the short-term exponent alpha1.

Implementation follows Peng et al. (1995, Chaos 5:82):

1. profile   y(k) = sum_{i<=k} (x_i - mean(x))
2. split y into floor(N/n) non-overlapping boxes of length n (forward only)
3. remove a least-squares linear trend inside each box
4. F(n) = sqrt( mean over all retained samples of the squared residuals )
5. alpha = slope of log F(n) vs log n (ordinary least squares)

For HRV the "short-term" exponent alpha1 is the slope over box sizes
n = 4..16 beats, which is the convention used by Kubios HRV and by the
DFA-alpha1 threshold literature (Gronwald, Rogers et al.).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ALPHA1_SCALES: tuple[int, ...] = tuple(range(4, 17))


@dataclass(frozen=True)
class DFAResult:
    alpha: float
    scales: np.ndarray
    fluctuations: np.ndarray
    intercept: float


def fluctuation_function(x: np.ndarray, scales=ALPHA1_SCALES, order: int = 1) -> np.ndarray:
    """Return F(n) for every box size n in ``scales``."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be 1-D")
    scales = np.asarray(scales, dtype=int)
    if scales.min() < order + 2:
        raise ValueError("smallest scale must exceed polynomial order + 1")
    if scales.max() > x.size // 2:
        raise ValueError(f"series too short ({x.size}) for max scale {scales.max()}")
    y = np.cumsum(x - x.mean())
    out = np.empty(scales.size)
    for j, n in enumerate(scales):
        m = y.size // n
        boxes = y[: m * n].reshape(m, n)
        t = np.arange(n, dtype=float)
        # Vandermonde least squares for all boxes at once: boxes.T = V @ coef
        V = np.vander(t, order + 1)
        coef, *_ = np.linalg.lstsq(V, boxes.T, rcond=None)
        resid = boxes.T - V @ coef
        out[j] = np.sqrt(np.mean(resid**2))
    return out


def dfa(x: np.ndarray, scales=ALPHA1_SCALES, order: int = 1) -> DFAResult:
    """DFA scaling exponent over ``scales`` (default 4..16 beats = alpha1)."""
    scales = np.asarray(scales, dtype=int)
    x = np.asarray(x, dtype=float)
    # constant (or numerically constant) input has no fluctuations to scale
    if x.size == 0 or np.ptp(x) <= 1e-9 * max(1.0, float(np.abs(x).max())):
        return DFAResult(float("nan"), scales, np.full(scales.size, np.nan), float("nan"))
    F = fluctuation_function(x, scales, order)
    if np.any(F <= 0):
        return DFAResult(float("nan"), scales, F, float("nan"))
    slope, intercept = np.polyfit(np.log(scales), np.log(F), 1)
    return DFAResult(float(slope), scales, F, float(intercept))


def alpha1(rr: np.ndarray) -> float:
    """Short-term DFA exponent (box sizes 4-16 beats) of an RR series."""
    return dfa(rr, ALPHA1_SCALES).alpha
