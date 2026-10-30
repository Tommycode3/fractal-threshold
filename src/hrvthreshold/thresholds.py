"""HRV thresholds from alpha1.

HRVT1 (aerobic threshold proxy): alpha1 = 0.75  (Rogers et al. 2021, Front. Physiol.)
HRVT2 (anaerobic threshold proxy): alpha1 = 0.50 (Rogers et al. 2021, J. Funct. Morphol. Kinesiol.)

Method: ordinary least-squares regression of alpha1 on an intensity
variable x (heart rate by default; power or speed also work) over the
*descending* part of an incremental test, then solve for x at the target
alpha1. Windows with too many corrected beats are excluded.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

HRVT1_ALPHA = 0.75
HRVT2_ALPHA = 0.50


@dataclass(frozen=True)
class ThresholdResult:
    hrvt1: float  # intensity (e.g. bpm) at alpha1 = 0.75
    hrvt2: float  # intensity at alpha1 = 0.50
    slope: float
    intercept: float
    r2: float
    n_windows: int
    extrapolated: tuple[bool, bool]
    reliable: bool  # r2 >= min_r2 and neither threshold extrapolated


def estimate_thresholds(
    alpha1: np.ndarray,
    x: np.ndarray,
    alpha_range: tuple[float, float] = (0.25, 1.05),
    targets: tuple[float, float] = (HRVT1_ALPHA, HRVT2_ALPHA),
    min_r2: float = 0.5,
) -> ThresholdResult:
    alpha1 = np.asarray(alpha1, float)
    x = np.asarray(x, float)
    m = np.isfinite(alpha1) & np.isfinite(x)
    m &= (alpha1 >= alpha_range[0]) & (alpha1 <= alpha_range[1])
    if m.sum() < 5:
        raise ValueError("need >= 5 valid windows inside alpha_range")
    xs, ys = x[m], alpha1[m]
    slope, intercept = np.polyfit(xs, ys, 1)
    pred = slope * xs + intercept
    ss_res = np.sum((ys - pred) ** 2)
    ss_tot = np.sum((ys - ys.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    if slope >= 0:
        raise ValueError("alpha1 does not decrease with intensity: no threshold")
    sol = [(a - intercept) / slope for a in targets]
    ext = tuple(bool(s < xs.min() or s > xs.max()) for s in sol)
    reliable = bool(np.isfinite(r2) and r2 >= min_r2 and not any(ext))
    return ThresholdResult(sol[0], sol[1], float(slope), float(intercept), float(r2), int(m.sum()), ext, reliable)
