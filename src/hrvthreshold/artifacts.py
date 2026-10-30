"""RR-interval artifact detection and correction.

This is deliberately a *simple, documented* method, NOT the Lipponen &
Tarvainen (2019) beat-classification algorithm used by Kubios "automatic"
correction. Two rules:

1. Physiological range: RR outside [min_rr, max_rr] ms is an artifact.
2. Local-median rule: RR deviating from the median of its centred
   neighbourhood (``window`` beats) by more than ``threshold`` (relative,
   default 20 %) is an artifact. This is close in spirit to the Kubios
   "threshold" filters and to the 20 % rule common in HRV software.

Flagged beats are replaced by cubic-spline interpolation over the beat
index of the accepted beats (``method="interpolate"``) or removed
(``method="delete"``). Interpolation preserves the beat count, which keeps
the 4..16-beat DFA boxes aligned with the original rhythm; it is the
default. Always report ``artifact_fraction``: Rogers et al. (2021,
Sensors 21:821) showed that artifacts and their correction measurably
distort alpha1, so a window with more than a few percent corrected beats
should not be trusted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import median_filter


@dataclass(frozen=True)
class CorrectionResult:
    rr: np.ndarray  # corrected RR (ms)
    flagged: np.ndarray  # boolean mask on the *input* beats
    artifact_fraction: float
    method: str


def detect_artifacts(
    rr_ms: np.ndarray,
    threshold: float = 0.20,
    window: int = 11,
    min_rr: float = 250.0,
    max_rr: float = 2000.0,
) -> np.ndarray:
    rr = np.asarray(rr_ms, dtype=float)
    bad = (rr < min_rr) | (rr > max_rr) | ~np.isfinite(rr)
    # neighbourhood median computed after range-invalid beats are bridged
    filled = rr.copy()
    if bad.any() and (~bad).any():
        filled[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(~bad), rr[~bad])
    med = median_filter(filled, size=window, mode="nearest")
    with np.errstate(invalid="ignore"):
        bad |= np.abs(rr - med) > threshold * med
    return bad


def correct_rr(
    rr_ms: np.ndarray,
    threshold: float = 0.20,
    window: int = 11,
    method: str = "interpolate",
    min_rr: float = 250.0,
    max_rr: float = 2000.0,
    extra_flags: np.ndarray | None = None,
) -> CorrectionResult:
    """Detect and correct artifacts.

    ``extra_flags``: optional boolean mask of intervals known to be non-NN
    (e.g. adjacent to annotated ectopic beats); they are corrected too.
    """
    rr = np.asarray(rr_ms, dtype=float)
    if method not in ("interpolate", "delete"):
        raise ValueError("method must be 'interpolate' or 'delete'")
    flagged = detect_artifacts(rr, threshold, window, min_rr, max_rr)
    if extra_flags is not None:
        flagged = flagged | np.asarray(extra_flags, bool)
    frac = float(flagged.mean()) if rr.size else 0.0
    if not flagged.any():
        return CorrectionResult(rr.copy(), flagged, frac, method)
    if method == "delete":
        return CorrectionResult(rr[~flagged], flagged, frac, method)
    good = ~flagged
    if good.sum() < 4:
        raise ValueError("too few valid beats to interpolate")
    idx = np.arange(rr.size)
    cs = CubicSpline(idx[good], rr[good], extrapolate=True)
    out = rr.copy()
    out[flagged] = np.clip(cs(idx[flagged]), rr[good].min(), rr[good].max())
    return CorrectionResult(out, flagged, frac, method)
