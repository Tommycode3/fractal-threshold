"""Time-varying alpha1: sliding windows over the beat timeline.

Default: 120 s windows recomputed every 5 s, the setting used by
Rogers et al. (2021, Front. Physiol. 11:596567) via Kubios' time-varying
analysis. Windows are defined in *time* (seconds), so the number of beats
per window grows with heart rate (~120 beats at 60 bpm, ~340 at 170 bpm).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .artifacts import correct_rr
from .dfa import ALPHA1_SCALES, dfa


@dataclass(frozen=True)
class Alpha1Series:
    t: np.ndarray  # window END time (s from first beat)
    alpha1: np.ndarray
    hr: np.ndarray  # mean HR in window (bpm)
    artifact_fraction: np.ndarray  # fraction of corrected beats in window
    n_beats: np.ndarray

    def valid(self, max_artifacts: float = 0.05) -> np.ndarray:
        return np.isfinite(self.alpha1) & (self.artifact_fraction <= max_artifacts)


def alpha1_timeseries(
    rr_ms: np.ndarray,
    window_s: float = 120.0,
    step_s: float = 5.0,
    correct: bool = True,
    min_beats: int = 50,
    scales=ALPHA1_SCALES,
    **correction_kwargs,
) -> Alpha1Series:
    rr = np.asarray(rr_ms, dtype=float)
    if correct:
        res = correct_rr(rr, method="interpolate", **correction_kwargs)
        rr, flagged = res.rr, res.flagged
    else:
        flagged = np.zeros(rr.size, bool)
    t_beat = np.cumsum(rr) / 1000.0
    t_beat -= t_beat[0]
    ends = np.arange(window_s, t_beat[-1] + 1e-9, step_s)
    a, hr, af, nb = (np.full(ends.size, np.nan) for _ in range(4))
    lo = np.searchsorted(t_beat, ends - window_s, side="left")
    hi = np.searchsorted(t_beat, ends, side="right")
    for k, (i, j) in enumerate(zip(lo, hi)):
        seg = rr[i:j]
        nb[k] = seg.size
        if seg.size < max(min_beats, 2 * max(scales)):
            continue
        a[k] = dfa(seg, scales).alpha
        hr[k] = 60000.0 / seg.mean()
        af[k] = flagged[i:j].mean()
    return Alpha1Series(ends, a, hr, af, nb)
