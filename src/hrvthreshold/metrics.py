"""Standard HRV metrics (Task Force 1996 definitions) for recovery tracking.

Frequency domain uses the Lomb-Scargle periodogram directly on the
unevenly sampled tachogram (no resampling/interpolation). The periodogram
is linearly detrended first and scaled so that its integral over
(0, f_max] equals the variance of the detrended RR series (Parseval), which
gives absolute band powers in ms^2. LF/HF and normalised units are
independent of that scaling.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import detrend, lombscargle

VLF = (0.0033, 0.04)
LF = (0.04, 0.15)
HF = (0.15, 0.40)


def time_domain(rr_ms: np.ndarray) -> dict:
    rr = np.asarray(rr_ms, float)
    d = np.diff(rr)
    return {
        "n_beats": int(rr.size),
        "mean_rr_ms": float(rr.mean()),
        "mean_hr_bpm": float(60000.0 / rr.mean()),
        "sdnn_ms": float(rr.std(ddof=1)),
        "rmssd_ms": float(np.sqrt(np.mean(d**2))),
        "pnn50_pct": float(100.0 * np.mean(np.abs(d) > 50.0)),
    }


def poincare(rr_ms: np.ndarray) -> dict:
    rr = np.asarray(rr_ms, float)
    x, y = rr[:-1], rr[1:]
    sd1 = np.std((y - x) / np.sqrt(2), ddof=1)
    sd2 = np.std((y + x) / np.sqrt(2), ddof=1)
    return {"sd1_ms": float(sd1), "sd2_ms": float(sd2), "sd1_sd2": float(sd1 / sd2)}


def lomb_scargle_psd(rr_ms: np.ndarray, f_max: float = 0.5, n_freq: int = 1024):
    rr = np.asarray(rr_ms, float)
    t = np.cumsum(rr) / 1000.0
    t -= t[0]
    x = detrend(rr, type="linear")
    f = np.linspace(f_max / n_freq, f_max, n_freq)
    p = lombscargle(t, x - x.mean(), 2 * np.pi * f)
    df = f[1] - f[0]
    p = p * (np.var(x) / (p.sum() * df))
    return f, p


def _band(f, p, band):
    m = (f >= band[0]) & (f < band[1])
    return float(np.trapezoid(p[m], f[m])) if m.sum() > 1 else 0.0


def frequency_domain(rr_ms: np.ndarray) -> dict:
    f, p = lomb_scargle_psd(rr_ms)
    lf, hf, vlf = _band(f, p, LF), _band(f, p, HF), _band(f, p, VLF)
    return {
        "vlf_ms2": vlf,
        "lf_ms2": lf,
        "hf_ms2": hf,
        "lf_hf": lf / hf if hf > 0 else float("nan"),
        "lf_nu": 100 * lf / (lf + hf) if lf + hf > 0 else float("nan"),
        "hf_nu": 100 * hf / (lf + hf) if lf + hf > 0 else float("nan"),
        "lf_peak_hz": float(f[(f >= LF[0]) & (f < LF[1])][np.argmax(p[(f >= LF[0]) & (f < LF[1])])]),
        "hf_peak_hz": float(f[(f >= HF[0]) & (f < HF[1])][np.argmax(p[(f >= HF[0]) & (f < HF[1])])]),
    }


def all_metrics(rr_ms: np.ndarray) -> dict:
    from .dfa import alpha1

    out = time_domain(rr_ms)
    out.update(poincare(rr_ms))
    out.update(frequency_domain(rr_ms))
    out["dfa_alpha1"] = alpha1(rr_ms)
    return out
