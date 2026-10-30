"""Matplotlib figures (Agg backend; import only when plotting)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .thresholds import HRVT1_ALPHA, HRVT2_ALPHA  # noqa: E402

INK = "#1f2328"
MUTED = "#8c959f"
C1 = "#0969da"
C2 = "#cf222e"
C3 = "#1a7f37"


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(alpha=0.25, lw=0.6)


def plot_alpha1(series, title="", path=None, thresholds=None):
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=130)
    v = series.valid()
    t_min = series.t / 60
    ax.axhspan(HRVT1_ALPHA, 1.6, color=C3, alpha=0.06)
    ax.axhspan(HRVT2_ALPHA, HRVT1_ALPHA, color="#bf8700", alpha=0.07)
    ax.axhspan(0, HRVT2_ALPHA, color=C2, alpha=0.06)
    ax.axhline(HRVT1_ALPHA, color=INK, ls="--", lw=0.9)
    ax.axhline(HRVT2_ALPHA, color=INK, ls=":", lw=0.9)
    ax.text(t_min[0], HRVT1_ALPHA + 0.02, "α1 = 0.75 (HRVT1, aerobic)", fontsize=8, color=INK)
    ax.text(t_min[0], HRVT2_ALPHA + 0.02, "α1 = 0.50 (HRVT2, anaerobic)", fontsize=8, color=INK)
    ax.plot(t_min[v], series.alpha1[v], color=C1, lw=1.6, label="DFA α1 (120 s window, 5 s step)")
    if (~v & np.isfinite(series.alpha1)).any():
        bad = ~v & np.isfinite(series.alpha1)
        ax.plot(t_min[bad], series.alpha1[bad], "x", color=MUTED, ms=3, label="window > 5 % artifacts")
    ax.set_xlabel("time (min, window end)")
    ax.set_ylabel("DFA α1")
    ax.set_ylim(0, max(1.6, np.nanmax(series.alpha1) + 0.1))
    _style(ax)
    ax2 = ax.twinx()
    ax2.plot(t_min, series.hr, color=C2, lw=1.0, alpha=0.7, label="HR")
    ax2.set_ylabel("HR (bpm)", color=C2)
    ax2.spines["top"].set_visible(False)
    if thresholds is not None:
        ax.set_title(f"{title}\nHRVT1 ≈ {thresholds.hrvt1:.0f} bpm · HRVT2 ≈ {thresholds.hrvt2:.0f} bpm "
                     f"(linear fit α1~HR, R² = {thresholds.r2:.2f})", fontsize=10, loc="left")
    else:
        ax.set_title(title, fontsize=10, loc="left")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right", frameon=False)
    fig.tight_layout()
    if path:
        fig.savefig(path)
        plt.close(fig)
    return fig


def plot_poincare(rr_ms, title="", path=None, ax=None):
    from .metrics import poincare

    rr = np.asarray(rr_ms, float)
    p = poincare(rr)
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.6, 4.6), dpi=130)
    else:
        fig = ax.figure
    ax.scatter(rr[:-1], rr[1:], s=3, alpha=0.35, color=C1, lw=0)
    lo, hi = np.percentile(rr, [0.5, 99.5])
    pad = 0.1 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=MUTED, lw=0.8)
    m = rr.mean()
    from matplotlib.patches import Ellipse

    ax.add_patch(Ellipse((m, m), 2 * p["sd2_ms"] * 2, 2 * p["sd1_ms"] * 2, angle=45, fill=False, color=C2, lw=1.4))
    lo, hi = min(lo, m - 2.2 * p["sd2_ms"]), max(hi, m + 2.2 * p["sd2_ms"])
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal")
    ax.set_xlabel("RR$_n$ (ms)")
    ax.set_ylabel("RR$_{n+1}$ (ms)")
    ax.set_title(f"{title}\nSD1 = {p['sd1_ms']:.1f} ms · SD2 = {p['sd2_ms']:.1f} ms (ellipse = ±2 SD)",
                 fontsize=9, loc="left")
    _style(ax)
    if path:
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
    return fig
