"""Regenerate every figure in docs/figures and results/synthetic_validation.json.

Synthetic figures need nothing else; PhysioNet figures need
results/alpha1/*.csv from scripts/validate_physionet.py plus network
(Poincare plots re-read the reference annotations).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hrvthreshold import alpha1, alpha1_timeseries, correct_rr, dfa, estimate_thresholds  # noqa: E402
from hrvthreshold.plots import C1, C2, C3, INK, MUTED, _style, plot_alpha1, plot_poincare  # noqa: E402
from hrvthreshold.synth import brownian_noise, noise_with_alpha, power_law_noise, synthetic_ramp, white_noise  # noqa: E402
from hrvthreshold.windows import Alpha1Series  # noqa: E402

FIG = ROOT / "docs" / "figures"
RES = ROOT / "results"


def synthetic():
    import nolds

    rng = np.random.default_rng(2026)
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(10, 4.2), dpi=130)
    N = 4096
    wide = np.unique(np.logspace(np.log10(4), np.log10(N // 10), 15).astype(int))
    out = {"N": N, "reps": 50, "signals": {}}
    for name, gen, col, exp in (("white", white_noise, MUTED, 0.5), ("pink 1/f", lambda n, r: power_law_noise(n, 1, r), C1, 1.0),
                                ("Brownian", brownian_noise, C2, 1.5)):
        r1 = [alpha1(gen(N, rng)) for _ in range(50)]
        rw = [dfa(gen(N, rng), wide).alpha for _ in range(50)]
        out["signals"][name] = {"expected": exp, "alpha1_4_16_mean": float(np.mean(r1)), "alpha1_4_16_sd": float(np.std(r1)),
                                "wide_range_mean": float(np.mean(rw)), "wide_range_sd": float(np.std(rw))}
        res = dfa(gen(N, rng), wide)
        Fn = res.fluctuations / res.fluctuations[0]
        a0.loglog(res.scales, Fn, "o", color=col, ms=4)
        a0.loglog(res.scales, (res.scales / res.scales[0]) ** res.alpha, "-", color=col, lw=1,
                  label=f"{name}: α = {res.alpha:.2f} (theory {exp})")
    a0.axvspan(4, 16, color=C3, alpha=0.08, label="α1 range, n = 4…16 beats")
    a0.set_xlabel("box size n (beats)")
    a0.set_ylabel("F(n) / F(4)")
    a0.set_title("Fluctuation function, synthetic noise (N = 4096)", fontsize=10, loc="left")
    a0.legend(fontsize=7.5, frameon=False)
    _style(a0)

    true = np.array([0.3, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0, 1.25, 1.5])
    m, sd, ref_diff = [], [], 0.0
    for a in true:
        est = []
        for _ in range(200):
            x = noise_with_alpha(240, a, rng)  # 240 beats = 2 min at 120 bpm
            est.append(alpha1(x))
        for _ in range(5):
            x = noise_with_alpha(240, a, rng)
            ref_diff = max(ref_diff, abs(alpha1(x) - nolds.dfa(x, nvals=list(range(4, 17)), overlap=False, fit_exp="poly")))
        m.append(np.mean(est))
        sd.append(np.std(est))
    m, sd = np.array(m), np.array(sd)
    out["calibration_240_beats"] = [{"true_alpha": float(t), "alpha1_mean": float(a), "alpha1_sd": float(s)}
                                    for t, a, s in zip(true, m, sd)]
    out["max_abs_diff_vs_nolds"] = float(ref_diff)
    a1.plot([0.2, 1.6], [0.2, 1.6], color=MUTED, lw=0.8, label="identity")
    a1.errorbar(true, m, yerr=sd, fmt="o-", color=C1, ms=4, capsize=3, lw=1.2, label="α1 estimate (mean ± SD, 200 reps)")
    a1.axhline(0.75, color=INK, ls="--", lw=0.8)
    a1.axhline(0.5, color=INK, ls=":", lw=0.8)
    a1.set_xlabel("true spectral exponent α = (β + 1)/2")
    a1.set_ylabel("estimated DFA α1 (n = 4…16)")
    a1.set_title(f"Single 240-beat window · max |ours − nolds| = {ref_diff:.0e}", fontsize=10, loc="left")
    a1.legend(fontsize=8, frameon=False)
    _style(a1)
    fig.tight_layout()
    fig.savefig(FIG / "synthetic_validation.png")
    plt.close(fig)

    # synthetic incremental test
    rr, a_true, hr_true = synthetic_ramp(seed=1)
    s = alpha1_timeseries(rr)
    v = s.valid()
    th = estimate_thresholds(s.alpha1[v], s.hr[v])
    plot_alpha1(s, title="SYNTHETIC incremental test (generator: HR 95→185 bpm, spectral α 1.1→0.3)", path=FIG / "synthetic_ramp.png", thresholds=th)
    out["synthetic_ramp"] = {"hrvt1_est": th.hrvt1, "hrvt2_est": th.hrvt2,
                             "hrvt1_true": 95 + (1.1 - 0.75) / 0.8 * 90, "hrvt2_true": 95 + (1.1 - 0.5) / 0.8 * 90,
                             "r2": th.r2}

    # jitter
    jit = [0, 0.5, 1, 2, 3, 4, 6, 8]
    fig, ax = plt.subplots(figsize=(6.4, 4), dpi=130)
    out["jitter"] = {}
    for label, base, sdv, col in (("rest-like: 60 bpm, RR SD 40 ms", 1000, 40, C1), ("exercise-like: 150 bpm, RR SD 4 ms", 400, 4, C2)):
        rows = []
        for j in jit:
            vals = []
            for _ in range(40):
                r = base + sdv * noise_with_alpha(240, 1.0, rng)
                b = np.cumsum(r) + rng.normal(0, j, r.size)
                vals.append(alpha1(np.diff(b)))
            rows.append(np.mean(vals))
        out["jitter"][label] = dict(zip([str(j) for j in jit], map(float, rows)))
        ax.plot(jit, rows, "o-", color=col, label=label, ms=4)
    ax.axhline(0.75, color=INK, ls="--", lw=0.8)
    ax.axhline(0.5, color=INK, ls=":", lw=0.8)
    ax.axvline(1000 / 256, color=MUTED, lw=0.8)
    ax.text(1000 / 256 + 0.1, 1.05, "1 sample @ 256 Hz", fontsize=7, color=MUTED)
    ax.set_xlabel("beat-detection timing jitter σ (ms)")
    ax.set_ylabel("DFA α1 (true α = 1.0)")
    ax.set_title("Timing jitter fakes a threshold crossing at high HR", fontsize=10, loc="left")
    ax.legend(fontsize=8, frameon=False)
    _style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "jitter.png")
    plt.close(fig)
    (RES / "synthetic_validation.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


def _load_series(name):
    d = np.loadtxt(RES / "alpha1" / f"{name}.csv", delimiter=",", skiprows=1, ndmin=2)
    return Alpha1Series(d[:, 0], d[:, 1], d[:, 2], d[:, 3], np.full(d.shape[0], np.nan))


def physionet():
    val = json.loads((RES / "physionet_validation.json").read_text())
    # 1. one stress test, alpha1 vs time
    s = _load_series("stdb_306")
    v = s.valid()
    k = int(np.nanargmax(np.where(v, s.hr, np.nan)))
    th = estimate_thresholds(s.alpha1[v][: np.sum(v[: k + 1])], s.hr[v][: np.sum(v[: k + 1])])
    plot_alpha1(s, title="REAL: MIT-BIH ST Change DB record 306 (exercise stress test, reference beat labels)",
                path=FIG / "alpha1_stress_test_306.png", thresholds=th)

    # 2. alpha1 vs HR, all stress tests reaching 140 bpm + rest distribution
    hi = [r["record"] for r in val["stress_stdb"] if r.get("hr_max", 0) >= 140]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=130, gridspec_kw={"width_ratios": [3, 1]})
    cmap = plt.get_cmap("tab20")
    for i, rec in enumerate(hi):
        s = _load_series(f"stdb_{rec}")
        v = s.valid()
        ax.plot(s.hr[v], s.alpha1[v], ".", ms=2, alpha=0.5, color=cmap(i % 20), label=rec)
    ax.axhline(0.75, color=INK, ls="--", lw=0.8)
    ax.axhline(0.5, color=INK, ls=":", lw=0.8)
    ax.set_xlabel("HR (bpm, window mean)")
    ax.set_ylabel("DFA α1")
    ax.set_title(f"REAL: α1 vs HR, {len(hi)} stress tests reaching ≥ 140 bpm (each colour = 1 record)", fontsize=10, loc="left")
    ax.legend(fontsize=6, ncol=4, frameon=False, markerscale=4)
    _style(ax)
    rest = [r["alpha1_median"] for r in val["rest_fantasia"]]
    lo = [r["alpha1_at_lowest_decile_hr"] for r in val["stress_stdb"] if r["record"] in hi]
    pk = [r["alpha1_at_highest_decile_hr"] for r in val["stress_stdb"] if r["record"] in hi]
    for x, d, c in ((0, rest, MUTED), (1, lo, C1), (2, pk, C2)):
        bx.plot(np.full(len(d), x) + rng_jit(len(d)), d, "o", color=c, ms=4, alpha=0.8)
        bx.plot([x - 0.25, x + 0.25], [np.median(d)] * 2, color=INK, lw=1.5)
    for a, b in zip(lo, pk):
        bx.plot([1, 2], [a, b], color=MUTED, lw=0.5, alpha=0.6)
    bx.set_xticks([0, 1, 2], ["rest\n(Fantasia)", "stress test\nlowest HR", "stress test\nhighest HR"], fontsize=8)
    bx.axhline(0.75, color=INK, ls="--", lw=0.8)
    bx.axhline(0.5, color=INK, ls=":", lw=0.8)
    bx.set_ylabel("median α1 per record")
    _style(bx)
    fig.tight_layout()
    fig.savefig(FIG / "alpha1_vs_hr_physionet.png")
    plt.close(fig)

    # 3. Poincare: rest vs peak exercise
    import wfdb

    fig, axs = plt.subplots(1, 2, figsize=(9, 4.6), dpi=130)
    ann = wfdb.rdann("f1y01", "ecg", pn_dir="fantasia")
    rr = np.diff(ann.sample[np.array(ann.symbol) == "N"]) / ann.fs * 1000
    rr = correct_rr(rr).rr
    plot_poincare(rr[np.cumsum(rr) <= 300_000], title="REAL rest: Fantasia f1y01, first 5 min", ax=axs[0])
    ann = wfdb.rdann("306", "atr", pn_dir="stdb")
    rr = correct_rr(np.diff(ann.sample[np.array(ann.symbol) == "N"]) / ann.fs * 1000).rr
    t = np.cumsum(rr) / 1000
    s = _load_series("stdb_306")
    tpk = s.t[np.nanargmax(s.hr)]
    plot_poincare(rr[(t > tpk - 300) & (t <= tpk)], title="REAL exercise: stdb 306, 5 min to peak HR", ax=axs[1])
    fig.tight_layout()
    fig.savefig(FIG / "poincare_rest_vs_exercise.png")
    plt.close(fig)


def rng_jit(n, _r=np.random.default_rng(0)):
    return _r.uniform(-0.12, 0.12, n)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    which = sys.argv[1:] or ["synthetic", "physionet"]
    if "synthetic" in which:
        synthetic()
    if "physionet" in which:
        physionet()
