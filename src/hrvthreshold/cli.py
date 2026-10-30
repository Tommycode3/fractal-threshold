"""Command-line interface: ``hrv-threshold``."""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from . import __version__
from .artifacts import correct_rr
from .io import read_rr
from .metrics import all_metrics
from .thresholds import estimate_thresholds
from .windows import alpha1_timeseries


def _load(args):
    rr = read_rr(args.file, args.column)
    corr = correct_rr(rr, threshold=args.artifact_threshold)
    return rr, corr


def cmd_metrics(args) -> int:
    rr, corr = _load(args)
    out = {"file": args.file, "artifact_fraction": corr.artifact_fraction}
    out.update(all_metrics(corr.rr))
    _emit(out, args)
    if args.plot:
        from .plots import plot_poincare

        plot_poincare(corr.rr, title=args.file, path=args.plot)
    return 0


def cmd_alpha1(args) -> int:
    rr, corr = _load(args)
    s = alpha1_timeseries(rr, window_s=args.window, step_s=args.step, threshold=args.artifact_threshold)
    v = s.valid(args.max_artifacts)
    out = {
        "file": args.file,
        "n_beats": int(rr.size),
        "artifact_fraction": corr.artifact_fraction,
        "window_s": args.window,
        "step_s": args.step,
        "n_windows": int(s.t.size),
        "n_valid_windows": int(v.sum()),
        "alpha1_median": float(np.nanmedian(s.alpha1[v])) if v.any() else None,
    }
    thr = None
    if args.thresholds:
        try:
            thr = estimate_thresholds(s.alpha1[v], s.hr[v])
            out["thresholds"] = {
                "hrvt1_bpm": round(float(thr.hrvt1), 1), "hrvt2_bpm": round(float(thr.hrvt2), 1), "r2": round(thr.r2, 3),
                "n_windows": thr.n_windows, "extrapolated": list(thr.extrapolated),
                "reliable": thr.reliable,
            }
        except ValueError as e:
            out["thresholds"] = {"error": str(e)}
    if args.csv:
        np.savetxt(args.csv, np.column_stack([s.t, s.alpha1, s.hr, s.artifact_fraction]),
                   delimiter=",", header="t_end_s,alpha1,hr_bpm,artifact_fraction", comments="", fmt="%.5g")
    _emit(out, args)
    if args.plot:
        from .plots import plot_alpha1

        plot_alpha1(s, title=args.file, path=args.plot, thresholds=thr)
    return 0


def _emit(out, args):
    if args.json:
        print(json.dumps(out, indent=2))
    else:
        for k, v in out.items():
            print(f"{k:>22}: {v:.4g}" if isinstance(v, float) else f"{k:>22}: {v}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="hrv-threshold", description="DFA-alpha1 HRV thresholds and recovery metrics")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("file", help="RR file: .txt/.csv (ms or s) or .fit")
        sp.add_argument("--column", help="RR column name in a CSV with header")
        sp.add_argument("--artifact-threshold", type=float, default=0.20,
                        help="relative deviation from local median flagged as artifact (default 0.20)")
        sp.add_argument("--json", action="store_true", help="JSON output")
        sp.add_argument("--plot", help="save a PNG figure here")

    m = sub.add_parser("metrics", help="time/frequency/Poincare/alpha1 over the whole recording (rest/recovery)")
    common(m)
    m.set_defaults(func=cmd_metrics)

    a = sub.add_parser("alpha1", help="time-varying DFA alpha1 (exercise), optional HRVT1/HRVT2")
    common(a)
    a.add_argument("--window", type=float, default=120.0, help="window length s (default 120)")
    a.add_argument("--step", type=float, default=5.0, help="step s (default 5)")
    a.add_argument("--max-artifacts", type=float, default=0.05, help="drop windows above this artifact fraction")
    a.add_argument("--thresholds", action="store_true", help="estimate HR at alpha1 = 0.75 and 0.50")
    a.add_argument("--csv", help="write the alpha1 time series to CSV")
    a.set_defaults(func=cmd_alpha1)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
