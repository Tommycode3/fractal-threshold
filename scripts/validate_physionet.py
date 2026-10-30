"""Verifier (c): DFA-alpha1 on real, openly downloadable PhysioNet recordings.

1. REST - Fantasia database (doi:10.13026/C2RG61; Iyengar et al. 1996),
   young subjects f1y01..f1y10, ~2 h supine rest. RR from the reference
   beat annotations (*.ecg).
2. EXERCISE STRESS TESTS - MIT-BIH ST Change Database (doi:10.13026/C2ZW2H;
   Albrecht 1983), 28 ECG records "most of which were recorded during
   exercise stress tests", with reference beat annotations (*.atr).
   Intervals touching a non-normal beat are treated as artifacts (NN only).
   Patients, not athletes; no gas-exchange reference -> we can show alpha1
   *behaviour* vs HR, not validate VT1/VT2 accuracy.
3. FIELD ECG - Wrist PPG During Exercise (doi:10.13026/C2PQ1X; Jarchi &
   Casson 2017), chest-ECG channel during walk/run/cycling. No beat
   annotations: two R-peak detectors compared (wfdb XQRS vs a band-pass
   + adaptive-threshold detector with parabolic sub-sample refinement).

Writes results/physionet_validation.json and results/alpha1/*.csv.
Needs network + `pip install -e .[validate]`. Not run in CI.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import wfdb
from scipy.ndimage import maximum_filter1d
from scipy.signal import butter, find_peaks, sosfiltfilt
from wfdb import processing

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hrvthreshold import all_metrics, alpha1_timeseries, correct_rr, estimate_thresholds  # noqa: E402
from hrvthreshold.dfa import alpha1 as alpha1_fn  # noqa: E402

OUT = ROOT / "results"
BEATS = list("NLRBAaJSVrFejnE/fQ")
NORMAL = list("NLRBej")  # sinus-origin beat labels (incl. bundle-branch-block beats)
FANTASIA = [f"f1y{i:02d}" for i in range(1, 11)]
STDB = [str(i) for i in range(300, 328)]
WRIST = [
    "s1_walk", "s1_low_resistance_bike", "s1_high_resistance_bike",
    "s2_walk", "s2_low_resistance_bike", "s2_high_resistance_bike",
    "s3_walk", "s3_low_resistance_bike", "s3_high_resistance_bike", "s3_run",
    "s4_run", "s5_low_resistance_bike", "s5_run",
    "s6_walk", "s6_low_resistance_bike", "s6_run",
    "s8_walk", "s8_run", "s9_walk",
]


# --------------------------------------------------------------------- helpers
def nolds_max_diff(rr, s, k=20):
    """Max |ours - nolds| over k evenly spaced *valid* windows of the same series."""
    import nolds

    t = np.cumsum(rr) / 1000
    t -= t[0]
    diffs = []
    ok = np.flatnonzero(s.valid(0.05))
    if ok.size == 0:
        return None
    for i in ok[np.linspace(0, ok.size - 1, min(k, ok.size)).astype(int)]:
        seg = rr[(t >= s.t[i] - 120) & (t <= s.t[i])]
        if seg.size >= 50:
            ref = nolds.dfa(seg, nvals=list(range(4, 17)), overlap=False, fit_exp="poly")
            diffs.append(abs(alpha1_fn(seg) - ref))
    return float(max(diffs)) if diffs else None


def save_series(name, s):
    np.savetxt(OUT / "alpha1" / f"{name}.csv", np.column_stack([s.t, s.alpha1, s.hr, s.artifact_fraction]),
               delimiter=",", header="t_end_s,alpha1,hr_bpm,artifact_fraction", comments="", fmt="%.5g")


def ann_rr(rec, ext, pn_dir):
    ann = wfdb.rdann(rec, ext, pn_dir=pn_dir)
    sym = np.array(ann.symbol)
    b = np.isin(sym, BEATS)
    samp, sym = ann.sample[b], sym[b]
    rr = np.diff(samp) / ann.fs * 1000.0
    normal = np.isin(sym, NORMAL)
    non_nn = ~(normal[:-1] & normal[1:])
    return rr, non_nn


def detect_bandpass(sig, fs):
    sos = butter(3, [8, 30], btype="band", fs=fs, output="sos")
    b = sosfiltfilt(sos, np.nan_to_num(sig))
    pol = -1.0 if abs(np.percentile(b, 0.5)) > abs(np.percentile(b, 99.5)) else 1.0
    e = pol * b
    env = maximum_filter1d(e, int(2 * fs))
    pk, _ = find_peaks(e, height=0.4 * env, distance=int(0.33 * fs))
    pk = pk[(pk > 0) & (pk < e.size - 1)]
    # parabolic sub-sample refinement of the peak position
    y0, y1, y2 = e[pk - 1], e[pk], e[pk + 1]
    den = y0 - 2 * y1 + y2
    delta = np.where(den != 0, 0.5 * (y0 - y2) / np.where(den == 0, 1, den), 0.0)
    return pk + np.clip(delta, -0.5, 0.5)


def detect_xqrs(sig, fs):
    sig = np.nan_to_num(sig - np.nanmedian(sig))
    q = processing.xqrs_detect(sig, fs=fs, verbose=False)
    q = processing.correct_peaks(sig, q, search_radius=int(0.05 * fs), smooth_window_size=int(0.15 * fs),
                                 peak_dir="compare")
    return np.unique(q[q >= 0]).astype(float)


def series_summary(rr, extra=None):
    corr = correct_rr(rr, extra_flags=extra)
    s = alpha1_timeseries(rr, extra_flags=extra)
    v = s.valid(0.05)
    return corr, s, v


# --------------------------------------------------------------------- parts
def part_rest():
    rows = []
    for rec in FANTASIA:
        rr, non_nn = ann_rr(rec, "ecg", "fantasia")
        corr, s, v = series_summary(rr, non_nn)
        save_series(rec, s)
        m = all_metrics(corr.rr[np.cumsum(corr.rr) <= 300_000])  # first 5 min
        rows.append({
            "record": rec, "duration_min": round(float(rr.sum() / 60000), 1),
            "artifact_pct": round(100 * corr.artifact_fraction, 2),
            "valid_windows": f"{int(v.sum())}/{v.size}",
            "hr_median": round(float(np.median(s.hr[v])), 1),
            "alpha1_median": round(float(np.median(s.alpha1[v])), 3),
            "rmssd_5min_ms": round(m["rmssd_ms"], 1), "lf_hf_5min": round(m["lf_hf"], 2),
            "sd1_ms": round(m["sd1_ms"], 1), "sd2_ms": round(m["sd2_ms"], 1),
            "max_abs_diff_vs_nolds": nolds_max_diff(corr.rr, s),
        })
        print("rest", rows[-1], flush=True)
    return rows


def part_stress():
    rows = []
    for rec in STDB:
        rr, non_nn = ann_rr(rec, "atr", "stdb")
        corr, s, v = series_summary(rr, non_nn)
        save_series(f"stdb_{rec}", s)
        hr, a = s.hr[v], s.alpha1[v]
        row = {
            "record": rec, "duration_min": round(float(rr.sum() / 60000), 1),
            "artifact_pct": round(100 * corr.artifact_fraction, 2),
            "valid_windows": f"{int(v.sum())}/{v.size}",
            "max_abs_diff_vs_nolds": nolds_max_diff(corr.rr, s),
        }
        if v.sum() >= 10:
            lo, hi = hr <= np.percentile(hr, 10), hr >= np.percentile(hr, 90)
            row.update({
                "hr_min": round(float(hr.min()), 1), "hr_max": round(float(hr.max()), 1),
                "alpha1_at_lowest_decile_hr": round(float(np.median(a[lo])), 3),
                "alpha1_at_highest_decile_hr": round(float(np.median(a[hi])), 3),
                "spearman_alpha1_hr": round(float(_spearman(a, hr)), 3),
            })
            # thresholds on the ascending limb only (start -> HR peak)
            k = int(np.argmax(hr))
            if hr.max() >= 140 and k >= 20:
                try:
                    th = estimate_thresholds(a[: k + 1], hr[: k + 1])
                    row["thresholds_ascending_limb"] = {
                        "hrvt1_bpm": round(th.hrvt1, 1), "hrvt2_bpm": round(th.hrvt2, 1),
                        "hrvt1_pct_hrpeak": round(100 * th.hrvt1 / hr.max(), 1),
                        "hrvt2_pct_hrpeak": round(100 * th.hrvt2 / hr.max(), 1),
                        "r2": round(th.r2, 3), "n_windows": th.n_windows, "extrapolated": list(th.extrapolated),
                        "reliable": th.reliable,
                    }
                except ValueError as e:
                    row["thresholds_ascending_limb"] = {"error": str(e)}
        rows.append(row)
        print("stress", row, flush=True)
    return rows


def part_field():
    rows = []
    for rec in WRIST:
        r = wfdb.rdrecord(rec, pn_dir="wrist", channel_names=["chest_ecg"])
        sig, fs = r.p_signal[:, 0], r.fs
        row = {"record": rec, "activity": rec.split("_", 1)[1], "duration_min": round(len(sig) / fs / 60, 1)}
        for name, det in (("bandpass", detect_bandpass), ("xqrs", detect_xqrs)):
            pk = det(sig, fs)
            rr = np.diff(pk) / fs * 1000.0
            corr, s, v = series_summary(rr)
            if name == "bandpass":
                save_series(f"wrist_{rec}", s)
            row[name] = {
                "artifact_pct": round(100 * corr.artifact_fraction, 1),
                "valid_windows": f"{int(v.sum())}/{v.size}",
                "hr_median": round(float(np.nanmedian(s.hr)), 1),
                "alpha1_median": round(float(np.median(s.alpha1[v])), 3) if v.any() else None,
            }
        rows.append(row)
        print("field", row, flush=True)
    return rows


def build_summary(rest, stress, field):
    a_rest = [r["alpha1_median"] for r in rest]
    hi = [r for r in stress if "alpha1_at_highest_decile_hr" in r and r["hr_max"] >= 140]
    drop = [r["alpha1_at_lowest_decile_hr"] - r["alpha1_at_highest_decile_hr"] for r in hi]
    rel = [r["thresholds_ascending_limb"] for r in hi
           if r.get("thresholds_ascending_limb", {}).get("reliable")]
    disagree = [r for r in field if r["bandpass"]["alpha1_median"] is not None and r["xqrs"]["alpha1_median"] is not None
                and abs(r["bandpass"]["alpha1_median"] - r["xqrs"]["alpha1_median"]) > 0.15]
    return {
        "rest_fantasia_alpha1": {"n": len(a_rest), "mean": round(float(np.mean(a_rest)), 3),
                                 "sd": round(float(np.std(a_rest, ddof=1)), 3),
                                 "min": float(min(a_rest)), "max": float(max(a_rest))},
        "stress_tests_reaching_hr140": {
            "n": len(hi),
            "alpha1_lowest_hr_decile_mean": round(float(np.mean([r["alpha1_at_lowest_decile_hr"] for r in hi])), 3),
            "alpha1_highest_hr_decile_mean": round(float(np.mean([r["alpha1_at_highest_decile_hr"] for r in hi])), 3),
            "n_records_alpha1_drop_gt_0.2": int(sum(d > 0.2 for d in drop)),
            "spearman_alpha1_hr_median": round(float(np.median([r["spearman_alpha1_hr"] for r in hi])), 3),
            "n_reliable_threshold_fits": len(rel),
            "hrvt1_pct_hrpeak_median": round(float(np.median([t["hrvt1_pct_hrpeak"] for t in rel])), 1) if rel else None,
            "hrvt2_pct_hrpeak_median": round(float(np.median([t["hrvt2_pct_hrpeak"] for t in rel])), 1) if rel else None,
        },
        "field_wrist_ecg": {
            "n_records": len(field),
            "records_with_any_valid_window_bandpass": sum(r["bandpass"]["alpha1_median"] is not None for r in field),
            "records_with_any_valid_window_xqrs": sum(r["xqrs"]["alpha1_median"] is not None for r in field),
            "records_where_detectors_disagree_by_gt_0.15": [r["record"] for r in disagree],
        },
        "max_abs_diff_vs_nolds_all_records": max(x for x in
                                                  [r["max_abs_diff_vs_nolds"] for r in rest + stress] if x is not None),
    }


def _spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return np.corrcoef(ra, rb)[0, 1]


def main():
    (OUT / "alpha1").mkdir(parents=True, exist_ok=True)
    rest, stress, field = part_rest(), part_stress(), part_field()
    summary = build_summary(rest, stress, field)
    res = {"summary": summary, "rest_fantasia": rest, "stress_stdb": stress, "field_wrist_chest_ecg": field}
    (OUT / "physionet_validation.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
