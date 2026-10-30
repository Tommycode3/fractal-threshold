import json

import numpy as np
import pytest

from hrvthreshold import (
    alpha1_timeseries, correct_rr, estimate_thresholds, frequency_domain, poincare, read_rr, time_domain,
)
from hrvthreshold.cli import main
from hrvthreshold.synth import noise_with_alpha, synthetic_ramp

from .fitwriter import write_hrv_fit


# ---------- artifacts ----------
def test_artifact_detection_and_interpolation():
    rng = np.random.default_rng(0)
    rr = 800 + 20 * noise_with_alpha(500, 1.0, rng)
    dirty = rr.copy()
    idx = [50, 120, 121, 300]
    dirty[50] = 1600  # missed beat
    dirty[120], dirty[121] = 400, 400  # extra (split) beat
    dirty[300] = 100  # out of physiological range
    res = correct_rr(dirty)
    assert set(idx) <= set(np.flatnonzero(res.flagged))
    assert res.flagged.sum() <= len(idx) + 4  # few false positives
    assert np.max(np.abs(res.rr[idx] - rr[idx])) < 80
    assert res.rr.size == rr.size


def test_artifact_delete_method():
    rr = np.full(100, 800.0)
    rr[10] = 2000
    res = correct_rr(rr, method="delete")
    assert res.rr.size == 99


def test_clean_series_untouched():
    rng = np.random.default_rng(3)
    rr = 900 + 15 * noise_with_alpha(1000, 1.0, rng)
    res = correct_rr(rr)
    assert res.artifact_fraction == 0.0
    np.testing.assert_array_equal(res.rr, rr)


# ---------- time / frequency / poincare ----------
def test_time_domain_hand_values():
    rr = np.array([800, 850, 790, 900, 880.0])
    d = np.diff(rr)
    m = time_domain(rr)
    assert m["rmssd_ms"] == pytest.approx(np.sqrt(np.mean(d**2)))
    assert m["sdnn_ms"] == pytest.approx(np.std(rr, ddof=1))
    assert m["pnn50_pct"] == pytest.approx(100 * np.mean(np.abs(d) > 50))


def test_poincare_identities():
    # SD1^2 = 0.5 var(diff); SD1^2 + SD2^2 = 2 var(RR) (approximately, for long series)
    rng = np.random.default_rng(5)
    rr = 850 + 30 * noise_with_alpha(5000, 1.0, rng)
    p = poincare(rr)
    assert p["sd1_ms"] ** 2 == pytest.approx(0.5 * np.var(np.diff(rr), ddof=1), rel=1e-6)
    assert p["sd1_ms"] ** 2 + p["sd2_ms"] ** 2 == pytest.approx(2 * np.var(rr, ddof=1), rel=0.01)


def test_lomb_scargle_lf_hf_ratio():
    # tachogram = 1000 ms + A_lf sin(2pi 0.10 t) + A_hf sin(2pi 0.25 t)  -> LF/HF = (A_lf/A_hf)^2
    t, rr = 0.0, []
    while t < 600:
        v = 1000 + 30 * np.sin(2 * np.pi * 0.10 * t) + 15 * np.sin(2 * np.pi * 0.25 * t)
        rr.append(v)
        t += v / 1000
    fd = frequency_domain(np.array(rr))
    assert fd["lf_peak_hz"] == pytest.approx(0.10, abs=0.005)
    assert fd["hf_peak_hz"] == pytest.approx(0.25, abs=0.005)
    assert fd["lf_hf"] == pytest.approx(4.0, rel=0.15)
    # Parseval scaling: LF + HF ~ variance of the tachogram
    assert fd["lf_ms2"] + fd["hf_ms2"] == pytest.approx(np.var(rr), rel=0.1)


# ---------- windowing + thresholds ----------
def test_window_geometry():
    rr = np.full(1200, 1000.0)  # 20 min at 60 bpm
    s = alpha1_timeseries(rr + np.random.default_rng(0).normal(0, 10, rr.size))
    assert s.t[0] == pytest.approx(120.0)
    assert np.all(np.diff(s.t) == pytest.approx(5.0))
    assert np.nanmedian(s.n_beats) == pytest.approx(120, abs=1)
    assert np.nanmedian(s.hr) == pytest.approx(60, abs=0.5)


def test_synthetic_ramp_alpha1_declines_and_thresholds_ordered():
    rr, _, _ = synthetic_ramp(seed=1)
    s = alpha1_timeseries(rr)
    v = s.valid()
    first, last = s.alpha1[v][:20].mean(), s.alpha1[v][-20:].mean()
    assert first > 0.9 and last < 0.6
    th = estimate_thresholds(s.alpha1[v], s.hr[v])
    assert 110 < th.hrvt1 < th.hrvt2 < 200
    assert th.r2 > 0.6


def test_thresholds_exact_on_linear_data():
    hr = np.linspace(100, 180, 50)
    a = 2.0 - 0.01 * hr  # 0.75 at 125, 0.50 at 150
    th = estimate_thresholds(a, hr)
    assert th.hrvt1 == pytest.approx(125)
    assert th.hrvt2 == pytest.approx(150)
    assert th.r2 == pytest.approx(1.0)


# ---------- IO ----------
def test_read_text_ms_and_seconds(tmp_path):
    p1 = tmp_path / "rr.txt"
    p1.write_text("800\n810\n# comment\n790\n")
    np.testing.assert_allclose(read_rr(p1), [800, 810, 790])
    p2 = tmp_path / "rr_s.txt"
    p2.write_text("0.8\n0.81\n")
    np.testing.assert_allclose(read_rr(p2), [800, 810])


def test_read_polar_sensor_logger_csv(tmp_path):
    p = tmp_path / "polar.txt"
    p.write_text("Phone timestamp;RR-interval [ms]\n2026-01-01T10:00:00.000;812\n"
                 "2026-01-01T10:00:00.800;798\n2026-01-01T10:00:01.600;805\n")
    np.testing.assert_allclose(read_rr(p), [812, 798, 805])


def test_read_fit_roundtrip(tmp_path):
    pytest.importorskip("fitdecode")
    rr_s = [0.812, 0.798, 0.805, 0.790, 0.801, 0.777, 0.765]  # 7 -> 2 msgs, 3 invalid pads
    p = tmp_path / "act.fit"
    write_hrv_fit(p, rr_s)
    np.testing.assert_allclose(read_rr(p), np.array(rr_s) * 1000)


# ---------- CLI ----------
def test_cli_alpha1_json(tmp_path, capsys):
    rr, _, _ = synthetic_ramp(seed=2)
    p = tmp_path / "ramp.txt"
    np.savetxt(p, rr, fmt="%.1f")
    assert main(["alpha1", str(p), "--thresholds", "--json", "--plot", str(tmp_path / "a.png")]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["n_valid_windows"] > 100
    assert out["thresholds"]["hrvt1_bpm"] < out["thresholds"]["hrvt2_bpm"]
    assert (tmp_path / "a.png").stat().st_size > 10_000


def test_cli_metrics(tmp_path, capsys):
    rng = np.random.default_rng(9)
    p = tmp_path / "rest.csv"
    np.savetxt(p, 950 + 40 * noise_with_alpha(400, 1.0, rng), fmt="%.1f")
    assert main(["metrics", str(p), "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    for k in ("rmssd_ms", "sdnn_ms", "pnn50_pct", "lf_hf", "sd1_ms", "dfa_alpha1"):
        assert np.isfinite(out[k])
