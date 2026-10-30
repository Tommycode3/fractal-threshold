"""Verifier (a): synthetic signals with known DFA exponent.

Note on tolerances: with the alpha1 box range (4..16 beats) DFA-1 has a
known small-scale bias (Kantelhardt et al. 2001): white noise reads ~0.58,
not 0.50. The threshold literature (Kubios alpha1) uses the same estimator,
so the bias is part of the calibration of 0.75 / 0.50. We therefore test:
  * alpha1 range: white 0.50 +/- 0.10, pink 1.00 +/- 0.06, Brownian 1.50 +/- 0.06
  * wide range (4..N/10, log-spaced): all three within +/- 0.05
"""

import numpy as np
import pytest

from hrvthreshold import alpha1, dfa
from hrvthreshold.dfa import fluctuation_function
from hrvthreshold.synth import brownian_noise, noise_with_alpha, power_law_noise, white_noise

N = 4096
REPS = 30


def _mean_alpha(gen, scales=None):
    rng = np.random.default_rng(42)
    if scales is None:
        return np.mean([alpha1(gen(N, rng)) for _ in range(REPS)])
    return np.mean([dfa(gen(N, rng), scales).alpha for _ in range(REPS)])


GENS = {
    "white": (lambda n, r: white_noise(n, r), 0.5),
    "pink": (lambda n, r: power_law_noise(n, 1.0, r), 1.0),
    "brownian": (lambda n, r: brownian_noise(n, r), 1.5),
}


@pytest.mark.parametrize("name,tol", [("white", 0.10), ("pink", 0.06), ("brownian", 0.06)])
def test_alpha1_range_synthetic(name, tol):
    gen, expected = GENS[name]
    assert abs(_mean_alpha(gen) - expected) < tol


@pytest.mark.parametrize("name", ["white", "pink", "brownian"])
def test_wide_range_synthetic(name):
    gen, expected = GENS[name]
    scales = np.unique(np.logspace(np.log10(4), np.log10(N // 10), 15).astype(int))
    assert abs(_mean_alpha(gen, scales) - expected) < 0.05


def test_monotone_in_true_alpha():
    rng = np.random.default_rng(7)
    est = [np.mean([alpha1(noise_with_alpha(1024, a, rng)) for _ in range(20)]) for a in (0.3, 0.5, 0.75, 1.0, 1.25)]
    assert all(np.diff(est) > 0)


def test_scale_and_offset_invariance():
    rng = np.random.default_rng(1)
    x = power_law_noise(600, 1.0, rng)
    assert alpha1(x) == pytest.approx(alpha1(800 + 37 * x), abs=1e-10)


def test_fluctuation_function_hand_computed():
    # y = cumsum of mean-removed x; with n=4 on x = [1,-1]*4 the profile is
    # [1,0,1,0,...]: per box, residual of linear fit to [1,0,1,0]
    x = np.array([1.0, -1.0] * 4)
    F = fluctuation_function(x, scales=[4])
    box = np.array([1.0, 0.0, 1.0, 0.0])
    t = np.arange(4)
    c = np.polyfit(t, box, 1)
    expected = np.sqrt(np.mean((box - np.polyval(c, t)) ** 2))
    assert F[0] == pytest.approx(expected)


def test_too_short_raises():
    with pytest.raises(ValueError):
        alpha1(np.arange(20.0))


def test_constant_input_is_nan():
    assert np.isnan(alpha1(np.full(200, 800.0)))


def test_beat_timing_jitter_lowers_alpha1():
    """Fiducial-point jitter e_k on beat times adds (e_{k+1} - e_k) to RR:
    an anti-correlated error that drags alpha1 toward 0. At exercise HR,
    where RR fluctuations are only a few ms, a few ms of jitter dominates."""
    rng = np.random.default_rng(11)
    rr = 400 + 4.0 * noise_with_alpha(4000, 1.0, rng)  # ~150 bpm, SD 4 ms
    beats = np.cumsum(rr)
    a = [alpha1(np.diff(beats + rng.normal(0, s, beats.size))) for s in (0.0, 1.0, 2.0, 4.0)]
    assert all(np.diff(a) < 0)
    assert a[0] > 0.9 and a[-1] < 0.5
