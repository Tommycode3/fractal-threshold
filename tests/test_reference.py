"""Verifier (b): cross-check against an independent implementation (nolds).

nolds.dfa with overlap=False, linear detrending and an OLS fit of
log F(n) over n = 4..16 is the same estimator as ours; results must agree
to floating-point precision.
"""

import numpy as np
import pytest

from hrvthreshold import alpha1
from hrvthreshold.synth import noise_with_alpha

nolds = pytest.importorskip("nolds")


@pytest.mark.parametrize("a", [0.3, 0.5, 0.75, 1.0, 1.5])
@pytest.mark.parametrize("n", [150, 240, 1000])
def test_matches_nolds(a, n):
    rng = np.random.default_rng(int(a * 100) + n)
    x = 800 + 30 * noise_with_alpha(n, a, rng)
    ref = nolds.dfa(x, nvals=list(range(4, 17)), overlap=False, order=1, fit_exp="poly")
    assert alpha1(x) == pytest.approx(ref, abs=1e-9)
