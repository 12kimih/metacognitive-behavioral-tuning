import math

import numpy as np
import pytest

from mbt.stats import clogit, positions


def test_ks_d_of_known_samples():
    assert positions.ks_uniform([0.5]) == pytest.approx(0.5)
    assert positions.ks_uniform([0.0, 0.0]) == pytest.approx(1.0)
    evenly = [(i + 0.5) / 100 for i in range(100)]
    assert positions.ks_uniform(evenly) == pytest.approx(0.005)
    assert positions.tenths([0.05, 0.95, 1.0]) == pytest.approx(
        [100 / 3, 0, 0, 0, 0, 0, 0, 0, 0, 200 / 3])


def test_matched_pairs_reduce_to_logistic_regression_on_differences():
    # For 1:1 pairs the conditional likelihood is logistic regression of 1 on (x_case - x_ctrl)
    # without an intercept, so the unpenalised fit must agree with a direct Newton solution.
    rng = np.random.default_rng(0)
    n = 300
    x_case = rng.binomial(1, 0.6, n).astype(float)
    x_ctrl = rng.binomial(1, 0.4, n).astype(float)
    outcome, X, groups = [], [], []
    for i in range(n):
        outcome += [1, 0]
        X += [[x_case[i]], [x_ctrl[i]]]
        groups += [i, i]
    fit = clogit.clogit_firth(outcome, X, groups, firth=False)
    d = x_case - x_ctrl
    b = 0.0
    for _ in range(100):
        p = 1 / (1 + np.exp(-b * d))
        b += float(np.sum(d * (1 - p)) / np.sum(d * d * p * (1 - p)))
    assert fit["beta"][0] == pytest.approx(b, rel=1e-6)
    discordant_up = int(np.sum(d > 0))
    discordant_down = int(np.sum(d < 0))
    assert b == pytest.approx(math.log(discordant_up / discordant_down), rel=1e-6)


def test_firth_keeps_the_estimate_finite_under_separation():
    outcome = [1, 0] * 5
    X = [[1.0], [0.0]] * 5           # the behavior is present exactly in every correct trace
    groups = [i // 2 for i in range(10)]
    fit = clogit.clogit_firth(outcome, X, groups)
    assert fit["converged"] and math.isfinite(fit["beta"][0]) and fit["beta"][0] > 0
    res = clogit.contrast(fit, [1])
    assert res["ci"][0] < res["odds_ratio"] < res["ci"][1]


def test_holm():
    assert clogit.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
