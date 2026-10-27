"""Conditional logistic regression with Firth's penalty, in NumPy (Appendix H of the paper).

The paper asks whether a behavior is associated with a correct answer WITHIN the same problem and
training condition. Each (problem, condition) is a group holding a correct and an incorrect trace
(437 such pairs in the paper). Conditioning on the number of correct traces in each group removes
the group effects exactly (Breslow and Day 1980), so a group whose members all have the same
outcome carries no information and is dropped. With few discordant pairs the ordinary estimate can
diverge, so the log likelihood is penalised by half the log determinant of the information
(Firth 1993; Heinze and Puhr 2010).

    fit = clogit_firth(outcome, X, groups)
    contrast(fit, [1, 0])   # odds ratio, Wald 95% interval and p for the first column
"""

from __future__ import annotations

import itertools
import math

import numpy as np

Z95 = 1.959963984540054


def _groups(outcome, X, groups):
    """Per informative group: its covariate rows, all subsets of the size of its number of
    positive outcomes, and the members with outcome 1."""
    X = np.asarray(X, float)
    members: dict = {}
    for i, g in enumerate(groups):
        members.setdefault(g, []).append(i)
    out = []
    for idx in members.values():
        m = sum(int(outcome[i]) for i in idx)
        if 0 < m < len(idx):
            subsets = [list(s) for s in itertools.combinations(range(len(idx)), m)]
            chosen = [k for k, i in enumerate(idx) if outcome[i]]
            out.append((X[idx], subsets, chosen))
    return out


def conditional_loglik(beta, data):
    """Conditional log likelihood, score and information, exact by enumerating subsets."""
    k = len(beta)
    ll, score, info = 0.0, np.zeros(k), np.zeros((k, k))
    for X, subsets, chosen in data:
        sums = np.asarray([X[s].sum(axis=0) for s in subsets])
        eta = sums @ beta
        top = eta.max()
        w = np.exp(eta - top)
        w /= w.sum()
        mean = w @ sums
        observed = X[chosen].sum(axis=0)
        ll += observed @ beta - (top + math.log(np.exp(eta - top).sum()))
        score += observed - mean
        centred = sums - mean
        info += (centred * w[:, None]).T @ centred
    return ll, score, info


def clogit_firth(outcome, X, groups, *, firth: bool = True, max_iter: int = 200) -> dict | None:
    """Fit; return `{"beta", "cov", "converged", "groups"}` or None when nothing varies.

    A column that is constant inside every informative group is removed by the conditioning; it
    is left out of the fit and reported as NaN.
    """
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    data = _groups(outcome, X, groups)
    if not data or all(np.ptp(Xg, axis=0).max() == 0 for Xg, _s, _c in data):
        return None
    keep = [j for j in range(X.shape[1]) if any(np.ptp(Xg[:, j]) > 0 for Xg, _s, _c in data)]
    if len(keep) < X.shape[1]:
        fit = clogit_firth(outcome, X[:, keep], groups, firth=firth, max_iter=max_iter)
        beta = np.full(X.shape[1], math.nan)
        cov = np.full((X.shape[1], X.shape[1]), math.nan)
        beta[keep] = fit["beta"]
        cov[np.ix_(keep, keep)] = fit["cov"]
        return {**fit, "beta": beta, "cov": cov,
                "dropped": [j for j in range(X.shape[1]) if j not in keep]}
    k = X.shape[1]

    def objective(beta):
        ll, _score, info = conditional_loglik(beta, data)
        if not firth:
            return ll
        sign, logdet = np.linalg.slogdet(info)
        return ll + 0.5 * logdet if sign > 0 else -math.inf

    beta, ok, h = np.zeros(k), False, 1e-5
    for _ in range(max_iter):
        _ll, score, info = conditional_loglik(beta, data)
        gradient = score
        if firth:
            # The penalty's gradient by central differences of the penalised objective.
            gradient = np.asarray([(objective(beta + h * e) - objective(beta - h * e)) / (2 * h)
                                   for e in np.eye(k)])
        step = np.linalg.solve(info, gradient)
        biggest = np.max(np.abs(step))
        if biggest > 2:
            step *= 2 / biggest
        now = objective(beta)
        while objective(beta + step) < now - 1e-12 and np.max(np.abs(step)) > 1e-10:
            step /= 2
        beta = beta + step
        if np.max(np.abs(step)) < 1e-8:
            ok = True
            break
    _ll, _score, info = conditional_loglik(beta, data)
    return {"beta": beta, "cov": np.linalg.inv(info), "converged": ok, "groups": len(data)}


def normal_two_sided_p(z: float) -> float:
    return math.erfc(abs(z) / math.sqrt(2))


def contrast(fit: dict | None, c) -> dict:
    """The combination c . beta with its odds ratio, Wald 95% interval and two-sided p."""
    nan = math.nan
    if fit is None:
        return {"coef": nan, "se": nan, "odds_ratio": nan, "ci": (nan, nan), "p": nan,
                "converged": False, "groups": 0}
    c = np.asarray(c, float)
    on = np.flatnonzero(c)
    beta, cov = fit["beta"][on], fit["cov"][np.ix_(on, on)]
    if np.isnan(beta).any():
        return {**contrast(None, c), "groups": fit["groups"]}
    b = float(c[on] @ beta)
    se = float(math.sqrt(c[on] @ cov @ c[on]))
    return {"coef": b, "se": se, "odds_ratio": math.exp(b),
            "ci": (math.exp(b - Z95 * se), math.exp(b + Z95 * se)),
            "p": normal_two_sided_p(b / se), "converged": fit["converged"],
            "groups": fit["groups"]}


def holm(pvalues: list[float]) -> list[float]:
    """Holm's step-down adjustment, in the input order."""
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    out, running = [0.0] * len(pvalues), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[i]))
        out[i] = running
    return out


def behavior_association(present, correct, groups) -> dict:
    """The Table H1 estimate for one behavior: logit P(correct) = group effect + b * present."""
    return contrast(clogit_firth(correct, np.asarray(present, float)[:, None], groups), [1])
