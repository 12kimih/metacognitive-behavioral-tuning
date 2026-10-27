"""Where insertions fall in the original reasoning (Section 4.2 and Appendix F of the paper).

An insertion's position is pi = l / |r|, where l counts the characters of the original reasoning
before it and |r| all of them. Over the M insertions of a strategy pooled over traces, the
Kolmogorov-Smirnov statistic against uniform placement is

    D = sup_u | F(u) - u |,   F(u) = (1/M) * #{ l : pi_l <= u }.

Lower D means more even placement. `tenths` gives the share of insertions in each tenth of [0, 1]
(Figure 3a), which would be 10% everywhere under uniform placement.
"""

from __future__ import annotations

import statistics


def ks_uniform(positions: list[float]) -> float:
    """D against the uniform distribution on [0, 1], exact for the empirical distribution."""
    xs = sorted(min(1.0, max(0.0, float(p))) for p in positions)
    m = len(xs)
    if m == 0:
        return float("nan")
    d = 0.0
    for i, x in enumerate(xs):
        d = max(d, (i + 1) / m - x, x - i / m)
    return d


def tenths(positions: list[float]) -> list[float]:
    """Share of insertions (in percent) in each tenth [0, 0.1), ..., [0.9, 1.0]."""
    counts = [0] * 10
    for p in positions:
        counts[min(9, max(0, int(float(p) * 10)))] += 1
    total = sum(counts) or 1
    return [100.0 * c / total for c in counts]


def summary(positions: list[float]) -> dict:
    share = tenths(positions)
    return {"insertions": len(positions),
            "median_position": statistics.median(positions) if positions else float("nan"),
            "first_tenth_pct": share[0],
            "middle_40_to_90_pct": sum(share[4:9]),
            "ks_d": ks_uniform(positions),
            "tenths_pct": share}
