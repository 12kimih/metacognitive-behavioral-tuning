"""Pre-compute per-sample difficulty tier for MHQA / math benchmarks.

Output: data/sample_difficulty.csv with columns
  benchmark, sample_id, hf_id, tier, L_star

The tier mapping uses metadata that the original dataset authors attached to
each sample (HotpotQA `level`, MuSiQue `id` prefix, 2Wiki `type`, MATH-500
`level`). The 1-indexed sample_id matches the convention used by
src/mbt/tasks/{hotpotqa,musique,2wikimultihopqa,math500}.py:format_columns,
i.e. `idx + 1` over the dataset iteration order.

Usage::

    uv run python scripts/build_difficulty.py
    uv run python scripts/build_difficulty.py --benchmarks musique 2wikimultihopqa
    uv run python scripts/build_difficulty.py --out data/sample_difficulty.csv

See docs/scoring_redesign.md §3.2.1 / §5.3 for the rationale.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from datasets import load_dataset

L_STAR = {"easy": 1, "medium": 3, "hard": 4}


def hotpotqa_tier(row: dict) -> str:
    """HotpotQA tier with a fallback for the hard-only validation split.

    The training split has a balanced 'level' distribution (Yang et al. 2018:
    easy 17,972 / medium 56,814 / hard 15,661), so we trust the author label
    when it is 'easy' or 'medium'. The official `distractor` validation
    split, however, marks every example as 'hard', making the label
    uninformative for evaluation reporting.

    To recover sub-difficulty within the validation split, we fall back on
    two structural signals that are present on every example regardless of
    split:
      - `type`: 'bridge' (a true multi-hop join) vs 'comparison'
        (a yes/no or attribute comparison that typically resolves in one step).
      - `supporting_facts` list length: the number of sentence-level
        evidence pointers, which is a direct proxy for the number of
        reasoning hops actually required.

    Sub-tiering when level is 'hard' or absent:
      - 3 or more supporting facts -> hard (genuine 3+ hop)
      - 2 supporting facts and type == 'comparison' -> easy (yes/no compare)
      - 2 supporting facts and type == 'bridge' -> medium (canonical 2-hop)
    """
    level = row.get("level")
    if level in {"easy", "medium"}:
        return level
    sf = row.get("supporting_facts") or {}
    if isinstance(sf, dict):
        sf_len = len(sf.get("title", []))
    else:
        sf_len = len(sf)
    qtype = row.get("type", "bridge")
    if sf_len >= 3:
        return "hard"
    if qtype == "comparison":
        return "easy"
    return "medium"


def musique_tier(row: dict) -> str:
    """MuSiQue dataset-relative tier (Trivedi et al. 2022).

    The dataset has no 1-hop questions by design (its smallest unit is a
    2-hop join). On the validation split (size 2,417) the prefix
    distribution is 2hop 1,252 / 3hop* 760 / 4hop* 405, so an absolute
    hop -> tier mapping (2hop->medium, 3hop+->hard) leaves the 'easy' tier
    empty.

    Instead we tier *relative to MuSiQue's own range*: 2-hop questions are
    the easiest reasoning structure available in this dataset, and 3-hop /
    4-hop are progressively harder. This keeps all three tiers populated,
    which is what conditional appropriateness needs.

    Sub-tiering:
      - id starts with '4hop' -> hard
      - id starts with '3hop' -> medium
      - id starts with '2hop' -> easy
      - answerable == False (MuSiQue-Full contrast subset) -> hard
    """
    if not row.get("answerable", True):
        return "hard"
    sid = str(row.get("id", ""))
    if sid.startswith("4hop"):
        return "hard"
    if sid.startswith("3hop"):
        return "medium"
    if sid.startswith("2hop"):
        return "easy"
    return "medium"


def twowiki_tier(row: dict) -> str:
    """Ho et al. (2020): 'type' partitions samples into reasoning structures."""
    return {
        "comparison": "easy",
        "inference": "medium",
        "compositional": "medium",
        "bridge_comparison": "hard",
    }.get(row.get("type", "compositional"), "medium")


def math500_tier(row: dict) -> str:
    """Hendrycks MATH 'level' is an integer in 1..5; tier mapping is monotonic."""
    try:
        lvl = int(row.get("level", 3))
    except (TypeError, ValueError):
        return "medium"
    if lvl <= 2:
        return "easy"
    if lvl == 3:
        return "medium"
    return "hard"


HANDLERS: list[tuple[str, str, str | None, str, callable]] = [
    ("hotpotqa", "hotpotqa/hotpot_qa", "distractor", "validation", hotpotqa_tier),
    ("musique", "dgslibisey/MuSiQue", None, "validation", musique_tier),
    ("2wikimultihopqa", "framolfese/2WikiMultihopQA", None, "validation", twowiki_tier),
    ("math500", "HuggingFaceH4/MATH-500", None, "test", math500_tier),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/sample_difficulty.csv", help="output csv path")
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=None,
        help="subset of benchmarks to process (default: all)",
    )
    args = parser.parse_args()

    selected = set(args.benchmarks) if args.benchmarks else None
    rows: list[dict] = []

    for bench, hf_id, cfg, split, handler in HANDLERS:
        if selected is not None and bench not in selected:
            continue
        cfg_str = cfg if cfg is not None else "<default>"
        print(f"[build_difficulty] loading {bench}: {hf_id} (config={cfg_str}, split={split})")
        ds = load_dataset(hf_id, cfg, split=split)
        for idx, ex in enumerate(ds):
            tier = handler(ex)
            rows.append(
                {
                    "benchmark": bench,
                    "sample_id": idx + 1,
                    "hf_id": str(ex.get("id", idx + 1)),
                    "tier": tier,
                    "L_star": L_STAR.get(tier, 3),
                }
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["benchmark", "sample_id", "hf_id", "tier", "L_star"])
        writer.writeheader()
        writer.writerows(rows)

    summary = Counter((r["benchmark"], r["tier"]) for r in rows)
    print(f"[build_difficulty] wrote {len(rows)} rows to {out}")
    print("[build_difficulty] tier distribution:")
    for (bench, tier), n in sorted(summary.items()):
        print(f"    {bench:20s} {tier:6s} {n}")


if __name__ == "__main__":
    main()
