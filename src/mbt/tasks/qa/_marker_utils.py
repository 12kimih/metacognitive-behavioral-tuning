"""Paragraph-marker helpers for the rrp-score (RRP) judge.

Convention: a `reasoning_trace` is split on blank lines into paragraphs, then a
zero-padded ASCII marker `[[Mk]]` is inserted IMMEDIATELY AFTER paragraph k for
k = 1..N-1. The last paragraph has no trailing marker; the judge uses the
literal sentinel `[[M_END]]` to point at it. The judge uses `[[M00]]` to mean
"no paragraph derives the gold answer".

Marker → 1-indexed sentence position mapping is intentionally conservative:
"answer derived in paragraph k" maps to the LAST sentence of paragraph k, so
`first_correct / total_sentences` keeps its meaning as a progress rate (rounds
slightly *late*, which is the safe direction for an overthinking metric).
"""
from __future__ import annotations

import re

_MARKER_RE = re.compile(r"\[\[M(\d+)\]\]")
SENT_END = re.compile(r"(?<=[.!?])\s+")


def split_paragraphs(trace: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", trace.strip()) if p.strip()]


def inject_markers(paragraphs: list[str]) -> tuple[str, list[str]]:
    """Insert `[[Mk]]` after paragraph k for k = 1..N-1. Last paragraph: no marker.

    Returns `(marked_text, marker_ids)` where `marker_ids[k-1]` follows
    paragraph k (1-indexed); `len(marker_ids) == len(paragraphs) - 1`.
    """
    marker_ids = [f"M{i:02d}" for i in range(1, len(paragraphs))]
    parts: list[str] = []
    for k, p in enumerate(paragraphs):
        parts.append(p)
        if k < len(paragraphs) - 1:
            parts.append(f"[[{marker_ids[k]}]]")
    return "\n\n".join(parts), marker_ids


def marker_to_paragraph_index(token: str, num_paragraphs: int) -> int:
    """`[[M00]]`/`NONE` → 0; `[[M_END]]` → num_paragraphs; `[[Mk]]` → k (clamped)."""
    t = token.strip().upper()
    if t in ("M00", "[[M00]]", "NONE"):
        return 0
    if t in ("M_END", "[[M_END]]"):
        return num_paragraphs
    mo = _MARKER_RE.search(t)
    if not mo:
        return 0
    k = int(mo.group(1))
    return max(0, min(num_paragraphs, k))


def first_correct_sentence_from_paragraph(paragraphs: list[str], k: int) -> int:
    """Last sentence of paragraph k (1-indexed). k=0 -> -1 (sentinel: not found)."""
    if k <= 0:
        return -1
    cum = 0
    for i in range(k):
        sents = [s for s in SENT_END.split(paragraphs[i]) if s.strip()]
        cum += max(1, len(sents))
    return cum
