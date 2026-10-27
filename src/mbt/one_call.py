"""MBT-R and MBT-I: markers in, replies read back, passages put in.

MBT-I numbers the paragraphs of the REASONING with markers `[[#k]]`, asks in one call for a list
of (marker, behavior, passage), and inserts each passage after its paragraph by rule. MBT-R asks
in one call for the whole REASONING and RESPONSE back. A reply that cannot be used raises
`UnusableReply` with the reason, and the caller asks again; there is no fallback to the original.

A paragraph is what a blank line separates. Separators are kept exactly, so removing the inserted
spans from an MBT-I result gives the original REASONING back byte for byte (`original_kept`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from mbt import trajectory as tf
from mbt.prompts import BEHAVIORS

_SEPARATOR = re.compile(r"(\n[ \t]*\n(?:[ \t]*\n)*)")
MARKER = "[[#{k}]]"
_ANY_MARKER = re.compile(r"\[\[\s*#\s*\d+\s*\]\]")
JOIN = "\n\n"

_INSERTION = "===== INSERTION ====="
_NOTES = "===== NOTES ====="
_AFTER = re.compile(r"^AFTER:\s*(?:\[\[)?\s*#?\s*(\d+)\s*(?:\]\])?\s*$")
_BEHAVIOUR = re.compile(r"^BEHAVIOU?R:\s*(\S+)\s*$")
_REWRITE_SECTIONS = ("PROBLEM", "REASONING", "RESPONSE", "NOTES")
END_LINE = "===== END OF trajectory.md ====="


class UnusableReply(ValueError):
    """The editor answered, and the answer cannot be put into the trace."""


# ---------------------------------------------------------------------- markers (MBT-I)
def pieces(reasoning: str) -> list[str]:
    """`[paragraph, separator, paragraph, ...]`, so that `"".join(pieces(r)) == r`."""
    return _SEPARATOR.split(reasoning)


def paragraph_indices(parts: list[str]) -> list[int]:
    """Indices of the non-blank paragraphs in `pieces`, which are the ones that get a marker."""
    return [i for i in range(0, len(parts), 2) if parts[i].strip()]


def mark(reasoning: str) -> tuple[str, int]:
    """The REASONING with `[[#k]]` on its own line after the k-th paragraph, and the count."""
    if _ANY_MARKER.search(reasoning):
        raise ValueError("the trace already holds text shaped like a marker")
    parts = pieces(reasoning)
    out = list(parts)
    where = paragraph_indices(parts)
    for k, i in enumerate(where, start=1):
        out[i] = parts[i] + "\n" + MARKER.format(k=k)
    return "".join(out), len(where)


@dataclass(frozen=True)
class Insertion:
    marker: int
    behavior: str
    text: str


@dataclass
class Applied:
    reasoning: str
    spans: list[tuple[int, int, Insertion]] = field(default_factory=list)


def apply(reasoning: str, insertions: list[Insertion]) -> Applied:
    """Each passage after its marker's paragraph, in the given order, nothing else touched."""
    parts = pieces(reasoning)
    where = paragraph_indices(parts)
    by_marker: dict[int, list[Insertion]] = {}
    for ins in insertions:
        if not 1 <= ins.marker <= len(where):
            raise UnusableReply(f"marker {ins.marker} is not one of 1..{len(where)}")
        by_marker.setdefault(ins.marker, []).append(ins)
    at_part = {where[k - 1]: items for k, items in by_marker.items()}
    out: list[str] = []
    spans: list[tuple[int, int, Insertion]] = []
    length = 0
    for i, part in enumerate(parts):
        out.append(part)
        length += len(part)
        for ins in at_part.get(i, []):
            added = JOIN + ins.text
            spans.append((length, length + len(added), ins))
            out.append(added)
            length += len(added)
    return Applied("".join(out), spans)


def remove_spans(text: str, spans: list[tuple[int, int, object]]) -> str:
    out, cursor = [], 0
    for start, stop, _ in sorted(spans, key=lambda s: s[0]):
        out.append(text[cursor:start])
        cursor = stop
    out.append(text[cursor:])
    return "".join(out)


def original_kept(original: str, applied: Applied) -> bool:
    return remove_spans(applied.reasoning, applied.spans) == original


def positions(original: str, applied: Applied) -> list[float]:
    """Each insertion's position in the original REASONING: the share of the original's
    non-whitespace characters that come before it (the unit of `mbt.commits`)."""
    from mbt.commits import nonspace  # noqa: PLC0415

    total = max(1, nonspace(original))
    out, removed = [], 0
    for start, stop, _ in sorted(applied.spans, key=lambda s: s[0]):
        out.append(nonspace(original[:start - removed]) / total)
        removed += stop - start
    return out


# ---------------------------------------------------------------------- reading replies
def _without_the_end_line(content: str) -> str:
    lines = content.rstrip().split("\n")
    if lines and lines[-1].strip() == END_LINE:
        return "\n".join(lines[:-1])
    return content


def parse_insert_reply(content: str, markers: int) -> tuple[list[Insertion], str | None]:
    """`(insertions in reply order, notes)` from an MBT-I reply, or `UnusableReply`."""
    lines = _without_the_end_line(content).split("\n")
    heads = [i for i, line in enumerate(lines) if line.strip() in (_INSERTION, _NOTES)]
    if not heads:
        raise UnusableReply("no INSERTION and no NOTES section")
    before = "\n".join(lines[: heads[0]]).strip()
    if before:
        raise UnusableReply(f"text before the first section line: {before[:120]!r}")
    notes: str | None = None
    found: list[Insertion] = []
    for n, start in enumerate(heads):
        stop = heads[n + 1] if n + 1 < len(heads) else len(lines)
        body = lines[start + 1: stop]
        if lines[start].strip() == _NOTES:
            if notes is not None:
                raise UnusableReply("two NOTES sections")
            if n + 1 != len(heads):
                raise UnusableReply("an INSERTION after the NOTES")
            notes = "\n".join(body).strip()
            continue
        while body and not body[0].strip():
            body = body[1:]
        if len(body) < 3:
            raise UnusableReply(f"insertion {len(found) + 1} has fewer than three lines")
        after, name = _AFTER.match(body[0].strip()), _BEHAVIOUR.match(body[1].strip())
        if not after:
            raise UnusableReply(f"insertion {len(found) + 1}: no AFTER line, got {body[0][:80]!r}")
        if not name:
            raise UnusableReply(f"insertion {len(found) + 1}: no BEHAVIOUR line, got {body[1][:80]!r}")
        marker, key = int(after.group(1)), name.group(1)
        if not 1 <= marker <= markers:
            raise UnusableReply(f"insertion {len(found) + 1}: marker {marker} is not one of 1..{markers}")
        if key not in BEHAVIORS:
            raise UnusableReply(f"insertion {len(found) + 1}: {key!r} is not a behavior the brief prints")
        text = "\n".join(body[2:]).strip("\n").rstrip()
        if not text:
            raise UnusableReply(f"insertion {len(found) + 1}: the passage is empty")
        if _ANY_MARKER.search(text):
            raise UnusableReply(f"insertion {len(found) + 1}: the passage carries a marker")
        if any(tf.LOOKS_LIKE_A_HEADER.match(line) for line in text.split("\n")):
            raise UnusableReply(f"insertion {len(found) + 1}: the passage holds a section line")
        found.append(Insertion(marker, key, text))
    return found, notes


_BRACKETED = re.compile(r"^\s*[\[\(<].*[\]\)>]\s*$")
_ELISION = re.compile(r"(?i)unchanged|omitted|reproduc|verbatim|as before|as above|rest of|"
                      r"remaining|continues|not repeated|same as|\.\.\.|\u2026|truncat|elided|"
                      r"identical")


def _stand_in(section: str, original_lines: set[str] | None) -> str | None:
    """The first bracketed line that speaks of eliding text and is not in the original."""
    for line in section.split("\n"):
        if original_lines is not None and line in original_lines:
            continue
        if _BRACKETED.match(line) and _ELISION.search(line):
            return line.strip()
    return None


def parse_rewrite_reply(content: str, problem: str | None = None,
                        original: dict[str, str] | None = None) -> dict[str, str | None]:
    """`{"REASONING", "RESPONSE", "NOTES"}` from an MBT-R reply, or `UnusableReply`.

    This is the acceptance function acc(o) of the paper: both sections present, non-empty, in
    order, and no line that stands in for original text ("(rest omitted)").
    """
    original_lines = (set("\n".join([original.get("REASONING", ""),
                                     original.get("RESPONSE", "")]).split("\n"))
                      if original else None)
    lines = _without_the_end_line(content).split("\n")
    heads: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        if not tf.LOOKS_LIKE_A_HEADER.match(line.strip()):
            continue
        name = line.strip().strip("=").strip()
        if name not in _REWRITE_SECTIONS:
            raise UnusableReply(f"a section line the reply may not hold: {line.strip()!r}")
        heads.append((i, name))
    if not heads:
        raise UnusableReply("no section lines at all")
    before = "\n".join(lines[: heads[0][0]]).strip()
    if before:
        raise UnusableReply(f"text before the first section line: {before[:120]!r}")
    got: dict[str, str | None] = {}
    for n, (start, name) in enumerate(heads):
        if name in got:
            raise UnusableReply(f"{name} appears twice")
        stop = heads[n + 1][0] if n + 1 < len(heads) else len(lines)
        got[name] = "\n".join(lines[start + 1: stop]).strip("\n")
    if "PROBLEM" in got:
        if problem is None or (got["PROBLEM"] or "").strip() != problem.strip():
            raise UnusableReply("the reply carries a PROBLEM section that is not the problem")
        del got["PROBLEM"]
    order = [name for _, name in heads if name != "PROBLEM"]
    if order[:2] != ["REASONING", "RESPONSE"]:
        raise UnusableReply(f"sections in the order {order}, not REASONING, RESPONSE, NOTES")
    for name in ("REASONING", "RESPONSE"):
        if not (got.get(name) or "").strip():
            raise UnusableReply(f"{name} is empty")
        stand_in = _stand_in(got[name] or "", original_lines)
        if stand_in:
            raise UnusableReply(f"{name} holds a line that stands in for text: {stand_in[:120]!r}")
    got.setdefault("NOTES", None)
    return got
