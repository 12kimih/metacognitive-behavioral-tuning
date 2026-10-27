"""The MBT-A commit history: subject grammar, and insertion positions by replaying the commits.

The agent records each edit as one commit (Figure C3 of the paper):

    <name>: <where> -- <what you did>   one insertion under the behavior <name>
    <name>: none -- <why>               the behavior has no place in this trace (empty commit)
    fix: <what you are repairing>       a repair of one of the agent's own earlier insertions
    before: <anything>                  the opening snapshot of the unedited trace

`replay` walks the commits in order and maps every line the agent added back to a position in the
original REASONING, so each insertion gets the position pi = l / |r| of the paper (Section 4.2),
where l counts the original characters before it and |r| all of them. Characters are counted
without whitespace, so a reflowed blank line does not move a position.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

from mbt import trajectory as tf

_NAME = r"[a-z][a-z0-9]*(?:-[a-z0-9]+)+"
_INSERTION = re.compile(r"^(" + _NAME + r")\s*:\s*(.+?)\s+--\s+(.*)$", re.S)
_NONE = re.compile(r"^(" + _NAME + r")\s*:\s*none\b\s*(?:--\s*)?(.*)$", re.I | re.S)
_FIX = re.compile(r"^fix\s*:\s*(.*)$", re.I | re.S)
_BEFORE = re.compile(r"^before\s*:", re.I)


@dataclass(frozen=True)
class Subject:
    kind: str                 # insertion, none, fix, before, unparsed
    behavior: str | None
    where: str | None
    what: str | None


def parse_subject(subject: str) -> Subject:
    s = subject.strip()
    if _BEFORE.match(s):
        return Subject("before", None, None, None)
    if (m := _FIX.match(s)):
        return Subject("fix", None, None, m.group(1).strip())
    if (m := _NONE.match(s)):
        return Subject("none", m.group(1).lower(), None, m.group(2).strip())
    if (m := _INSERTION.match(s)):
        return Subject("insertion", m.group(1).lower(), m.group(2).strip(), m.group(3).strip())
    return Subject("unparsed", None, None, s)


@dataclass
class Event:
    commit: str
    subject: Subject
    position: float | None     # share of the original REASONING before the first added text
    added_lines: int
    removed_lines: int


def nonspace(text: str) -> int:
    """Characters that are not whitespace, the unit positions are measured in."""
    return len(text) - sum(1 for ch in text if ch.isspace())


#: For each line of the current REASONING: (non-whitespace characters of the original before
#: the line, whether the line is original text).
Origin = tuple[int, bool]


def carry(origins: list[Origin], parent: list[str], child: list[str],
          total: int) -> tuple[list[Origin], int | None]:
    """One commit: the origins of `child`'s lines from `parent`'s, and where in the original the
    commit's first new text went (None when it added no non-blank line).

    The brief has the agent split a host line at a sentence boundary to put a passage inside it,
    which a line diff sees as the whole host line replaced. A child line that is a piece of an
    original parent line in the same changed block therefore keeps its place inside that line,
    and a line of new text sits right after the original text that precedes it.
    """
    out: list[Origin] = []
    first: int | None = None
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, parent, child,
                                                       autojunk=False).get_opcodes():
        if tag == "equal":
            out.extend(origins[i1:i2])
            continue
        running = origins[i1][0] if i1 < len(origins) else total
        hosts = [(parent[i], origins[i][0]) for i in range(i1, i2)
                 if origins[i][1] and parent[i].strip()]
        for line in child[j1:j2]:
            piece = next(((h, at) for h, at in hosts if line.strip() and line.strip() in h), None)
            if piece:
                h, at = piece
                start = at + nonspace(h[:h.index(line.strip())])
                out.append((start, True))
                running = start + nonspace(line)
            else:
                out.append((running, False))
                if first is None and line.strip():
                    first = running
    return out, first


def _reasoning_lines(text: str) -> list[str]:
    try:
        return tf.parse(text)["REASONING"].splitlines()
    except tf.NotATrajectoryFile:
        return []


def replay(versions: list[tuple[str, str, str]]) -> list[Event]:
    """Events for each commit after the first.

    `versions` is `[(sha, subject, file text at that commit), ...]` oldest first, the first being
    the `before:` snapshot. Positions are fractions of the original REASONING's non-whitespace
    characters, carried commit by commit (`carry`).
    """
    if not versions:
        return []
    lines = _reasoning_lines(versions[0][2])
    total = max(1, sum(nonspace(x) for x in lines))
    origins: list[Origin] = []
    at = 0
    for line in lines:
        origins.append((at, True))
        at += nonspace(line)
    events: list[Event] = []
    for sha, subject, text in versions[1:]:
        now = _reasoning_lines(text)
        origins, first = carry(origins, lines, now, total)
        matcher = difflib.SequenceMatcher(None, lines, now, autojunk=False)
        added = sum(j2 - j1 for op, _i1, _i2, j1, j2 in matcher.get_opcodes() if op != "equal")
        removed = sum(i2 - i1 for op, i1, i2, _j1, _j2 in matcher.get_opcodes() if op != "equal")
        lines = now
        events.append(Event(sha, parse_subject(subject),
                            None if first is None else min(1.0, first / total), added, removed))
    return events


def insertion_positions(events: list[Event]) -> list[tuple[str, float | None]]:
    """`(behavior, position)` for each insertion commit; None when it added no reasoning text."""
    return [(e.subject.behavior or "", e.position) for e in events
            if e.subject.kind == "insertion"]
