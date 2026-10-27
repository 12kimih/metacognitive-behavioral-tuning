"""The two annotation schemes of the paper (Appendices B and H), applied by an LLM annotator.

`undesirable`: 14 undesirable behaviors (D1 to D14) and 3 desirable ones (P1 to P3), coded as
present or absent per trace (Table B1 and Figure 1).

`target`: the seven target behaviors (B1 to B7), coded as present or absent with a count of
occurrences and how many start in each third of the trace (Figure H1 and Table H1).

The annotator receives one user message: the instructions, the codebook and the trace with its
lines numbered `L<n>`. It does not see whether the answer is correct or which condition produced
the trace. The paper used gpt-6.1-sol with reasoning effort medium.
"""

from __future__ import annotations

import json
import re

from mbt import prompts
from mbt.client import Client

UNDESIRABLE_CODES = [f"D{i}" for i in range(1, 15)] + ["P1", "P2", "P3"]
TARGET = list(prompts.BEHAVIORS)
THIRDS = ("early", "mid", "late")
CLOSING_CHECKS = ("conditions", "second route", "consistency", "form")
SCHEMES = ("undesirable", "target")


def bundle(identifier: str, problem: str, reasoning: str, response: str) -> str:
    """The trace as the annotator sees it."""
    return (f"=== {identifier} ===\n--- PROBLEM ---\n{problem}\n--- TRACE ---\n"
            f"<think>\n{reasoning}\n</think>\n\n{response}\n")


def numbered(text: str) -> str:
    return "\n".join(f"L{i}: {line}" for i, line in enumerate(text.splitlines(), start=1))


def trace_thirds(text: str) -> list[str]:
    """The third of each line of a bundle by character position within the TRACE part."""
    lines = text.splitlines()
    try:
        start = lines.index("--- TRACE ---") + 1
    except ValueError:
        start = 0
    body = lines[start:]
    total = sum(len(x) + 1 for x in body) or 1
    out, seen = ["problem"] * start, 0
    for line in body:
        out.append(THIRDS[min(2, 3 * seen // total)])
        seen += len(line) + 1
    return out


def thirds_note(text: str) -> str:
    marks = trace_thirds(text)
    parts = []
    for third in THIRDS:
        nums = [i for i, m in enumerate(marks, start=1) if m == third]
        parts.append(f"{third}: L{nums[0]}-L{nums[-1]}" if nums
                     else f"{third}: (no line starts here)")
    return "\n".join(parts)


def message(scheme: str, text: str) -> str:
    """The annotator's user message for one bundle."""
    base = "annotation/"
    if scheme == "undesirable":
        return (prompts.text(base + "undesirable_instructions.txt")
                + prompts.text(base + "undesirable_behaviors_codebook.md")
                + "\n\nTRACE:\n" + numbered(text))
    if scheme == "target":
        head = prompts.text(base + "target_instructions.txt").format(
            thirds=thirds_note(text), names=", ".join(TARGET))
        return (head + prompts.text(base + "target_behaviors_codebook.md")
                + "\n\nTRACE:\n" + numbered(text))
    raise ValueError(f"scheme must be one of {SCHEMES}")


def json_object(reply: str) -> dict | None:
    match = re.search(r"\{.*\}", reply, flags=re.S)
    if not match:
        return None
    for candidate in (match.group(0), re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", match.group(0))):
        try:
            obj = json.loads(candidate)
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            continue
    return None


def parse(scheme: str, reply: str) -> dict | None:
    """The reply as one flat row of codes, or None when it does not follow the format."""
    obj = json_object(reply)
    if obj is None:
        return None
    if scheme == "undesirable":
        codes = obj.get("codes")
        if not isinstance(codes, dict) or any(c not in codes for c in UNDESIRABLE_CODES):
            return None
        row = {c: int(bool((codes[c] or {}).get("present"))) for c in UNDESIRABLE_CODES}
        row["D1f"] = int(bool(obj.get("D1f")))
        row["D8c"] = int(bool(obj.get("D8c")))
        return row
    got = obj.get("behaviours")
    if not isinstance(got, dict):
        return None
    row: dict = {}
    for name in TARGET:
        b = got.get(name)
        if not isinstance(b, dict):
            return None
        try:
            thirds = b.get("thirds") or {}
            row[name] = int(bool(int(b.get("present", 0))))
            row[f"{name}#count"] = int(b.get("count", 0))
            row[f"{name}#thirds"] = [int(thirds.get(t, 0)) for t in THIRDS]
        except (TypeError, ValueError):
            return None
    checks = obj.get("closing_checks") or []
    row["closing_checks"] = sorted({c for c in checks if c in CLOSING_CHECKS}) \
        if isinstance(checks, list) else []
    return row


def annotate(client: Client, scheme: str, row: dict) -> dict:
    """Annotate one trace row `{id, problem, reasoning, response}`."""
    text = bundle(row["id"], row["problem"], row["reasoning"], row["response"])
    result = client.chat([{"role": "user", "content": message(scheme, text)}])
    codes = parse(scheme, result.content)
    return {"id": row["id"], "scheme": scheme, "parse_error": codes is None,
            "codes": codes, "reply": result.content, "usage": result.usage,
            "request": result.request}
