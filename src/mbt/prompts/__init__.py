"""The prompt texts of the paper, exactly as they were sent.

`mbt_a_brief.md` is the first user message of the MBT-A agent. `mbt_r_instructions.md` and
`mbt_i_instructions.md` are the same brief with the parts about the repository, the editing tool
and the commits taken out, preceded by a short framing and followed by the reply format. All
three carry the same seven behavior entries byte for byte. `tests/test_prompts.py` checks the
SHA-256 of each text against the values recorded with the paper's editing runs.
"""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources

#: SHA-256 of each text as recorded by the editing runs of the paper.
RECORDED_SHA256 = {
    "mbt_a_brief": "956574de7e5672f6dcc140162816ba9a0b375c0818daf8a6eae19dbe6b531321",
    # instructions + file-facts template + after-file template, joined by newlines
    "mbt_r": "22d32f8f12ac39aadfcadb9a680f0f842ce4703479981fae0c6bc2f73eb44253",
    "mbt_i": "78bd692d454e667f08b9414398406f941c42be3e00a65e3c2a7dd294e1d7a534",
}

#: The seven target behaviors, by the name the instructions print and the paper's name.
BEHAVIORS = {
    "subgoal-when-absent": "subgoal setting",
    "logical-check-in-deduction": "logical check",
    "checkable-verification": "substantive verification",
    "deep-named-return": "targeted backtracking",
    "continue-the-dropped-line": "resuming abandoned paths",
    "catch-and-correct": "self-correction",
    "closing-decision": "sufficiency judgment",
}

METHOD_FILES = {"rewrite": "mbt_r_instructions.md", "insert": "mbt_i_instructions.md"}


@lru_cache(maxsize=None)
def text(name: str) -> str:
    """A prompt file of this package by file name."""
    return resources.files(__package__).joinpath(name).read_text(encoding="utf-8")


def generation(problem: str) -> str:
    """The teacher's user message (no system prompt), Figure D1 of the paper."""
    return text("generation.txt").replace("{problem}", problem)


def mbt_a_brief() -> str:
    return text("mbt_a_brief.md")


@lru_cache(maxsize=None)
def one_call_templates() -> dict:
    return json.loads(text("one_call_templates.json"))


def one_call_instructions(method: str) -> str:
    if method not in METHOD_FILES:
        raise ValueError(f"method must be one of {tuple(METHOD_FILES)}, not {method!r}")
    return text(METHOD_FILES[method])


def one_call_identity_text(method: str) -> str:
    """What `RECORDED_SHA256` hashes for a one-call method."""
    t = one_call_templates()
    return "\n".join([one_call_instructions(method), t["file_facts"][method],
                      t["after_file"][method]])


def one_call_message(method: str, file_text: str, markers: int = 0) -> str:
    """The single user message of MBT-R (`rewrite`) or MBT-I (`insert`).

    The instructions, a line of facts about this file, the file exactly as shown, and the
    closing reminder of the reply format.
    """
    from mbt import trajectory  # noqa: PLC0415

    sections = trajectory.parse(file_text)
    reasoning, response = sections["REASONING"], sections["RESPONSE"]
    t = one_call_templates()
    facts = t["file_facts"][method].format(
        chars=len(reasoning), lines=len(reasoning.split("\n")),
        response_chars=len(response), markers=markers).rstrip("\n")
    return "\n".join([one_call_instructions(method).rstrip("\n"), "", facts, "",
                      file_text.rstrip("\n"), "", t["after_file"][method].rstrip("\n"), ""])
