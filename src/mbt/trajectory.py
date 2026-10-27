"""One trace as one editable file (`trajectory.md`), and the exact way back.

The file holds three sections in the order the teacher produced them: the PROBLEM as it was put
to the teacher, the REASONING (the text inside the thinking block) and the RESPONSE (the text
after it). Everything else is a comment line that the parser ignores. There is no reference
answer in the file and no place to put one.

The section delimiter is a line of five equals signs on each side of the section name. Such a
line does not occur in the teacher traces of the released data; `write` refuses a trace that
contains one instead of escaping it, so a file always parses back to exactly what was written.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

SECTIONS = ("PROBLEM", "REASONING", "RESPONSE")
HEADER = re.compile(r"^={5} (" + "|".join(SECTIONS) + r") ={5}$")
LOOKS_LIKE_A_HEADER = re.compile(r"^={5}[^=]*={5}$")
COMMENT = "#"
TITLE = "trajectory file, format 1"
FILE_NAME = "trajectory.md"


class NotATrajectoryFile(ValueError):
    """The text does not parse back to the three sections."""


@dataclass(frozen=True)
class Trajectory:
    """A problem, the teacher's reasoning and the teacher's response."""

    identifier: str
    problem: str
    reasoning: str
    response: str
    rollout: int = 0

    def header_lines(self) -> list[str]:
        return [
            f"{COMMENT} {TITLE}",
            f"{COMMENT} id      {self.identifier}",
            f"{COMMENT} sample  {self.identifier}",
            f"{COMMENT} rollout {self.rollout}",
            f"{COMMENT}",
            f"{COMMENT} THREE SECTIONS AND NOTHING ELSE IS CONTENT. A line beginning with '{COMMENT}'",
            f"{COMMENT} outside a section is ignored; inside one it is part of the text.",
            f"{COMMENT} Do not add, remove or reword a '===== ... =====' line.",
        ]


def render(trajectory: Trajectory) -> str:
    """The file text. `parse(render(t))` gives back the three sections of `t`."""
    parts = list(trajectory.header_lines())
    for name, body in (("PROBLEM", trajectory.problem),
                       ("REASONING", trajectory.reasoning),
                       ("RESPONSE", trajectory.response)):
        parts += ["", f"===== {name} =====", body.rstrip("\n")]
    return "\n".join(parts) + "\n"


def parse(text: str) -> dict[str, str]:
    """The three sections by name. Raises `NotATrajectoryFile` when the text is not one."""
    found: dict[str, list[str]] = {}
    current: str | None = None
    for line in text.splitlines():
        match = HEADER.match(line)
        if match:
            current = match.group(1)
            if current in found:
                raise NotATrajectoryFile(f"{current} appears twice")
            found[current] = []
            continue
        if current is not None:
            found[current].append(line)
    missing = [name for name in SECTIONS if name not in found]
    if missing:
        raise NotATrajectoryFile(f"no {', '.join(missing)} section")
    return {name: "\n".join(found[name]).strip("\n") for name in SECTIONS}


def unsafe_lines(trajectory: Trajectory) -> list[str]:
    """Lines of the content that would be read back as a section header."""
    out: list[str] = []
    for body in (trajectory.problem, trajectory.reasoning, trajectory.response):
        out += [line for line in body.splitlines() if LOOKS_LIKE_A_HEADER.match(line)]
    return out


def to_text(trajectory: Trajectory) -> str:
    """`render`, after refusing a trace whose own text would break the parse."""
    unsafe = unsafe_lines(trajectory)
    if unsafe:
        raise NotATrajectoryFile(
            f"{trajectory.identifier}: {len(unsafe)} line(s) would parse as a section header, "
            f"for example {unsafe[0]!r}")
    return render(trajectory)


def write(path: str | Path, trajectory: Trajectory) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_text(trajectory), encoding="utf-8")
    return path


def read(path: str | Path) -> dict[str, str]:
    return parse(Path(path).read_text(encoding="utf-8"))
