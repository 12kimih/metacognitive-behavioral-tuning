"""MBT-A: an agent edits `trajectory.md` in a git repository, one commit per edit.

The agent is Claude Code running in an isolated sandbox (Docker or E2B) without web search or web
fetch. Its model is the editor, reached through an Anthropic-compatible endpoint, so the same
model writes the passages as in MBT-R and MBT-I. The working directory holds only `.git` and
`trajectory.md`, whose first commit is the unedited trace. The brief is the first and only user
message.

After the agent stops, the edited file and the full commit history are read back. The commits
are replayed to give each insertion its behavior and its position in the original reasoning.
"""

from __future__ import annotations

import json
import os
import re
import shlex
from dataclasses import asdict, dataclass
from pathlib import Path

from mbt import commits, prompts
from mbt import trajectory as tf

#: Inside the sandbox: the repository the agent works in, and a directory outside it that holds
#: the brief and receives the agent's transcript. Backends may move both.
WORKDIR = "/workspace"
IO_DIR = "/tmp/mbt"
GIT_AUTHOR = ("editing agent", "agent@example.invalid")
BEFORE_SUBJECT = "before: the teacher trace as generated"


@dataclass
class AgentSettings:
    """The harness settings of the paper (Appendix C), all overridable."""

    model: str = "deepseek-flash"
    base_url: str = "https://api.deepseek.com/anthropic"
    api_key_env: str = "DEEPSEEK_API_KEY"
    effort: str = "max"
    max_output_tokens: int = 128000
    max_retries: int = 10
    auto_compact_window: int = 917504
    timeout_sec: int = 3600
    sandbox_lifetime_sec: int = 7200
    cpus: float = 1.0
    memory_mb: int = 2048
    claude_code_version: str = "2.1.283"

    def env(self) -> dict[str, str]:
        """The environment of the agent process. The key is read from the host environment."""
        key = os.environ.get(self.api_key_env, "")
        if not key:
            raise SystemExit(f"the environment variable {self.api_key_env} is not set")
        return {
            "ANTHROPIC_BASE_URL": self.base_url,
            "ANTHROPIC_AUTH_TOKEN": key,
            "ANTHROPIC_API_KEY": "",
            "ANTHROPIC_MODEL": self.model,
            "ANTHROPIC_DEFAULT_OPUS_MODEL": self.model,
            "ANTHROPIC_DEFAULT_SONNET_MODEL": self.model,
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": self.model,
            "CLAUDE_CODE_SUBAGENT_MODEL": self.model,
            "CLAUDE_CODE_EFFORT_LEVEL": self.effort,
            "CLAUDE_CODE_MAX_OUTPUT_TOKENS": str(self.max_output_tokens),
            "CLAUDE_CODE_MAX_RETRIES": str(self.max_retries),
            "CLAUDE_CODE_AUTO_COMPACT_WINDOW": str(self.auto_compact_window),
            "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "DISABLE_TELEMETRY": "1",
            "DISABLE_ERROR_REPORTING": "1",
            "DISABLE_AUTOUPDATER": "1",
        }

    def public(self) -> dict:
        return asdict(self)


def setup_script(workdir: str = WORKDIR) -> str:
    """Put `trajectory.md` under git with the unedited trace as the first commit."""
    name, email = GIT_AUTHOR
    return " && ".join([
        f"cd {shlex.quote(workdir)}",
        "git init -q",
        f"git config user.name {shlex.quote(name)}",
        f"git config user.email {shlex.quote(email)}",
        f"git add {tf.FILE_NAME}",
        f"git commit -q -m {shlex.quote(BEFORE_SUBJECT)}",
    ])


def agent_command(settings: AgentSettings, claude: str = "claude", workdir: str = WORKDIR,
                  io_dir: str = IO_DIR) -> str:
    """The shell command that runs one agent turn inside the sandbox, bounded by `timeout`."""
    argv = [claude, "-p", "--output-format", "stream-json", "--verbose",
            "--permission-mode", "bypassPermissions",
            "--setting-sources", "user", "--strict-mcp-config",
            "--disallowedTools", "WebSearch", "WebFetch",
            "--effort", settings.effort]
    run = " ".join(shlex.quote(a) for a in argv)
    io = shlex.quote(io_dir)
    return (f"cd {shlex.quote(workdir)} && timeout -k 30 {int(settings.timeout_sec)} {run} "
            f"< {io}/brief.md > {io}/agent.out 2> {io}/agent.err")


def safe_name(identifier: str) -> str:
    """A file-system-safe directory name for a trace id."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", identifier)[:200] or "trace"


#: The arguments of the `git log` whose output `collect` reads.
GIT_LOG_ARGS = ["log", "--reverse", "--format=%H%x09%s"]


def collect(log: str, show) -> dict:
    """The commit history and the replayed insertions.

    `log` is the output of `git` with `GIT_LOG_ARGS`; `show(sha)` returns `trajectory.md` at that
    commit.
    """
    versions = []
    for line in log.splitlines():
        if "\t" not in line:
            continue
        sha, subject = line.split("\t", 1)
        versions.append((sha, subject, show(sha)))
    events = commits.replay(versions)
    rows = [{"commit": e.commit[:12], "kind": e.subject.kind, "behavior": e.subject.behavior,
             "where": e.subject.where, "what": e.subject.what,
             "position": None if e.position is None else round(e.position, 6),
             "added_lines": e.added_lines, "removed_lines": e.removed_lines} for e in events]
    return {"commits": [{"commit": s[:12], "subject": subj} for s, subj, _ in versions],
            "events": rows,
            "insertions": [{"behavior": b, "position": None if p is None else round(p, 6)}
                           for b, p in commits.insertion_positions(events)]}


def run(row: dict, backend, settings: AgentSettings, artifacts: Path | None = None) -> dict:
    """Edit one trace with one agent run. `backend` is a `mbt.sandbox` backend instance.

    When `artifacts` is given, the agent's full `stream-json` transcript, its stderr and a git
    bundle of the repository (every commit) are kept under `artifacts/<id>/`.
    """
    trace = tf.Trajectory(identifier=row["id"], problem=row["problem"],
                          reasoning=row["reasoning"], response=row["response"])
    text = tf.to_text(trace)
    result = backend.run_agent(file_text=text, brief=prompts.mbt_a_brief(), settings=settings)
    stream = result.get("agent_stream") or ""
    out = {"id": row["id"], "method": "mbt-a", "problem": row["problem"],
           "settings": settings.public(),
           "backend": backend.name, "exit_code": result.get("exit_code"),
           "timed_out": result.get("timed_out"), "seconds": result.get("seconds"),
           "agent": summarize_stream(stream)}
    if artifacts is not None:
        keep = Path(artifacts) / safe_name(row["id"])
        keep.mkdir(parents=True, exist_ok=True)
        (keep / "agent.jsonl").write_text(stream, encoding="utf-8")
        (keep / "agent.stderr").write_text(result.get("agent_stderr") or "", encoding="utf-8")
        if result.get("repo_bundle"):
            (keep / "repo.bundle").write_bytes(result["repo_bundle"])
        out["artifacts"] = str(keep)
    final = out["agent"]["result"]
    if result.get("exit_code") != 0 or result.get("timed_out") or not final or final.get("is_error"):
        out.update(edited=False, reason="the agent did not complete successfully")
        return out
    edited_text = result.get("file_text")
    try:
        sections = tf.parse(edited_text or "")
    except tf.NotATrajectoryFile as why:
        out.update(edited=False, reason=f"the edited file does not parse: {why}")
        return out
    if sections["PROBLEM"] != tf.parse(text)["PROBLEM"]:
        out.update(edited=False, reason="the editor changed the problem")
        return out
    history = collect(result.get("git_log", ""), result["show"]) if result.get("show") else {}
    out.update(edited=True, reasoning=sections["REASONING"], response=sections["RESPONSE"],
               problem_unchanged=sections["PROBLEM"] == tf.parse(text)["PROBLEM"], **history)
    return out


def summarize_stream(stream: str) -> dict:
    """Turns and the final result line of a Claude Code `stream-json` transcript."""
    turns, final = 0, None
    for line in stream.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "assistant":
            turns += 1
        elif event.get("type") == "result":
            final = {k: event.get(k) for k in ("subtype", "is_error", "num_turns",
                                                "duration_ms", "total_cost_usd", "usage")}
    return {"assistant_messages": turns, "result": final}
