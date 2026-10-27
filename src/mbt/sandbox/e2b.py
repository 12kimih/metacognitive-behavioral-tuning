"""MBT-A in an E2B cloud sandbox, as in the paper.

Needs `pip install "metacognitive-behavioral-tuning[e2b]"` and `E2B_API_KEY`. With a template built
from `docker/e2b.Dockerfile` (`e2b template build`, see the README) Claude Code is already
installed; with the default template it is installed with npm when the sandbox starts. The
paper's sandboxes had 1 CPU and 2 GiB of memory, which the template sets (`docker/e2b.toml`).
"""

from __future__ import annotations

import shlex
import time

from mbt import agentic
from mbt import trajectory as tf

HOME = "/home/user"
WORKDIR = f"{HOME}/workspace"
IO_DIR = f"{HOME}/.mbt"


class E2BBackend:
    name = "e2b"

    def __init__(self, template: str | None = None, install: bool | None = None):
        try:
            import e2b  # noqa: F401, PLC0415
        except ImportError as error:
            raise SystemExit('the E2B backend needs the e2b package: '
                             'pip install "metacognitive-behavioral-tuning[e2b]"') from error
        self.template = template
        # A custom template carries Claude Code already; the default one does not.
        self.install = (template is None) if install is None else install

    def _create(self, settings: agentic.AgentSettings):
        from e2b import Sandbox  # noqa: PLC0415

        options = {"timeout": settings.sandbox_lifetime_sec}
        if self.template:
            options["template"] = self.template
        create = getattr(Sandbox, "create", None)
        return create(**options) if callable(create) else Sandbox(**options)

    @staticmethod
    def _run(sbx, cmd: str, timeout: int, envs: dict | None = None):
        """Run a command; return (exit code, stdout, stderr) without raising on a nonzero exit."""
        try:
            r = sbx.commands.run(cmd, timeout=timeout, envs=envs or {})
            return r.exit_code, r.stdout or "", r.stderr or ""
        except Exception as error:                                    # noqa: BLE001
            code = getattr(error, "exit_code", None)
            return (code if code is not None else -1, getattr(error, "stdout", "") or "",
                    getattr(error, "stderr", "") or str(error))

    def run_agent(self, *, file_text: str, brief: str, settings: agentic.AgentSettings) -> dict:
        sbx = self._create(settings)
        try:
            self._run(sbx, f"mkdir -p {WORKDIR} {IO_DIR}", 120)
            sbx.files.write(f"{WORKDIR}/{tf.FILE_NAME}", file_text)
            sbx.files.write(f"{IO_DIR}/brief.md", brief)
            claude = "claude"
            if self.install:
                prefix = f"{HOME}/.npm-global"
                code, _out, err = self._run(
                    sbx, f"npm install -g --prefix {prefix} "
                         f"@anthropic-ai/claude-code@{shlex.quote(settings.claude_code_version)}",
                    900)
                if code != 0:
                    raise RuntimeError(f"installing Claude Code failed: {err[-500:]}")
                claude = f"{prefix}/bin/claude"
            code, _out, err = self._run(sbx, agentic.setup_script(WORKDIR), 300)
            if code != 0:
                raise RuntimeError(f"git setup failed: {err[-500:]}")
            started = time.time()
            exit_code, _out, _err = self._run(
                sbx, agentic.agent_command(settings, claude=claude, workdir=WORKDIR, io_dir=IO_DIR),
                settings.timeout_sec + 120, envs=settings.env())
            seconds = round(time.time() - started, 1)
            result = {"exit_code": exit_code, "timed_out": exit_code in (124, 137),
                      "seconds": seconds,
                      "agent_stream": _read(sbx, f"{IO_DIR}/agent.out"),
                      "agent_stderr": _read(sbx, f"{IO_DIR}/agent.err"),
                      "file_text": _read(sbx, f"{WORKDIR}/{tf.FILE_NAME}")}
            git = "git -C " + shlex.quote(WORKDIR) + " "
            code, log, _ = self._run(sbx, git + " ".join(shlex.quote(a)
                                                         for a in agentic.GIT_LOG_ARGS), 120)
            if code == 0:
                result["git_log"] = log
                shows = {}
                for line in log.splitlines():
                    sha = line.split("\t", 1)[0]
                    _c, text, _e = self._run(sbx, git + f"show {sha}:{tf.FILE_NAME}", 120)
                    shows[sha] = text
                result["show"] = shows.__getitem__
                code, _o, _e = self._run(sbx, git + f"bundle create {IO_DIR}/repo.bundle --all", 300)
                if code == 0:
                    data = sbx.files.read(f"{IO_DIR}/repo.bundle", format="bytes")
                    result["repo_bundle"] = bytes(data)
            return result
        finally:
            try:
                sbx.kill()
            except Exception:                                         # noqa: BLE001
                pass


def _read(sbx, path: str) -> str:
    try:
        return sbx.files.read(path)
    except Exception:                                                 # noqa: BLE001
        return ""
