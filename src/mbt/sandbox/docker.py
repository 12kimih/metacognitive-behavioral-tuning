"""MBT-A in a local Docker container.

The image (`docker/Dockerfile`) holds Node.js, git and a pinned Claude Code. Each trace gets a
fresh container with a host directory mounted as the repository and another as the I/O directory,
limited to the paper's 1 CPU and 2 GiB of memory, running as the calling user. The API key reaches
the container through an `--env-file` readable only by the caller, never on a command line.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

from mbt import agentic
from mbt import trajectory as tf

DEFAULT_IMAGE = "mbt-editor:latest"


class DockerBackend:
    name = "docker"

    def __init__(self, image: str = DEFAULT_IMAGE, network: str | None = None,
                 keep_workdir: bool = False):
        if shutil.which("docker") is None:
            raise SystemExit("docker is not on PATH")
        self.image = image
        self.network = network
        self.keep_workdir = keep_workdir

    def _git(self, repo: Path, *args: str) -> str:
        return subprocess.run(["git", "-C", str(repo), *args], check=True, text=True,
                              capture_output=True).stdout

    def run_agent(self, *, file_text: str, brief: str, settings: agentic.AgentSettings) -> dict:
        root = Path(tempfile.mkdtemp(prefix="mbt-a-"))
        try:
            return self._run_in(root, file_text=file_text, brief=brief, settings=settings)
        finally:
            (root / "agent.env").unlink(missing_ok=True)
            if not self.keep_workdir:
                shutil.rmtree(root)

    def _run_in(self, root: Path, *, file_text: str, brief: str,
                settings: agentic.AgentSettings) -> dict:
        work, io = root / "workspace", root / "io"
        work.mkdir()
        io.mkdir()
        (work / tf.FILE_NAME).write_text(file_text, encoding="utf-8")
        (io / "brief.md").write_text(brief, encoding="utf-8")
        env_file = root / "agent.env"
        fd = os.open(env_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            for k, v in {**settings.env(), "HOME": "/tmp/home"}.items():
                fh.write(f"{k}={v}\n")
        name = f"mbt-a-{uuid.uuid4().hex[:12]}"
        script = (f"mkdir -p /tmp/home && {agentic.setup_script()} && "
                  f"{agentic.agent_command(settings)}; echo $? > {agentic.IO_DIR}/exit_code")
        argv = ["docker", "run", "--rm", "--name", name,
                "--cpus", str(settings.cpus), "--memory", f"{settings.memory_mb}m",
                "--memory-swap", f"{settings.memory_mb}m", "--pids-limit", "1024",
                "--security-opt", "no-new-privileges",
                "--user", f"{os.getuid()}:{os.getgid()}",
                "--env-file", str(env_file),
                "-v", f"{work}:{agentic.WORKDIR}", "-v", f"{io}:{agentic.IO_DIR}",
                "-w", agentic.WORKDIR]
        if self.network:
            argv += ["--network", self.network]
        argv += [self.image, "bash", "-lc", script]
        started = time.time()
        timed_out = False
        try:
            subprocess.run(argv, check=False, timeout=settings.sandbox_lifetime_sec,
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except subprocess.TimeoutExpired:
            timed_out = True
            subprocess.run(["docker", "kill", name], check=False, capture_output=True)
        finally:
            subprocess.run(["docker", "rm", "-f", name], check=False, capture_output=True,
                           timeout=30)
            env_file.unlink(missing_ok=True)
        seconds = round(time.time() - started, 1)
        try:
            exit_code = int((io / "exit_code").read_text().strip())
        except (OSError, ValueError):
            exit_code = None
        timed_out = timed_out or exit_code in (124, 137)
        result = {"exit_code": exit_code, "timed_out": timed_out, "seconds": seconds,
                  "agent_stream": _read(io / "agent.out"),
                  "agent_stderr": _read(io / "agent.err"),
                  "file_text": _read(work / tf.FILE_NAME)}
        if (work / ".git").is_dir():
            result["git_log"] = self._git(work, *agentic.GIT_LOG_ARGS)
            shows = {}
            for line in result["git_log"].splitlines():
                sha = line.split("\t", 1)[0]
                shows[sha] = self._git(work, "show", f"{sha}:{tf.FILE_NAME}")
            result["show"] = shows.__getitem__
            bundle = root / "repo.bundle"
            subprocess.run(["git", "-C", str(work), "bundle", "create", str(bundle), "--all"],
                           check=False, capture_output=True)
            if bundle.exists():
                result["repo_bundle"] = bundle.read_bytes()
        return result


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
