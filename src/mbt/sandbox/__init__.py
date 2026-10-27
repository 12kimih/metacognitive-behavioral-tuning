"""Where the MBT-A agent runs: a local Docker container or an E2B cloud sandbox.

Both backends do the same five things for one trace: write `trajectory.md` and the brief, put the
file under git with the unedited trace as the first commit, run Claude Code once with the brief as
its prompt, and read back the edited file, the commit history and the agent's transcript. Both
return the same dictionary to `mbt.agentic.run`.
"""

from __future__ import annotations


def backend(name: str, **options):
    if name == "docker":
        from mbt.sandbox.docker import DockerBackend  # noqa: PLC0415

        return DockerBackend(**options)
    if name == "e2b":
        from mbt.sandbox.e2b import E2BBackend  # noqa: PLC0415

        return E2BBackend(**options)
    raise SystemExit(f"unknown sandbox backend {name!r}; one of docker, e2b")
