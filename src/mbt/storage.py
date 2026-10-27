"""Exclusive output writers, recoverable JSONL tails and atomic replacement."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import tempfile
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def bind_output(path: Path, rows: list[dict], context: dict) -> None:
    """Bind an output to its semantic inputs, operation and local implementation.

    The caller holds locked(path). Context is hashed, never serialized verbatim,
    so credentials supplied in an endpoint configuration cannot enter this record.
    """
    package = Path(__file__).parent
    code = {p.relative_to(package).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(package.rglob("*"))
            if p.is_file() and p.suffix in (".py", ".txt", ".json", ".md")}
    environment = {"python": platform.python_version()}
    for name in ("httpx", "numpy", "e2b"):
        try:
            environment[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            environment[name] = None
    expected = {"schema_version": 1, "inputs": {r["id"]: _digest(r) for r in rows},
                "context_sha256": _digest(context), "code": code, "environment": environment}
    manifest = path.with_name(path.name + ".manifest.json")
    if manifest.exists():
        recorded = json.loads(manifest.read_text(encoding="utf-8"))
        if any(recorded.get(k) != v for k, v in expected.items()):
            raise ValueError("Output identity changed; restore its original inputs, code and "
                             "configuration or choose a new output path")
        if not path.exists():
            raise ValueError("Recorded output file is missing; preserve its manifest and use a new path")
        return
    if path.exists() and path.stat().st_size:
        raise ValueError("Existing output has no identity manifest; resume with its original "
                         "code or choose a new output path")
    path.touch(exist_ok=True)
    with atomic_text(manifest) as handle:
        json.dump(expected | {"run_id": uuid.uuid4().hex,
                              "created_at": datetime.now(timezone.utc).isoformat()}, handle,  # noqa: UP017 - Python 3.10
                  sort_keys=True, indent=2)
        handle.write("\n")


@contextmanager
def locked(path: Path):
    """Reject a concurrent writer, including callers using a symlink alias."""
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.with_name(target.name + ".lock").open("a+b") as handle:
        if os.name == "nt":
            import msvcrt

            if handle.seek(0, os.SEEK_END) == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            def acquire():
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)

            def release():
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            def acquire():
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)

            def release():
                fcntl.flock(handle, fcntl.LOCK_UN)
        try:
            acquire()
        except OSError as error:
            raise RuntimeError(f"Cannot acquire exclusive output lease: {target}") from error
        try:
            yield
        finally:
            handle.seek(0)
            release()


def recover_tail(path: Path) -> None:
    """Repair only the final record, under the caller's output lease."""
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open("r+b") as handle:
        end = handle.seek(0, os.SEEK_END)
        handle.seek(end - 1)
        if handle.read(1) == b"\n":
            return
        position, tail = end, b""
        while position:
            size = min(position, 65536)
            position -= size
            handle.seek(position)
            tail = handle.read(size) + tail
            boundary = tail.rfind(b"\n")
            if boundary >= 0:
                position += boundary + 1
                tail = tail[boundary + 1:]
                break
        try:
            json.loads(tail)
        except (json.JSONDecodeError, UnicodeDecodeError):
            recovery = path.parent / ".recovery"
            recovery.mkdir(exist_ok=True)
            backup = recovery / f"{path.name}.{hashlib.sha256(tail).hexdigest()}.tail"
            try:
                with backup.open("xb") as saved:
                    saved.write(tail)
                    saved.flush()
                    os.fsync(saved.fileno())
            except FileExistsError:
                if backup.read_bytes() != tail:
                    raise ValueError(f"Recovery fragment mismatch: {backup}") from None
            handle.truncate(position)
        else:
            handle.seek(end)
            handle.write(b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def refuse_same_file(source: Path, destination: Path) -> None:
    if source.resolve() == destination.resolve() or (
        source.exists() and destination.exists() and source.samefile(destination)
    ):
        raise ValueError("Input and output must be different files")


@contextmanager
def atomic_text(path: Path):
    """Keep an existing output intact if conversion or flushing fails."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
