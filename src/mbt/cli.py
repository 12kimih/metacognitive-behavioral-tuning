"""Command line: `mbt <command> --input IN.jsonl --output OUT.jsonl [options]`.

    mbt generate   problems -> Unedited teacher traces
    mbt rewrite    traces   -> MBT-R edits (whole-trace rewriting, one call)
    mbt insert     traces   -> MBT-I edits (marker-based insertion, one call)
    mbt agentic    traces   -> MBT-A edits (Claude Code agent in Docker or E2B)
    mbt annotate   traces   -> behavior codes under either annotation scheme
    mbt sft        traces   -> chat-format SFT rows (user prompt, assistant reasoning and response)
    mbt positions  edits    -> where insertions fall (share per tenth, KS D)
    mbt prompt     NAME     -> print a prompt exactly as sent

Every command that calls a model writes one output line per input row as soon as it finishes
and skips ids already present in the output, so rerunning the same command resumes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from mbt import endpoints, storage


# ---------------------------------------------------------------------- jsonl
def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out = set()
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.add(json.loads(line)["id"])
    return out


def run_rows(rows: list[dict], output: Path, work, workers: int, limit: int | None,
             *, context: dict) -> None:
    if workers < 1 or (limit is not None and limit < 0):
        raise ValueError("workers must be positive and limit must be nonnegative")
    identifiers = [row["id"] for row in rows]
    if any(not isinstance(ident, str) or not ident for ident in identifiers):
        raise ValueError("Input row IDs must be nonempty strings")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Input row IDs must be unique")
    output.parent.mkdir(parents=True, exist_ok=True)
    with storage.locked(output):
        storage.bind_output(output, rows, context)
        storage.recover_tail(output)
        skip = done_ids(output)
        if skip - set(identifiers):
            raise ValueError("Output contains row IDs outside the recorded input")
        todo = [r for r in rows if r["id"] not in skip]
        if limit is not None:
            todo = todo[:limit]
        print(f"{len(rows)} rows, {len(skip)} already done, {len(todo)} to run", file=sys.stderr)
        lock = threading.Lock()
        failures = 0
        with output.open("a", encoding="utf-8") as fh, ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(work, row): row["id"] for row in todo}
            for n, future in enumerate(as_completed(futures), start=1):
                ident = futures[future]
                try:
                    result = future.result()
                    if result.get("id") != ident:
                        raise ValueError(f"Result ID does not match input ID {ident}")
                except Exception as error:                               # noqa: BLE001
                    failures += 1
                    print(f"[{n}/{len(todo)}] {ident}: {type(error).__name__}: {error}",
                          file=sys.stderr)
                    continue
                with lock:
                    fh.write(json.dumps(result, ensure_ascii=False) + "\n")
                    fh.flush()
                    os.fsync(fh.fileno())
                state = "edited" if result.get("edited", True) else "not edited"
                print(f"[{n}/{len(todo)}] {ident}: {state}", file=sys.stderr)
        if failures:
            raise RuntimeError(f"{failures} rows failed; completed rows are saved; rerun to retry")

# ---------------------------------------------------------------------- options
def add_io(p: argparse.ArgumentParser) -> None:
    p.add_argument("--input", required=True, help="input JSONL")
    p.add_argument("--output", required=True, help="output JSONL (appended, resumable)")
    p.add_argument("--workers", type=int, default=8, help="parallel requests or sandboxes")
    p.add_argument("--limit", type=int, help="run at most this many new rows")


def add_endpoint(p: argparse.ArgumentParser, default: str) -> None:
    g = p.add_argument_group("endpoint")
    g.add_argument("--endpoint", default=default, choices=sorted(endpoints.PRESETS),
                   help=f"provider preset (default {default})")
    g.add_argument("--model", help="model id at the endpoint")
    g.add_argument("--base-url", help="override the preset's base URL")
    g.add_argument("--api", choices=["openai", "anthropic"], help="override the wire format")
    g.add_argument("--api-key-env", help="environment variable holding the key ('' for none)")
    g.add_argument("--temperature", type=float)
    g.add_argument("--top-p", type=float)
    g.add_argument("--max-tokens", type=int)
    g.add_argument("--reasoning-effort", help="e.g. max, high, medium")
    g.add_argument("--extra-body", help="JSON object merged into every request body")
    g.add_argument("--no-paper-sampling", action="store_true",
                   help="do not apply temperature 1.0, top_p 0.95, reasoning effort max")
    g.add_argument("--retries", type=int, default=10, help="HTTP retries per request")


def client_from(args):
    from mbt.client import Client  # noqa: PLC0415

    ep = endpoints.resolve(args.endpoint, model=args.model, base_url=args.base_url, api=args.api,
                           api_key_env=args.api_key_env, temperature=args.temperature,
                           top_p=args.top_p, max_tokens=args.max_tokens,
                           reasoning_effort=args.reasoning_effort, extra_body=args.extra_body,
                           paper_sampling=not args.no_paper_sampling)
    return Client(ep, retries=args.retries)


def client_context(args, client) -> dict:
    return {"command": args.command, "endpoint": asdict(client.endpoint), "retries": args.retries}


# ---------------------------------------------------------------------- commands
def cmd_generate(args) -> None:
    from mbt import methods  # noqa: PLC0415

    client = client_from(args)
    run_rows(read_jsonl(args.input), Path(args.output),
             lambda row: methods.generate(client, row), args.workers, args.limit,
             context=client_context(args, client))


def cmd_one_call(args) -> None:
    from mbt import methods  # noqa: PLC0415

    client = client_from(args)
    work = methods.rewrite if args.command == "rewrite" else methods.insert
    run_rows(read_jsonl(args.input), Path(args.output), lambda row: work(client, row),
             args.workers, args.limit, context=client_context(args, client))


def cmd_agentic(args) -> None:
    from mbt import agentic, sandbox  # noqa: PLC0415

    settings = agentic.AgentSettings()
    for field in ("model", "base_url", "api_key_env", "effort", "timeout_sec",
                  "claude_code_version", "sandbox_lifetime_sec"):
        value = getattr(args, field)
        if value is not None:
            setattr(settings, field, value)
    if args.sandbox == "docker":
        backend = sandbox.backend("docker", image=args.image, network=args.network)
    else:
        backend = sandbox.backend("e2b", template=args.template)
    artifacts = Path(args.artifacts) if args.artifacts else None
    run_rows(read_jsonl(args.input), Path(args.output),
             lambda row: agentic.run(row, backend, settings, artifacts), args.workers, args.limit,
             context={"command": "agentic", "settings": settings.public(), "sandbox": args.sandbox,
                      "image": args.image, "network": args.network, "template": args.template})


def cmd_annotate(args) -> None:
    from mbt import annotate  # noqa: PLC0415

    # The paper's annotator: gpt-6.1-sol, reasoning effort medium, provider default sampling.
    args.model = args.model or "gpt-6.1-sol"
    args.reasoning_effort = args.reasoning_effort or "medium"
    args.no_paper_sampling = True
    client = client_from(args)
    run_rows(read_jsonl(args.input), Path(args.output),
             lambda row: annotate.annotate(client, args.scheme, row), args.workers, args.limit,
             context=client_context(args, client) | {"scheme": args.scheme})


def cmd_sft(args) -> None:
    from mbt import prompts  # noqa: PLC0415

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    storage.refuse_same_file(Path(args.input), out)
    with storage.locked(out), storage.atomic_text(out) as fh:
        for row in read_jsonl(args.input):
            if not row.get("edited", True) or row.get("reasoning") is None:
                continue
            assistant = f"<think>\n{row['reasoning']}\n</think>\n\n{row['response']}"
            fh.write(json.dumps({"id": row["id"], "messages": [
                {"role": "user", "content": prompts.generation(row["problem"])},
                {"role": "assistant", "content": assistant}]}, ensure_ascii=False) + "\n")
            kept += 1
    print(f"wrote {kept} rows to {out}", file=sys.stderr)


def cmd_positions(args) -> None:
    from mbt.stats import positions  # noqa: PLC0415

    pooled, by_behavior = [], {}
    for row in read_jsonl(args.input):
        for ins in row.get("insertions") or []:
            if ins.get("position") is None:
                continue
            pooled.append(float(ins["position"]))
            by_behavior.setdefault(ins.get("behavior"), []).append(float(ins["position"]))
    report = {"all": positions.summary(pooled),
              "by_behavior": {k: positions.summary(v) for k, v in sorted(by_behavior.items())}}
    print(json.dumps(report, indent=1))


def cmd_prompt(args) -> None:
    from mbt import prompts  # noqa: PLC0415

    names = {"generation": lambda: prompts.generation("{problem}"),
             "mbt-a": prompts.mbt_a_brief,
             "mbt-r": lambda: prompts.one_call_instructions("rewrite"),
             "mbt-i": lambda: prompts.one_call_instructions("insert"),
             "deletion": lambda: prompts.text("deletion_instructions.md")}
    sys.stdout.write(names[args.name]())


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="mbt", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("generate", help="Unedited teacher traces for problems")
    add_io(p)
    add_endpoint(p, "deepseek")
    p.set_defaults(func=cmd_generate)

    for name, help_text in (("rewrite", "MBT-R, whole-trace rewriting"),
                            ("insert", "MBT-I, marker-based insertion")):
        p = sub.add_parser(name, help=help_text)
        add_io(p)
        add_endpoint(p, "deepseek")
        p.set_defaults(func=cmd_one_call)

    p = sub.add_parser("agentic", help="MBT-A, agentic editing in a sandbox")
    add_io(p)
    p.add_argument("--sandbox", choices=["docker", "e2b"], default="docker")
    p.add_argument("--image", default="mbt-editor:latest", help="Docker image (docker/Dockerfile)")
    p.add_argument("--network", help="Docker network for the container")
    p.add_argument("--template", help="E2B template name or id (docker/e2b.Dockerfile)")
    p.add_argument("--artifacts", help="directory for transcripts and git bundles")
    p.add_argument("--model", help="editor model id (default deepseek-flash)")
    p.add_argument("--base-url", help="Anthropic-compatible base URL "
                                      "(default https://api.deepseek.com/anthropic)")
    p.add_argument("--api-key-env", help="variable holding the key (default DEEPSEEK_API_KEY)")
    p.add_argument("--effort", help="Claude Code effort level (default max)")
    p.add_argument("--timeout-sec", type=int, help="limit of one agent run (default 3600)")
    p.add_argument("--sandbox-lifetime-sec", type=int, help="sandbox lifetime (default 7200)")
    p.add_argument("--claude-code-version", help="Claude Code version for the E2B install")
    p.set_defaults(func=cmd_agentic)

    p = sub.add_parser("annotate", help="code traces with an annotation scheme")
    add_io(p)
    p.add_argument("--scheme", choices=["undesirable", "target"], required=True)
    add_endpoint(p, "openai")
    p.set_defaults(func=cmd_annotate)

    p = sub.add_parser("sft", help="chat-format SFT rows from traces or edits")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(func=cmd_sft)

    p = sub.add_parser("positions", help="insertion position statistics of an edit file")
    p.add_argument("--input", required=True)
    p.set_defaults(func=cmd_positions)

    p = sub.add_parser("prompt", help="print a prompt exactly as sent")
    p.add_argument("name", choices=["generation", "mbt-a", "mbt-r", "mbt-i", "deletion"])
    p.set_defaults(func=cmd_prompt)

    args = ap.parse_args(argv)
    if hasattr(args, "input") and hasattr(args, "output"):
        storage.refuse_same_file(Path(args.input), Path(args.output))
    args.func(args)


if __name__ == "__main__":
    main()
