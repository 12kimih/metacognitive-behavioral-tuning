"""Unedited generation, MBT-R and MBT-I, one trace at a time.

Each function takes one input row and returns one output row. `mbt.cli` runs them over a JSONL
file with a thread pool, writes one output line per input as it finishes, and skips ids already
in the output file, so an interrupted run resumes where it stopped.

Every attempt, with its request settings, finish reason, token usage and the raw reply text, is
kept in the output row under `attempts`, so a failed edit says why it failed.
"""

from __future__ import annotations

from typing import Any

from mbt import one_call, prompts
from mbt import trajectory as tf
from mbt.client import ChatResult, Client, EndpointError

#: The paper's limits for the one-call editors (Section 3.3 and Appendix D).
MAX_CALLS = 10
LENGTH_STREAK = 5
EDITOR_MAX_TOKENS = 131072


def _attempt(result: ChatResult | None, error: str | None = None, **extra: Any) -> dict:
    if result is None:
        return {"usable": False, "reason": error, **extra}
    return {"finish_reason": result.finish_reason, "usage": result.usage,
            "seconds": result.seconds, "request": result.request, "reasoning": result.reasoning,
            "reply": result.content, **extra}


# ---------------------------------------------------------------------- Unedited
def generate(client: Client, row: dict, *, max_tokens: int | None = None) -> dict:
    """One teacher trace for one problem: `{id, problem, reasoning, response, ...}`."""
    message = prompts.generation(row["problem"])
    result = client.chat([{"role": "user", "content": message}], max_tokens=max_tokens)
    return {"id": row["id"], "problem": row["problem"], "reasoning": result.reasoning,
            "response": result.content, "finish_reason": result.finish_reason,
            "usage": result.usage, "request": result.request}


def _file_of(row: dict) -> tuple[tf.Trajectory, str]:
    trace = tf.Trajectory(identifier=row["id"], problem=row["problem"],
                          reasoning=row["reasoning"], response=row["response"])
    return trace, tf.to_text(trace)


def _ask(client: Client, message: str, read, attempts: list[dict]) -> Any:
    """Call until `read(reply)` accepts, at most MAX_CALLS times, and stop early after
    LENGTH_STREAK consecutive replies that hit the output limit."""
    streak = 0
    for n in range(1, MAX_CALLS + 1):
        try:
            result = client.chat([{"role": "user", "content": message}],
                                 max_tokens=client.endpoint.max_tokens or EDITOR_MAX_TOKENS)
        except EndpointError as error:
            attempts.append(_attempt(None, f"request failed: {error}"[:2000], attempt=n))
            continue
        entry = _attempt(result, attempt=n)
        if result.finish_reason not in ("stop", "end_turn"):
            entry.update(usable=False, reason="the reply did not finish normally")
            attempts.append(entry)
            streak = streak + 1 if result.hit_length_limit else 0
            if streak >= LENGTH_STREAK:
                break
            continue
        try:
            value = read(result.content)
            entry.update(usable=True)
            attempts.append(entry)
            return value
        except one_call.UnusableReply as why:
            entry.update(usable=False, reason=str(why))
            attempts.append(entry)
        streak = streak + 1 if result.hit_length_limit else 0
        if streak >= LENGTH_STREAK:
            break
    return None


# ---------------------------------------------------------------------- MBT-R
def rewrite(client: Client, row: dict) -> dict:
    """MBT-R: the whole trace in one call, the whole edited REASONING and RESPONSE back."""
    trace, text = _file_of(row)
    original = {"PROBLEM": trace.problem, "REASONING": trace.reasoning, "RESPONSE": trace.response}
    message = prompts.one_call_message("rewrite", text)
    attempts: list[dict] = []
    got = _ask(client, message,
               lambda reply: one_call.parse_rewrite_reply(reply, trace.problem, original), attempts)
    out = {"id": row["id"], "method": "mbt-r", "problem": row["problem"],
           "edited": got is not None, "attempts": attempts}
    if got is not None:
        out.update(reasoning=got["REASONING"], response=got["RESPONSE"], notes=got["NOTES"])
    return out


# ---------------------------------------------------------------------- MBT-I
def insert(client: Client, row: dict) -> dict:
    """MBT-I: the trace with paragraph markers in one call, a list of insertions back."""
    trace, _ = _file_of(row)
    marked, markers = one_call.mark(trace.reasoning)
    shown = tf.to_text(tf.Trajectory(identifier=trace.identifier, problem=trace.problem,
                                     reasoning=marked, response=trace.response))
    message = prompts.one_call_message("insert", shown, markers)

    def read(reply: str):
        insertions, notes = one_call.parse_insert_reply(reply, markers)
        applied = one_call.apply(trace.reasoning, insertions)
        if not one_call.original_kept(trace.reasoning, applied):
            raise one_call.UnusableReply("removing the passages does not give the original")
        return applied, notes

    attempts: list[dict] = []
    got = _ask(client, message, read, attempts)
    out = {"id": row["id"], "method": "mbt-i", "problem": row["problem"],
           "edited": got is not None, "markers": markers, "attempts": attempts}
    if got is not None:
        applied, notes = got
        where = one_call.positions(trace.reasoning, applied)
        out.update(reasoning=applied.reasoning, response=trace.response, notes=notes,
                   insertions=[{"behavior": ins.behavior, "marker": ins.marker,
                                "position": round(pos, 6), "text": ins.text}
                               for (_s, _e, ins), pos in zip(
                                   sorted(applied.spans, key=lambda s: s[0]), where)])
    return out
