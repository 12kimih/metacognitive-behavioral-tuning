import copy
import json
import os
import re
from pathlib import Path

from datasets import Dataset, concatenate_datasets, load_from_disk

from mbt.registry import register_task
from mbt.tasks.qa._marker_utils import (
    SENT_END,
    first_correct_sentence_from_paragraph,
    inject_markers,
    marker_to_paragraph_index,
    split_paragraphs,
)
from mbt.tasks.qa.prompt_templates import RRP_SCORE_TEMPLATE

TASK_NAME = "qa.rrp_score"
NUM_PROC = int(os.environ.get("OMP_NUM_THREADS", "8"))
NUM_SAMPLES = None

_ANSWER_RE = re.compile(r"ANSWER_PARAGRAPH_MARKER:\s*(\[\[M\w+\]\])", re.IGNORECASE)
_REDUN_RE = re.compile(r"REDUNDANT_PARAGRAPH_MARKERS:\s*(.+?)(?:\n|$)", re.IGNORECASE)
_MARKER_TOK = re.compile(r"\[\[M(?:\d+|_END|00)\]\]|\[\[M\w+\]\]")
_CONF_RE = re.compile(r"CONFIDENCE:\s*([0-9.]+)", re.IGNORECASE)


@register_task(TASK_NAME)
class Task:
    """Reasoning Regulation Plane (RRP) judge task — paragraph-marker variant.

    For each rollout, the trace is split on blank lines into paragraphs, then
    `[[Mk]]` markers are injected after each non-final paragraph before the
    judge sees it. The judge returns:
      - ANSWER_PARAGRAPH_MARKER (`[[Mk]]` / `[[M_END]]` / `[[M00]]`)
      - REDUNDANT_PARAGRAPH_MARKERS (list of `[[Mk]]` or NONE)
      - CONFIDENCE (float in [0,1]) — stored only; not used in the per-sample R formula.

    Postprocess derives the `first_correct`, `redundant_fraction`,
    `redundant_sentences`, and `total_sentences` columns deterministically from
    the marker outputs. Aux columns: `confidence`, `redundant_paragraphs`
    (list[int], 1..N), `answer_paragraph` (int).

    v2 redundancy semantics (Round-6 redesign — docs/neurips2026_revision_changes.md):
      - The judge classifies EVERY paragraph (1..N) as PROGRESS / VERIFICATION /
        REDUNDANT, with VERIFICATION requiring concrete external-evidence appeal
        (a)-(d) per the prompt. Verification is explicitly excluded from REDUNDANT.
      - `redundant_paragraphs` may include any index in [1, N], including paragraphs
        BEFORE the answer-deriving paragraph (pre-arrival looping) and the last
        paragraph (referenced by [[M_END]]).
      - `redundant_fraction = redundant_sentences / total_sentences` — sentence-
        level proportion, unit-consistent with `first_correct` (also sentence-
        indexed). Earlier formulations used a paragraph-level ratio, which mixed
        units between numerator and denominator.

    Downstream metrics:
      arrival     rho_i        = first_correct_i / total_sentences_i        (1 if first_correct = -1)
      redundancy  delta_r_i    = redundant_fraction_i  (sentence-level)
      regulation  R_i          = harmonic mean of (1-rho_i) and (1-delta_r_i)
                               = 2 * (1-rho_i) * (1-delta_r_i) /
                                 ((1-rho_i) + (1-delta_r_i))
                  (F1-style; standard combination for two quality dimensions
                  per Van Rijsbergen 1979)

    Length-aware variants (anchored at the base model's mean trace length T_base
    on the same benchmark), used for the cross-method regulation plane:
      rho_la_i     = min(1, first_correct_i / T_base)
      delta_la_i   = min(1, redundant_sentences_i / T_base)
      R_la_i       = harmonic mean of (1-rho_la_i) and (1-delta_la_i)
    """

    def __init__(self, task_config: dict) -> None:
        super().__init__()
        self.task_config: dict = task_config
        self.num_proc: int = task_config.get("num_proc", NUM_PROC)
        self.num_samples: int | None = task_config.get("num_samples", NUM_SAMPLES)

    def preprocess(self, root_dir: Path) -> Path:
        self.task_dir = root_dir / TASK_NAME.split(".")[-1].replace("_", "-")
        self.task_dir.mkdir(parents=True, exist_ok=True)
        with (self.task_dir / "task_config.json").open("w", encoding="utf-8") as f:
            json.dump(self.task_config, f, ensure_ascii=False, indent=4)

        self.dataset: Dataset = load_from_disk(str(root_dir / "results"))
        if self.num_samples is not None:
            self.dataset = self.dataset.select(range(self.num_samples))
        requests = self.dataset.map(
            build_prompt,
            with_indices=True,
            remove_columns=self.dataset.column_names,
            num_proc=self.num_proc,
        )
        requests.save_to_disk(str(self.task_dir / "requests"))
        return self.task_dir

    def postprocess(self, api_dir: Path) -> None:
        responses: Dataset = load_from_disk(str(api_dir / "responses"))
        # Re-derive paragraph context per row from `self.dataset["reasoning_trace"]`
        # so we can map markers back to deterministic sentence positions without
        # piping state through `requests/`.
        traces = self.dataset["reasoning_trace"]

        def _parse_with_context(example: dict, idx: int) -> dict:
            paragraphs = split_paragraphs(traces[idx])
            return parse_response(example, paragraphs)

        responses = responses.map(
            _parse_with_context,
            with_indices=True,
            remove_columns=responses.column_names,
            num_proc=self.num_proc,
        )
        self.dataset = concatenate_datasets([self.dataset, responses], axis=1)
        self.dataset.save_to_disk(str(api_dir / "results"))


def format_messages(messages: list[dict[str, str]], **kwargs) -> list[dict[str, str]]:
    for message in messages:
        message["content"] = message["content"].format(**kwargs)
    return messages


def build_prompt(example: dict, idx: int) -> dict:
    paragraphs = split_paragraphs(example["reasoning_trace"])
    marked, _ = inject_markers(paragraphs)
    prompt = format_messages(
        copy.deepcopy(RRP_SCORE_TEMPLATE),
        question=example["question"],
        answer=example["answer"],
        marked_reasoning_trace=marked,
    )
    return {
        "sample_id": example["sample_id"],
        "rollout_id": example["rollout_id"],
        "request_id": idx + 1,
        "prompt": prompt,
    }


def parse_response(example: dict, paragraphs: list[str]) -> dict:
    response = example.get("response") or {}
    choices = response.get("choices") or []
    raw = (choices[0].get("message", {}).get("content") if choices else "") or ""

    n = len(paragraphs)
    am = _ANSWER_RE.search(raw)
    rm = _REDUN_RE.search(raw)
    cm = _CONF_RE.search(raw)

    answer_marker_raw = am.group(1) if am else "[[M00]]"
    k = marker_to_paragraph_index(answer_marker_raw, n)
    first_correct = first_correct_sentence_from_paragraph(paragraphs, k)

    redundant_paragraphs: list[int] = []
    if rm:
        # v2 whole-trace semantics: redundant markers are accepted across the full
        # trace [1, n], independently of the answer-deriving paragraph k. The judge
        # may emit [[Mj]] for j in 1..n-1 or [[M_END]] (= n) to flag the last
        # paragraph. [[M00]] / out-of-range markers are dropped silently.
        body = rm.group(1).strip()
        if body.upper() != "NONE":
            for tok in _MARKER_TOK.findall(body):
                kk = marker_to_paragraph_index(tok, n)
                if 1 <= kk <= n:
                    redundant_paragraphs.append(kk)
    redundant_paragraphs = sorted(set(redundant_paragraphs))

    # Sentence-level redundancy: count sentences in each redundant paragraph and
    # divide by total sentences. Unit-consistent with `first_correct` (which is
    # also a sentence position). The earlier paragraph-fraction formulation used
    # `len(redundant_paragraphs) / n`, which mixed paragraph and sentence units.
    sent_counts = []
    for p in paragraphs:
        sents = [s for s in SENT_END.split(p) if s.strip()]
        sent_counts.append(max(1, len(sents)))
    total_sentences = sum(sent_counts) if sent_counts else 0
    redundant_sentences = sum(sent_counts[p - 1] for p in redundant_paragraphs if 1 <= p <= n)
    redundant_fraction = (redundant_sentences / total_sentences) if total_sentences > 0 else 0.0
    redundant_fraction = max(0.0, min(1.0, redundant_fraction))

    try:
        confidence = float(cm.group(1)) if cm else 0.5
    except ValueError:
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))

    return {
        "first_correct": first_correct,
        "redundant_fraction": redundant_fraction,
        "redundant_sentences": redundant_sentences,
        "total_sentences": total_sentences,
        "confidence": confidence,
        "answer_paragraph": k,
        "redundant_paragraphs": redundant_paragraphs,
    }
