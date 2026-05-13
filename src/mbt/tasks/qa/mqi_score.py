import copy
import csv
import json
import os
import re
from pathlib import Path

from datasets import Dataset, concatenate_datasets, load_from_disk

from mbt.registry import register_task
from mbt.tasks.qa.prompt_templates import MQI_SCORE_TEMPLATE

TASK_NAME = "qa.mqi_score"
NUM_PROC = int(os.environ.get("OMP_NUM_THREADS", "8"))
NUM_SAMPLES = None
DEFAULT_DIFFICULTY_CSV = "data/sample_difficulty.csv"
DEFAULT_TIER = "medium"

_LOBS_RE = re.compile(r"L_OBS:\s*(\d+)", re.IGNORECASE)
_PHASES_RE = re.compile(r"PHASES:\s*([0-9,\s]+|NONE)", re.IGNORECASE)
_CONF_RE = re.compile(r"CONFIDENCE:\s*([0-9.]+)", re.IGNORECASE)


@register_task(TASK_NAME)
class Task:
    """Metacognition Quality Index (MQI) judge task.

    For each rollout, asks an LLM judge for three values per trace:
      - l_obs (int 0..5): the 0-5 phase-presence rubric score
        (System-1 direct answer at 0; fully integrated five-phase reasoning at 5).
      - phases (list[int]): indices of distinctly present phases in {1..5}.
      - confidence (float in [0,1]): final-answer verbalized confidence.

    The judge prompt is conditioned on a per-sample `difficulty_tier` looked
    up from a pre-computed csv (see scripts/build_difficulty.py), so easy
    questions are not penalised for short, direct answers.

    This task only emits the per-sample judge tuple. The length-aware MQI
    used in the NeurIPS 2026 paper is synthesised offline as

        MQI_i        = L_obs_i * T_base / T_i        (length-aware combination)
        MQI-bar      = (1/n) * sum_i MQI_i           (aggregate)

    where T_i is the average token count of method i's output and T_base is
    the analogous mean-token length of the same-scale un-tuned base model on
    the same benchmark (e.g., 731 / 1173 / 1342 tokens for Qwen3-0.6B/1.7B/4B
    on MuSiQue). The plotter `scripts/plot_neurips_figures.py:figure6_mqi_bars`
    is the canonical implementation; per-phase presence ratios are reported
    alongside in TableA6.

    See docs/scoring_redesign_marker_variant.md and the paper's §4.2 for the
    full definition.
    """

    def __init__(self, task_config: dict) -> None:
        super().__init__()
        self.task_config: dict = task_config
        try:
            self.benchmark: str = task_config["benchmark"]
        except KeyError as e:
            raise ValueError(
                "qa.mqi_score requires task_config['benchmark'] (e.g. 'musique', 'hotpotqa', '2wikimultihopqa', 'math500')."
            ) from e
        self.difficulty_csv: str = task_config.get("difficulty_csv", DEFAULT_DIFFICULTY_CSV)
        self.default_tier: str = task_config.get("default_tier", DEFAULT_TIER)
        self.num_proc: int = task_config.get("num_proc", NUM_PROC)
        self.num_samples: int | None = task_config.get("num_samples", NUM_SAMPLES)

    def preprocess(self, root_dir: Path) -> Path:
        self.task_dir = root_dir / TASK_NAME.split(".")[-1].replace("_", "-")
        self.task_dir.mkdir(parents=True, exist_ok=True)
        with (self.task_dir / "task_config.json").open("w", encoding="utf-8") as f:
            json.dump(self.task_config, f, ensure_ascii=False, indent=4)

        self.tier_lookup = _load_difficulty(self.difficulty_csv, self.benchmark, self.default_tier)

        self.dataset: Dataset = load_from_disk(str(root_dir / "results"))
        if self.num_samples is not None:
            self.dataset = self.dataset.select(range(self.num_samples))
        requests = self.dataset.map(
            build_prompt,
            with_indices=True,
            remove_columns=self.dataset.column_names,
            num_proc=1,
            fn_kwargs={"tier_lookup": self.tier_lookup, "default_tier": self.default_tier},
        )
        requests.save_to_disk(str(self.task_dir / "requests"))
        return self.task_dir

    def postprocess(self, api_dir: Path) -> None:
        responses: Dataset = load_from_disk(str(api_dir / "responses"))
        responses = responses.map(
            parse_response,
            remove_columns=responses.column_names,
            num_proc=self.num_proc,
        )
        self.dataset = concatenate_datasets([self.dataset, responses], axis=1)
        self.dataset.save_to_disk(str(api_dir / "results"))


def _load_difficulty(csv_path: str, benchmark: str, default_tier: str) -> dict[int, str]:
    """Load `data/sample_difficulty.csv` and return {sample_id: tier} for one benchmark.

    Returns an empty dict (so build_prompt falls back to default_tier) if the
    csv does not exist; the task still runs but every sample uses default_tier.
    """
    p = Path(csv_path)
    if not p.exists():
        print(f"[qa.mqi_score] difficulty csv not found at {p}; falling back to default_tier='{default_tier}' for all samples")
        return {}
    lookup: dict[int, str] = {}
    with p.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("benchmark") != benchmark:
                continue
            try:
                sid = int(row["sample_id"])
            except (KeyError, ValueError):
                continue
            lookup[sid] = row.get("tier", default_tier)
    return lookup


def format_messages(messages: list[dict[str, str]], **kwargs) -> list[dict[str, str]]:
    for message in messages:
        message["content"] = message["content"].format(**kwargs)
    return messages


def build_prompt(example: dict, idx: int, tier_lookup: dict, default_tier: str) -> dict:
    sid = int(example["sample_id"])
    tier = tier_lookup.get(sid, default_tier)
    prompt = format_messages(
        copy.deepcopy(MQI_SCORE_TEMPLATE),
        question=example["question"],
        answer=example["answer"],
        reasoning_trace=example["reasoning_trace"],
        difficulty_tier=tier,
    )
    return {
        "sample_id": example["sample_id"],
        "rollout_id": example["rollout_id"],
        "request_id": idx + 1,
        "prompt": prompt,
    }


def parse_response(example: dict) -> dict:
    response = example.get("response") or {}
    choices = response.get("choices") or []
    raw = (choices[0].get("message", {}).get("content") if choices else "") or ""

    lm = _LOBS_RE.search(raw)
    try:
        l_obs = int(lm.group(1)) if lm else 0
    except ValueError:
        l_obs = 0
    l_obs = max(0, min(5, l_obs))

    pm = _PHASES_RE.search(raw)
    if pm:
        s = pm.group(1).strip().upper()
        if s == "NONE":
            phases: list[int] = []
        else:
            phases = []
            for token in s.split(","):
                token = token.strip()
                if token.isdigit():
                    p = int(token)
                    if 1 <= p <= 5 and p not in phases:
                        phases.append(p)
    else:
        phases = []

    cm = _CONF_RE.search(raw)
    try:
        confidence = float(cm.group(1)) if cm else 0.5
    except ValueError:
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))

    return {"l_obs": l_obs, "phases": phases, "confidence": confidence}
