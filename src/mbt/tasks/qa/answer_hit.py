import copy
import json
import os
from pathlib import Path

from datasets import Dataset, concatenate_datasets, load_from_disk

from mbt.registry import register_task
from mbt.tasks.qa.prompt_templates import ANSWER_HIT_TEMPLATE

TASK_NAME = "qa.answer_hit"
NUM_PROC = int(os.environ.get("OMP_NUM_THREADS", "8"))
NUM_SAMPLES = None


@register_task(TASK_NAME)
class Task:
    """
    Task wrapper for evaluating answer hits in QA reasoning traces.

    This class loads a dataset of reasoning traces (typically generated from a previous
    QA task) and prepares it for an "answer hit" evaluation. This involves checking
    if the correct answer (or its aliases) appears as a substring within the model's
    reasoning trace (`reasoning_trace`) and optionally using an LLM-based judge to
    verify the hit.

    Args:
      task_config (dict): A configuration dictionary containing the following keys:
        - num_proc (int, optional): Number of CPU workers for dataset processing. Defaults to NUM_PROC.
        - num_samples (int, optional): If set, limits the number of samples processed. Defaults to NUM_SAMPLES.
    """

    def __init__(self, task_config: dict) -> None:
        super().__init__()
        self.task_config: dict = task_config
        self.num_proc: int = task_config.get("num_proc", NUM_PROC)
        self.num_samples: int | None = task_config.get("num_samples", NUM_SAMPLES)

    def preprocess(self, root_dir: Path) -> Path | None:
        self.task_dir = root_dir / TASK_NAME.split(".")[-1].replace("_", "-")
        self.task_dir.mkdir(parents=True, exist_ok=True)
        with (self.task_dir / "task_config.json").open("w", encoding="utf-8") as f:
            json.dump(self.task_config, f, ensure_ascii=False, indent=4)

        self.dataset: Dataset = load_from_disk(str(root_dir / "results"))
        if self.num_samples is not None:
            self.dataset = self.dataset.select(range(self.num_samples))
        self.dataset = self.dataset.map(compute_substring_match, fn_kwargs={"reference_columns": ["answer", "answer_aliases"]}, num_proc=self.num_proc)
        requests = self.dataset.map(build_prompt, with_indices=True, remove_columns=self.dataset.column_names, num_proc=self.num_proc)
        requests.save_to_disk(str(self.task_dir / "requests"))
        return self.task_dir

    def postprocess(self, api_dir: Path) -> None:
        responses: Dataset = load_from_disk(str(api_dir / "responses"))
        responses = responses.map(lambda example: {"judgment": example["response"][0].strip()}, remove_columns=responses.column_names, num_proc=self.num_proc)
        self.dataset = concatenate_datasets([self.dataset, responses], axis=1)
        self.dataset = self.dataset.map(lambda example: {"answer_hit": 1.0 if example["judgment"].upper() == "YES" else 0.0}, remove_columns="judgment", num_proc=self.num_proc)
        self.dataset.save_to_disk(str(api_dir / "results"))


def compute_substring_match(example: dict, reference_columns: list[str]) -> dict:
    references = [item for col in reference_columns if col in example for item in ([example[col]] if isinstance(example[col], str) else example[col])]
    return {"substring_match": 1.0 if any(reference.lower().strip() in example["reasoning_trace"].lower() for reference in references) else 0.0}


def format_messages(messages: list[dict[str, str]], **kwargs) -> list[dict[str, str]]:
    for message in messages:
        message["content"] = message["content"].format(**kwargs)
    return messages


def build_prompt(example: dict, idx: int) -> dict:
    variables = {key: example[key] for key in ["question", "answer", "reasoning_trace"]}
    variables["answer"] = variables["answer"] + f" (aliases: {', '.join(example['answer_aliases'])})" if ("answer_aliases" in example) and example["answer_aliases"] else variables["answer"]
    prompt = format_messages(copy.deepcopy(ANSWER_HIT_TEMPLATE), **variables)
    return {"sample_id": example["sample_id"], "rollout_id": example["rollout_id"], "request_id": idx + 1, "prompt": prompt}
