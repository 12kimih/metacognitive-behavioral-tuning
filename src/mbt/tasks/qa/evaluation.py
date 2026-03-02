import copy
import json
import os
from pathlib import Path

from datasets import Dataset, concatenate_datasets, load_from_disk

from mbt.registry import register_task
from mbt.tasks.qa.metrics import aggregate_metrics, compute_metrics
from mbt.tasks.qa.prompt_templates import EVALUATION_TEMPLATE

TASK_NAME = "qa.evaluation"
METRICS = ["exact_match", "substring_match", "f1_score", "llm_as_judge"]
NUM_PROC = int(os.environ.get("OMP_NUM_THREADS", "8"))
NUM_SAMPLES = None


@register_task(TASK_NAME)
class Task:
    """
    Task wrapper for evaluating Question Answering (QA) model outputs.

    This class loads prediction results from a previous task step and computes various
    evaluation metrics. It supports both deterministic metrics (Exact Match, Substring Match,
    F1 Score) and model-based evaluation ("LLM as a Judge").

    If "llm_as_judge" is included in the metrics, it constructs evaluation prompts
    comparing the predicted answer to the ground truth. Otherwise, it computes and
    saves the deterministic metrics directly.

    Args:
      task_config (dict): A configuration dictionary containing the following keys:
        - metrics (list[str], optional): A list of metrics to compute. Options: "exact_match", "substring_match", "f1_score", "llm_as_judge". Defaults to METRICS.
        - num_proc (int, optional): Number of CPU workers for metric computation. Defaults to NUM_PROC.
        - num_samples (int, optional): If set, limits the number of samples evaluated. Defaults to NUM_SAMPLES.
    """

    def __init__(self, task_config: dict) -> None:
        super().__init__()
        self.task_config: dict = task_config
        self.metrics: list[str] = task_config.get("metrics", METRICS)
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
        self.dataset = self.dataset.map(compute_metrics, fn_kwargs={"metrics": self.metrics, "prediction_column": "predicted_answer", "reference_columns": ["answer", "answer_aliases"]}, num_proc=self.num_proc)

        if "llm_as_judge" in self.metrics:
            requests = self.dataset.map(build_prompt, with_indices=True, remove_columns=self.dataset.column_names, num_proc=self.num_proc)
            requests.save_to_disk(str(self.task_dir / "requests"))
            return self.task_dir
        else:
            self.dataset.save_to_disk(str(self.task_dir / "results"))
            results = aggregate_metrics(self.dataset, self.metrics, prediction_column="predicted_answer")
            with (self.task_dir / "results.json").open("w", encoding="utf-8") as f:
                f.write(json.dumps(results, ensure_ascii=False, indent=4))
            return None

    def postprocess(self, api_dir: Path) -> None:
        responses: Dataset = load_from_disk(str(api_dir / "responses"))
        responses = responses.map(lambda example: {"judgment": example["response"][0].strip()}, remove_columns=responses.column_names, num_proc=self.num_proc)
        self.dataset = concatenate_datasets([self.dataset, responses], axis=1)
        self.dataset = self.dataset.map(lambda example: {"llm_as_judge": 1.0 if example["judgment"].upper() == "A" else 0.0}, num_proc=self.num_proc)
        self.dataset.save_to_disk(str(api_dir / "results"))
        results = aggregate_metrics(self.dataset, self.metrics, prediction_column="predicted_answer")
        with (api_dir / "results.json").open("w", encoding="utf-8") as f:
            f.write(json.dumps(results, ensure_ascii=False, indent=4))


def format_messages(messages: list[dict[str, str]], **kwargs) -> list[dict[str, str]]:
    for message in messages:
        message["content"] = message["content"].format(**kwargs)
    return messages


def build_prompt(example: dict, idx: int) -> dict:
    variables = {key: example[key] for key in ["question", "answer", "predicted_answer"]}
    variables["answer"] = variables["answer"] + f" (aliases: {', '.join(example['answer_aliases'])})" if ("answer_aliases" in example) and example["answer_aliases"] else variables["answer"]
    prompt = format_messages(copy.deepcopy(EVALUATION_TEMPLATE), **variables)
    return {"sample_id": example["sample_id"], "rollout_id": example["rollout_id"], "request_id": idx + 1, "prompt": prompt}
