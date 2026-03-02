import copy
import json
import os
from pathlib import Path

from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk

from mbt.registry import register_task
from mbt.tasks.qa.prompt_templates import METACOGNITION_SCORE_TEMPLATE

TASK_NAME = "qa.metacognition_score"
SOLUTION_NAME = "metacognitive-behavioral-tuning/solutions-gpt-oss-120b"
SOLUTION_CONFIG = "musique"
SOLUTION_SPLIT = "validation"
NUM_PROC = int(os.environ.get("OMP_NUM_THREADS", "8"))
NUM_SAMPLES = None


@register_task(TASK_NAME)
class Task:
    """
    Task wrapper for evaluating metacognitive reasoning quality (Version 1).

    This class evaluates a model's reasoning trace by comparing it against a gold standard
    solution. It loads the generated results (containing the model's `reasoning_trace`)
    and merges them with a reference solution dataset. It then constructs prompts for an
    LLM judge using the `METACOGNITION_SCORE_TEMPLATE` to assign a score, assessing how
    well the generated reasoning aligns with the logic of the reference solution.

    Args:
      task_config (dict): A configuration dictionary containing the following keys:
        - solution_name (str, optional): The Hugging Face dataset path for the reference solution. Defaults to "metacognitive-behavioral-tuning/solutions-gpt-oss-120b".
        - solution_config (str, optional): The configuration name for the reference dataset. Defaults to "musique".
        - solution_split (str, optional): The split of the reference dataset to load. Defaults to "validation".
        - num_proc (int, optional): Number of CPU workers for dataset processing. Defaults to NUM_PROC.
        - num_samples (int, optional): If set, limits the number of samples processed. Defaults to NUM_SAMPLES.
    """

    def __init__(self, task_config: dict) -> None:
        super().__init__()
        self.task_config: dict = task_config
        self.solution_name: str = task_config.get("solution_name", SOLUTION_NAME)
        self.solution_config: str = task_config.get("solution_config", SOLUTION_CONFIG)
        self.solution_split: str = task_config.get("solution_split", SOLUTION_SPLIT)
        self.num_proc: int = task_config.get("num_proc", NUM_PROC)
        self.num_samples: int | None = task_config.get("num_samples", NUM_SAMPLES)

    def preprocess(self, root_dir: Path) -> Path | None:
        self.task_dir = root_dir / TASK_NAME.split(".")[-1].replace("_", "-")
        self.task_dir.mkdir(parents=True, exist_ok=True)
        with (self.task_dir / "task_config.json").open("w", encoding="utf-8") as f:
            json.dump(self.task_config, f, ensure_ascii=False, indent=4)

        self.dataset: Dataset = load_from_disk(str(root_dir / "results"))
        self.solution: Dataset = load_dataset(self.solution_name, name=self.solution_config, split=self.solution_split)
        self.dataset = concatenate_datasets([self.dataset, self.solution.select_columns(["solution_prompt", "solution"])], axis=1)
        if self.num_samples is not None:
            self.dataset = self.dataset.select(range(self.num_samples))
        requests = self.dataset.map(build_prompt, with_indices=True, remove_columns=self.dataset.column_names, num_proc=self.num_proc)
        requests.save_to_disk(str(self.task_dir / "requests"))
        return self.task_dir

    def postprocess(self, api_dir: Path) -> None:
        responses: Dataset = load_from_disk(str(api_dir / "responses"))
        responses = responses.map(lambda example: {"metacognition_score": safe_parse_int(example["response"][0])}, remove_columns=responses.column_names, num_proc=self.num_proc)
        self.dataset = concatenate_datasets([self.dataset.remove_columns(["solution_prompt", "solution"]), responses], axis=1)
        self.dataset.save_to_disk(str(api_dir / "results"))


def format_messages(messages: list[dict[str, str]], **kwargs) -> list[dict[str, str]]:
    for message in messages:
        message["content"] = message["content"].format(**kwargs)
    return messages


def build_prompt(example: dict, idx: int) -> dict:
    prompt = example["solution_prompt"] + [{"role": "assistant", "content": example["solution"]}] + format_messages(copy.deepcopy(METACOGNITION_SCORE_TEMPLATE), reasoning_trace=example["reasoning_trace"])
    return {"sample_id": example["sample_id"], "rollout_id": example["rollout_id"], "request_id": idx + 1, "prompt": prompt}


def safe_parse_int(value):
    if value is None:
        return 0
    try:
        return int(value.strip())
    except ValueError:
        return 0
