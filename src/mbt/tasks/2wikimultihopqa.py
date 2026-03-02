import copy
import json
import os
import re
from pathlib import Path

from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk

from mbt.registry import register_task
from mbt.tasks.qa.metrics import extract_answer
from mbt.tasks.qa.prompt_templates import MBT_S_TEMPLATE, METACOGNITIVE_PROMPT_TEMPLATE, PROMPT_TEMPLATE, SOLUTION_TEMPLATE

TASK_NAME = "2wikimultihopqa"
DATASET_NAME = "framolfese/2WikiMultihopQA"
DATASET_CONFIG = None
DATASET_SPLIT = "validation"
METACOGNITIVE_PROMPT = False
SOLUTION = False
MBT_S = False
SKIP_FORMAT_COLUMNS = False
NUM_PROC = int(os.environ.get("OMP_NUM_THREADS", "8"))
NUM_SAMPLES = None


@register_task(TASK_NAME)
class Task:
    """
    Task wrapper for the 2WikiMultihopQA dataset.

    This class handles the loading and processing of the 2WikiMultihopQA dataset, which is designed
    for multi-hop reasoning across Wikipedia articles. It formats the context (titles and sentences),
    constructs prompts for various inference modes (standard, metacognitive, solution generation),
    and extracts answers from model outputs.

    Args:
      task_config (dict): A configuration dictionary containing the following keys:
        - dataset_name (str, optional): The Hugging Face dataset path. Defaults to "framolfese/2WikiMultihopQA".
        - dataset_config (str, optional): The specific dataset configuration name. Defaults to DATASET_CONFIG.
        - dataset_split (str, optional): The dataset split to load (e.g., "validation"). Defaults to DATASET_SPLIT.
        - metacognitive_prompt (bool, optional): If True, uses `METACOGNITIVE_PROMPT_TEMPLATE` for inference. Defaults to METACOGNITIVE_PROMPT.
        - solution (bool, optional): If True, prepares the task for generating standard solutions. Defaults to SOLUTION.
        - mbt_s (bool, optional): If True, prepares the task for generating MBT-S samples. Defaults to MBT_S.
        - skip_format_columns (bool, optional): If True, skips the column formatting step. Defaults to SKIP_FORMAT_COLUMNS.
        - num_proc (int, optional): Number of CPU workers for dataset processing. Defaults to NUM_PROC.
        - num_samples (int, optional): If set, limits the number of samples processed. Defaults to NUM_SAMPLES.
    """

    def __init__(self, task_config: dict) -> None:
        super().__init__()
        self.task_config: dict = task_config
        self.dataset_name: str = task_config.get("dataset_name", DATASET_NAME)
        self.dataset_config: str | None = task_config.get("dataset_config", DATASET_CONFIG)
        self.dataset_split: str = task_config.get("dataset_split", DATASET_SPLIT)
        self.metacognitive_prompt: bool = task_config.get("metacognitive_prompt", METACOGNITIVE_PROMPT)
        self.solution: bool = task_config.get("solution", SOLUTION)
        self.mbt_s: bool = task_config.get("mbt_s", MBT_S)
        self.skip_format_columns: bool = task_config.get("skip_format_columns", SKIP_FORMAT_COLUMNS)
        self.num_proc: int = task_config.get("num_proc", NUM_PROC)
        self.num_samples: int | None = task_config.get("num_samples", NUM_SAMPLES)

    def preprocess(self, root_dir: Path) -> Path:
        self.task_dir = (root_dir / "solution") if self.solution else (root_dir / "mbt-s") if self.mbt_s else root_dir
        self.task_dir.mkdir(parents=True, exist_ok=True)
        with (self.task_dir / "task_config.json").open("w", encoding="utf-8") as f:
            json.dump(self.task_config, f, ensure_ascii=False, indent=4)

        self.dataset: Dataset = load_dataset(self.dataset_name, name=self.dataset_config, split=self.dataset_split, num_proc=self.num_proc)
        if self.num_samples is not None:
            self.dataset = self.dataset.select(range(self.num_samples))
        if not self.skip_format_columns:
            self.dataset = self.dataset.map(format_columns, with_indices=True, remove_columns=self.dataset.column_names, num_proc=self.num_proc)
        requests = self.dataset.map(build_prompt, with_indices=True, fn_kwargs={"metacognitive_prompt": self.metacognitive_prompt, "solution": self.solution, "mbt_s": self.mbt_s}, remove_columns=self.dataset.column_names, num_proc=self.num_proc)
        requests.save_to_disk(str(self.task_dir / "requests"))
        return self.task_dir

    def postprocess(self, api_dir: Path) -> None:
        responses: Dataset = load_from_disk(str(api_dir / "responses"))
        self.dataset = concatenate_datasets([self.dataset, responses.select_columns(["prompt", "response", "valid"])], axis=1)
        if self.solution:
            self.dataset = self.dataset.map(lambda example: {"solution": example["response"][0].strip()}, remove_columns="response", num_proc=self.num_proc).rename_column("prompt", "solution_prompt")
            self.dataset = self.dataset.select_columns([key for key in ["sample_id", "metadata", "question", "answer", "answer_aliases", "solution_prompt", "solution"] if key in self.dataset.column_names])
        elif self.mbt_s:
            self.dataset = self.dataset.map(add_prompt, num_proc=self.num_proc)
            self.dataset = self.dataset.map(lambda example: {"synthesized_trace": example["response"][0].strip()}, remove_columns="response", num_proc=self.num_proc)
            self.dataset = self.dataset.select_columns([key for key in ["sample_id", "metadata", "question", "answer", "answer_aliases", "prompt", "synthesized_trace"] if key in self.dataset.column_names])
        else:
            self.dataset = self.dataset.map(expand_rollouts, batched=True, batch_size=1, remove_columns=self.dataset.column_names, num_proc=self.num_proc)
            self.dataset = self.dataset.map(lambda example: {"reasoning_trace": match.group(1).strip() if (match := re.search(r"<think>(.*)</think>", example["response"], re.DOTALL)) else ""}, num_proc=self.num_proc)
            self.dataset = self.dataset.map(lambda example: {"predicted_answer": extract_answer(example["response"])}, num_proc=self.num_proc)
            self.dataset = self.dataset.select_columns([key for key in ["sample_id", "metadata", "question", "answer", "answer_aliases", "rollout_id", "prompt", "response", "valid", "reasoning_trace", "predicted_answer"] if key in self.dataset.column_names])
        self.dataset.save_to_disk(str(api_dir / "results"))


def format_columns(example: dict, idx: int) -> dict:
    question = example.pop("question")
    answer = example.pop("answer")
    return {"sample_id": idx + 1, "metadata": example, "question": question, "answer": answer}


def format_messages(messages: list[dict[str, str]], **kwargs) -> list[dict[str, str]]:
    for message in messages:
        message["content"] = message["content"].format(**kwargs)
    return messages


def build_prompt(example: dict, idx: int, metacognitive_prompt: bool, solution: bool, mbt_s: bool) -> dict:
    variables = {key: example[key] for key in ["question", "answer"]}
    variables["context"] = "\n".join([f"Document [{idx}] (Title: {title}) {' '.join(sentences)}" for idx, (title, sentences) in enumerate(zip(example["metadata"]["context"]["title"], example["metadata"]["context"]["sentences"], strict=False))])
    prompt_template = SOLUTION_TEMPLATE if solution else MBT_S_TEMPLATE if mbt_s else METACOGNITIVE_PROMPT_TEMPLATE if metacognitive_prompt else PROMPT_TEMPLATE
    prompt = format_messages(copy.deepcopy(prompt_template), **variables)
    return {"sample_id": example["sample_id"], "request_id": idx + 1, "prompt": prompt}


def add_prompt(example: dict) -> dict:
    variables = {key: example[key] for key in ["question", "answer"]}
    variables["context"] = "\n".join([f"Document [{idx}] (Title: {title}) {' '.join(sentences)}" for idx, (title, sentences) in enumerate(zip(example["metadata"]["context"]["title"], example["metadata"]["context"]["sentences"], strict=False))])
    prompt_template = PROMPT_TEMPLATE
    prompt = format_messages(copy.deepcopy(prompt_template), **variables)
    return {"prompt": prompt}


def expand_rollouts(batch: dict) -> dict:
    response = batch.pop("response")[0]
    valid = batch.pop("valid")[0]
    batch = {key: batch[key] * len(response) for key in batch}
    return batch | {"rollout_id": list(range(1, len(response) + 1)), "response": [r for r in response], "valid": [v for v in valid]}
