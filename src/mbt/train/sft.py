import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from accelerate.logging import get_logger
from datasets import Dataset, disable_caching, load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, set_seed
from transformers.models.auto.modeling_auto import MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES
from trl import ModelConfig, ScriptArguments, SFTConfig, SFTTrainer, TrlParser, get_kbit_device_map, get_peft_config, get_quantization_config

from mbt.train import chat_templates


@dataclass
class CustomScriptArguments(ScriptArguments):
    mode: str = field(default="mbt-r", metadata={"help": "Which mode to run in (e.g., distill, mbt-s, mbt-r)."})
    num_rollouts: int | None = field(default=None, metadata={"help": "The number of rollout samples (model responses) to use for training."})
    num_traces: int | None = field(default=None, metadata={"help": "The number of refined traces to use for training."})
    wandb_project: str = field(default="mbt", metadata={"help": "The name of the project for Weights & Biases logging."})
    wandb_run_group: str | None = field(default=None, metadata={"help": "A group name for organizing runs in Weights & Biases."})
    wandb_tags: str | None = field(default=None, metadata={"help": "Comma-separated tags for the run in Weights & Biases."})
    train_seed: int | None = field(default=42, metadata={"help": "Random seed for training to ensure reproducibility."})

    def __post_init__(self):
        os.environ["WANDB_PROJECT"] = self.wandb_project
        if self.wandb_run_group is not None:
            os.environ["WANDB_RUN_GROUP"] = self.wandb_run_group
        if self.wandb_tags is not None:
            os.environ["WANDB_TAGS"] = self.wandb_tags


@dataclass
class CustomSFTConfig(SFTConfig):
    def __post_init__(self):
        if self.dataset_num_proc is None:
            self.dataset_num_proc = int(os.environ.get("OMP_NUM_THREADS", "8"))
        if self.dataloader_num_workers is None:
            self.dataloader_num_workers = int(os.environ.get("OMP_NUM_THREADS", "8"))
        super().__post_init__()


def get_chat_template(model_name: str) -> str | None:
    assignment = {
        "Qwen3": chat_templates.QWEN3,
    }
    model_name = model_name.split("/")[-1]
    for model_prefix in assignment:
        if model_name.startswith(model_prefix):
            return assignment[model_prefix]
    return None


def tokenize(example: dict, script_args, tokenizer, chat_template: str, logger) -> dict:
    completion = (
        [{"role": "assistant", "content": example["response"]}]
        if script_args.mode == "distill"
        else [{"role": "assistant", "content": f"<think>\n{example['synthesized_trace']}\n</think>\n\n<answer>{example['answer']}</answer>"}]
        if script_args.mode == "mbt-s"
        else [{"role": "assistant", "content": f"<think>\n{example['refined_trace']}\n</think>\n\n<answer>{example['answer']}</answer>"}]
    )
    prompt_ids = tokenizer.apply_chat_template(example["prompt"], chat_template=chat_template, add_generation_prompt=True)
    prompt_completion_ids = tokenizer.apply_chat_template(example["prompt"] + completion, chat_template=chat_template, add_generation_prompt=False)
    if prompt_completion_ids[: len(prompt_ids)] != prompt_ids:
        logger.warning("Mismatch between tokenized prompt and the start of tokenized prompt+completion. This may be due to unexpected tokenizer behavior, whitespace issues, or special token handling. Verify that the tokenizer is processing text consistently.")
    completion_mask = [0] * len(prompt_ids) + [1] * (len(prompt_completion_ids) - len(prompt_ids))
    return {"input_ids": prompt_completion_ids, "completion_mask": completion_mask}


def decode(example: dict, tokenizer) -> dict[str, str]:
    prompt_ids = []
    completion_ids = []

    for id, mask in zip(example["input_ids"], example["completion_mask"], strict=False):
        if mask == 0:
            prompt_ids.append(id)
        else:
            completion_ids.append(id)

    return {
        "prompt": tokenizer.decode(prompt_ids, skip_special_tokens=False),
        "completion": tokenizer.decode(completion_ids, skip_special_tokens=False),
    }


def main():
    dataclass_types = (CustomScriptArguments, CustomSFTConfig, ModelConfig)
    parser = TrlParser(dataclass_types)
    script_args, training_args, model_args, _ = parser.parse_args_and_config(return_remaining_strings=True)

    if script_args.wandb_tags is not None:
        training_args.run_name = script_args.wandb_tags
    (Path(training_args.output_dir) / "logs").mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
        handlers=[logging.FileHandler(str(Path(training_args.output_dir) / "logs" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S.log")))],
    )
    logger = get_logger(__name__)

    logger.info(f"script_args: {script_args}", main_process_only=True)
    logger.info(f"training_args: {training_args}", main_process_only=True)
    logger.info(f"model_args: {model_args}", main_process_only=True)

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        dtype=model_args.dtype,
    )
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        # Passing None would not be treated the same as omitting the argument, so we include it only when valid.
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = quantization_config

    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    valid_image_text_architectures = MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES.values()

    if config.architectures and any(arch in valid_image_text_architectures for arch in config.architectures):
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(model_args.model_name_or_path, **model_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)

    tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path, trust_remote_code=model_args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    disable_caching()

    dataset: Dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config, split=script_args.dataset_train_split)
    if script_args.mode == "distill":
        dataset = dataset.filter(lambda example: example["valid"] and example["evaluation"], num_proc=training_args.dataset_num_proc)
        dataset = dataset.filter(lambda example: example["rollout_id"] <= script_args.num_rollouts, num_proc=training_args.dataset_num_proc) if script_args.num_rollouts is not None else dataset
    elif script_args.mode == "mbt-r":
        dataset = dataset.filter(lambda example: example["valid"], num_proc=training_args.dataset_num_proc)
        dataset = dataset.filter(lambda example: example["rollout_id"] <= script_args.num_rollouts, num_proc=training_args.dataset_num_proc) if script_args.num_rollouts is not None else dataset
        dataset = dataset.filter(lambda example: example["trace_id"] <= script_args.num_traces, num_proc=training_args.dataset_num_proc) if script_args.num_traces is not None else dataset
    train_dataset = dataset.map(tokenize, fn_kwargs={"script_args": script_args, "tokenizer": tokenizer, "chat_template": get_chat_template(model.config._name_or_path), "logger": logger}, remove_columns=dataset.column_names, num_proc=training_args.dataset_num_proc)
    train_sample = decode(train_dataset[0], tokenizer)
    for k, v in train_sample.items():
        logger.info(f"{k}: {v}", main_process_only=True)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
    )

    logger.info(f"model: {trainer.model}", main_process_only=True)
    if model_args.use_peft:
        logger.info(f"peft: {trainer.model.get_model_status()}", main_process_only=True)
        if getattr(trainer.accelerator.state, "fsdp_plugin", None):
            from peft.utils.other import fsdp_auto_wrap_policy

            fsdp_plugin = trainer.accelerator.state.fsdp_plugin
            fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(trainer.model)

    if script_args.train_seed is not None:
        set_seed(script_args.train_seed)
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    main()
