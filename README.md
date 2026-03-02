# MBT

This repository contains the official implementation and open-source code for the paper **"Mirroring the Mind: Distilling Human-Like Metacognitive Strategies into Large Language Models"**.

**MBT** is a post-training framework designed to enhance the structural stability and efficiency of Large Language Models (LLMs). Unlike standard reasoning distillation, MBT explicitly injects metacognitive behaviors—such as understanding and filtering, planning, execution and monitoring, self-correction, and verification—to prevent reasoning hallucinations.

## 🌟 Features

* **Metacognitive Tasks**: Specialized pipeline for generating **MBT-S** (Synthesized) and **MBT-R** (Rewritten) reasoning traces.
* **Targeted Evaluation**: Custom tasks for scoring **Overthinking Score**, **Underthinking Score**, and **Metacognition Score** using LLM-as-a-Judge.
* **Streamlined Pipeline**: A unified entry point (`mbt`) managing data preprocessing, high-throughput inference (vLLM), and result post-processing.
* **Training Support**: Specialized Supervised Fine-Tuning (SFT) scripts supporting various distillation modes (RS, MBT-S, MBT-R).
* **Modern Stack**: Built on **[uv](https://github.com/astral-sh/uv)** for fast dependency management and **vLLM** for efficient inference.

## 📚 Datasets

We have publicly released all experimental data required to reproduce our results, including raw rollouts, solutions, and the curated training datasets for **MBT-S**, **MBT-R**, and **Distill-R**.

You can access the full collection on our Hugging Face organization page:
👉 **[https://huggingface.co/metacognitive-behavioral-tuning](https://huggingface.co/metacognitive-behavioral-tuning)**

The collection includes:
* **Training Samples**: Curated datasets for MBT-S (Synthesized) and MBT-R (Rewritten) specifically on **HotpotQA**.
* **Raw Rollouts**: Raw training data used for constructing **self-distill** and **gpt-oss-distill** baselines via Rejection Sampling.
* **Solutions**: Gold standard solutions used in multi-turn prompting. They are provided in the first turn to effectively guide the reasoning trace towards the correct answer during **MBT-R** generation. Additionally, they are used during evaluation to assist the judge model in accurately assessing **Overthinking**, **Underthinking**, and **Metacognition** scores.

---

## 🛠️ Installation

We use **`uv`** for dependency management. It replaces `pip` and `poetry` for a faster, more reliable experience.

### 1. Install `uv`

If you do not have `uv` installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone the Repository

```bash
git clone https://github.com/metacognitive-behavioral-tuning/MBT.git
cd MBT
```

### 3. Install Dependencies

Initialize the environment and install dependencies. We strongly recommend installing with the `vllm` extra for inference.

```bash
# Create virtual env and install dependencies with vLLM support
uv sync --extra vllm
```

---

## 🔑 Authentication & Setup

Before running scripts, you must authenticate with Hugging Face (to access gated models/datasets) and Weights & Biases (for experiment logging).

We provide a helper script for this:

```bash
bash scripts/setup_auth.sh
```

Or run manually:

```bash
uv run hf auth login
uv run wandb login
```

### Download Assets

Download the required models (e.g., Qwen series, GPT-OSS) and datasets (MuSiQue, 2WikiMultiHopQA, HotpotQA) to your local cache:

```bash
uv run scripts/download.py
```

---

## 🚀 Usage: Inference & Evaluation Pipeline

The core logic is handled by the `mbt` module. The general syntax is:

```bash
uv run --extra vllm mbt \
    --task-name "<TASK>" \
    --task-config '<JSON>' \
    --api-name "vllm.chat" \
    --api-config '<JSON>' \
    --script-config '{"root_dir": "output/..."}'
```

### 1. Supported Tasks

We currently support the following multi-hop QA benchmarks:

* `musique`
* `2wikimultihopqa`
* `hotpotqa`

And specific evaluation/scoring tasks:

* `qa.evaluation`
* `qa.answer_hit`
* `qa.overthinking_score` / `qa.underthinking_score` / `qa.metacognition_score`

### 2. Generation Examples

#### Standard Inference

Running inference on the validation set of MuSiQue using Qwen3-4B:

```bash
uv run --extra vllm mbt \
    --task-name "musique" \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-4B", "model_kwargs": {"model": "Qwen/Qwen3-4B", "config": "configs/vllm/defaults.yaml", "max_model_len": 40960, "tensor_parallel_size": 4}, "request_kwargs": {"max_completion_tokens": 32768, "temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 32, "num_shards": 32, "validation_max_retries": 0, "strict_validation": false}' \
    --script-config '{"root_dir": "output/musique/validation"}'
```

#### Generating MBT-S Samples

To generate synthesized metacognitive traces (MBT-S) using a strong teacher model (e.g., gpt-oss-120b):

```bash
uv run --extra vllm mbt \
    --task-name "hotpotqa" \
    --task-config '{"dataset_split": "train", "mbt_s": true}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "gpt-oss-120b-high", "model_kwargs": {"model": "openai/gpt-oss-120b", "config": "configs/vllm/gpt-oss.yaml", "max_model_len": 40960, "tensor_parallel_size": 4}, "request_kwargs": {"max_completion_tokens": 32768, "temperature": 0.6, "top_p": 0.95, "n": 1, "reasoning_effort": "high", "extra_body": {"top_k": 20}}, "num_threads": 32, "num_shards": 32}' \
    --script-config '{"root_dir": "output/hotpotqa/train"}'
```

#### Metacognitive Prompting

To run inference with a system prompt that encourages metacognition (without fine-tuning):

```bash
uv run --extra vllm mbt \
    --task-name "hotpotqa" \
    --task-config '{"dataset_split": "validation", "metacognitive_prompt": true}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-4B/metacognitive-prompt", "model_kwargs": {"model": "Qwen/Qwen3-4B", "config": "configs/vllm/defaults.yaml", "max_model_len": 40960, "tensor_parallel_size": 4}, "request_kwargs": {"max_completion_tokens": 32768, "temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 32, "num_shards": 32, "validation_max_retries": 0, "strict_validation": false}' \
    --script-config '{"root_dir": "output/hotpotqa/validation"}'
```

### 3. Evaluation Examples

#### Standard QA Evaluation

Computes metrics like Exact Match (EM), F1, and LLM-as-a-Judge.

```bash
uv run --extra vllm mbt \
    --task-name "qa.evaluation" \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "gpt-oss-120b-high", "model_kwargs": {"model": "openai/gpt-oss-120b", "config": "configs/vllm/gpt-oss.yaml", "max_model_len": 40960, "tensor_parallel_size": 4}, "request_kwargs": {"max_completion_tokens": 32768, "temperature": 0.6, "top_p": 0.95, "n": 1, "reasoning_effort": "high", "extra_body": {"top_k": 20}}, "num_threads": 32, "num_shards": 32}' \
    --script-config '{"root_dir": "output/hotpotqa/validation/Qwen3-4B"}'
```

#### Scoring Behavioral Quality (Overthinking/Underthinking/Metacognition)

To evaluate the quality of the reasoning traces generated in previous steps (e.g., detecting redundancy):

```bash
uv run --extra vllm mbt \
    --task-name "qa.overthinking_score" \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "gpt-oss-120b-high", "model_kwargs": {"model": "openai/gpt-oss-120b", "config": "configs/vllm/gpt-oss.yaml", "max_model_len": 40960, "tensor_parallel_size": 4}, "request_kwargs": {"max_completion_tokens": 32768, "temperature": 0.6, "top_p": 0.95, "n": 1, "reasoning_effort": "high", "extra_body": {"top_k": 20}}, "num_threads": 32, "num_shards": 32, "allow_none": true}' \
    --script-config '{"root_dir": "output/musique/validation/Qwen3-4B"}'
```

---

## 🏋️ Training (Supervised Fine-Tuning)

We provide a flexible `sft.py` script to train models using **Rejection Sampling (RS)**, **MBT-S**, or **MBT-R** samples.

### SFT Modes

The training script supports a `--mode` argument to handle different data formats:

* **`RS`**: Standard Rejection Sampling on correct responses.
* **`MBT-S`**: Trains on synthesized metacognitive traces.
* **`MBT-R`**: Trains on rewritten/refined traces.

### Running with `uv`

```bash
uv run accelerate launch --config_file configs/accelerate/multi_gpu.yaml --main_process_port $(shuf -i 49152-65535 -n 1) src/mbt/train/sft.py --config configs/sft.yaml --model_name_or_path Qwen/Qwen3-0.6B --dataset_name metacognitive-behavioral-tuning/mbt-r-hotpotqa --dataset_config Qwen3-0.6B --wandb_tags Qwen3-0.6B,mbt-r,sft,1e-4,128 --output_dir output/train/Qwen3-0.6B/mbt-r/sft/1e-4/128 --learning_rate 1e-4 --per_device_train_batch_size 2 --gradient_accumulation_steps 16
```

### Running on Slurm

If you have a Slurm cluster, you can use the provided `scripts/slurm/sft.slurm`.

**Example: MBT-R Training**

```bash
sbatch --cpus-per-task=32 --gres=gpu:4 scripts/slurm/sft.slurm --config configs/sft.yaml --model_name_or_path Qwen/Qwen3-0.6B --dataset_name metacognitive-behavioral-tuning/mbt-r-hotpotqa --dataset_config Qwen3-0.6B --wandb_tags Qwen3-0.6B,mbt-r,sft,1e-4,128 --output_dir output/train/Qwen3-0.6B/mbt-r/sft/1e-4/128 --learning_rate 1e-4 --per_device_train_batch_size 2 --gradient_accumulation_steps 16
```

---

## 📂 Project Structure

```text
MBT/
├── src/mbt/                # Core package
│   ├── apis/               # API wrappers
│   ├── tasks/              # Task definitions
│   ├── train/              # Training utilities
│   ├── download.py         # Utility script to download required models and datasets
│   ├── main.py             # Main entry point for executing tasks and API pipelines
│   └── registry.py         # Dynamic registry for tasks and APIs
├── scripts/
│   ├── slurm/              # Cluster submission scripts
│   ├── main.sh             # Bash script for batch running inference tasks
│   ├── sft.sh              # Bash script for batch running SFT training
│   └── setup_auth.sh       # Helper script for authentication
├── configs/                # Configuration YAMLs
└── pyproject.toml          # Dependencies and project metadata
```

## ⚖️ License

This project is licensed under the **Apache-2.0 License**. See the [LICENSE](LICENSE) file for details.
