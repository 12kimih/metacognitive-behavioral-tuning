# MBT

Official implementation of the paper **"Metacognitive Behavioral Tuning of Large Language Models for Multi-Hop Question Answering"**.

**MBT** (Metacognitive Behavioral Tuning) is a post-training framework that injects a five-phase metacognitive structure into reasoning traces — *understanding & filtering*, *planning*, *execution & monitoring*, *self-correction*, *verification* — so that valid intermediate conclusions are recognized and preserved rather than overridden by continued exploration. MBT has two formulations:

- **MBT-S** synthesizes new metacognitive traces from scratch.
- **MBT-R** rewrites the student's own traces into a metacognitive form.

`mbt` is the codebase that runs MBT. It unifies (1) data-generation rollouts on multi-hop QA / math benchmarks, (2) SFT training on three distillation modes, (3) judge-based scoring (Accuracy-Efficiency Score / Reach-Redundancy Profile / Metacognitive Quality Index) — all behind a single `mbt` CLI that orchestrates vLLM, OpenAI, or HuggingFace backends.

---

## Table of Contents

- [1. Quick links](#1-quick-links)
- [2. Installation](#2-installation)
- [3. Authentication & asset download](#3-authentication--asset-download)
- [4. Architecture in one screen](#4-architecture-in-one-screen)
- [5. Quickstart (10-minute smoke test)](#5-quickstart-10-minute-smoke-test)
- [6. Reproducing the paper, step by step](#6-reproducing-the-paper-step-by-step)
- [7. Task reference](#7-task-reference)
- [8. API backend reference](#8-api-backend-reference)
- [9. SFT training reference](#9-sft-training-reference)
- [10. Configs](#10-configs)
- [11. Scoring metrics (AES / RRP / MQI)](#11-scoring-metrics-aes--rrp--mqi)
- [12. Project layout](#12-project-layout)
- [13. SLURM submission](#13-slurm-submission)
- [14. Troubleshooting](#14-troubleshooting)
- [15. License & citation](#15-license--citation)

---

## 1. Quick links

| What | Where |
|---|---|
| HF Hub organization | <https://huggingface.co/metacognitive-behavioral-tuning> |
| MBT-R training data | `metacognitive-behavioral-tuning/mbt-r-hotpotqa` |
| MBT-S training data | `metacognitive-behavioral-tuning/mbt-s-gpt-oss-120b` |
| Distill-R baseline | `metacognitive-behavioral-tuning/distill-r-hotpotqa` |
| Rejection-Sampling baseline | `metacognitive-behavioral-tuning/rollouts-hotpotqa` |
| Gold solutions (for MBT-R + scoring) | `metacognitive-behavioral-tuning/solutions-gpt-oss-120b` |
| Paper-table launch catalog (single host) | `scripts/tasks/local/*.sh` |
| Paper-table launch catalog (SLURM cluster) | `scripts/tasks/slurm/*.sh` |

---

## 2. Installation

### 2.1. Prerequisites

| Component | Version |
|---|---|
| Python | 3.12 |
| CUDA | 12.8 (driver 535+) |
| GPU | NVIDIA, ≥ 24 GB VRAM for 4B-scale; multi-GPU recommended for 8B+ |
| Disk | ≥ 500 GB free for full paper reproduction |
| Package manager | [`uv`](https://github.com/astral-sh/uv) (replaces pip / poetry) |

### 2.2. Install `uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2.3. Clone the repository

```bash
git clone git@github.com:12kimih/MBT.git
cd MBT
```

### 2.4. Install dependencies

```bash
uv sync
```

This creates `.venv/` and installs the full stack (vLLM 0.19.1, TRL 1.3+, PEFT, flash-attn 2.8, Liger Kernel, etc.) plus the `mbt` console script.

Confirm with:

```bash
uv run mbt --help
# usage: mbt [-h] --task-name TASK_NAME ...
```

---

## 3. Authentication & asset download

### 3.1. Secrets

Copy `.env.example` to `.env` and fill in the keys you need:

```bash
cp .env.example .env
```

```dotenv
HF_TOKEN=hf_...              # required: HuggingFace gated models/datasets
OPENAI_API_KEY=sk-...        # required if you use openai.* backends
WANDB_API_KEY=...            # required for SFT logging (used by sft.py)
```

`.env` is loaded automatically by `mbt` on startup unless `script_config.load_dotenv=false`.

### 3.2. One-line login (recommended)

```bash
bash scripts/setup_auth.sh
# → runs:  uv run hf auth login   +   uv run wandb login
```

### 3.3. Pre-fetch models & datasets

Pull every model + dataset referenced by the paper-table pipeline to your local HF cache (`$HF_HOME`):

```bash
uv run python scripts/download.py
```

You can subset:

```bash
uv run python scripts/download.py --skip-models
uv run python scripts/download.py --skip-datasets
uv run python scripts/download.py --models "Qwen/Qwen3-4B"
```

The script retries 10× with 60s backoff per repo; failures are skipped, not fatal.

---

## 4. Architecture in one screen

```
┌──────────────────────────────────────────────────────────┐
│  mbt <task-name>                                         │
│  ─────────────────                                       │
│  task.preprocess(root_dir)  →  task_dir/requests/        │  ← prompts
│           │                                              │
│           ▼                                              │
│  api.process(task_dir)      →  task_dir/<model>/responses/│  ← inference
│           │                                              │
│           ▼                                              │
│  task.postprocess(api_dir)  →  task_dir/<model>/results/  │  ← derived columns
└──────────────────────────────────────────────────────────┘
```

- **Tasks** define *what data to prompt with and how to interpret responses*. They live in `src/mbt/tasks/`. 8 tasks are registered: 3 rollout-generation (`hotpotqa`, `musique`, `2wikimultihopqa`) and 5 analysis (`qa.mbt_r`, `qa.evaluation`, `qa.answer_hit`, `qa.rrp_score`, `qa.mqi_score`).
- **APIs** define *how to run inference*. They live in `src/mbt/apis/`. 2 registered: `vllm.chat` (default, local server) and `openai.chat` (hosted SDK).
- **Training** (`src/mbt/train/sft.py`) is **not** driven by the `mbt` CLI — it is invoked separately via `accelerate launch`.

Output layout on disk (every task + API agree):

```
{root_dir}/
  <task-dir>/                # e.g. hotpotqa/, mbt-r/, evaluation/
    task_config.json
    requests/                # HF Dataset of prompts
    {model_name}/
      api_config.json
      stats.json
      logs/{timestamp}.log
      cache/response_{shard}/
      responses/             # final HF Dataset (requests + response/valid)
      results/               # task.postprocess output
      results.json           # judge-aggregate metrics (eval tasks only)
```

Subsequent tasks consume the previous step's `results/` directly via `load_from_disk`.

---

## 5. Quickstart (10-minute smoke test)

Generate 16 HotpotQA rollouts with Qwen3-0.6B and score them:

```bash
# Step 1 — rollout (≈ 5 min on 1× A100)
uv run mbt \
    --task-name hotpotqa \
    --task-config '{"dataset_split": "validation", "num_samples": 16}' \
    --api-name vllm.chat \
    --api-config '{
        "model_name": "Qwen3-0.6B",
        "model_kwargs": {"config": "configs/vllm/qwen3-0.6b.yaml"},
        "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1}
    }' \
    --script-config '{"root_dir": "output/_smoke/hotpotqa"}'

# Step 2 — evaluation (deterministic short-circuit, no judge → just EM/F1)
uv run mbt \
    --task-name qa.evaluation \
    --task-config '{"metrics": ["exact_match", "substring_match", "f1_score"]}' \
    --script-config '{"root_dir": "output/_smoke/hotpotqa/Qwen3-0.6B"}'

cat output/_smoke/hotpotqa/Qwen3-0.6B/results.json
```

When `metrics` does not contain `"llm_as_judge"`, `qa.evaluation` short-circuits and writes `results.json` directly — no judge model needed.

---

## 6. Reproducing the paper, step by step

The full paper-table reproduction is encoded as a flat shell-script catalog under `scripts/tasks/`. Two mirrored layouts:

- `scripts/tasks/local/` — direct execution on the current host via `uv run`.
- `scripts/tasks/slurm/` — same matrix, submitted via `sbatch` to a SLURM cluster.

Pick one and run end-to-end. Below is the local variant; replace `local/` with `slurm/` for cluster mode.

### Phase 1 — Base rollouts on the three QA benchmarks

For each base model × {`musique`, `hotpotqa`, `2wikimultihopqa`} × {`validation`, `train`}:

```bash
bash scripts/tasks/local/rollout.sh
```

Cells:
- 141 invocations total (4 models × 3 datasets × 2 splits + variant rollouts).
- Output: `output/{dataset}/{split}/{model_name}/results/` (with `reasoning_trace`, `predicted_answer` columns).

You can also run one cell manually:

```bash
uv run mbt \
    --task-name musique \
    --task-config '{"dataset_split": "train"}' \
    --api-name vllm.chat \
    --api-config '{
        "model_name": "Qwen3-4B",
        "model_kwargs": {"config": "configs/vllm/qwen3-4b.yaml"},
        "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}},
        "num_threads": 1024
    }' \
    --script-config '{"root_dir": "output/musique/train"}'
```

### Phase 2 — Gold solutions (gpt-oss-120b teacher)

```bash
bash scripts/tasks/local/solution.sh
```

Runs each top-level QA task with `task_config.solution=true`, which switches the prompt to `SOLUTION_TEMPLATE` and stores `solution_prompt`, `solution` (gold reasoning) under `output/{dataset}/train/solution/gpt-oss-120b-high/results/`.

Example one cell:

```bash
uv run mbt \
    --task-name hotpotqa \
    --task-config '{"dataset_split": "train", "solution": true}' \
    --api-name vllm.chat \
    --api-config '{
        "model_name": "gpt-oss-120b-high",
        "model_kwargs": {"config": "configs/vllm/gpt-oss-120b.yaml"},
        "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "reasoning_effort": "high"},
        "num_threads": 1024
    }' \
    --script-config '{"root_dir": "output/hotpotqa/train"}'
```

### Phase 3 — Generate MBT-S synthesized traces

```bash
bash scripts/tasks/local/mbt_s.sh
```

Same as Phase 2 but with `mbt_s=true` (uses `MBT_S_TEMPLATE`). Output:
`output/{dataset}/train/mbt-s/gpt-oss-120b-high/results/`
with `synthesized_trace` column. This is the SFT data for `--mode mbt-s`.

```bash
uv run mbt \
    --task-name hotpotqa \
    --task-config '{"dataset_split": "train", "mbt_s": true}' \
    --api-name vllm.chat \
    --api-config '{
        "model_name": "gpt-oss-120b-high",
        "model_kwargs": {"config": "configs/vllm/gpt-oss-120b.yaml"},
        "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "reasoning_effort": "high"},
        "num_threads": 1024
    }' \
    --script-config '{"root_dir": "output/hotpotqa/train"}'
```

### Phase 4 — MBT-R refinement (rewriting student traces)

```bash
bash scripts/tasks/local/mbt_r.sh
```

The `qa.mbt_r` task consumes the **previous student rollouts** (Phase 1 train split) and rewrites each `reasoning_trace` against the gold `solution` using `gpt-oss-120b-high`. Output:
`output/{dataset}/train/{student-model}/mbt-r/gpt-oss-120b-high/results/`
with `refined_trace`, `trace_id` columns. This is the SFT data for `--mode mbt-r`.

```bash
uv run mbt \
    --task-name qa.mbt_r \
    --task-config '{"solution_config": "hotpotqa", "solution_split": "train"}' \
    --api-name vllm.chat \
    --api-config '{
        "model_name": "gpt-oss-120b-high",
        "model_kwargs": {"config": "configs/vllm/gpt-oss-120b.yaml"},
        "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 4, "reasoning_effort": "high"},
        "num_threads": 1024
    }' \
    --script-config '{"root_dir": "output/hotpotqa/train/Qwen3-4B"}'
```

`request_kwargs.n=4` produces 4 refined traces per input; `expand_traces` emits one row per trace.

### Phase 5 — SFT training (18 cells: 6 modes × 3 model sizes)

```bash
bash scripts/tasks/local/sft.sh
```

6 modes per Qwen3-{0.6B, 1.7B, 4B}:
- `self-distill,sft` — RS on student-generated correct traces.
- `gpt-oss-distill,sft` — RS on gpt-oss-120b teacher rollouts.
- `mbt-s,sft` — full MBT-S synthesis.
- `mbt-r,sft` — MBT-R refinement.
- `distill-r,sft` — distill-R baseline.
- `direct-r,sft` — direct-R baseline.

Effective batch size: `per_device=2 × grad_accum=16 × num_gpus=4 = 128`. Learning rate `1e-4`. 1 epoch. Cosine schedule with `warmup_ratio=0.1`. Output dir: `output/train/{model}/{mode}/sft/1e-4/128/`.

Single-cell example:

```bash
uv run accelerate launch \
    --config_file configs/accelerate/multi_gpu.yaml \
    --main_process_port $(shuf -i 49152-65535 -n 1) \
    src/mbt/train/sft.py \
    --config configs/sft.yaml \
    --model_name_or_path Qwen/Qwen3-4B \
    --dataset_name metacognitive-behavioral-tuning/mbt-r-hotpotqa \
    --dataset_config Qwen3-4B \
    --mode mbt-r \
    --wandb_tags Qwen3-4B,mbt-r,sft,1e-4,128 \
    --output_dir output/train/Qwen3-4B/mbt-r/sft/1e-4/128 \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16
```

### Phase 6 — Trained-model rollouts on validation

After Phase 5 finishes, rerun rollout.sh (already includes "Section 2" cells that point `model_kwargs.model` at the local SFT checkpoint dirs):

```bash
bash scripts/tasks/local/rollout.sh   # Section 2 cells reuse the trained ckpts
```

### Phase 7 — Evaluation (EM / substring / F1 / LLM-as-judge)

```bash
bash scripts/tasks/local/evaluation.sh   # 156 cells: 52 variants × 3 datasets
```

Judge model: `gemma-4-31b-it`. Single-cell:

```bash
uv run mbt \
    --task-name qa.evaluation \
    --api-name vllm.chat \
    --api-config '{
        "model_name": "gemma-4-31b-it",
        "model_kwargs": {"config": "configs/vllm/gemma-4-31b-it.yaml"},
        "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1}
    }' \
    --script-config '{"root_dir": "output/musique/validation/Qwen3-4B/mbt-r/sft/1e-4/128"}'
```

### Phase 8 — Auxiliary scoring (paper's Section 4 metrics)

```bash
bash scripts/tasks/local/answer_hit.sh        # answer_hit
bash scripts/tasks/local/build_difficulty.sh  # pre-computer for mqi
bash scripts/tasks/local/rrp_score.sh         # RRP (paper Section 4.2)
bash scripts/tasks/local/mqi_score.sh         # MQI (paper Section 4.3)
```

> `mqi_score.sh` **requires `build_difficulty.sh` to have run first** — it writes `data/sample_difficulty.csv`.

### Phase 9 — Aggregate results

Each task writes deterministic per-cell artifacts under `output/<dataset>/<split>/<model>/<task>/<judge>/results/` (HF Dataset) plus `results.json` (judge-aggregate metrics) for evaluation tasks. Pull the columns you need into a DataFrame with `datasets.load_from_disk`:

```python
from datasets import load_from_disk
ds = load_from_disk("output/musique/validation/Qwen3-4B/rrp-score/gemma-4-31b-it/results")
print(ds.column_names)
# → ['sample_id', 'reasoning_trace', 'predicted_answer',
#    'first_correct', 'redundant_fraction', 'confidence',
#    'answer_paragraph', 'redundant_paragraphs', ...]
```

The paper's main numbers (AES, RRP, MQI) are simple aggregations over these columns — see §11 for which column maps to which metric.

---

## 7. Task reference

Every task accepts a JSON config via `--task-config '{...}'`. Defaults are module constants (see `src/mbt/tasks/<task>.py`). Common fields below; per-task extras follow.

### 7.1. Common task config fields

| Field | Type | Default | Description |
|---|---|---|---|
| `dataset_name` | str | task-specific | HF Hub dataset path. |
| `dataset_config` | str \| null | task-specific | HF dataset subconfig. |
| `dataset_split` | str | `"validation"` | Split to load. |
| `num_proc` | int | `$OMP_NUM_THREADS` or 8 | `dataset.map` parallelism. |
| `num_samples` | int \| null | null | If set, slices to the first N samples (for debugging). |
| `skip_format_columns` | bool | false | Skip the standardization step (only QA tasks). |

### 7.2. Top-level QA tasks — `hotpotqa`, `musique`, `2wikimultihopqa`

Extra mode flags (mutually exclusive in practice):

| Flag | Output mode | Prompt template |
|---|---|---|
| (none) | rollouts (`reasoning_trace`, `predicted_answer`) | `PROMPT_TEMPLATE` |
| `metacognitive_prompt: true` | rollouts with metacognition system prompt | `METACOGNITIVE_PROMPT_TEMPLATE` |
| `solution: true` | gold solutions (`solution`, `solution_prompt`) | `SOLUTION_TEMPLATE` |
| `mbt_s: true` | synthesized MBT-S trace (`synthesized_trace`) | `MBT_S_TEMPLATE` |

Output directory layout:
- default mode → `{root_dir}/`
- `solution` → `{root_dir}/solution/`
- `mbt_s` → `{root_dir}/mbt-s/`

### 7.3. `qa.mbt_r` — MBT-R refinement

Reads `{root_dir}/results/` (a prior rollout's results) and rewrites each `reasoning_trace` against gold `solution`. Extra config:

| Field | Default |
|---|---|
| `solution_name` | `"metacognitive-behavioral-tuning/solutions-gpt-oss-120b"` |
| `solution_config` | `"hotpotqa"` |
| `solution_split` | `"train"` |

Outputs N copies per request (controlled by `api_config.request_kwargs.n`) with `refined_trace`, `trace_id`.

### 7.4. `qa.evaluation` — deterministic + judge metrics

| Field | Default |
|---|---|
| `metrics` | `["exact_match", "substring_match", "f1_score", "llm_as_judge"]` |

If `metrics` excludes `"llm_as_judge"`, preprocess short-circuits — no API needed, no `--api-name` required.

### 7.5. `qa.answer_hit` — answer-derivation judgment

No extra config beyond common fields. Postprocess parses judge response `== "YES"` → `answer_hit=1.0`.

### 7.6. `qa.rrp_score` — RRP (Reach-Redundancy Profile)

Marker-based regulation judge. No extra config. Output columns: `first_correct`, `redundant_fraction`, `confidence`, `redundant_paragraphs`, `answer_paragraph`.

### 7.7. `qa.mqi_score` — MQI (length-aware Metacognitive Quality Index)

| Field | Default |
|---|---|
| `difficulty_csv` | `"data/sample_difficulty.csv"` |
| `default_tier` | `"medium"` |

Requires `scripts/build_difficulty.py` to have been run first (it writes the difficulty CSV). Output columns: `l_obs`, `phases`, `confidence`.

---

## 8. API backend reference

Pass via `--api-name "<name>"` and `--api-config '{...}'`.

| Registered name | Transport | Best for |
|---|---|---|
| `vllm.chat` | local `vllm serve` + OpenAI-compat HTTP | **default for paper-table runs** |
| `openai.chat` | hosted SDK `chat.completions.create` | hosted closed models (GPT-4o, gemini, etc.) |

### 8.1. `vllm.chat` — full api_config reference

| Key | Type | Default | Description |
|---|---|---|---|
| `model_name` (**required**) | str | — | Output subdir under `{task_dir}`. |
| `model_kwargs` (**required**) | dict | — | Forwarded to `vllm serve`. Must include `config: <path-to-yaml>`; yaml seeds and explicit keys override. |
| `request_kwargs` | dict | `{}` | Forwarded to `client.chat.completions.create`. Common: `temperature`, `top_p`, `n`, `max_completion_tokens`, `extra_body`. |
| `num_threads` | int | 1 | Worker pool size for concurrent client requests. |
| `num_proc` | int | `$OMP_NUM_THREADS` or 8 | Dataset map parallelism. |
| `max_retries` | int | 0 | Retries when `finish_reason ∈ retry_on`. |
| `retry_on` | list[str] | `["length", "content_filter"]` | Trigger values. |
| `log_ratio` | float | 0.01 | Progress-log frequency as ratio of total. |
| `cache_ratio` | float | 0.1 | Cache-flush frequency. |
| `sample_ratio` | float | 0.1 | Per-shard sample-log frequency. |
| `client_timeout` | int (sec) | 300 | OpenAI HTTPX request timeout. |
| `client_max_retries` | int | 20 | OpenAI client network retries. |
| `health_check_timeout` | int (sec) | 3600 | `vllm serve /health` polling budget. |
| `dry_run` | bool | false | Skip server boot + worker loop. |
| `seed` | int | 42 | Forwarded into request_kwargs if not set. |

The `vllm serve` command is built from `model_kwargs` by mapping each kv → CLI flag (e.g. `tensor_parallel_size: 4` → `--tensor-parallel-size 4`). Yaml files under `configs/vllm/<model>.yaml` provide pre-tuned presets.

### 8.2. Example: hosted OpenAI

```bash
uv run mbt \
    --task-name qa.evaluation \
    --api-name openai.chat \
    --api-config '{
        "model_name": "gpt-4o-judge",
        "request_kwargs": {"model": "gpt-4o-mini", "temperature": 0.0, "max_completion_tokens": 1024}
    }' \
    --script-config '{"root_dir": "output/musique/validation/Qwen3-4B"}'
```


---

## 9. SFT training reference

Driver: `src/mbt/train/sft.py`. Launched via `accelerate launch`, **not** through the `mbt` CLI. Reads `configs/sft.yaml` for defaults, accepts CLI overrides.

### 9.1. Critical CLI flags

| Flag | Default | Description |
|---|---|---|
| `--config` | — | Path to base SFT yaml (use `configs/sft.yaml`). |
| `--mode` | `mbt-r` | One of `distill` \| `mbt-s` \| `mbt-r`. Selects which column becomes the completion target. |
| `--model_name_or_path` | `Qwen/Qwen3-4B` | Base model. |
| `--dataset_name` | `metacognitive-behavioral-tuning/mbt-r-hotpotqa` | HF Hub training dataset. |
| `--dataset_config` | `Qwen3-4B` | Sub-config (per-model dataset slice). |
| `--output_dir` | yaml default | Where checkpoints go. |
| `--learning_rate` | `1e-4` | Peak LR (cosine schedule). |
| `--per_device_train_batch_size` | 2 | Per-GPU batch. |
| `--gradient_accumulation_steps` | 16 | Effective batch = pdb × gas × n_gpus. |
| `--num_train_epochs` | 1 | |
| `--max_length` | 32768 | Tokenized sequence length cap. |
| `--use_peft` | false | Set true for LoRA. `--lora_r`, `--lora_alpha`, `--lora_target_modules` etc. follow. |
| `--num_rollouts` | null | Filter dataset to `rollout_id <= N`. |
| `--num_traces` | null | Filter dataset to `trace_id <= N`. |
| `--wandb_project` | `mbt` | W&B project name. |
| `--wandb_run_group` | null | W&B run group. |
| `--wandb_tags` | null | Comma-separated W&B tags. |
| `--train_seed` | 42 | RNG seed for training. |

### 9.2. Mode → completion target

| Mode | Completion built from |
|---|---|
| `distill` | `example["response"]` (raw rollout). |
| `mbt-s` | `<think>\n{synthesized_trace}\n</think>\n\n<answer>{answer}</answer>` |
| `mbt-r` | `<think>\n{refined_trace}\n</think>\n\n<answer>{answer}</answer>` |

`tokenize` produces a `completion_mask` so `completion_only_loss: true` in `configs/sft.yaml` applies loss only to assistant tokens.

### 9.3. Example: full fine-tune Qwen3-4B on MBT-R

```bash
uv run accelerate launch \
    --config_file configs/accelerate/multi_gpu.yaml \
    --main_process_port $(shuf -i 49152-65535 -n 1) \
    src/mbt/train/sft.py \
    --config configs/sft.yaml \
    --model_name_or_path Qwen/Qwen3-4B \
    --dataset_name metacognitive-behavioral-tuning/mbt-r-hotpotqa \
    --dataset_config Qwen3-4B \
    --mode mbt-r \
    --output_dir output/train/Qwen3-4B/mbt-r/sft/1e-4/128 \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --wandb_tags Qwen3-4B,mbt-r,sft,1e-4,128
```

### 9.4. Example: LoRA fine-tune (memory-constrained)

```bash
uv run accelerate launch \
    --config_file configs/accelerate/fsdp_qlora.yaml \
    src/mbt/train/sft.py \
    --config configs/sft.yaml \
    --model_name_or_path Qwen/Qwen3-4B \
    --dataset_name metacognitive-behavioral-tuning/mbt-r-hotpotqa \
    --dataset_config Qwen3-4B \
    --mode mbt-r \
    --use_peft true \
    --lora_r 64 \
    --lora_alpha 32 \
    --lora_target_modules all-linear \
    --output_dir output/train/Qwen3-4B/mbt-r/lora/1e-4
```

---

## 10. Configs

### 10.1. `configs/sft.yaml`

Three TRL dataclasses flattened into one YAML namespace. Key defaults:

- `completion_only_loss: true` (loss only on assistant tokens via `completion_mask`).
- `dataset_kwargs.skip_prepare_dataset: true` (tokenize step pre-builds input_ids).
- `attn_implementation: flash_attention_2`.
- `use_liger_kernel: true`.
- `gradient_checkpointing: true` with `use_reentrant: false`.
- `bf16: true`, `tf32: true`, `optim: adamw_torch_fused`.

### 10.2. `configs/accelerate/*.yaml`

Choose one with `--config_file`:

| File | Distribution | Use for |
|---|---|---|
| `single_gpu.yaml` | `NO` (1 GPU) | debugging |
| `multi_gpu.yaml` | DDP, 4 GPUs | **default** for SFT |
| `fsdp.yaml` | FSDP `FULL_SHARD` + `TRANSFORMER_BASED_WRAP` | very large models |
| `fsdp_qlora.yaml` | FSDP + 4-bit QLoRA | big model on small VRAM |

### 10.3. `configs/vllm/*.yaml`

Per-model `vllm serve` flag presets. Reference via `model_kwargs.config`. Includes presets for `qwen3-{0.6b,1.7b,4b,8b,14b,32b}.yaml`, `gpt-oss-{20b,120b}.yaml`, `gemma-4-31b-it.yaml`, `llama-4-scout-fp8.yaml`, plus several Qwen3.5 / Qwen3.6 / Nemotron / Mistral / DeepSeek variants.

Example (`configs/vllm/qwen3-4b.yaml`):

```yaml
model: Qwen/Qwen3-4B
max_model_len: 32768
gpu-memory-utilization: 0.9
tensor-parallel-size: 4
trust-remote-code: true
reasoning-parser: qwen3
```

Add per-model overrides via `model_kwargs` at call site:

```json
{"model_kwargs": {"config": "configs/vllm/qwen3-4b.yaml", "max_model_len": 40960, "tensor_parallel_size": 2}}
```

---

## 11. Scoring metrics (AES / RRP / MQI)

The paper introduces three trace-quality metrics. Each maps to one or more tasks:

| Metric | Paper | Task(s) | Output column(s) |
|---|---|---|---|
| **EM / Substring / F1** | §3 | `qa.evaluation` (deterministic short-circuit) | `exact_match`, `substring_match`, `f1_score` |
| **LLM-as-judge accuracy** | §3 | `qa.evaluation` (with `llm_as_judge` in metrics) | `llm_as_judge` |
| **Answer-hit rate** | §3 | `qa.answer_hit` | `answer_hit`, `substring_match` |
| **Accuracy-Efficiency Score (AES)** | §4.1 | computed downstream from the EM + token-count columns | derived from token count + EM |
| **Reach-Redundancy Profile (RRP)** | §4.2 | `qa.rrp_score` | `first_correct`, `redundant_fraction`, `confidence`, `answer_paragraph`, `redundant_paragraphs` |
| **Metacognitive Quality Index (MQI)** | §4.3 | `qa.mqi_score` | `l_obs`, `phases`, `confidence` |

For RRP and MQI implementation details, see `docs/scoring_redesign_marker_variant.md`.

---

## 12. Project layout

```
MBT/
├── src/mbt/                # Core package (Python 3.12, console-script: `mbt`)
│   ├── main.py              # Pipeline orchestrator (preprocess → API → postprocess)
│   ├── registry.py          # @register_task / @register_api decorators
│   ├── apis/                # Inference backends
│   │   ├── vllm/chat.py     # local `vllm serve` + OpenAI-compat HTTP
│   │   └── openai/chat.py   # hosted OpenAI SDK
│   ├── tasks/               # Task definitions
│   │   ├── hotpotqa.py      # rollout / solution / MBT-S modes
│   │   ├── musique.py
│   │   ├── 2wikimultihopqa.py
│   │   └── qa/              # 5 analysis tasks (mbt_r + evaluation + answer_hit + rrp_score + mqi_score)
│   └── train/               # SFT trainer (TRL)
├── configs/
│   ├── sft.yaml             # TRL SFT config (3 dataclasses flat)
│   ├── accelerate/          # Distributed launchers
│   └── vllm/                # Per-model `vllm serve` presets
├── scripts/
│   ├── setup_auth.sh        # hf + wandb login
│   ├── download.py          # Pre-fetch HF models & datasets
│   ├── build_difficulty.py  # Pre-compute per-sample difficulty (MQI input)
│   ├── slurm/               # *.slurm SBATCH entry points
│   └── tasks/
│       ├── local/           # Single-host catalog (paper-table replication)
│       └── slurm/           # Same catalog, sbatch-driven
├── pyproject.toml           # Project metadata + dependencies (uv-managed)
├── uv.lock                  # Pinned dependency lock
└── README.md                # ← you are here
```

---

## 13. SLURM submission

Every `*.sh` in `scripts/tasks/local/` has a `slurm/` twin that wraps each cell as:

```
sbatch --cpus-per-task=32 --gres=gpu:4 scripts/slurm/<entry>.slurm <args>
```

`scripts/slurm/*.slurm` headers leave `--partition`, `--qos`, and `--time` blank — edit those for your cluster before submitting. `OMP_NUM_THREADS` is computed automatically:

```bash
((SLURM_GPUS_ON_NODE > 0)) && export OMP_NUM_THREADS=$((SLURM_CPUS_ON_NODE / SLURM_GPUS_ON_NODE))
```

Submit the full reproduction matrix:

```bash
bash scripts/tasks/slurm/rollout.sh
bash scripts/tasks/slurm/solution.sh
bash scripts/tasks/slurm/mbt_s.sh
bash scripts/tasks/slurm/mbt_r.sh
bash scripts/tasks/slurm/sft.sh
bash scripts/tasks/slurm/rollout.sh    # Section 2 — trained variants
bash scripts/tasks/slurm/evaluation.sh
bash scripts/tasks/slurm/answer_hit.sh
bash scripts/tasks/slurm/build_difficulty.sh
bash scripts/tasks/slurm/rrp_score.sh
bash scripts/tasks/slurm/mqi_score.sh
```

---

## 14. Troubleshooting

### 14.1. vLLM hangs at startup / OOMs

Check `<api_dir>/logs/<timestamp>_server.log`. Common fixes:
- Reduce `gpu-memory-utilization` in the model's `configs/vllm/<model>.yaml` (e.g. 0.9 → 0.85).
- Reduce `max_model_len` (e.g. 32768 → 16384) to free KV-cache.
- If an orphaned vLLM worker is holding the port from a previous crashed run, locate and kill it manually: `pgrep -fu "$USER" 'vllm serve' | xargs -r kill -TERM`.

### 14.2. `ModuleNotFoundError: No module named 'mbt'`

You're not in the `uv` env. Either use `uv run` for all commands, or `source .venv/bin/activate`.

### 14.3. Task is registered but never runs

Inspect logs — the most common cause is a syntax error in any other module under `mbt.apis.*` or `mbt.tasks.*`. `recursive_import` fails silently for siblings of a broken module. Run:

```bash
uv run python -c "
from mbt.main import recursive_import
from mbt.registry import TASK_REGISTRY, API_REGISTRY
recursive_import('mbt.apis')
recursive_import('mbt.tasks')
print(len(TASK_REGISTRY), 'tasks;', len(API_REGISTRY), 'apis')
"
# Should print: 8 tasks; 2 apis
```

### 14.4. `/tmp` is `noexec` (some shared clusters)

vLLM uses Triton + torch.inductor caches that default to `/tmp`. Redirect them to a writable+exec mount before launching:

```bash
export TRITON_CACHE_DIR="$HOME/.cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$HOME/.cache/torchinductor"
export TMPDIR="$HOME/.cache/tmp"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TMPDIR"
```

### 14.5. Pre-fetched models still get re-downloaded

Make sure `$HF_HOME` is exported in the shell that launches `mbt` (some clusters reset env on `srun`). `download.py` respects `HF_HOME`.

---

## 15. License & citation

This project is licensed under **Apache-2.0**. See [LICENSE](LICENSE).

If you use this code or the released checkpoints / datasets, please cite:

```bibtex
@misc{kim2026metacognitivebehavioraltuninglarge,
      title={Metacognitive Behavioral Tuning of Large Language Models for Multi-Hop Question Answering},
      author={Ik-hwan Kim and Hyeongrok Han and Mingi Jung and Sangwon Yu and Jinseok Hong and Sang Hun Kim and Yoonyoung Choi and Sungroh Yoon},
      year={2026},
      eprint={2602.22508},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2602.22508},
}
```
