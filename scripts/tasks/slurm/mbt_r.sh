#!/usr/bin/env bash
set -uo pipefail
FAILED=0
trap 'FAILED=$((FAILED+1))' ERR

# qa.mbt_r — refine each rollout's reasoning_trace against the gold solution.
# Judge: gpt-oss-120b-high (kept reasoning-rich for refinement; not gemma).
# Output: output/<dataset>/train/<rollout-model>/mbt-r/gpt-oss-120b-high/results/
# (refined_trace column). Used as SFT data for the mbt-r training mode.

# >>> musique train mbt-r refinement >>>

# Qwen3-0.6B

sbatch scripts/slurm/main.slurm \
    --task-name "qa.mbt_r" \
    --task-config '{"solution_config": "musique", "solution_split": "train"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "gpt-oss-120b-high", "model_kwargs": {"config": "configs/vllm/gpt-oss-120b.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}, "reasoning_effort": "high"}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/musique/train/Qwen3-0.6B"}'

# Qwen3-1.7B

sbatch scripts/slurm/main.slurm \
    --task-name "qa.mbt_r" \
    --task-config '{"solution_config": "musique", "solution_split": "train"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "gpt-oss-120b-high", "model_kwargs": {"config": "configs/vllm/gpt-oss-120b.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}, "reasoning_effort": "high"}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/musique/train/Qwen3-1.7B"}'

# Qwen3-4B

sbatch scripts/slurm/main.slurm \
    --task-name "qa.mbt_r" \
    --task-config '{"solution_config": "musique", "solution_split": "train"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "gpt-oss-120b-high", "model_kwargs": {"config": "configs/vllm/gpt-oss-120b.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}, "reasoning_effort": "high"}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/musique/train/Qwen3-4B"}'

# gpt-oss-120b-high (self-refine)

sbatch scripts/slurm/main.slurm \
    --task-name "qa.mbt_r" \
    --task-config '{"solution_config": "musique", "solution_split": "train"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "gpt-oss-120b-high", "model_kwargs": {"config": "configs/vllm/gpt-oss-120b.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}, "reasoning_effort": "high"}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/musique/train/gpt-oss-120b-high"}'

# <<< musique <<<

# >>> 2wikimultihopqa train mbt-r refinement >>>

# Qwen3-0.6B

sbatch scripts/slurm/main.slurm \
    --task-name "qa.mbt_r" \
    --task-config '{"solution_config": "2wikimultihopqa", "solution_split": "train"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "gpt-oss-120b-high", "model_kwargs": {"config": "configs/vllm/gpt-oss-120b.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}, "reasoning_effort": "high"}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/2wikimultihopqa/train/Qwen3-0.6B"}'

# Qwen3-1.7B

sbatch scripts/slurm/main.slurm \
    --task-name "qa.mbt_r" \
    --task-config '{"solution_config": "2wikimultihopqa", "solution_split": "train"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "gpt-oss-120b-high", "model_kwargs": {"config": "configs/vllm/gpt-oss-120b.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}, "reasoning_effort": "high"}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/2wikimultihopqa/train/Qwen3-1.7B"}'

# Qwen3-4B

sbatch scripts/slurm/main.slurm \
    --task-name "qa.mbt_r" \
    --task-config '{"solution_config": "2wikimultihopqa", "solution_split": "train"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "gpt-oss-120b-high", "model_kwargs": {"config": "configs/vllm/gpt-oss-120b.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}, "reasoning_effort": "high"}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/2wikimultihopqa/train/Qwen3-4B"}'

# gpt-oss-120b-high (self-refine)

sbatch scripts/slurm/main.slurm \
    --task-name "qa.mbt_r" \
    --task-config '{"solution_config": "2wikimultihopqa", "solution_split": "train"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "gpt-oss-120b-high", "model_kwargs": {"config": "configs/vllm/gpt-oss-120b.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}, "reasoning_effort": "high"}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/2wikimultihopqa/train/gpt-oss-120b-high"}'

# <<< 2wikimultihopqa <<<

# >>> hotpotqa train mbt-r refinement >>>

# Qwen3-0.6B

sbatch scripts/slurm/main.slurm \
    --task-name "qa.mbt_r" \
    --task-config '{"solution_config": "hotpotqa", "solution_split": "train"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "gpt-oss-120b-high", "model_kwargs": {"config": "configs/vllm/gpt-oss-120b.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}, "reasoning_effort": "high"}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/hotpotqa/train/Qwen3-0.6B"}'

# Qwen3-1.7B

sbatch scripts/slurm/main.slurm \
    --task-name "qa.mbt_r" \
    --task-config '{"solution_config": "hotpotqa", "solution_split": "train"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "gpt-oss-120b-high", "model_kwargs": {"config": "configs/vllm/gpt-oss-120b.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}, "reasoning_effort": "high"}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/hotpotqa/train/Qwen3-1.7B"}'

# Qwen3-4B

sbatch scripts/slurm/main.slurm \
    --task-name "qa.mbt_r" \
    --task-config '{"solution_config": "hotpotqa", "solution_split": "train"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "gpt-oss-120b-high", "model_kwargs": {"config": "configs/vllm/gpt-oss-120b.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}, "reasoning_effort": "high"}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/hotpotqa/train/Qwen3-4B"}'

# gpt-oss-120b-high (self-refine)

sbatch scripts/slurm/main.slurm \
    --task-name "qa.mbt_r" \
    --task-config '{"solution_config": "hotpotqa", "solution_split": "train"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "gpt-oss-120b-high", "model_kwargs": {"config": "configs/vllm/gpt-oss-120b.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}, "reasoning_effort": "high"}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/hotpotqa/train/gpt-oss-120b-high"}'

# <<< hotpotqa <<<

echo "[summary] failed=$FAILED"
exit $FAILED
