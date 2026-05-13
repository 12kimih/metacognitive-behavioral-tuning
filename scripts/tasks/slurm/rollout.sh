#!/usr/bin/env bash
set -uo pipefail
FAILED=0
trap 'FAILED=$((FAILED+1))' ERR

# Generate rollouts for every (dataset, variant, split) in the paper matrix.
#
# Section 1: base Qwen3-{0.6B,1.7B,4B} + gpt-oss-120b-high on QA validation + train.
#   Train-set rollouts feed the SFT data pipeline; trained-variant rollouts are
#   validation-only.
# Section 2: prompt / SFT / GRPO / external variants on validation.
#   - prompt: served from base HF, task-config metacognitive_prompt=true.
#   - sft: model_kwargs.model overrides yaml's model with the local checkpoint.
#   - grpo_only / sft_grpo: same shape, requires the GRPO checkpoint to exist.
#   - external (shorter-better/token-skip/limopro): not auto-emitted; use the
#     baseline method's own pipeline.

# >>> musique/validation — base rollouts >>>

# Qwen3-0.6B

sbatch scripts/slurm/main.slurm \
    --task-name musique \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-0.6B", "model_kwargs": {"config": "configs/vllm/qwen3-0.6b.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/musique/validation"}'

# Qwen3-1.7B

sbatch scripts/slurm/main.slurm \
    --task-name musique \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-1.7B", "model_kwargs": {"config": "configs/vllm/qwen3-1.7b.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/musique/validation"}'

# Qwen3-4B

sbatch scripts/slurm/main.slurm \
    --task-name musique \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-4B", "model_kwargs": {"config": "configs/vllm/qwen3-4b.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/musique/validation"}'

# gpt-oss-120b-high

sbatch scripts/slurm/main.slurm \
    --task-name musique \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "gpt-oss-120b-high", "model_kwargs": {"config": "configs/vllm/gpt-oss-120b.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}, "reasoning_effort": "high"}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/musique/validation"}'

# <<< musique/validation <<<

# >>> musique/train — base rollouts >>>

# Qwen3-0.6B

sbatch scripts/slurm/main.slurm \
    --task-name musique \
    --task-config '{"dataset_split": "train"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-0.6B", "model_kwargs": {"config": "configs/vllm/qwen3-0.6b.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/musique/train"}'

# Qwen3-1.7B

sbatch scripts/slurm/main.slurm \
    --task-name musique \
    --task-config '{"dataset_split": "train"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-1.7B", "model_kwargs": {"config": "configs/vllm/qwen3-1.7b.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/musique/train"}'

# Qwen3-4B

sbatch scripts/slurm/main.slurm \
    --task-name musique \
    --task-config '{"dataset_split": "train"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-4B", "model_kwargs": {"config": "configs/vllm/qwen3-4b.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/musique/train"}'

# gpt-oss-120b-high

sbatch scripts/slurm/main.slurm \
    --task-name musique \
    --task-config '{"dataset_split": "train"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "gpt-oss-120b-high", "model_kwargs": {"config": "configs/vllm/gpt-oss-120b.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}, "reasoning_effort": "high"}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/musique/train"}'

# <<< musique/train <<<

# >>> 2wikimultihopqa/validation — base rollouts >>>

# Qwen3-0.6B

sbatch scripts/slurm/main.slurm \
    --task-name 2wikimultihopqa \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-0.6B", "model_kwargs": {"config": "configs/vllm/qwen3-0.6b.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/2wikimultihopqa/validation"}'

# Qwen3-1.7B

sbatch scripts/slurm/main.slurm \
    --task-name 2wikimultihopqa \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-1.7B", "model_kwargs": {"config": "configs/vllm/qwen3-1.7b.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/2wikimultihopqa/validation"}'

# Qwen3-4B

sbatch scripts/slurm/main.slurm \
    --task-name 2wikimultihopqa \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-4B", "model_kwargs": {"config": "configs/vllm/qwen3-4b.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/2wikimultihopqa/validation"}'

# gpt-oss-120b-high

sbatch scripts/slurm/main.slurm \
    --task-name 2wikimultihopqa \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "gpt-oss-120b-high", "model_kwargs": {"config": "configs/vllm/gpt-oss-120b.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}, "reasoning_effort": "high"}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/2wikimultihopqa/validation"}'

# <<< 2wikimultihopqa/validation <<<

# >>> 2wikimultihopqa/train — base rollouts >>>

# Qwen3-0.6B

sbatch scripts/slurm/main.slurm \
    --task-name 2wikimultihopqa \
    --task-config '{"dataset_split": "train"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-0.6B", "model_kwargs": {"config": "configs/vllm/qwen3-0.6b.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/2wikimultihopqa/train"}'

# Qwen3-1.7B

sbatch scripts/slurm/main.slurm \
    --task-name 2wikimultihopqa \
    --task-config '{"dataset_split": "train"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-1.7B", "model_kwargs": {"config": "configs/vllm/qwen3-1.7b.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/2wikimultihopqa/train"}'

# Qwen3-4B

sbatch scripts/slurm/main.slurm \
    --task-name 2wikimultihopqa \
    --task-config '{"dataset_split": "train"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-4B", "model_kwargs": {"config": "configs/vllm/qwen3-4b.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/2wikimultihopqa/train"}'

# gpt-oss-120b-high

sbatch scripts/slurm/main.slurm \
    --task-name 2wikimultihopqa \
    --task-config '{"dataset_split": "train"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "gpt-oss-120b-high", "model_kwargs": {"config": "configs/vllm/gpt-oss-120b.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}, "reasoning_effort": "high"}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/2wikimultihopqa/train"}'

# <<< 2wikimultihopqa/train <<<

# >>> hotpotqa/validation — base rollouts >>>

# Qwen3-0.6B

sbatch scripts/slurm/main.slurm \
    --task-name hotpotqa \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-0.6B", "model_kwargs": {"config": "configs/vllm/qwen3-0.6b.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/hotpotqa/validation"}'

# Qwen3-1.7B

sbatch scripts/slurm/main.slurm \
    --task-name hotpotqa \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-1.7B", "model_kwargs": {"config": "configs/vllm/qwen3-1.7b.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/hotpotqa/validation"}'

# Qwen3-4B

sbatch scripts/slurm/main.slurm \
    --task-name hotpotqa \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-4B", "model_kwargs": {"config": "configs/vllm/qwen3-4b.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/hotpotqa/validation"}'

# gpt-oss-120b-high

sbatch scripts/slurm/main.slurm \
    --task-name hotpotqa \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "gpt-oss-120b-high", "model_kwargs": {"config": "configs/vllm/gpt-oss-120b.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}, "reasoning_effort": "high"}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/hotpotqa/validation"}'

# <<< hotpotqa/validation <<<

# >>> hotpotqa/train — base rollouts >>>

# Qwen3-0.6B

sbatch scripts/slurm/main.slurm \
    --task-name hotpotqa \
    --task-config '{"dataset_split": "train"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-0.6B", "model_kwargs": {"config": "configs/vllm/qwen3-0.6b.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/hotpotqa/train"}'

# Qwen3-1.7B

sbatch scripts/slurm/main.slurm \
    --task-name hotpotqa \
    --task-config '{"dataset_split": "train"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-1.7B", "model_kwargs": {"config": "configs/vllm/qwen3-1.7b.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/hotpotqa/train"}'

# Qwen3-4B

sbatch scripts/slurm/main.slurm \
    --task-name hotpotqa \
    --task-config '{"dataset_split": "train"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-4B", "model_kwargs": {"config": "configs/vllm/qwen3-4b.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/hotpotqa/train"}'

# gpt-oss-120b-high

sbatch scripts/slurm/main.slurm \
    --task-name hotpotqa \
    --task-config '{"dataset_split": "train"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "gpt-oss-120b-high", "model_kwargs": {"config": "configs/vllm/gpt-oss-120b.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}, "reasoning_effort": "high"}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/hotpotqa/train"}'

# <<< hotpotqa/train <<<

# >>> musique/validation — trained / prompt / external variants >>>

# Qwen3-0.6B prompt

sbatch scripts/slurm/main.slurm \
    --task-name musique \
    --task-config '{"dataset_split": "validation", "metacognitive_prompt": true}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-0.6B/metacognitive-prompt", "model_kwargs": {"config": "configs/vllm/qwen3-0.6b.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/musique/validation"}'

# Qwen3-0.6B grpo

# (requires GRPO checkpoint at output/train/Qwen3-0.6B/grpo)

sbatch scripts/slurm/main.slurm \
    --task-name musique \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-0.6B/grpo", "model_kwargs": {"config": "configs/vllm/qwen3-0.6b.yaml", "model": "output/train/Qwen3-0.6B/grpo"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/musique/validation"}'

# Qwen3-0.6B self-distill,sft

sbatch scripts/slurm/main.slurm \
    --task-name musique \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-0.6B/self-distill/sft/1e-4/128", "model_kwargs": {"config": "configs/vllm/qwen3-0.6b.yaml", "model": "output/train/Qwen3-0.6B/self-distill/sft/1e-4/128"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/musique/validation"}'

# Qwen3-0.6B gpt-oss-distill,sft

sbatch scripts/slurm/main.slurm \
    --task-name musique \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-0.6B/gpt-oss-distill/sft/1e-4/128", "model_kwargs": {"config": "configs/vllm/qwen3-0.6b.yaml", "model": "output/train/Qwen3-0.6B/gpt-oss-distill/sft/1e-4/128"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/musique/validation"}'

# Qwen3-0.6B gpt-oss-distill,grpo

# (requires GRPO checkpoint at output/train/Qwen3-0.6B/gpt-oss-distill/grpo)

sbatch scripts/slurm/main.slurm \
    --task-name musique \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-0.6B/gpt-oss-distill/grpo", "model_kwargs": {"config": "configs/vllm/qwen3-0.6b.yaml", "model": "output/train/Qwen3-0.6B/gpt-oss-distill/grpo"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/musique/validation"}'

# Qwen3-0.6B mbt-s,sft

sbatch scripts/slurm/main.slurm \
    --task-name musique \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-0.6B/mbt-s/sft/1e-4/128", "model_kwargs": {"config": "configs/vllm/qwen3-0.6b.yaml", "model": "output/train/Qwen3-0.6B/mbt-s/sft/1e-4/128"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/musique/validation"}'

# Qwen3-0.6B mbt-s,grpo

# (requires GRPO checkpoint at output/train/Qwen3-0.6B/mbt-s/grpo)

sbatch scripts/slurm/main.slurm \
    --task-name musique \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-0.6B/mbt-s/grpo", "model_kwargs": {"config": "configs/vllm/qwen3-0.6b.yaml", "model": "output/train/Qwen3-0.6B/mbt-s/grpo"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/musique/validation"}'

# Qwen3-0.6B mbt-r,sft

sbatch scripts/slurm/main.slurm \
    --task-name musique \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-0.6B/mbt-r/sft/1e-4/128", "model_kwargs": {"config": "configs/vllm/qwen3-0.6b.yaml", "model": "output/train/Qwen3-0.6B/mbt-r/sft/1e-4/128"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/musique/validation"}'

# Qwen3-0.6B mbt-r,grpo

# (requires GRPO checkpoint at output/train/Qwen3-0.6B/mbt-r/grpo)

sbatch scripts/slurm/main.slurm \
    --task-name musique \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-0.6B/mbt-r/grpo", "model_kwargs": {"config": "configs/vllm/qwen3-0.6b.yaml", "model": "output/train/Qwen3-0.6B/mbt-r/grpo"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/musique/validation"}'

# Qwen3-0.6B distill-r,sft

sbatch scripts/slurm/main.slurm \
    --task-name musique \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-0.6B/distill-r/sft/1e-4/128", "model_kwargs": {"config": "configs/vllm/qwen3-0.6b.yaml", "model": "output/train/Qwen3-0.6B/distill-r/sft/1e-4/128"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/musique/validation"}'

# Qwen3-0.6B direct-r,sft

sbatch scripts/slurm/main.slurm \
    --task-name musique \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-0.6B/direct-r/sft/1e-4/128", "model_kwargs": {"config": "configs/vllm/qwen3-0.6b.yaml", "model": "output/train/Qwen3-0.6B/direct-r/sft/1e-4/128"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/musique/validation"}'

# Qwen3-0.6B shorter-better

# external baseline (shorter-better) — implement separately;

# expected rollout output dir: output/musique/validation/Qwen3-0.6B/shorter-better

# (no SBATCH cell emitted — populate via the method's own pipeline)

# Qwen3-0.6B token-skip

# external baseline (token-skip) — implement separately;

# expected rollout output dir: output/musique/validation/Qwen3-0.6B/token-skip

# (no SBATCH cell emitted — populate via the method's own pipeline)

# Qwen3-0.6B limopro

# external baseline (limopro) — implement separately;

# expected rollout output dir: output/musique/validation/Qwen3-0.6B/limopro

# (no SBATCH cell emitted — populate via the method's own pipeline)

# Qwen3-1.7B prompt

sbatch scripts/slurm/main.slurm \
    --task-name musique \
    --task-config '{"dataset_split": "validation", "metacognitive_prompt": true}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-1.7B/metacognitive-prompt", "model_kwargs": {"config": "configs/vllm/qwen3-1.7b.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/musique/validation"}'

# Qwen3-1.7B grpo

# (requires GRPO checkpoint at output/train/Qwen3-1.7B/grpo)

sbatch scripts/slurm/main.slurm \
    --task-name musique \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-1.7B/grpo", "model_kwargs": {"config": "configs/vllm/qwen3-1.7b.yaml", "model": "output/train/Qwen3-1.7B/grpo"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/musique/validation"}'

# Qwen3-1.7B self-distill,sft

sbatch scripts/slurm/main.slurm \
    --task-name musique \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-1.7B/self-distill/sft/1e-4/128", "model_kwargs": {"config": "configs/vllm/qwen3-1.7b.yaml", "model": "output/train/Qwen3-1.7B/self-distill/sft/1e-4/128"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/musique/validation"}'

# Qwen3-1.7B gpt-oss-distill,sft

sbatch scripts/slurm/main.slurm \
    --task-name musique \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-1.7B/gpt-oss-distill/sft/1e-4/128", "model_kwargs": {"config": "configs/vllm/qwen3-1.7b.yaml", "model": "output/train/Qwen3-1.7B/gpt-oss-distill/sft/1e-4/128"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/musique/validation"}'

# Qwen3-1.7B gpt-oss-distill,grpo

# (requires GRPO checkpoint at output/train/Qwen3-1.7B/gpt-oss-distill/grpo)

sbatch scripts/slurm/main.slurm \
    --task-name musique \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-1.7B/gpt-oss-distill/grpo", "model_kwargs": {"config": "configs/vllm/qwen3-1.7b.yaml", "model": "output/train/Qwen3-1.7B/gpt-oss-distill/grpo"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/musique/validation"}'

# Qwen3-1.7B mbt-s,sft

sbatch scripts/slurm/main.slurm \
    --task-name musique \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-1.7B/mbt-s/sft/1e-4/128", "model_kwargs": {"config": "configs/vllm/qwen3-1.7b.yaml", "model": "output/train/Qwen3-1.7B/mbt-s/sft/1e-4/128"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/musique/validation"}'

# Qwen3-1.7B mbt-s,grpo

# (requires GRPO checkpoint at output/train/Qwen3-1.7B/mbt-s/grpo)

sbatch scripts/slurm/main.slurm \
    --task-name musique \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-1.7B/mbt-s/grpo", "model_kwargs": {"config": "configs/vllm/qwen3-1.7b.yaml", "model": "output/train/Qwen3-1.7B/mbt-s/grpo"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/musique/validation"}'

# Qwen3-1.7B mbt-r,sft

sbatch scripts/slurm/main.slurm \
    --task-name musique \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-1.7B/mbt-r/sft/1e-4/128", "model_kwargs": {"config": "configs/vllm/qwen3-1.7b.yaml", "model": "output/train/Qwen3-1.7B/mbt-r/sft/1e-4/128"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/musique/validation"}'

# Qwen3-1.7B mbt-r,grpo

# (requires GRPO checkpoint at output/train/Qwen3-1.7B/mbt-r/grpo)

sbatch scripts/slurm/main.slurm \
    --task-name musique \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-1.7B/mbt-r/grpo", "model_kwargs": {"config": "configs/vllm/qwen3-1.7b.yaml", "model": "output/train/Qwen3-1.7B/mbt-r/grpo"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/musique/validation"}'

# Qwen3-1.7B distill-r,sft

sbatch scripts/slurm/main.slurm \
    --task-name musique \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-1.7B/distill-r/sft/1e-4/128", "model_kwargs": {"config": "configs/vllm/qwen3-1.7b.yaml", "model": "output/train/Qwen3-1.7B/distill-r/sft/1e-4/128"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/musique/validation"}'

# Qwen3-1.7B direct-r,sft

sbatch scripts/slurm/main.slurm \
    --task-name musique \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-1.7B/direct-r/sft/1e-4/128", "model_kwargs": {"config": "configs/vllm/qwen3-1.7b.yaml", "model": "output/train/Qwen3-1.7B/direct-r/sft/1e-4/128"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/musique/validation"}'

# Qwen3-1.7B shorter-better

# external baseline (shorter-better) — implement separately;

# expected rollout output dir: output/musique/validation/Qwen3-1.7B/shorter-better

# (no SBATCH cell emitted — populate via the method's own pipeline)

# Qwen3-1.7B token-skip

# external baseline (token-skip) — implement separately;

# expected rollout output dir: output/musique/validation/Qwen3-1.7B/token-skip

# (no SBATCH cell emitted — populate via the method's own pipeline)

# Qwen3-1.7B limopro

# external baseline (limopro) — implement separately;

# expected rollout output dir: output/musique/validation/Qwen3-1.7B/limopro

# (no SBATCH cell emitted — populate via the method's own pipeline)

# Qwen3-4B prompt

sbatch scripts/slurm/main.slurm \
    --task-name musique \
    --task-config '{"dataset_split": "validation", "metacognitive_prompt": true}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-4B/metacognitive-prompt", "model_kwargs": {"config": "configs/vllm/qwen3-4b.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/musique/validation"}'

# Qwen3-4B grpo

# (requires GRPO checkpoint at output/train/Qwen3-4B/grpo)

sbatch scripts/slurm/main.slurm \
    --task-name musique \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-4B/grpo", "model_kwargs": {"config": "configs/vllm/qwen3-4b.yaml", "model": "output/train/Qwen3-4B/grpo"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/musique/validation"}'

# Qwen3-4B self-distill,sft

sbatch scripts/slurm/main.slurm \
    --task-name musique \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-4B/self-distill/sft/1e-4/128", "model_kwargs": {"config": "configs/vllm/qwen3-4b.yaml", "model": "output/train/Qwen3-4B/self-distill/sft/1e-4/128"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/musique/validation"}'

# Qwen3-4B gpt-oss-distill,sft

sbatch scripts/slurm/main.slurm \
    --task-name musique \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-4B/gpt-oss-distill/sft/1e-4/128", "model_kwargs": {"config": "configs/vllm/qwen3-4b.yaml", "model": "output/train/Qwen3-4B/gpt-oss-distill/sft/1e-4/128"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/musique/validation"}'

# Qwen3-4B gpt-oss-distill,grpo

# (requires GRPO checkpoint at output/train/Qwen3-4B/gpt-oss-distill/grpo)

sbatch scripts/slurm/main.slurm \
    --task-name musique \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-4B/gpt-oss-distill/grpo", "model_kwargs": {"config": "configs/vllm/qwen3-4b.yaml", "model": "output/train/Qwen3-4B/gpt-oss-distill/grpo"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/musique/validation"}'

# Qwen3-4B mbt-s,sft

sbatch scripts/slurm/main.slurm \
    --task-name musique \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-4B/mbt-s/sft/1e-4/128", "model_kwargs": {"config": "configs/vllm/qwen3-4b.yaml", "model": "output/train/Qwen3-4B/mbt-s/sft/1e-4/128"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/musique/validation"}'

# Qwen3-4B mbt-s,grpo

# (requires GRPO checkpoint at output/train/Qwen3-4B/mbt-s/grpo)

sbatch scripts/slurm/main.slurm \
    --task-name musique \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-4B/mbt-s/grpo", "model_kwargs": {"config": "configs/vllm/qwen3-4b.yaml", "model": "output/train/Qwen3-4B/mbt-s/grpo"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/musique/validation"}'

# Qwen3-4B mbt-r,sft

sbatch scripts/slurm/main.slurm \
    --task-name musique \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-4B/mbt-r/sft/1e-4/128", "model_kwargs": {"config": "configs/vllm/qwen3-4b.yaml", "model": "output/train/Qwen3-4B/mbt-r/sft/1e-4/128"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/musique/validation"}'

# Qwen3-4B mbt-r,grpo

# (requires GRPO checkpoint at output/train/Qwen3-4B/mbt-r/grpo)

sbatch scripts/slurm/main.slurm \
    --task-name musique \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-4B/mbt-r/grpo", "model_kwargs": {"config": "configs/vllm/qwen3-4b.yaml", "model": "output/train/Qwen3-4B/mbt-r/grpo"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/musique/validation"}'

# Qwen3-4B distill-r,sft

sbatch scripts/slurm/main.slurm \
    --task-name musique \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-4B/distill-r/sft/1e-4/128", "model_kwargs": {"config": "configs/vllm/qwen3-4b.yaml", "model": "output/train/Qwen3-4B/distill-r/sft/1e-4/128"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/musique/validation"}'

# Qwen3-4B direct-r,sft

sbatch scripts/slurm/main.slurm \
    --task-name musique \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-4B/direct-r/sft/1e-4/128", "model_kwargs": {"config": "configs/vllm/qwen3-4b.yaml", "model": "output/train/Qwen3-4B/direct-r/sft/1e-4/128"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/musique/validation"}'

# Qwen3-4B shorter-better

# external baseline (shorter-better) — implement separately;

# expected rollout output dir: output/musique/validation/Qwen3-4B/shorter-better

# (no SBATCH cell emitted — populate via the method's own pipeline)

# Qwen3-4B token-skip

# external baseline (token-skip) — implement separately;

# expected rollout output dir: output/musique/validation/Qwen3-4B/token-skip

# (no SBATCH cell emitted — populate via the method's own pipeline)

# Qwen3-4B limopro

# external baseline (limopro) — implement separately;

# expected rollout output dir: output/musique/validation/Qwen3-4B/limopro

# (no SBATCH cell emitted — populate via the method's own pipeline)

# <<< musique/validation <<<

# >>> 2wikimultihopqa/validation — trained / prompt / external variants >>>

# Qwen3-0.6B prompt

sbatch scripts/slurm/main.slurm \
    --task-name 2wikimultihopqa \
    --task-config '{"dataset_split": "validation", "metacognitive_prompt": true}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-0.6B/metacognitive-prompt", "model_kwargs": {"config": "configs/vllm/qwen3-0.6b.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/2wikimultihopqa/validation"}'

# Qwen3-0.6B grpo

# (requires GRPO checkpoint at output/train/Qwen3-0.6B/grpo)

sbatch scripts/slurm/main.slurm \
    --task-name 2wikimultihopqa \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-0.6B/grpo", "model_kwargs": {"config": "configs/vllm/qwen3-0.6b.yaml", "model": "output/train/Qwen3-0.6B/grpo"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/2wikimultihopqa/validation"}'

# Qwen3-0.6B self-distill,sft

sbatch scripts/slurm/main.slurm \
    --task-name 2wikimultihopqa \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-0.6B/self-distill/sft/1e-4/128", "model_kwargs": {"config": "configs/vllm/qwen3-0.6b.yaml", "model": "output/train/Qwen3-0.6B/self-distill/sft/1e-4/128"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/2wikimultihopqa/validation"}'

# Qwen3-0.6B gpt-oss-distill,sft

sbatch scripts/slurm/main.slurm \
    --task-name 2wikimultihopqa \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-0.6B/gpt-oss-distill/sft/1e-4/128", "model_kwargs": {"config": "configs/vllm/qwen3-0.6b.yaml", "model": "output/train/Qwen3-0.6B/gpt-oss-distill/sft/1e-4/128"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/2wikimultihopqa/validation"}'

# Qwen3-0.6B gpt-oss-distill,grpo

# (requires GRPO checkpoint at output/train/Qwen3-0.6B/gpt-oss-distill/grpo)

sbatch scripts/slurm/main.slurm \
    --task-name 2wikimultihopqa \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-0.6B/gpt-oss-distill/grpo", "model_kwargs": {"config": "configs/vllm/qwen3-0.6b.yaml", "model": "output/train/Qwen3-0.6B/gpt-oss-distill/grpo"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/2wikimultihopqa/validation"}'

# Qwen3-0.6B mbt-s,sft

sbatch scripts/slurm/main.slurm \
    --task-name 2wikimultihopqa \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-0.6B/mbt-s/sft/1e-4/128", "model_kwargs": {"config": "configs/vllm/qwen3-0.6b.yaml", "model": "output/train/Qwen3-0.6B/mbt-s/sft/1e-4/128"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/2wikimultihopqa/validation"}'

# Qwen3-0.6B mbt-s,grpo

# (requires GRPO checkpoint at output/train/Qwen3-0.6B/mbt-s/grpo)

sbatch scripts/slurm/main.slurm \
    --task-name 2wikimultihopqa \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-0.6B/mbt-s/grpo", "model_kwargs": {"config": "configs/vllm/qwen3-0.6b.yaml", "model": "output/train/Qwen3-0.6B/mbt-s/grpo"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/2wikimultihopqa/validation"}'

# Qwen3-0.6B mbt-r,sft

sbatch scripts/slurm/main.slurm \
    --task-name 2wikimultihopqa \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-0.6B/mbt-r/sft/1e-4/128", "model_kwargs": {"config": "configs/vllm/qwen3-0.6b.yaml", "model": "output/train/Qwen3-0.6B/mbt-r/sft/1e-4/128"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/2wikimultihopqa/validation"}'

# Qwen3-0.6B mbt-r,grpo

# (requires GRPO checkpoint at output/train/Qwen3-0.6B/mbt-r/grpo)

sbatch scripts/slurm/main.slurm \
    --task-name 2wikimultihopqa \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-0.6B/mbt-r/grpo", "model_kwargs": {"config": "configs/vllm/qwen3-0.6b.yaml", "model": "output/train/Qwen3-0.6B/mbt-r/grpo"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/2wikimultihopqa/validation"}'

# Qwen3-0.6B distill-r,sft

sbatch scripts/slurm/main.slurm \
    --task-name 2wikimultihopqa \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-0.6B/distill-r/sft/1e-4/128", "model_kwargs": {"config": "configs/vllm/qwen3-0.6b.yaml", "model": "output/train/Qwen3-0.6B/distill-r/sft/1e-4/128"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/2wikimultihopqa/validation"}'

# Qwen3-0.6B direct-r,sft

sbatch scripts/slurm/main.slurm \
    --task-name 2wikimultihopqa \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-0.6B/direct-r/sft/1e-4/128", "model_kwargs": {"config": "configs/vllm/qwen3-0.6b.yaml", "model": "output/train/Qwen3-0.6B/direct-r/sft/1e-4/128"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/2wikimultihopqa/validation"}'

# Qwen3-0.6B shorter-better

# external baseline (shorter-better) — implement separately;

# expected rollout output dir: output/2wikimultihopqa/validation/Qwen3-0.6B/shorter-better

# (no SBATCH cell emitted — populate via the method's own pipeline)

# Qwen3-0.6B token-skip

# external baseline (token-skip) — implement separately;

# expected rollout output dir: output/2wikimultihopqa/validation/Qwen3-0.6B/token-skip

# (no SBATCH cell emitted — populate via the method's own pipeline)

# Qwen3-0.6B limopro

# external baseline (limopro) — implement separately;

# expected rollout output dir: output/2wikimultihopqa/validation/Qwen3-0.6B/limopro

# (no SBATCH cell emitted — populate via the method's own pipeline)

# Qwen3-1.7B prompt

sbatch scripts/slurm/main.slurm \
    --task-name 2wikimultihopqa \
    --task-config '{"dataset_split": "validation", "metacognitive_prompt": true}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-1.7B/metacognitive-prompt", "model_kwargs": {"config": "configs/vllm/qwen3-1.7b.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/2wikimultihopqa/validation"}'

# Qwen3-1.7B grpo

# (requires GRPO checkpoint at output/train/Qwen3-1.7B/grpo)

sbatch scripts/slurm/main.slurm \
    --task-name 2wikimultihopqa \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-1.7B/grpo", "model_kwargs": {"config": "configs/vllm/qwen3-1.7b.yaml", "model": "output/train/Qwen3-1.7B/grpo"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/2wikimultihopqa/validation"}'

# Qwen3-1.7B self-distill,sft

sbatch scripts/slurm/main.slurm \
    --task-name 2wikimultihopqa \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-1.7B/self-distill/sft/1e-4/128", "model_kwargs": {"config": "configs/vllm/qwen3-1.7b.yaml", "model": "output/train/Qwen3-1.7B/self-distill/sft/1e-4/128"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/2wikimultihopqa/validation"}'

# Qwen3-1.7B gpt-oss-distill,sft

sbatch scripts/slurm/main.slurm \
    --task-name 2wikimultihopqa \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-1.7B/gpt-oss-distill/sft/1e-4/128", "model_kwargs": {"config": "configs/vllm/qwen3-1.7b.yaml", "model": "output/train/Qwen3-1.7B/gpt-oss-distill/sft/1e-4/128"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/2wikimultihopqa/validation"}'

# Qwen3-1.7B gpt-oss-distill,grpo

# (requires GRPO checkpoint at output/train/Qwen3-1.7B/gpt-oss-distill/grpo)

sbatch scripts/slurm/main.slurm \
    --task-name 2wikimultihopqa \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-1.7B/gpt-oss-distill/grpo", "model_kwargs": {"config": "configs/vllm/qwen3-1.7b.yaml", "model": "output/train/Qwen3-1.7B/gpt-oss-distill/grpo"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/2wikimultihopqa/validation"}'

# Qwen3-1.7B mbt-s,sft

sbatch scripts/slurm/main.slurm \
    --task-name 2wikimultihopqa \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-1.7B/mbt-s/sft/1e-4/128", "model_kwargs": {"config": "configs/vllm/qwen3-1.7b.yaml", "model": "output/train/Qwen3-1.7B/mbt-s/sft/1e-4/128"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/2wikimultihopqa/validation"}'

# Qwen3-1.7B mbt-s,grpo

# (requires GRPO checkpoint at output/train/Qwen3-1.7B/mbt-s/grpo)

sbatch scripts/slurm/main.slurm \
    --task-name 2wikimultihopqa \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-1.7B/mbt-s/grpo", "model_kwargs": {"config": "configs/vllm/qwen3-1.7b.yaml", "model": "output/train/Qwen3-1.7B/mbt-s/grpo"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/2wikimultihopqa/validation"}'

# Qwen3-1.7B mbt-r,sft

sbatch scripts/slurm/main.slurm \
    --task-name 2wikimultihopqa \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-1.7B/mbt-r/sft/1e-4/128", "model_kwargs": {"config": "configs/vllm/qwen3-1.7b.yaml", "model": "output/train/Qwen3-1.7B/mbt-r/sft/1e-4/128"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/2wikimultihopqa/validation"}'

# Qwen3-1.7B mbt-r,grpo

# (requires GRPO checkpoint at output/train/Qwen3-1.7B/mbt-r/grpo)

sbatch scripts/slurm/main.slurm \
    --task-name 2wikimultihopqa \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-1.7B/mbt-r/grpo", "model_kwargs": {"config": "configs/vllm/qwen3-1.7b.yaml", "model": "output/train/Qwen3-1.7B/mbt-r/grpo"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/2wikimultihopqa/validation"}'

# Qwen3-1.7B distill-r,sft

sbatch scripts/slurm/main.slurm \
    --task-name 2wikimultihopqa \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-1.7B/distill-r/sft/1e-4/128", "model_kwargs": {"config": "configs/vllm/qwen3-1.7b.yaml", "model": "output/train/Qwen3-1.7B/distill-r/sft/1e-4/128"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/2wikimultihopqa/validation"}'

# Qwen3-1.7B direct-r,sft

sbatch scripts/slurm/main.slurm \
    --task-name 2wikimultihopqa \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-1.7B/direct-r/sft/1e-4/128", "model_kwargs": {"config": "configs/vllm/qwen3-1.7b.yaml", "model": "output/train/Qwen3-1.7B/direct-r/sft/1e-4/128"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/2wikimultihopqa/validation"}'

# Qwen3-1.7B shorter-better

# external baseline (shorter-better) — implement separately;

# expected rollout output dir: output/2wikimultihopqa/validation/Qwen3-1.7B/shorter-better

# (no SBATCH cell emitted — populate via the method's own pipeline)

# Qwen3-1.7B token-skip

# external baseline (token-skip) — implement separately;

# expected rollout output dir: output/2wikimultihopqa/validation/Qwen3-1.7B/token-skip

# (no SBATCH cell emitted — populate via the method's own pipeline)

# Qwen3-1.7B limopro

# external baseline (limopro) — implement separately;

# expected rollout output dir: output/2wikimultihopqa/validation/Qwen3-1.7B/limopro

# (no SBATCH cell emitted — populate via the method's own pipeline)

# Qwen3-4B prompt

sbatch scripts/slurm/main.slurm \
    --task-name 2wikimultihopqa \
    --task-config '{"dataset_split": "validation", "metacognitive_prompt": true}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-4B/metacognitive-prompt", "model_kwargs": {"config": "configs/vllm/qwen3-4b.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/2wikimultihopqa/validation"}'

# Qwen3-4B grpo

# (requires GRPO checkpoint at output/train/Qwen3-4B/grpo)

sbatch scripts/slurm/main.slurm \
    --task-name 2wikimultihopqa \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-4B/grpo", "model_kwargs": {"config": "configs/vllm/qwen3-4b.yaml", "model": "output/train/Qwen3-4B/grpo"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/2wikimultihopqa/validation"}'

# Qwen3-4B self-distill,sft

sbatch scripts/slurm/main.slurm \
    --task-name 2wikimultihopqa \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-4B/self-distill/sft/1e-4/128", "model_kwargs": {"config": "configs/vllm/qwen3-4b.yaml", "model": "output/train/Qwen3-4B/self-distill/sft/1e-4/128"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/2wikimultihopqa/validation"}'

# Qwen3-4B gpt-oss-distill,sft

sbatch scripts/slurm/main.slurm \
    --task-name 2wikimultihopqa \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-4B/gpt-oss-distill/sft/1e-4/128", "model_kwargs": {"config": "configs/vllm/qwen3-4b.yaml", "model": "output/train/Qwen3-4B/gpt-oss-distill/sft/1e-4/128"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/2wikimultihopqa/validation"}'

# Qwen3-4B gpt-oss-distill,grpo

# (requires GRPO checkpoint at output/train/Qwen3-4B/gpt-oss-distill/grpo)

sbatch scripts/slurm/main.slurm \
    --task-name 2wikimultihopqa \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-4B/gpt-oss-distill/grpo", "model_kwargs": {"config": "configs/vllm/qwen3-4b.yaml", "model": "output/train/Qwen3-4B/gpt-oss-distill/grpo"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/2wikimultihopqa/validation"}'

# Qwen3-4B mbt-s,sft

sbatch scripts/slurm/main.slurm \
    --task-name 2wikimultihopqa \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-4B/mbt-s/sft/1e-4/128", "model_kwargs": {"config": "configs/vllm/qwen3-4b.yaml", "model": "output/train/Qwen3-4B/mbt-s/sft/1e-4/128"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/2wikimultihopqa/validation"}'

# Qwen3-4B mbt-s,grpo

# (requires GRPO checkpoint at output/train/Qwen3-4B/mbt-s/grpo)

sbatch scripts/slurm/main.slurm \
    --task-name 2wikimultihopqa \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-4B/mbt-s/grpo", "model_kwargs": {"config": "configs/vllm/qwen3-4b.yaml", "model": "output/train/Qwen3-4B/mbt-s/grpo"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/2wikimultihopqa/validation"}'

# Qwen3-4B mbt-r,sft

sbatch scripts/slurm/main.slurm \
    --task-name 2wikimultihopqa \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-4B/mbt-r/sft/1e-4/128", "model_kwargs": {"config": "configs/vllm/qwen3-4b.yaml", "model": "output/train/Qwen3-4B/mbt-r/sft/1e-4/128"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/2wikimultihopqa/validation"}'

# Qwen3-4B mbt-r,grpo

# (requires GRPO checkpoint at output/train/Qwen3-4B/mbt-r/grpo)

sbatch scripts/slurm/main.slurm \
    --task-name 2wikimultihopqa \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-4B/mbt-r/grpo", "model_kwargs": {"config": "configs/vllm/qwen3-4b.yaml", "model": "output/train/Qwen3-4B/mbt-r/grpo"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/2wikimultihopqa/validation"}'

# Qwen3-4B distill-r,sft

sbatch scripts/slurm/main.slurm \
    --task-name 2wikimultihopqa \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-4B/distill-r/sft/1e-4/128", "model_kwargs": {"config": "configs/vllm/qwen3-4b.yaml", "model": "output/train/Qwen3-4B/distill-r/sft/1e-4/128"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/2wikimultihopqa/validation"}'

# Qwen3-4B direct-r,sft

sbatch scripts/slurm/main.slurm \
    --task-name 2wikimultihopqa \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-4B/direct-r/sft/1e-4/128", "model_kwargs": {"config": "configs/vllm/qwen3-4b.yaml", "model": "output/train/Qwen3-4B/direct-r/sft/1e-4/128"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/2wikimultihopqa/validation"}'

# Qwen3-4B shorter-better

# external baseline (shorter-better) — implement separately;

# expected rollout output dir: output/2wikimultihopqa/validation/Qwen3-4B/shorter-better

# (no SBATCH cell emitted — populate via the method's own pipeline)

# Qwen3-4B token-skip

# external baseline (token-skip) — implement separately;

# expected rollout output dir: output/2wikimultihopqa/validation/Qwen3-4B/token-skip

# (no SBATCH cell emitted — populate via the method's own pipeline)

# Qwen3-4B limopro

# external baseline (limopro) — implement separately;

# expected rollout output dir: output/2wikimultihopqa/validation/Qwen3-4B/limopro

# (no SBATCH cell emitted — populate via the method's own pipeline)

# <<< 2wikimultihopqa/validation <<<

# >>> hotpotqa/validation — trained / prompt / external variants >>>

# Qwen3-0.6B prompt

sbatch scripts/slurm/main.slurm \
    --task-name hotpotqa \
    --task-config '{"dataset_split": "validation", "metacognitive_prompt": true}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-0.6B/metacognitive-prompt", "model_kwargs": {"config": "configs/vllm/qwen3-0.6b.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/hotpotqa/validation"}'

# Qwen3-0.6B grpo

# (requires GRPO checkpoint at output/train/Qwen3-0.6B/grpo)

sbatch scripts/slurm/main.slurm \
    --task-name hotpotqa \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-0.6B/grpo", "model_kwargs": {"config": "configs/vllm/qwen3-0.6b.yaml", "model": "output/train/Qwen3-0.6B/grpo"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/hotpotqa/validation"}'

# Qwen3-0.6B self-distill,sft

sbatch scripts/slurm/main.slurm \
    --task-name hotpotqa \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-0.6B/self-distill/sft/1e-4/128", "model_kwargs": {"config": "configs/vllm/qwen3-0.6b.yaml", "model": "output/train/Qwen3-0.6B/self-distill/sft/1e-4/128"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/hotpotqa/validation"}'

# Qwen3-0.6B gpt-oss-distill,sft

sbatch scripts/slurm/main.slurm \
    --task-name hotpotqa \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-0.6B/gpt-oss-distill/sft/1e-4/128", "model_kwargs": {"config": "configs/vllm/qwen3-0.6b.yaml", "model": "output/train/Qwen3-0.6B/gpt-oss-distill/sft/1e-4/128"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/hotpotqa/validation"}'

# Qwen3-0.6B gpt-oss-distill,grpo

# (requires GRPO checkpoint at output/train/Qwen3-0.6B/gpt-oss-distill/grpo)

sbatch scripts/slurm/main.slurm \
    --task-name hotpotqa \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-0.6B/gpt-oss-distill/grpo", "model_kwargs": {"config": "configs/vllm/qwen3-0.6b.yaml", "model": "output/train/Qwen3-0.6B/gpt-oss-distill/grpo"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/hotpotqa/validation"}'

# Qwen3-0.6B mbt-s,sft

sbatch scripts/slurm/main.slurm \
    --task-name hotpotqa \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-0.6B/mbt-s/sft/1e-4/128", "model_kwargs": {"config": "configs/vllm/qwen3-0.6b.yaml", "model": "output/train/Qwen3-0.6B/mbt-s/sft/1e-4/128"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/hotpotqa/validation"}'

# Qwen3-0.6B mbt-s,grpo

# (requires GRPO checkpoint at output/train/Qwen3-0.6B/mbt-s/grpo)

sbatch scripts/slurm/main.slurm \
    --task-name hotpotqa \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-0.6B/mbt-s/grpo", "model_kwargs": {"config": "configs/vllm/qwen3-0.6b.yaml", "model": "output/train/Qwen3-0.6B/mbt-s/grpo"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/hotpotqa/validation"}'

# Qwen3-0.6B mbt-r,sft

sbatch scripts/slurm/main.slurm \
    --task-name hotpotqa \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-0.6B/mbt-r/sft/1e-4/128", "model_kwargs": {"config": "configs/vllm/qwen3-0.6b.yaml", "model": "output/train/Qwen3-0.6B/mbt-r/sft/1e-4/128"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/hotpotqa/validation"}'

# Qwen3-0.6B mbt-r,grpo

# (requires GRPO checkpoint at output/train/Qwen3-0.6B/mbt-r/grpo)

sbatch scripts/slurm/main.slurm \
    --task-name hotpotqa \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-0.6B/mbt-r/grpo", "model_kwargs": {"config": "configs/vllm/qwen3-0.6b.yaml", "model": "output/train/Qwen3-0.6B/mbt-r/grpo"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/hotpotqa/validation"}'

# Qwen3-0.6B distill-r,sft

sbatch scripts/slurm/main.slurm \
    --task-name hotpotqa \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-0.6B/distill-r/sft/1e-4/128", "model_kwargs": {"config": "configs/vllm/qwen3-0.6b.yaml", "model": "output/train/Qwen3-0.6B/distill-r/sft/1e-4/128"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/hotpotqa/validation"}'

# Qwen3-0.6B direct-r,sft

sbatch scripts/slurm/main.slurm \
    --task-name hotpotqa \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-0.6B/direct-r/sft/1e-4/128", "model_kwargs": {"config": "configs/vllm/qwen3-0.6b.yaml", "model": "output/train/Qwen3-0.6B/direct-r/sft/1e-4/128"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/hotpotqa/validation"}'

# Qwen3-0.6B shorter-better

# external baseline (shorter-better) — implement separately;

# expected rollout output dir: output/hotpotqa/validation/Qwen3-0.6B/shorter-better

# (no SBATCH cell emitted — populate via the method's own pipeline)

# Qwen3-0.6B token-skip

# external baseline (token-skip) — implement separately;

# expected rollout output dir: output/hotpotqa/validation/Qwen3-0.6B/token-skip

# (no SBATCH cell emitted — populate via the method's own pipeline)

# Qwen3-0.6B limopro

# external baseline (limopro) — implement separately;

# expected rollout output dir: output/hotpotqa/validation/Qwen3-0.6B/limopro

# (no SBATCH cell emitted — populate via the method's own pipeline)

# Qwen3-1.7B prompt

sbatch scripts/slurm/main.slurm \
    --task-name hotpotqa \
    --task-config '{"dataset_split": "validation", "metacognitive_prompt": true}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-1.7B/metacognitive-prompt", "model_kwargs": {"config": "configs/vllm/qwen3-1.7b.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/hotpotqa/validation"}'

# Qwen3-1.7B grpo

# (requires GRPO checkpoint at output/train/Qwen3-1.7B/grpo)

sbatch scripts/slurm/main.slurm \
    --task-name hotpotqa \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-1.7B/grpo", "model_kwargs": {"config": "configs/vllm/qwen3-1.7b.yaml", "model": "output/train/Qwen3-1.7B/grpo"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/hotpotqa/validation"}'

# Qwen3-1.7B self-distill,sft

sbatch scripts/slurm/main.slurm \
    --task-name hotpotqa \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-1.7B/self-distill/sft/1e-4/128", "model_kwargs": {"config": "configs/vllm/qwen3-1.7b.yaml", "model": "output/train/Qwen3-1.7B/self-distill/sft/1e-4/128"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/hotpotqa/validation"}'

# Qwen3-1.7B gpt-oss-distill,sft

sbatch scripts/slurm/main.slurm \
    --task-name hotpotqa \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-1.7B/gpt-oss-distill/sft/1e-4/128", "model_kwargs": {"config": "configs/vllm/qwen3-1.7b.yaml", "model": "output/train/Qwen3-1.7B/gpt-oss-distill/sft/1e-4/128"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/hotpotqa/validation"}'

# Qwen3-1.7B gpt-oss-distill,grpo

# (requires GRPO checkpoint at output/train/Qwen3-1.7B/gpt-oss-distill/grpo)

sbatch scripts/slurm/main.slurm \
    --task-name hotpotqa \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-1.7B/gpt-oss-distill/grpo", "model_kwargs": {"config": "configs/vllm/qwen3-1.7b.yaml", "model": "output/train/Qwen3-1.7B/gpt-oss-distill/grpo"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/hotpotqa/validation"}'

# Qwen3-1.7B mbt-s,sft

sbatch scripts/slurm/main.slurm \
    --task-name hotpotqa \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-1.7B/mbt-s/sft/1e-4/128", "model_kwargs": {"config": "configs/vllm/qwen3-1.7b.yaml", "model": "output/train/Qwen3-1.7B/mbt-s/sft/1e-4/128"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/hotpotqa/validation"}'

# Qwen3-1.7B mbt-s,grpo

# (requires GRPO checkpoint at output/train/Qwen3-1.7B/mbt-s/grpo)

sbatch scripts/slurm/main.slurm \
    --task-name hotpotqa \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-1.7B/mbt-s/grpo", "model_kwargs": {"config": "configs/vllm/qwen3-1.7b.yaml", "model": "output/train/Qwen3-1.7B/mbt-s/grpo"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/hotpotqa/validation"}'

# Qwen3-1.7B mbt-r,sft

sbatch scripts/slurm/main.slurm \
    --task-name hotpotqa \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-1.7B/mbt-r/sft/1e-4/128", "model_kwargs": {"config": "configs/vllm/qwen3-1.7b.yaml", "model": "output/train/Qwen3-1.7B/mbt-r/sft/1e-4/128"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/hotpotqa/validation"}'

# Qwen3-1.7B mbt-r,grpo

# (requires GRPO checkpoint at output/train/Qwen3-1.7B/mbt-r/grpo)

sbatch scripts/slurm/main.slurm \
    --task-name hotpotqa \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-1.7B/mbt-r/grpo", "model_kwargs": {"config": "configs/vllm/qwen3-1.7b.yaml", "model": "output/train/Qwen3-1.7B/mbt-r/grpo"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/hotpotqa/validation"}'

# Qwen3-1.7B distill-r,sft

sbatch scripts/slurm/main.slurm \
    --task-name hotpotqa \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-1.7B/distill-r/sft/1e-4/128", "model_kwargs": {"config": "configs/vllm/qwen3-1.7b.yaml", "model": "output/train/Qwen3-1.7B/distill-r/sft/1e-4/128"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/hotpotqa/validation"}'

# Qwen3-1.7B direct-r,sft

sbatch scripts/slurm/main.slurm \
    --task-name hotpotqa \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-1.7B/direct-r/sft/1e-4/128", "model_kwargs": {"config": "configs/vllm/qwen3-1.7b.yaml", "model": "output/train/Qwen3-1.7B/direct-r/sft/1e-4/128"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/hotpotqa/validation"}'

# Qwen3-1.7B shorter-better

# external baseline (shorter-better) — implement separately;

# expected rollout output dir: output/hotpotqa/validation/Qwen3-1.7B/shorter-better

# (no SBATCH cell emitted — populate via the method's own pipeline)

# Qwen3-1.7B token-skip

# external baseline (token-skip) — implement separately;

# expected rollout output dir: output/hotpotqa/validation/Qwen3-1.7B/token-skip

# (no SBATCH cell emitted — populate via the method's own pipeline)

# Qwen3-1.7B limopro

# external baseline (limopro) — implement separately;

# expected rollout output dir: output/hotpotqa/validation/Qwen3-1.7B/limopro

# (no SBATCH cell emitted — populate via the method's own pipeline)

# Qwen3-4B prompt

sbatch scripts/slurm/main.slurm \
    --task-name hotpotqa \
    --task-config '{"dataset_split": "validation", "metacognitive_prompt": true}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-4B/metacognitive-prompt", "model_kwargs": {"config": "configs/vllm/qwen3-4b.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/hotpotqa/validation"}'

# Qwen3-4B grpo

# (requires GRPO checkpoint at output/train/Qwen3-4B/grpo)

sbatch scripts/slurm/main.slurm \
    --task-name hotpotqa \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-4B/grpo", "model_kwargs": {"config": "configs/vllm/qwen3-4b.yaml", "model": "output/train/Qwen3-4B/grpo"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/hotpotqa/validation"}'

# Qwen3-4B self-distill,sft

sbatch scripts/slurm/main.slurm \
    --task-name hotpotqa \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-4B/self-distill/sft/1e-4/128", "model_kwargs": {"config": "configs/vllm/qwen3-4b.yaml", "model": "output/train/Qwen3-4B/self-distill/sft/1e-4/128"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/hotpotqa/validation"}'

# Qwen3-4B gpt-oss-distill,sft

sbatch scripts/slurm/main.slurm \
    --task-name hotpotqa \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-4B/gpt-oss-distill/sft/1e-4/128", "model_kwargs": {"config": "configs/vllm/qwen3-4b.yaml", "model": "output/train/Qwen3-4B/gpt-oss-distill/sft/1e-4/128"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/hotpotqa/validation"}'

# Qwen3-4B gpt-oss-distill,grpo

# (requires GRPO checkpoint at output/train/Qwen3-4B/gpt-oss-distill/grpo)

sbatch scripts/slurm/main.slurm \
    --task-name hotpotqa \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-4B/gpt-oss-distill/grpo", "model_kwargs": {"config": "configs/vllm/qwen3-4b.yaml", "model": "output/train/Qwen3-4B/gpt-oss-distill/grpo"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/hotpotqa/validation"}'

# Qwen3-4B mbt-s,sft

sbatch scripts/slurm/main.slurm \
    --task-name hotpotqa \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-4B/mbt-s/sft/1e-4/128", "model_kwargs": {"config": "configs/vllm/qwen3-4b.yaml", "model": "output/train/Qwen3-4B/mbt-s/sft/1e-4/128"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/hotpotqa/validation"}'

# Qwen3-4B mbt-s,grpo

# (requires GRPO checkpoint at output/train/Qwen3-4B/mbt-s/grpo)

sbatch scripts/slurm/main.slurm \
    --task-name hotpotqa \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-4B/mbt-s/grpo", "model_kwargs": {"config": "configs/vllm/qwen3-4b.yaml", "model": "output/train/Qwen3-4B/mbt-s/grpo"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/hotpotqa/validation"}'

# Qwen3-4B mbt-r,sft

sbatch scripts/slurm/main.slurm \
    --task-name hotpotqa \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-4B/mbt-r/sft/1e-4/128", "model_kwargs": {"config": "configs/vllm/qwen3-4b.yaml", "model": "output/train/Qwen3-4B/mbt-r/sft/1e-4/128"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/hotpotqa/validation"}'

# Qwen3-4B mbt-r,grpo

# (requires GRPO checkpoint at output/train/Qwen3-4B/mbt-r/grpo)

sbatch scripts/slurm/main.slurm \
    --task-name hotpotqa \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-4B/mbt-r/grpo", "model_kwargs": {"config": "configs/vllm/qwen3-4b.yaml", "model": "output/train/Qwen3-4B/mbt-r/grpo"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/hotpotqa/validation"}'

# Qwen3-4B distill-r,sft

sbatch scripts/slurm/main.slurm \
    --task-name hotpotqa \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-4B/distill-r/sft/1e-4/128", "model_kwargs": {"config": "configs/vllm/qwen3-4b.yaml", "model": "output/train/Qwen3-4B/distill-r/sft/1e-4/128"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/hotpotqa/validation"}'

# Qwen3-4B direct-r,sft

sbatch scripts/slurm/main.slurm \
    --task-name hotpotqa \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-4B/direct-r/sft/1e-4/128", "model_kwargs": {"config": "configs/vllm/qwen3-4b.yaml", "model": "output/train/Qwen3-4B/direct-r/sft/1e-4/128"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/hotpotqa/validation"}'

# Qwen3-4B shorter-better

# external baseline (shorter-better) — implement separately;

# expected rollout output dir: output/hotpotqa/validation/Qwen3-4B/shorter-better

# (no SBATCH cell emitted — populate via the method's own pipeline)

# Qwen3-4B token-skip

# external baseline (token-skip) — implement separately;

# expected rollout output dir: output/hotpotqa/validation/Qwen3-4B/token-skip

# (no SBATCH cell emitted — populate via the method's own pipeline)

# Qwen3-4B limopro

# external baseline (limopro) — implement separately;

# expected rollout output dir: output/hotpotqa/validation/Qwen3-4B/limopro

# (no SBATCH cell emitted — populate via the method's own pipeline)

# <<< hotpotqa/validation <<<

echo "[summary] failed=$FAILED"
exit $FAILED
