#!/usr/bin/env bash
set -uo pipefail
FAILED=0
trap 'FAILED=$((FAILED+1))' ERR

# Generate gold reasoning solutions on QA train sets using gpt-oss-120b-high.
# Output: output/<dataset>/train/solution/gpt-oss-120b-high/results/ — consumed
# by mbt_r / metacognition_score / over+underthinking_score as the reference solution.

# >>> qa solutions (gpt-oss-120b-high) >>>

# musique

sbatch scripts/slurm/main.slurm \
    --task-name musique \
    --task-config '{"dataset_split": "train", "solution": true}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "gpt-oss-120b-high", "model_kwargs": {"config": "configs/vllm/gpt-oss-120b.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}, "reasoning_effort": "high"}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/musique/train"}'

# 2wikimultihopqa

sbatch scripts/slurm/main.slurm \
    --task-name 2wikimultihopqa \
    --task-config '{"dataset_split": "train", "solution": true}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "gpt-oss-120b-high", "model_kwargs": {"config": "configs/vllm/gpt-oss-120b.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}, "reasoning_effort": "high"}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/2wikimultihopqa/train"}'

# hotpotqa

sbatch scripts/slurm/main.slurm \
    --task-name hotpotqa \
    --task-config '{"dataset_split": "train", "solution": true}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "gpt-oss-120b-high", "model_kwargs": {"config": "configs/vllm/gpt-oss-120b.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}, "reasoning_effort": "high"}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/hotpotqa/train"}'

# <<< qa solutions <<<

echo "[summary] failed=$FAILED"
exit $FAILED
