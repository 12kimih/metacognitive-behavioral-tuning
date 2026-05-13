#!/usr/bin/env bash
set -uo pipefail
FAILED=0
trap 'FAILED=$((FAILED+1))' ERR

# >>> progress shim >>>
TOTAL_CELLS=$(grep -cE '^uv run ' "$0")
CELL_IDX=0
_progress() {
    case "$BASH_COMMAND" in
        "uv run "*)
            CELL_IDX=$((CELL_IDX + 1))
            local label
            label=$(printf '%s' "$BASH_COMMAND" \
                | grep -oP '"root_dir"\s*:\s*"\K[^"]+' | head -1)
            if [[ -z "$label" ]]; then
                label=$(printf '%s' "$BASH_COMMAND" \
                    | grep -oP -- '--output_dir\s+\K\S+' | head -1)
            fi
            if [[ -z "$label" ]]; then
                label=$(printf '%.80s' "$BASH_COMMAND")
            fi
            printf '\n>>> [%d/%d] %s\n    $ %s\n\n' \
                "$CELL_IDX" "$TOTAL_CELLS" "$label" \
                "$BASH_COMMAND" >&2
            ;;
    esac
}
trap _progress DEBUG
# <<< progress shim <<<

# LOCAL variant — runs each cell directly via `uv run` on the current host.
# For SLURM-driven submission see scripts/tasks/slurm/<same-name>.sh.

# qa.mbt_r — refine each rollout's reasoning_trace against the gold solution.
# Judge: gpt-oss-120b-high (kept reasoning-rich for refinement; not gemma).
# Output: output/<dataset>/train/<rollout-model>/mbt-r/gpt-oss-120b-high/results/
# (refined_trace column). Used as SFT data for the mbt-r training mode.

# >>> musique train mbt-r refinement >>>

# Qwen3-0.6B

uv run mbt \
    --task-name "qa.mbt_r" \
    --task-config '{"solution_config": "musique", "solution_split": "train"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "gpt-oss-120b-high", "model_kwargs": {"config": "configs/vllm/gpt-oss-120b.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}, "reasoning_effort": "high"}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/musique/train/Qwen3-0.6B"}'

# Qwen3-1.7B

uv run mbt \
    --task-name "qa.mbt_r" \
    --task-config '{"solution_config": "musique", "solution_split": "train"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "gpt-oss-120b-high", "model_kwargs": {"config": "configs/vllm/gpt-oss-120b.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}, "reasoning_effort": "high"}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/musique/train/Qwen3-1.7B"}'

# Qwen3-4B

uv run mbt \
    --task-name "qa.mbt_r" \
    --task-config '{"solution_config": "musique", "solution_split": "train"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "gpt-oss-120b-high", "model_kwargs": {"config": "configs/vllm/gpt-oss-120b.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}, "reasoning_effort": "high"}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/musique/train/Qwen3-4B"}'

# gpt-oss-120b-high (self-refine)

uv run mbt \
    --task-name "qa.mbt_r" \
    --task-config '{"solution_config": "musique", "solution_split": "train"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "gpt-oss-120b-high", "model_kwargs": {"config": "configs/vllm/gpt-oss-120b.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}, "reasoning_effort": "high"}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/musique/train/gpt-oss-120b-high"}'

# <<< musique <<<

# >>> 2wikimultihopqa train mbt-r refinement >>>

# Qwen3-0.6B

uv run mbt \
    --task-name "qa.mbt_r" \
    --task-config '{"solution_config": "2wikimultihopqa", "solution_split": "train"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "gpt-oss-120b-high", "model_kwargs": {"config": "configs/vllm/gpt-oss-120b.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}, "reasoning_effort": "high"}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/2wikimultihopqa/train/Qwen3-0.6B"}'

# Qwen3-1.7B

uv run mbt \
    --task-name "qa.mbt_r" \
    --task-config '{"solution_config": "2wikimultihopqa", "solution_split": "train"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "gpt-oss-120b-high", "model_kwargs": {"config": "configs/vllm/gpt-oss-120b.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}, "reasoning_effort": "high"}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/2wikimultihopqa/train/Qwen3-1.7B"}'

# Qwen3-4B

uv run mbt \
    --task-name "qa.mbt_r" \
    --task-config '{"solution_config": "2wikimultihopqa", "solution_split": "train"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "gpt-oss-120b-high", "model_kwargs": {"config": "configs/vllm/gpt-oss-120b.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}, "reasoning_effort": "high"}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/2wikimultihopqa/train/Qwen3-4B"}'

# gpt-oss-120b-high (self-refine)

uv run mbt \
    --task-name "qa.mbt_r" \
    --task-config '{"solution_config": "2wikimultihopqa", "solution_split": "train"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "gpt-oss-120b-high", "model_kwargs": {"config": "configs/vllm/gpt-oss-120b.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}, "reasoning_effort": "high"}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/2wikimultihopqa/train/gpt-oss-120b-high"}'

# <<< 2wikimultihopqa <<<

# >>> hotpotqa train mbt-r refinement >>>

# Qwen3-0.6B

uv run mbt \
    --task-name "qa.mbt_r" \
    --task-config '{"solution_config": "hotpotqa", "solution_split": "train"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "gpt-oss-120b-high", "model_kwargs": {"config": "configs/vllm/gpt-oss-120b.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}, "reasoning_effort": "high"}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/hotpotqa/train/Qwen3-0.6B"}'

# Qwen3-1.7B

uv run mbt \
    --task-name "qa.mbt_r" \
    --task-config '{"solution_config": "hotpotqa", "solution_split": "train"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "gpt-oss-120b-high", "model_kwargs": {"config": "configs/vllm/gpt-oss-120b.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}, "reasoning_effort": "high"}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/hotpotqa/train/Qwen3-1.7B"}'

# Qwen3-4B

uv run mbt \
    --task-name "qa.mbt_r" \
    --task-config '{"solution_config": "hotpotqa", "solution_split": "train"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "gpt-oss-120b-high", "model_kwargs": {"config": "configs/vllm/gpt-oss-120b.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}, "reasoning_effort": "high"}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/hotpotqa/train/Qwen3-4B"}'

# gpt-oss-120b-high (self-refine)

uv run mbt \
    --task-name "qa.mbt_r" \
    --task-config '{"solution_config": "hotpotqa", "solution_split": "train"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "gpt-oss-120b-high", "model_kwargs": {"config": "configs/vllm/gpt-oss-120b.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}, "reasoning_effort": "high"}, "num_threads": 1024}' \
    --script-config '{"root_dir": "output/hotpotqa/train/gpt-oss-120b-high"}'

# <<< hotpotqa <<<

echo "[summary] failed=$FAILED"
exit $FAILED
