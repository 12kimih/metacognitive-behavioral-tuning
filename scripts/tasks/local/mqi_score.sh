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

# qa.mqi_score (MQI) across the full paper-table model matrix on QA validation.
# Run scripts/tasks/build_difficulty.sh first (writes data/sample_difficulty.csv).
# Emits: l_obs (0..5) + phases (list[int] in {1..5}) + confidence (float in [0,1]).
# Confidence is stored only — never multiplied into per-sample MQI; used at dataset level via
# M_K = γ((c_i, correct_i)) (Goodman-Kruskal). See docs/scoring_redesign_marker_variant.md §2.5.
# Judge: gemma-4-31b-it.

# >>> musique >>>

# Qwen3-0.6B base

uv run mbt \
    --task-name qa.mqi_score \
    --task-config '{"benchmark": "musique", "difficulty_csv": "data/sample_difficulty.csv"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "gemma-4-31b-it", "model_kwargs": {"config": "configs/vllm/gemma-4-31b-it.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024, "client_timeout": 120}' \
    --script-config '{"root_dir": "output/musique/validation/Qwen3-0.6B"}'

# Qwen3-0.6B prompt

uv run mbt \
    --task-name qa.mqi_score \
    --task-config '{"benchmark": "musique", "difficulty_csv": "data/sample_difficulty.csv"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "gemma-4-31b-it", "model_kwargs": {"config": "configs/vllm/gemma-4-31b-it.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024, "client_timeout": 120}' \
    --script-config '{"root_dir": "output/musique/validation/Qwen3-0.6B/metacognitive-prompt"}'

# Qwen3-0.6B grpo

uv run mbt \
    --task-name qa.mqi_score \
    --task-config '{"benchmark": "musique", "difficulty_csv": "data/sample_difficulty.csv"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "gemma-4-31b-it", "model_kwargs": {"config": "configs/vllm/gemma-4-31b-it.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024, "client_timeout": 120}' \
    --script-config '{"root_dir": "output/musique/validation/Qwen3-0.6B/grpo"}'

# Qwen3-0.6B self-distill,sft

uv run mbt \
    --task-name qa.mqi_score \
    --task-config '{"benchmark": "musique", "difficulty_csv": "data/sample_difficulty.csv"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "gemma-4-31b-it", "model_kwargs": {"config": "configs/vllm/gemma-4-31b-it.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024, "client_timeout": 120}' \
    --script-config '{"root_dir": "output/musique/validation/Qwen3-0.6B/self-distill/sft/1e-4/128"}'

# Qwen3-0.6B gpt-oss-distill,sft

uv run mbt \
    --task-name qa.mqi_score \
    --task-config '{"benchmark": "musique", "difficulty_csv": "data/sample_difficulty.csv"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "gemma-4-31b-it", "model_kwargs": {"config": "configs/vllm/gemma-4-31b-it.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024, "client_timeout": 120}' \
    --script-config '{"root_dir": "output/musique/validation/Qwen3-0.6B/gpt-oss-distill/sft/1e-4/128"}'

# Qwen3-0.6B gpt-oss-distill,grpo

uv run mbt \
    --task-name qa.mqi_score \
    --task-config '{"benchmark": "musique", "difficulty_csv": "data/sample_difficulty.csv"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "gemma-4-31b-it", "model_kwargs": {"config": "configs/vllm/gemma-4-31b-it.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024, "client_timeout": 120}' \
    --script-config '{"root_dir": "output/musique/validation/Qwen3-0.6B/gpt-oss-distill/grpo"}'

# Qwen3-0.6B mbt-s,sft

uv run mbt \
    --task-name qa.mqi_score \
    --task-config '{"benchmark": "musique", "difficulty_csv": "data/sample_difficulty.csv"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "gemma-4-31b-it", "model_kwargs": {"config": "configs/vllm/gemma-4-31b-it.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024, "client_timeout": 120}' \
    --script-config '{"root_dir": "output/musique/validation/Qwen3-0.6B/mbt-s/sft/1e-4/128"}'

# Qwen3-0.6B mbt-s,grpo

uv run mbt \
    --task-name qa.mqi_score \
    --task-config '{"benchmark": "musique", "difficulty_csv": "data/sample_difficulty.csv"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "gemma-4-31b-it", "model_kwargs": {"config": "configs/vllm/gemma-4-31b-it.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024, "client_timeout": 120}' \
    --script-config '{"root_dir": "output/musique/validation/Qwen3-0.6B/mbt-s/grpo"}'

# Qwen3-0.6B mbt-r,sft

uv run mbt \
    --task-name qa.mqi_score \
    --task-config '{"benchmark": "musique", "difficulty_csv": "data/sample_difficulty.csv"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "gemma-4-31b-it", "model_kwargs": {"config": "configs/vllm/gemma-4-31b-it.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024, "client_timeout": 120}' \
    --script-config '{"root_dir": "output/musique/validation/Qwen3-0.6B/mbt-r/sft/1e-4/128"}'

# Qwen3-0.6B mbt-r,grpo

uv run mbt \
    --task-name qa.mqi_score \
    --task-config '{"benchmark": "musique", "difficulty_csv": "data/sample_difficulty.csv"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "gemma-4-31b-it", "model_kwargs": {"config": "configs/vllm/gemma-4-31b-it.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024, "client_timeout": 120}' \
    --script-config '{"root_dir": "output/musique/validation/Qwen3-0.6B/mbt-r/grpo"}'

# Qwen3-0.6B distill-r,sft

uv run mbt \
    --task-name qa.mqi_score \
    --task-config '{"benchmark": "musique", "difficulty_csv": "data/sample_difficulty.csv"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "gemma-4-31b-it", "model_kwargs": {"config": "configs/vllm/gemma-4-31b-it.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024, "client_timeout": 120}' \
    --script-config '{"root_dir": "output/musique/validation/Qwen3-0.6B/distill-r/sft/1e-4/128"}'

# Qwen3-0.6B direct-r,sft

uv run mbt \
    --task-name qa.mqi_score \
    --task-config '{"benchmark": "musique", "difficulty_csv": "data/sample_difficulty.csv"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "gemma-4-31b-it", "model_kwargs": {"config": "configs/vllm/gemma-4-31b-it.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024, "client_timeout": 120}' \
    --script-config '{"root_dir": "output/musique/validation/Qwen3-0.6B/direct-r/sft/1e-4/128"}'

# Qwen3-0.6B shorter-better

uv run mbt \
    --task-name qa.mqi_score \
    --task-config '{"benchmark": "musique", "difficulty_csv": "data/sample_difficulty.csv"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "gemma-4-31b-it", "model_kwargs": {"config": "configs/vllm/gemma-4-31b-it.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024, "client_timeout": 120}' \
    --script-config '{"root_dir": "output/musique/validation/Qwen3-0.6B/shorter-better"}'

# Qwen3-0.6B token-skip

uv run mbt \
    --task-name qa.mqi_score \
    --task-config '{"benchmark": "musique", "difficulty_csv": "data/sample_difficulty.csv"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "gemma-4-31b-it", "model_kwargs": {"config": "configs/vllm/gemma-4-31b-it.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024, "client_timeout": 120}' \
    --script-config '{"root_dir": "output/musique/validation/Qwen3-0.6B/token-skip"}'

# Qwen3-0.6B limopro

uv run mbt \
    --task-name qa.mqi_score \
    --task-config '{"benchmark": "musique", "difficulty_csv": "data/sample_difficulty.csv"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "gemma-4-31b-it", "model_kwargs": {"config": "configs/vllm/gemma-4-31b-it.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024, "client_timeout": 120}' \
    --script-config '{"root_dir": "output/musique/validation/Qwen3-0.6B/limopro"}'

# Qwen3-1.7B base

uv run mbt \
    --task-name qa.mqi_score \
    --task-config '{"benchmark": "musique", "difficulty_csv": "data/sample_difficulty.csv"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "gemma-4-31b-it", "model_kwargs": {"config": "configs/vllm/gemma-4-31b-it.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024, "client_timeout": 120}' \
    --script-config '{"root_dir": "output/musique/validation/Qwen3-1.7B"}'

# Qwen3-1.7B prompt

uv run mbt \
    --task-name qa.mqi_score \
    --task-config '{"benchmark": "musique", "difficulty_csv": "data/sample_difficulty.csv"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "gemma-4-31b-it", "model_kwargs": {"config": "configs/vllm/gemma-4-31b-it.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024, "client_timeout": 120}' \
    --script-config '{"root_dir": "output/musique/validation/Qwen3-1.7B/metacognitive-prompt"}'

# Qwen3-1.7B grpo

uv run mbt \
    --task-name qa.mqi_score \
    --task-config '{"benchmark": "musique", "difficulty_csv": "data/sample_difficulty.csv"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "gemma-4-31b-it", "model_kwargs": {"config": "configs/vllm/gemma-4-31b-it.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024, "client_timeout": 120}' \
    --script-config '{"root_dir": "output/musique/validation/Qwen3-1.7B/grpo"}'

# Qwen3-1.7B self-distill,sft

uv run mbt \
    --task-name qa.mqi_score \
    --task-config '{"benchmark": "musique", "difficulty_csv": "data/sample_difficulty.csv"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "gemma-4-31b-it", "model_kwargs": {"config": "configs/vllm/gemma-4-31b-it.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024, "client_timeout": 120}' \
    --script-config '{"root_dir": "output/musique/validation/Qwen3-1.7B/self-distill/sft/1e-4/128"}'

# Qwen3-1.7B gpt-oss-distill,sft

uv run mbt \
    --task-name qa.mqi_score \
    --task-config '{"benchmark": "musique", "difficulty_csv": "data/sample_difficulty.csv"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "gemma-4-31b-it", "model_kwargs": {"config": "configs/vllm/gemma-4-31b-it.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024, "client_timeout": 120}' \
    --script-config '{"root_dir": "output/musique/validation/Qwen3-1.7B/gpt-oss-distill/sft/1e-4/128"}'

# Qwen3-1.7B gpt-oss-distill,grpo

uv run mbt \
    --task-name qa.mqi_score \
    --task-config '{"benchmark": "musique", "difficulty_csv": "data/sample_difficulty.csv"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "gemma-4-31b-it", "model_kwargs": {"config": "configs/vllm/gemma-4-31b-it.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024, "client_timeout": 120}' \
    --script-config '{"root_dir": "output/musique/validation/Qwen3-1.7B/gpt-oss-distill/grpo"}'

# Qwen3-1.7B mbt-s,sft

uv run mbt \
    --task-name qa.mqi_score \
    --task-config '{"benchmark": "musique", "difficulty_csv": "data/sample_difficulty.csv"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "gemma-4-31b-it", "model_kwargs": {"config": "configs/vllm/gemma-4-31b-it.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024, "client_timeout": 120}' \
    --script-config '{"root_dir": "output/musique/validation/Qwen3-1.7B/mbt-s/sft/1e-4/128"}'

# Qwen3-1.7B mbt-s,grpo

uv run mbt \
    --task-name qa.mqi_score \
    --task-config '{"benchmark": "musique", "difficulty_csv": "data/sample_difficulty.csv"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "gemma-4-31b-it", "model_kwargs": {"config": "configs/vllm/gemma-4-31b-it.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024, "client_timeout": 120}' \
    --script-config '{"root_dir": "output/musique/validation/Qwen3-1.7B/mbt-s/grpo"}'

# Qwen3-1.7B mbt-r,sft

uv run mbt \
    --task-name qa.mqi_score \
    --task-config '{"benchmark": "musique", "difficulty_csv": "data/sample_difficulty.csv"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "gemma-4-31b-it", "model_kwargs": {"config": "configs/vllm/gemma-4-31b-it.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024, "client_timeout": 120}' \
    --script-config '{"root_dir": "output/musique/validation/Qwen3-1.7B/mbt-r/sft/1e-4/128"}'

# Qwen3-1.7B mbt-r,grpo

uv run mbt \
    --task-name qa.mqi_score \
    --task-config '{"benchmark": "musique", "difficulty_csv": "data/sample_difficulty.csv"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "gemma-4-31b-it", "model_kwargs": {"config": "configs/vllm/gemma-4-31b-it.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024, "client_timeout": 120}' \
    --script-config '{"root_dir": "output/musique/validation/Qwen3-1.7B/mbt-r/grpo"}'

# Qwen3-1.7B distill-r,sft

uv run mbt \
    --task-name qa.mqi_score \
    --task-config '{"benchmark": "musique", "difficulty_csv": "data/sample_difficulty.csv"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "gemma-4-31b-it", "model_kwargs": {"config": "configs/vllm/gemma-4-31b-it.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024, "client_timeout": 120}' \
    --script-config '{"root_dir": "output/musique/validation/Qwen3-1.7B/distill-r/sft/1e-4/128"}'

# Qwen3-1.7B direct-r,sft

uv run mbt \
    --task-name qa.mqi_score \
    --task-config '{"benchmark": "musique", "difficulty_csv": "data/sample_difficulty.csv"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "gemma-4-31b-it", "model_kwargs": {"config": "configs/vllm/gemma-4-31b-it.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024, "client_timeout": 120}' \
    --script-config '{"root_dir": "output/musique/validation/Qwen3-1.7B/direct-r/sft/1e-4/128"}'

# Qwen3-1.7B shorter-better

uv run mbt \
    --task-name qa.mqi_score \
    --task-config '{"benchmark": "musique", "difficulty_csv": "data/sample_difficulty.csv"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "gemma-4-31b-it", "model_kwargs": {"config": "configs/vllm/gemma-4-31b-it.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024, "client_timeout": 120}' \
    --script-config '{"root_dir": "output/musique/validation/Qwen3-1.7B/shorter-better"}'

# Qwen3-1.7B token-skip

uv run mbt \
    --task-name qa.mqi_score \
    --task-config '{"benchmark": "musique", "difficulty_csv": "data/sample_difficulty.csv"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "gemma-4-31b-it", "model_kwargs": {"config": "configs/vllm/gemma-4-31b-it.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024, "client_timeout": 120}' \
    --script-config '{"root_dir": "output/musique/validation/Qwen3-1.7B/token-skip"}'

# Qwen3-1.7B limopro

uv run mbt \
    --task-name qa.mqi_score \
    --task-config '{"benchmark": "musique", "difficulty_csv": "data/sample_difficulty.csv"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "gemma-4-31b-it", "model_kwargs": {"config": "configs/vllm/gemma-4-31b-it.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024, "client_timeout": 120}' \
    --script-config '{"root_dir": "output/musique/validation/Qwen3-1.7B/limopro"}'

# Qwen3-4B base

uv run mbt \
    --task-name qa.mqi_score \
    --task-config '{"benchmark": "musique", "difficulty_csv": "data/sample_difficulty.csv"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "gemma-4-31b-it", "model_kwargs": {"config": "configs/vllm/gemma-4-31b-it.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024, "client_timeout": 120}' \
    --script-config '{"root_dir": "output/musique/validation/Qwen3-4B"}'

# Qwen3-4B prompt

uv run mbt \
    --task-name qa.mqi_score \
    --task-config '{"benchmark": "musique", "difficulty_csv": "data/sample_difficulty.csv"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "gemma-4-31b-it", "model_kwargs": {"config": "configs/vllm/gemma-4-31b-it.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024, "client_timeout": 120}' \
    --script-config '{"root_dir": "output/musique/validation/Qwen3-4B/metacognitive-prompt"}'

# Qwen3-4B grpo

uv run mbt \
    --task-name qa.mqi_score \
    --task-config '{"benchmark": "musique", "difficulty_csv": "data/sample_difficulty.csv"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "gemma-4-31b-it", "model_kwargs": {"config": "configs/vllm/gemma-4-31b-it.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024, "client_timeout": 120}' \
    --script-config '{"root_dir": "output/musique/validation/Qwen3-4B/grpo"}'

# Qwen3-4B self-distill,sft

uv run mbt \
    --task-name qa.mqi_score \
    --task-config '{"benchmark": "musique", "difficulty_csv": "data/sample_difficulty.csv"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "gemma-4-31b-it", "model_kwargs": {"config": "configs/vllm/gemma-4-31b-it.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024, "client_timeout": 120}' \
    --script-config '{"root_dir": "output/musique/validation/Qwen3-4B/self-distill/sft/1e-4/128"}'

# Qwen3-4B gpt-oss-distill,sft

uv run mbt \
    --task-name qa.mqi_score \
    --task-config '{"benchmark": "musique", "difficulty_csv": "data/sample_difficulty.csv"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "gemma-4-31b-it", "model_kwargs": {"config": "configs/vllm/gemma-4-31b-it.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024, "client_timeout": 120}' \
    --script-config '{"root_dir": "output/musique/validation/Qwen3-4B/gpt-oss-distill/sft/1e-4/128"}'

# Qwen3-4B gpt-oss-distill,grpo

uv run mbt \
    --task-name qa.mqi_score \
    --task-config '{"benchmark": "musique", "difficulty_csv": "data/sample_difficulty.csv"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "gemma-4-31b-it", "model_kwargs": {"config": "configs/vllm/gemma-4-31b-it.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024, "client_timeout": 120}' \
    --script-config '{"root_dir": "output/musique/validation/Qwen3-4B/gpt-oss-distill/grpo"}'

# Qwen3-4B mbt-s,sft

uv run mbt \
    --task-name qa.mqi_score \
    --task-config '{"benchmark": "musique", "difficulty_csv": "data/sample_difficulty.csv"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "gemma-4-31b-it", "model_kwargs": {"config": "configs/vllm/gemma-4-31b-it.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024, "client_timeout": 120}' \
    --script-config '{"root_dir": "output/musique/validation/Qwen3-4B/mbt-s/sft/1e-4/128"}'

# Qwen3-4B mbt-s,grpo

uv run mbt \
    --task-name qa.mqi_score \
    --task-config '{"benchmark": "musique", "difficulty_csv": "data/sample_difficulty.csv"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "gemma-4-31b-it", "model_kwargs": {"config": "configs/vllm/gemma-4-31b-it.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024, "client_timeout": 120}' \
    --script-config '{"root_dir": "output/musique/validation/Qwen3-4B/mbt-s/grpo"}'

# Qwen3-4B mbt-r,sft

uv run mbt \
    --task-name qa.mqi_score \
    --task-config '{"benchmark": "musique", "difficulty_csv": "data/sample_difficulty.csv"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "gemma-4-31b-it", "model_kwargs": {"config": "configs/vllm/gemma-4-31b-it.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024, "client_timeout": 120}' \
    --script-config '{"root_dir": "output/musique/validation/Qwen3-4B/mbt-r/sft/1e-4/128"}'

# Qwen3-4B mbt-r,grpo

uv run mbt \
    --task-name qa.mqi_score \
    --task-config '{"benchmark": "musique", "difficulty_csv": "data/sample_difficulty.csv"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "gemma-4-31b-it", "model_kwargs": {"config": "configs/vllm/gemma-4-31b-it.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024, "client_timeout": 120}' \
    --script-config '{"root_dir": "output/musique/validation/Qwen3-4B/mbt-r/grpo"}'

# Qwen3-4B distill-r,sft

uv run mbt \
    --task-name qa.mqi_score \
    --task-config '{"benchmark": "musique", "difficulty_csv": "data/sample_difficulty.csv"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "gemma-4-31b-it", "model_kwargs": {"config": "configs/vllm/gemma-4-31b-it.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024, "client_timeout": 120}' \
    --script-config '{"root_dir": "output/musique/validation/Qwen3-4B/distill-r/sft/1e-4/128"}'

# Qwen3-4B direct-r,sft

uv run mbt \
    --task-name qa.mqi_score \
    --task-config '{"benchmark": "musique", "difficulty_csv": "data/sample_difficulty.csv"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "gemma-4-31b-it", "model_kwargs": {"config": "configs/vllm/gemma-4-31b-it.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024, "client_timeout": 120}' \
    --script-config '{"root_dir": "output/musique/validation/Qwen3-4B/direct-r/sft/1e-4/128"}'

# Qwen3-4B shorter-better

uv run mbt \
    --task-name qa.mqi_score \
    --task-config '{"benchmark": "musique", "difficulty_csv": "data/sample_difficulty.csv"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "gemma-4-31b-it", "model_kwargs": {"config": "configs/vllm/gemma-4-31b-it.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024, "client_timeout": 120}' \
    --script-config '{"root_dir": "output/musique/validation/Qwen3-4B/shorter-better"}'

# Qwen3-4B token-skip

uv run mbt \
    --task-name qa.mqi_score \
    --task-config '{"benchmark": "musique", "difficulty_csv": "data/sample_difficulty.csv"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "gemma-4-31b-it", "model_kwargs": {"config": "configs/vllm/gemma-4-31b-it.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024, "client_timeout": 120}' \
    --script-config '{"root_dir": "output/musique/validation/Qwen3-4B/token-skip"}'

# Qwen3-4B limopro

uv run mbt \
    --task-name qa.mqi_score \
    --task-config '{"benchmark": "musique", "difficulty_csv": "data/sample_difficulty.csv"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "gemma-4-31b-it", "model_kwargs": {"config": "configs/vllm/gemma-4-31b-it.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024, "client_timeout": 120}' \
    --script-config '{"root_dir": "output/musique/validation/Qwen3-4B/limopro"}'

# gpt-oss-120b-high base

uv run mbt \
    --task-name qa.mqi_score \
    --task-config '{"benchmark": "musique", "difficulty_csv": "data/sample_difficulty.csv"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "gemma-4-31b-it", "model_kwargs": {"config": "configs/vllm/gemma-4-31b-it.yaml"}, "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 1024, "client_timeout": 120}' \
    --script-config '{"root_dir": "output/musique/validation/gpt-oss-120b-high"}'

# <<< musique <<<

echo "[summary] failed=$FAILED"
exit $FAILED
