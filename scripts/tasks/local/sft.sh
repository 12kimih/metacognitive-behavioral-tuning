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

# SFT training across the 7 paper-table modes × Qwen3-{0.6B,1.7B,4B}.
# Trained on hotpotqa data (per the original paper convention) and evaluated on
# all 3 QA validation sets via the rollout/evaluation/score scripts. Output:
# output/train/<model>/<mode>/sft/1e-4/128/. Effective batch = 2*16 = 32, lr=1e-4.

# >>> self-distill >>>

# Qwen3-0.6B

uv run accelerate launch --config_file configs/accelerate/multi_gpu.yaml --main_process_port $(shuf -i 49152-65535 -n 1) src/mbt/train/sft.py \
    --config configs/sft.yaml \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --dataset_name metacognitive-behavioral-tuning/rollouts-hotpotqa \
    --dataset_config Qwen3-0.6B \
    --wandb_tags Qwen3-0.6B,self-distill,sft,1e-4,128 \
    --output_dir output/train/Qwen3-0.6B/self-distill/sft/1e-4/128 \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --mode distill

# Qwen3-1.7B

uv run accelerate launch --config_file configs/accelerate/multi_gpu.yaml --main_process_port $(shuf -i 49152-65535 -n 1) src/mbt/train/sft.py \
    --config configs/sft.yaml \
    --model_name_or_path Qwen/Qwen3-1.7B \
    --dataset_name metacognitive-behavioral-tuning/rollouts-hotpotqa \
    --dataset_config Qwen3-1.7B \
    --wandb_tags Qwen3-1.7B,self-distill,sft,1e-4,128 \
    --output_dir output/train/Qwen3-1.7B/self-distill/sft/1e-4/128 \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --mode distill

# Qwen3-4B

uv run accelerate launch --config_file configs/accelerate/multi_gpu.yaml --main_process_port $(shuf -i 49152-65535 -n 1) src/mbt/train/sft.py \
    --config configs/sft.yaml \
    --model_name_or_path Qwen/Qwen3-4B \
    --dataset_name metacognitive-behavioral-tuning/rollouts-hotpotqa \
    --dataset_config Qwen3-4B \
    --wandb_tags Qwen3-4B,self-distill,sft,1e-4,128 \
    --output_dir output/train/Qwen3-4B/self-distill/sft/1e-4/128 \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --mode distill

# <<< self-distill <<<

# >>> gpt-oss-distill >>>

# Qwen3-0.6B

uv run accelerate launch --config_file configs/accelerate/multi_gpu.yaml --main_process_port $(shuf -i 49152-65535 -n 1) src/mbt/train/sft.py \
    --config configs/sft.yaml \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --dataset_name metacognitive-behavioral-tuning/rollouts-hotpotqa \
    --dataset_config gpt-oss-120b-high \
    --wandb_tags Qwen3-0.6B,gpt-oss-distill,sft,1e-4,128 \
    --output_dir output/train/Qwen3-0.6B/gpt-oss-distill/sft/1e-4/128 \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --mode distill

# Qwen3-1.7B

uv run accelerate launch --config_file configs/accelerate/multi_gpu.yaml --main_process_port $(shuf -i 49152-65535 -n 1) src/mbt/train/sft.py \
    --config configs/sft.yaml \
    --model_name_or_path Qwen/Qwen3-1.7B \
    --dataset_name metacognitive-behavioral-tuning/rollouts-hotpotqa \
    --dataset_config gpt-oss-120b-high \
    --wandb_tags Qwen3-1.7B,gpt-oss-distill,sft,1e-4,128 \
    --output_dir output/train/Qwen3-1.7B/gpt-oss-distill/sft/1e-4/128 \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --mode distill

# Qwen3-4B

uv run accelerate launch --config_file configs/accelerate/multi_gpu.yaml --main_process_port $(shuf -i 49152-65535 -n 1) src/mbt/train/sft.py \
    --config configs/sft.yaml \
    --model_name_or_path Qwen/Qwen3-4B \
    --dataset_name metacognitive-behavioral-tuning/rollouts-hotpotqa \
    --dataset_config gpt-oss-120b-high \
    --wandb_tags Qwen3-4B,gpt-oss-distill,sft,1e-4,128 \
    --output_dir output/train/Qwen3-4B/gpt-oss-distill/sft/1e-4/128 \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --mode distill

# <<< gpt-oss-distill <<<

# >>> mbt-s >>>

# Qwen3-0.6B

uv run accelerate launch --config_file configs/accelerate/multi_gpu.yaml --main_process_port $(shuf -i 49152-65535 -n 1) src/mbt/train/sft.py \
    --config configs/sft.yaml \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --dataset_name metacognitive-behavioral-tuning/mbt-s-gpt-oss-120b \
    --dataset_config hotpotqa \
    --wandb_tags Qwen3-0.6B,mbt-s,sft,1e-4,128 \
    --output_dir output/train/Qwen3-0.6B/mbt-s/sft/1e-4/128 \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --mode mbt-s

# Qwen3-1.7B

uv run accelerate launch --config_file configs/accelerate/multi_gpu.yaml --main_process_port $(shuf -i 49152-65535 -n 1) src/mbt/train/sft.py \
    --config configs/sft.yaml \
    --model_name_or_path Qwen/Qwen3-1.7B \
    --dataset_name metacognitive-behavioral-tuning/mbt-s-gpt-oss-120b \
    --dataset_config hotpotqa \
    --wandb_tags Qwen3-1.7B,mbt-s,sft,1e-4,128 \
    --output_dir output/train/Qwen3-1.7B/mbt-s/sft/1e-4/128 \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --mode mbt-s

# Qwen3-4B

uv run accelerate launch --config_file configs/accelerate/multi_gpu.yaml --main_process_port $(shuf -i 49152-65535 -n 1) src/mbt/train/sft.py \
    --config configs/sft.yaml \
    --model_name_or_path Qwen/Qwen3-4B \
    --dataset_name metacognitive-behavioral-tuning/mbt-s-gpt-oss-120b \
    --dataset_config hotpotqa \
    --wandb_tags Qwen3-4B,mbt-s,sft,1e-4,128 \
    --output_dir output/train/Qwen3-4B/mbt-s/sft/1e-4/128 \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --mode mbt-s

# <<< mbt-s <<<

# >>> mbt-r >>>

# Qwen3-0.6B

uv run accelerate launch --config_file configs/accelerate/multi_gpu.yaml --main_process_port $(shuf -i 49152-65535 -n 1) src/mbt/train/sft.py \
    --config configs/sft.yaml \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --dataset_name metacognitive-behavioral-tuning/mbt-r-hotpotqa \
    --dataset_config Qwen3-0.6B \
    --wandb_tags Qwen3-0.6B,mbt-r,sft,1e-4,128 \
    --output_dir output/train/Qwen3-0.6B/mbt-r/sft/1e-4/128 \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16

# Qwen3-1.7B

uv run accelerate launch --config_file configs/accelerate/multi_gpu.yaml --main_process_port $(shuf -i 49152-65535 -n 1) src/mbt/train/sft.py \
    --config configs/sft.yaml \
    --model_name_or_path Qwen/Qwen3-1.7B \
    --dataset_name metacognitive-behavioral-tuning/mbt-r-hotpotqa \
    --dataset_config Qwen3-1.7B \
    --wandb_tags Qwen3-1.7B,mbt-r,sft,1e-4,128 \
    --output_dir output/train/Qwen3-1.7B/mbt-r/sft/1e-4/128 \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16

# Qwen3-4B

uv run accelerate launch --config_file configs/accelerate/multi_gpu.yaml --main_process_port $(shuf -i 49152-65535 -n 1) src/mbt/train/sft.py \
    --config configs/sft.yaml \
    --model_name_or_path Qwen/Qwen3-4B \
    --dataset_name metacognitive-behavioral-tuning/mbt-r-hotpotqa \
    --dataset_config Qwen3-4B \
    --wandb_tags Qwen3-4B,mbt-r,sft,1e-4,128 \
    --output_dir output/train/Qwen3-4B/mbt-r/sft/1e-4/128 \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16

# <<< mbt-r <<<

# >>> distill-r >>>

# Qwen3-0.6B

uv run accelerate launch --config_file configs/accelerate/multi_gpu.yaml --main_process_port $(shuf -i 49152-65535 -n 1) src/mbt/train/sft.py \
    --config configs/sft.yaml \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --dataset_name metacognitive-behavioral-tuning/distill-r-hotpotqa \
    --dataset_config Qwen3-0.6B \
    --wandb_tags Qwen3-0.6B,distill-r,sft,1e-4,128 \
    --output_dir output/train/Qwen3-0.6B/distill-r/sft/1e-4/128 \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16

# Qwen3-1.7B

uv run accelerate launch --config_file configs/accelerate/multi_gpu.yaml --main_process_port $(shuf -i 49152-65535 -n 1) src/mbt/train/sft.py \
    --config configs/sft.yaml \
    --model_name_or_path Qwen/Qwen3-1.7B \
    --dataset_name metacognitive-behavioral-tuning/distill-r-hotpotqa \
    --dataset_config Qwen3-1.7B \
    --wandb_tags Qwen3-1.7B,distill-r,sft,1e-4,128 \
    --output_dir output/train/Qwen3-1.7B/distill-r/sft/1e-4/128 \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16

# Qwen3-4B

uv run accelerate launch --config_file configs/accelerate/multi_gpu.yaml --main_process_port $(shuf -i 49152-65535 -n 1) src/mbt/train/sft.py \
    --config configs/sft.yaml \
    --model_name_or_path Qwen/Qwen3-4B \
    --dataset_name metacognitive-behavioral-tuning/distill-r-hotpotqa \
    --dataset_config Qwen3-4B \
    --wandb_tags Qwen3-4B,distill-r,sft,1e-4,128 \
    --output_dir output/train/Qwen3-4B/distill-r/sft/1e-4/128 \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16

# <<< distill-r <<<

# >>> direct-r >>>

# Qwen3-0.6B

uv run accelerate launch --config_file configs/accelerate/multi_gpu.yaml --main_process_port $(shuf -i 49152-65535 -n 1) src/mbt/train/sft.py \
    --config configs/sft.yaml \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --dataset_name metacognitive-behavioral-tuning/mbt-r-hotpotqa \
    --dataset_config gpt-oss-120b-high \
    --wandb_tags Qwen3-0.6B,direct-r,sft,1e-4,128 \
    --output_dir output/train/Qwen3-0.6B/direct-r/sft/1e-4/128 \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16

# Qwen3-1.7B

uv run accelerate launch --config_file configs/accelerate/multi_gpu.yaml --main_process_port $(shuf -i 49152-65535 -n 1) src/mbt/train/sft.py \
    --config configs/sft.yaml \
    --model_name_or_path Qwen/Qwen3-1.7B \
    --dataset_name metacognitive-behavioral-tuning/mbt-r-hotpotqa \
    --dataset_config gpt-oss-120b-high \
    --wandb_tags Qwen3-1.7B,direct-r,sft,1e-4,128 \
    --output_dir output/train/Qwen3-1.7B/direct-r/sft/1e-4/128 \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16

# Qwen3-4B

uv run accelerate launch --config_file configs/accelerate/multi_gpu.yaml --main_process_port $(shuf -i 49152-65535 -n 1) src/mbt/train/sft.py \
    --config configs/sft.yaml \
    --model_name_or_path Qwen/Qwen3-4B \
    --dataset_name metacognitive-behavioral-tuning/mbt-r-hotpotqa \
    --dataset_config gpt-oss-120b-high \
    --wandb_tags Qwen3-4B,direct-r,sft,1e-4,128 \
    --output_dir output/train/Qwen3-4B/direct-r/sft/1e-4/128 \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16

# <<< direct-r <<<

echo "[summary] failed=$FAILED"
exit $FAILED
