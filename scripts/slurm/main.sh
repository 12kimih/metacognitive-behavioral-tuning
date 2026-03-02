# Qwen3-0.6B
sbatch --cpus-per-task=32 --gres=gpu:4 scripts/slurm/main.slurm \
    --task-name "hotpotqa" \
    --task-config '{"dataset_split": "train"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-0.6B", "model_kwargs": {"model": "Qwen/Qwen3-0.6B", "config": "configs/vllm/defaults.yaml", "max_model_len": 40960, "tensor_parallel_size": 4}, "request_kwargs": {"max_completion_tokens": 32768, "temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 32, "num_shards": 32, "strict_validation": false}' \
    --script-config '{"root_dir": "output/hotpotqa/train"}'

sbatch --cpus-per-task=32 --gres=gpu:4 scripts/slurm/main.slurm \
    --task-name "musique" \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-0.6B", "model_kwargs": {"model": "Qwen/Qwen3-0.6B", "config": "configs/vllm/defaults.yaml", "max_model_len": 40960, "tensor_parallel_size": 4}, "request_kwargs": {"max_completion_tokens": 32768, "temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 32, "num_shards": 32, "validation_max_retries": 0, "strict_validation": false}' \
    --script-config '{"root_dir": "output/musique/validation"}'

sbatch --cpus-per-task=32 --gres=gpu:4 scripts/slurm/main.slurm \
    --task-name "2wikimultihopqa" \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-0.6B", "model_kwargs": {"model": "Qwen/Qwen3-0.6B", "config": "configs/vllm/defaults.yaml", "max_model_len": 40960, "tensor_parallel_size": 4}, "request_kwargs": {"max_completion_tokens": 32768, "temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 32, "num_shards": 32, "validation_max_retries": 0, "strict_validation": false}' \
    --script-config '{"root_dir": "output/2wikimultihopqa/validation"}'

sbatch --cpus-per-task=32 --gres=gpu:4 scripts/slurm/main.slurm \
    --task-name "hotpotqa" \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-0.6B", "model_kwargs": {"model": "Qwen/Qwen3-0.6B", "config": "configs/vllm/defaults.yaml", "max_model_len": 40960, "tensor_parallel_size": 4}, "request_kwargs": {"max_completion_tokens": 32768, "temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 32, "num_shards": 32, "validation_max_retries": 0, "strict_validation": false}' \
    --script-config '{"root_dir": "output/hotpotqa/validation"}'

# Qwen3-1.7B
sbatch --cpus-per-task=32 --gres=gpu:4 scripts/slurm/main.slurm \
    --task-name "hotpotqa" \
    --task-config '{"dataset_split": "train"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-1.7B", "model_kwargs": {"model": "Qwen/Qwen3-1.7B", "config": "configs/vllm/defaults.yaml", "max_model_len": 40960, "tensor_parallel_size": 4}, "request_kwargs": {"max_completion_tokens": 32768, "temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 32, "num_shards": 32, "strict_validation": false}' \
    --script-config '{"root_dir": "output/hotpotqa/train"}'

sbatch --cpus-per-task=32 --gres=gpu:4 scripts/slurm/main.slurm \
    --task-name "musique" \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-1.7B", "model_kwargs": {"model": "Qwen/Qwen3-1.7B", "config": "configs/vllm/defaults.yaml", "max_model_len": 40960, "tensor_parallel_size": 4}, "request_kwargs": {"max_completion_tokens": 32768, "temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 32, "num_shards": 32, "validation_max_retries": 0, "strict_validation": false}' \
    --script-config '{"root_dir": "output/musique/validation"}'

sbatch --cpus-per-task=32 --gres=gpu:4 scripts/slurm/main.slurm \
    --task-name "2wikimultihopqa" \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-1.7B", "model_kwargs": {"model": "Qwen/Qwen3-1.7B", "config": "configs/vllm/defaults.yaml", "max_model_len": 40960, "tensor_parallel_size": 4}, "request_kwargs": {"max_completion_tokens": 32768, "temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 32, "num_shards": 32, "validation_max_retries": 0, "strict_validation": false}' \
    --script-config '{"root_dir": "output/2wikimultihopqa/validation"}'

sbatch --cpus-per-task=32 --gres=gpu:4 scripts/slurm/main.slurm \
    --task-name "hotpotqa" \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-1.7B", "model_kwargs": {"model": "Qwen/Qwen3-1.7B", "config": "configs/vllm/defaults.yaml", "max_model_len": 40960, "tensor_parallel_size": 4}, "request_kwargs": {"max_completion_tokens": 32768, "temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 32, "num_shards": 32, "validation_max_retries": 0, "strict_validation": false}' \
    --script-config '{"root_dir": "output/hotpotqa/validation"}'

# Qwen3-4B
sbatch --cpus-per-task=32 --gres=gpu:4 scripts/slurm/main.slurm \
    --task-name "hotpotqa" \
    --task-config '{"dataset_split": "train"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-4B", "model_kwargs": {"model": "Qwen/Qwen3-4B", "config": "configs/vllm/defaults.yaml", "max_model_len": 40960, "tensor_parallel_size": 4}, "request_kwargs": {"max_completion_tokens": 32768, "temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 32, "num_shards": 32, "strict_validation": false}' \
    --script-config '{"root_dir": "output/hotpotqa/train"}'

sbatch --cpus-per-task=32 --gres=gpu:4 scripts/slurm/main.slurm \
    --task-name "musique" \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-4B", "model_kwargs": {"model": "Qwen/Qwen3-4B", "config": "configs/vllm/defaults.yaml", "max_model_len": 40960, "tensor_parallel_size": 4}, "request_kwargs": {"max_completion_tokens": 32768, "temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 32, "num_shards": 32, "validation_max_retries": 0, "strict_validation": false}' \
    --script-config '{"root_dir": "output/musique/validation"}'

sbatch --cpus-per-task=32 --gres=gpu:4 scripts/slurm/main.slurm \
    --task-name "2wikimultihopqa" \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-4B", "model_kwargs": {"model": "Qwen/Qwen3-4B", "config": "configs/vllm/defaults.yaml", "max_model_len": 40960, "tensor_parallel_size": 4}, "request_kwargs": {"max_completion_tokens": 32768, "temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 32, "num_shards": 32, "validation_max_retries": 0, "strict_validation": false}' \
    --script-config '{"root_dir": "output/2wikimultihopqa/validation"}'

sbatch --cpus-per-task=32 --gres=gpu:4 scripts/slurm/main.slurm \
    --task-name "hotpotqa" \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-4B", "model_kwargs": {"model": "Qwen/Qwen3-4B", "config": "configs/vllm/defaults.yaml", "max_model_len": 40960, "tensor_parallel_size": 4}, "request_kwargs": {"max_completion_tokens": 32768, "temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 32, "num_shards": 32, "validation_max_retries": 0, "strict_validation": false}' \
    --script-config '{"root_dir": "output/hotpotqa/validation"}'

# gpt-oss-120b
sbatch --cpus-per-task=32 --gres=gpu:4 scripts/slurm/main.slurm \
    --task-name "hotpotqa" \
    --task-config '{"dataset_split": "train"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "gpt-oss-120b-high", "model_kwargs": {"model": "openai/gpt-oss-120b", "config": "configs/vllm/gpt-oss.yaml", "max_model_len": 40960, "tensor_parallel_size": 4}, "request_kwargs": {"max_completion_tokens": 32768, "temperature": 0.6, "top_p": 0.95, "n": 1, "reasoning_effort": "high", "extra_body": {"top_k": 20}}, "num_threads": 32, "num_shards": 32, "apply_think_format": true, "strict_validation": false}' \
    --script-config '{"root_dir": "output/hotpotqa/train"}'

sbatch --cpus-per-task=32 --gres=gpu:4 scripts/slurm/main.slurm \
    --task-name "musique" \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "gpt-oss-120b-high", "model_kwargs": {"model": "openai/gpt-oss-120b", "config": "configs/vllm/gpt-oss.yaml", "max_model_len": 40960, "tensor_parallel_size": 4}, "request_kwargs": {"max_completion_tokens": 32768, "temperature": 0.6, "top_p": 0.95, "n": 1, "reasoning_effort": "high", "extra_body": {"top_k": 20}}, "num_threads": 32, "num_shards": 32, "apply_think_format": true, "validation_max_retries": 0, "strict_validation": false}' \
    --script-config '{"root_dir": "output/musique/validation"}'

sbatch --cpus-per-task=32 --gres=gpu:4 scripts/slurm/main.slurm \
    --task-name "2wikimultihopqa" \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "gpt-oss-120b-high", "model_kwargs": {"model": "openai/gpt-oss-120b", "config": "configs/vllm/gpt-oss.yaml", "max_model_len": 40960, "tensor_parallel_size": 4}, "request_kwargs": {"max_completion_tokens": 32768, "temperature": 0.6, "top_p": 0.95, "n": 1, "reasoning_effort": "high", "extra_body": {"top_k": 20}}, "num_threads": 32, "num_shards": 32, "apply_think_format": true, "validation_max_retries": 0, "strict_validation": false}' \
    --script-config '{"root_dir": "output/2wikimultihopqa/validation"}'

sbatch --cpus-per-task=32 --gres=gpu:4 scripts/slurm/main.slurm \
    --task-name "hotpotqa" \
    --task-config '{"dataset_split": "validation"}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "gpt-oss-120b-high", "model_kwargs": {"model": "openai/gpt-oss-120b", "config": "configs/vllm/gpt-oss.yaml", "max_model_len": 40960, "tensor_parallel_size": 4}, "request_kwargs": {"max_completion_tokens": 32768, "temperature": 0.6, "top_p": 0.95, "n": 1, "reasoning_effort": "high", "extra_body": {"top_k": 20}}, "num_threads": 32, "num_shards": 32, "apply_think_format": true, "validation_max_retries": 0, "strict_validation": false}' \
    --script-config '{"root_dir": "output/hotpotqa/validation"}'

# metacognitive-prompt
sbatch --cpus-per-task=32 --gres=gpu:4 scripts/slurm/main.slurm \
    --task-name "hotpotqa" \
    --task-config '{"dataset_split": "validation", "metacognitive_prompt": true}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "Qwen3-4B/metacognitive-prompt", "model_kwargs": {"model": "Qwen/Qwen3-4B", "config": "configs/vllm/defaults.yaml", "max_model_len": 40960, "tensor_parallel_size": 4}, "request_kwargs": {"max_completion_tokens": 32768, "temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}}, "num_threads": 32, "num_shards": 32, "validation_max_retries": 0, "strict_validation": false}' \
    --script-config '{"root_dir": "output/hotpotqa/validation"}'

# solution
sbatch --cpus-per-task=32 --gres=gpu:4 scripts/slurm/main.slurm \
    --task-name "hotpotqa" \
    --task-config '{"dataset_split": "train", "solution": true}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "gpt-oss-120b-high", "model_kwargs": {"model": "openai/gpt-oss-120b", "config": "configs/vllm/gpt-oss.yaml", "max_model_len": 40960, "tensor_parallel_size": 4}, "request_kwargs": {"max_completion_tokens": 32768, "temperature": 0.6, "top_p": 0.95, "n": 1, "reasoning_effort": "high", "extra_body": {"top_k": 20}}, "num_threads": 32, "num_shards": 32}' \
    --script-config '{"root_dir": "output/hotpotqa/train"}'

# mbt-s
sbatch --cpus-per-task=32 --gres=gpu:4 scripts/slurm/main.slurm \
    --task-name "hotpotqa" \
    --task-config '{"dataset_split": "train", "mbt_s": true}' \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "gpt-oss-120b-high", "model_kwargs": {"model": "openai/gpt-oss-120b", "config": "configs/vllm/gpt-oss.yaml", "max_model_len": 40960, "tensor_parallel_size": 4}, "request_kwargs": {"max_completion_tokens": 32768, "temperature": 0.6, "top_p": 0.95, "n": 1, "reasoning_effort": "high", "extra_body": {"top_k": 20}}, "num_threads": 32, "num_shards": 32}' \
    --script-config '{"root_dir": "output/hotpotqa/train"}'

# evaluation
sbatch --cpus-per-task=32 --gres=gpu:4 scripts/slurm/main.slurm \
    --task-name "qa.evaluation" \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "gpt-oss-120b-high", "model_kwargs": {"model": "openai/gpt-oss-120b", "config": "configs/vllm/gpt-oss.yaml", "max_model_len": 40960, "tensor_parallel_size": 4}, "request_kwargs": {"max_completion_tokens": 32768, "temperature": 0.6, "top_p": 0.95, "n": 1, "reasoning_effort": "high", "extra_body": {"top_k": 20}}, "num_threads": 32, "num_shards": 32}' \
    --script-config '{"root_dir": "output/musique/validation/Qwen3-4B"}'

sbatch --cpus-per-task=32 --gres=gpu:4 scripts/slurm/main.slurm \
    --task-name "qa.evaluation" \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "gpt-oss-120b-high", "model_kwargs": {"model": "openai/gpt-oss-120b", "config": "configs/vllm/gpt-oss.yaml", "max_model_len": 40960, "tensor_parallel_size": 4}, "request_kwargs": {"max_completion_tokens": 32768, "temperature": 0.6, "top_p": 0.95, "n": 1, "reasoning_effort": "high", "extra_body": {"top_k": 20}}, "num_threads": 32, "num_shards": 32}' \
    --script-config '{"root_dir": "output/2wikimultihopqa/validation/Qwen3-4B"}'

sbatch --cpus-per-task=32 --gres=gpu:4 scripts/slurm/main.slurm \
    --task-name "qa.evaluation" \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "gpt-oss-120b-high", "model_kwargs": {"model": "openai/gpt-oss-120b", "config": "configs/vllm/gpt-oss.yaml", "max_model_len": 40960, "tensor_parallel_size": 4}, "request_kwargs": {"max_completion_tokens": 32768, "temperature": 0.6, "top_p": 0.95, "n": 1, "reasoning_effort": "high", "extra_body": {"top_k": 20}}, "num_threads": 32, "num_shards": 32}' \
    --script-config '{"root_dir": "output/hotpotqa/validation/Qwen3-4B"}'

# answer-hit
sbatch --cpus-per-task=32 --gres=gpu:4 scripts/slurm/main.slurm \
    --task-name "qa.answer_hit" \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "gpt-oss-120b-high", "model_kwargs": {"model": "openai/gpt-oss-120b", "config": "configs/vllm/gpt-oss.yaml", "max_model_len": 40960, "tensor_parallel_size": 4}, "request_kwargs": {"max_completion_tokens": 32768, "temperature": 0.6, "top_p": 0.95, "n": 1, "reasoning_effort": "high", "extra_body": {"top_k": 20}}, "num_threads": 32, "num_shards": 32}' \
    --script-config '{"root_dir": "output/hotpotqa/validation/Qwen3-4B"}'

# mbt-r
sbatch --cpus-per-task=32 --gres=gpu:4 scripts/slurm/main.slurm \
    --task-name "qa.mbt_r" \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "gpt-oss-120b-high", "model_kwargs": {"model": "openai/gpt-oss-120b", "config": "configs/vllm/gpt-oss.yaml", "max_model_len": 40960, "tensor_parallel_size": 4}, "request_kwargs": {"max_completion_tokens": 32768, "temperature": 0.6, "top_p": 0.95, "n": 1, "reasoning_effort": "high", "extra_body": {"top_k": 20}}, "num_threads": 32, "num_shards": 32}' \
    --script-config '{"root_dir": "output/hotpotqa/train/Qwen3-4B"}'

# overthinking-score
sbatch --cpus-per-task=32 --gres=gpu:4 scripts/slurm/main.slurm \
    --task-name "qa.overthinking_score" \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "gpt-oss-120b-high", "model_kwargs": {"model": "openai/gpt-oss-120b", "config": "configs/vllm/gpt-oss.yaml", "max_model_len": 40960, "tensor_parallel_size": 4}, "request_kwargs": {"max_completion_tokens": 32768, "temperature": 0.6, "top_p": 0.95, "n": 1, "reasoning_effort": "high", "extra_body": {"top_k": 20}}, "num_threads": 32, "num_shards": 32, "allow_none": true}' \
    --script-config '{"root_dir": "output/musique/validation/Qwen3-4B"}'

# underthinking-score
sbatch --cpus-per-task=32 --gres=gpu:4 scripts/slurm/main.slurm \
    --task-name "qa.underthinking_score" \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "gpt-oss-120b-high", "model_kwargs": {"model": "openai/gpt-oss-120b", "config": "configs/vllm/gpt-oss.yaml", "max_model_len": 40960, "tensor_parallel_size": 4}, "request_kwargs": {"max_completion_tokens": 32768, "temperature": 0.6, "top_p": 0.95, "n": 1, "reasoning_effort": "high", "extra_body": {"top_k": 20}}, "num_threads": 32, "num_shards": 32, "allow_none": true}' \
    --script-config '{"root_dir": "output/musique/validation/Qwen3-4B"}'

# metacognition-score
sbatch --cpus-per-task=32 --gres=gpu:4 scripts/slurm/main.slurm \
    --task-name "qa.metacognition_score" \
    --api-name "vllm.chat" \
    --api-config '{"model_name": "gpt-oss-120b-high", "model_kwargs": {"model": "openai/gpt-oss-120b", "config": "configs/vllm/gpt-oss.yaml", "max_model_len": 40960, "tensor_parallel_size": 4}, "request_kwargs": {"max_completion_tokens": 32768, "temperature": 0.6, "top_p": 0.95, "n": 1, "reasoning_effort": "high", "extra_body": {"top_k": 20}}, "num_threads": 32, "num_shards": 32, "allow_none": true}' \
    --script-config '{"root_dir": "output/musique/validation/Qwen3-4B"}'
