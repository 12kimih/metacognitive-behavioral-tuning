#!/bin/bash
#SBATCH --job-name=debug                # Sets the name of the job.
#SBATCH --output=logs/%x_%j.log         # Specifies the file to which standard output and error are written. %x is the job name, %j is the job ID.
#SBATCH --open-mode=append              # Appends the output to the log file instead of overwriting it.
#SBATCH --partition=                    # Requests a specific partition (queue) for the job.
#SBATCH --qos=                          # Requests a specific Quality of Service for the job.
#SBATCH --nodes=1                       # Requests a specific number of nodes.
#SBATCH --ntasks-per-node=1             # Specifies the number of tasks to be launched on each node.
#SBATCH --cpus-per-task=                # Requests the number of CPU cores allocated to each task.
#SBATCH --gres=                         # Requests generic resources (e.g. gpu:N).
#SBATCH --time=                         # Sets a time limit for the job (D-HH:MM:SS).

set -x -e

((SLURM_GPUS_ON_NODE > 0)) && export OMP_NUM_THREADS=$((SLURM_CPUS_ON_NODE / SLURM_GPUS_ON_NODE)) || export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MASTER_PORT=$(shuf -i 49152-65535 -n 1)

if [ -d /etc/profile.d ]; then
  for i in /etc/profile.d/*.sh; do
    if [ -r $i ]; then
      . $i
    fi
  done
  unset i
fi

module load cuda/12.8.1-none-none-hpqvk2p

echo "START TIME: $(date)"

srun uv run vllm serve openai/gpt-oss-120b \
    --config configs/vllm/gpt-oss-120b.yaml \
    --host 0.0.0.0 \
    --port $MASTER_PORT \
    --tensor-parallel-size $SLURM_GPUS_ON_NODE \
    --pipeline-parallel-size 1 \
    --data-parallel-size 1 \
    --async-scheduling \
    --override-generation-config '{"max_new_tokens": null}'

echo "END TIME: $(date)"
