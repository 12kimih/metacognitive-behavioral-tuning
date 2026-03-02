#!/bin/bash
#SBATCH --job-name=sft                  # Sets the name of the job.
#SBATCH --output=logs/%x_%j.log         # Specifies the file to which standard output and error are written. %x is the job name, %j is the job ID.
#SBATCH --open-mode=append              # Appends the output to the log file instead of overwriting it.
#SBATCH --partition=my_partition        # Requests a specific partition (queue) for the job.
#SBATCH --qos=my_qos                    # Requests a specific Quality of Service for the job.
#SBATCH --nodes=1                       # Requests a specific number of nodes.
#SBATCH --ntasks-per-node=1             # Specifies the number of tasks to be launched on each node.

set -x -e

((SLURM_GPUS_ON_NODE > 0)) && export OMP_NUM_THREADS=$((SLURM_CPUS_ON_NODE / SLURM_GPUS_ON_NODE)) || export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MASTER_PORT=$(shuf -i 49152-65535 -n 1)

echo "START TIME: $(date)"

srun uv run accelerate launch --config_file configs/accelerate/multi_gpu.yaml --main_process_port $MASTER_PORT src/mbt/train/sft.py "$@"

echo "END TIME: $(date)"
