#!/bin/bash
#SBATCH --job-name=exp4_multilang
#SBATCH --partition=standard          # Standard partition
#SBATCH --nodes=1                     # Single node
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4                  # Request 4 GPUs
#SBATCH --cpus-per-task=16            # CPU cores for dataloaders
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/exp4_multilingual_%j.out
#SBATCH --error=logs/exp4_multilingual_%j.err

echo "=========================================="
echo "Experiment 4: Fine-tuning GPT-2 on 5 languages with 4 GPUs"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================="

# Activate Conda environment
source ~/miniconda3/bin/activate modalities

# Check GPU info
nvidia-smi

# Run training across all 4 GPUs
srun torchrun --standalone \
    --nnodes=1 \
    --nproc_per_node=4 \
    $(which modalities) run \
    --config_file_path configs/training_train3_all.yaml

echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="
