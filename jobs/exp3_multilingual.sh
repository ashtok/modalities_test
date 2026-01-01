#!/bin/bash
#SBATCH --job-name=exp3_multilang
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/exp3_multilingual_%j.out
#SBATCH --error=logs/exp3_multilingual_%j.err

echo "=========================================="
echo "Experiment 3: Fine-tuning GPT-2 on 5 languages"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================="

source ~/miniconda3/bin/activate modalities
nvidia-smi

srun torchrun --standalone \
    --nnodes=1 \
    --nproc_per_node=1 \
    $(which modalities) run \
    --config_file_path configs/training_train3_all.yaml

echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="
