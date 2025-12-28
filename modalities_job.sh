#!/bin/bash
#SBATCH --job-name=modalities_train
#SBATCH --partition=standard           # Adjust to your cluster's GPU partition
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1            # 1 task, torchrun spawns processes
#SBATCH --gres=gpu:1                   # Request 1 GPU
#SBATCH --cpus-per-task=4              # Adjust based on your needs
#SBATCH --mem=16G                      # Adjust based on your needs
#SBATCH --time=24:00:00                # Set appropriate time limit
#SBATCH --output=slurm-%j.out

# Activate your conda environment
source ~/miniconda3/bin/activate modalities

# Run the training command
# Note: CUDA_VISIBLE_DEVICES is handled automatically by Slurm when using --gres=gpu
srun torchrun --standalone \
    --nnodes=1 \
    --nproc_per_node=1 \
    $(which modalities) run \
    --config_file_path configs/pretraining_deu_small.yaml
