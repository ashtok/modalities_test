#!/bin/bash
#SBATCH --job-name=lm_eval_all
#SBATCH --partition=standard          # Standard partition
#SBATCH --nodes=1                     # Single node
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                  # Request 1 GPU (evaluation doesn't need multiple GPUs)
#SBATCH --cpus-per-task=8             # CPU cores
#SBATCH --mem=32G
#SBATCH --time=12:00:00               # Adjust based on your model size
#SBATCH --output=logs/lm_eval_all_%j.out
#SBATCH --error=logs/lm_eval_all_%j.err

echo "=========================================="
echo "LM Evaluation Harness: Evaluating All Models"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================="

# Activate Conda environment for lm-eval
source ~/miniconda3/bin/activate lm_eval

# Navigate to the working directory
cd ~/modalities_test

# Create results directory if it doesn't exist
mkdir -p results

# Check GPU info
echo "GPU Information:"
nvidia-smi
echo "=========================================="

# Exp1: German only (step 180000)
echo "Evaluating Exp1: German only (step 180000)..."
echo "Start time: $(date)"
python -m lm_eval \
  --model hf \
  --model_args pretrained=hf_models/exp1_german_only/exp1_german_only_2026-01-01__05-08-20_ab1b864305b4b8b1/hf_checkpoint_step180000 \
  --tasks ogx_hellaswagx_de,ogx_hellaswagx_es,ogx_hellaswagx_it,ogx_hellaswagx_fr \
  --device cuda:0 \
  --batch_size 8 \
  --output_path results/exp1_german_only_step180k_ogx_hellaswagx.json

echo "Exp1 completed at: $(date)"
echo "=========================================="

# Exp2: German-English (step 425000)
echo "Evaluating Exp2: German-English (step 425000)..."
echo "Start time: $(date)"
python -m lm_eval \
  --model hf \
  --model_args pretrained=hf_models/exp2_german_english/exp2_german_english_2026-01-01__05-08-20_5734a86bbe85efff/hf_checkpoint_step425000 \
  --tasks ogx_hellaswagx_de,ogx_hellaswagx_es,ogx_hellaswagx_it,ogx_hellaswagx_fr \
  --device cuda:0 \
  --batch_size 8 \
  --output_path results/exp2_german_english_step425k_ogx_hellaswagx.json

echo "Exp2 completed at: $(date)"
echo "=========================================="

# Exp3: Multilingual (step 420000)
echo "Evaluating Exp3: Multilingual (step 420000)..."
echo "Start time: $(date)"
python -m lm_eval \
  --model hf \
  --model_args pretrained=hf_models/exp3_multilingual/exp3_multilingual_2026-01-01__05-08-20_a91e58afaade00f6/hf_checkpoint_step420000 \
  --tasks ogx_hellaswagx_de,ogx_hellaswagx_es,ogx_hellaswagx_it,ogx_hellaswagx_fr \
  --device cuda:0 \
  --batch_size 8 \
  --output_path results/exp3_multilingual_step420k_ogx_hellaswagx.json

echo "Exp3 completed at: $(date)"
echo "=========================================="

echo "All evaluations completed!"
echo "End time: $(date)"
echo "Results saved in: ~/modalities_test/results/"
echo "=========================================="
