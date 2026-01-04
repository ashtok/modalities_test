# Multilingual GPT-2 Fine-Tuning with Modalities

Fine-tuning GPT-2 on 5 languages (German, English, French, Spanish, Italian) using the [Modalities](https://github.com/Modalities/modalities) framework to evaluate how training data composition affects downstream performance on HellaSwagX and FLORES200 benchmarks.

## Overview

Three experimental setups:
1. **German-only** (1.2M lines)
2. **German + English** (1.8M lines)  
3. **All 5 languages** (3.6M lines)

Models trained with FSDP and evaluated using lm-evaluation-harness.

## Project Structure

```
modalities_test/
├── configs/          # Training & tokenization configs
├── data/
│   ├── raw/         # Downloaded HPLT samples
│   ├── merged/      # Merged training files
│   └── preprocessed/ # Packed datasets
├── hf_models/       # HF checkpoints
├── jobs/            # SLURM scripts
├── results/         # Evaluation results
└── src/             # Python scripts
```

## Installation

```bash
# Setup environment
conda create -n modalities python=3.10
conda activate modalities
pip install modalities torch transformers requests

# Install evaluation framework
git clone https://github.com/EleutherAI/lm-evaluation-harness.git
cd lm-evaluation-harness && pip install -e . && cd ..
```

## Quick Start

### 1. Data Preparation

```bash
# Download & merge HPLT data (single command)
python src/download_and_merge.py
```

Downloads and samples HPLT data, then creates three merged datasets:
- `train1_deu.jsonl` (1.2M lines)
- `train2_deu_eng.jsonl` (1.8M lines)
- `train3_all.jsonl` (3.6M lines)

### 2. Preprocessing

```bash
# Create indices
python src/create_indexes.py

# Pack datasets
modalities data pack_encoded_data configs/tokenization_train1_deu.yaml
modalities data pack_encoded_data configs/tokenization_train2_deu_eng.yaml
modalities data pack_encoded_data configs/tokenization_train3_all.yaml
```

### 3. Training

```bash
# Generate configs
python src/generate_training_configs.py

# Submit jobs (SLURM)
sbatch jobs/exp1_german_only.sh
sbatch jobs/exp2_german_english.sh
sbatch jobs/exp3_multilingual.sh
```

**Training specs**: GPT-2 base, seq_len=512, batch_size=4, AdamW (lr=5e-5), FSDP with BF16

### 4. Evaluation

**HellaSwagX:**
```bash
python -m lm_eval \
  --model hf \
  --model_args pretrained=hf_models/exp1_german_only/.../hf_checkpoint_step180000 \
  --tasks ogx_hellaswagx_de,ogx_hellaswagx_es,ogx_hellaswagx_it,ogx_hellaswagx_fr \
  --device cuda:0 \
  --batch_size 8 \
  --output_path results/exp1_hellaswagx.json
```

**FLORES200:**
```bash
python -m lm_eval \
  --model hf \
  --model_args pretrained=hf_models/exp1_german_only/.../hf_checkpoint_step180000 \
  --tasks ogx_flores200-nll-deu_Latn,ogx_flores200-nll-eng_Latn,ogx_flores200-nll-spa_Latn,ogx_flores200-nll-ita_Latn,ogx_flores200-nll-fra_Latn \
  --device cuda:0 \
  --batch_size 8 \
  --output_path results/exp_flores200.json
```

## Results

**HellaSwagX**: No measurable improvement across the three setups under current data constraints. (Current results: https://github.com/ashtok/modalities_test/tree/main/results)

**FLORES200**: Evaluation blocked by GPU resource limits (`AssocGrpGRES`).

## Limitations

- **Data constraints**: Max 1.2M German lines, 600k per other language due to storage quotas
- **Limited scaling**: Current dataset sizes may not be sufficient to show multilingual training benefits
- **Evaluation blocked**: FLORES200 evaluation pending GPU resource availability

## Reproduction

Follow the Quick Start steps above. Total training time: ~40 hours across all experiments.

## Acknowledgments

- [Modalities Framework](https://github.com/Modalities/modalities)
- [HPLT Project](https://hplt-project.org/) for multilingual data
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- OpenGPT-X for benchmarks
