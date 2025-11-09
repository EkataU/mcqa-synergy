#!/bin/bash
#SBATCH --job-name=mod1_synergy_TEN
#SBATCH --output=mod1_synergy_TEN_%j.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00

echo "=== Module 1: P(TEN) layer-wise analysis ==="
echo "Job ID : ${SLURM_JOB_ID}"
echo "Host   : $(hostname)"
echo "Time   : $(date)"

echo "CUDA GPUs:"
nvidia-smi || echo "nvidia-smi not available"

source /scratch/ekata/pyenv/bin/activate

echo "Python path : $(which python)"
echo "Python ver  : $(python --version)"

# export HF_TOKEN="hf_ns#########"# <-- optional if not in ~/.bashrc

DATA_CSV=/scratch/ekata/projects/mcqa/mcqa-synergy/Data/belebele_sampled.csv
OUT_DIR=/scratch/ekata/projects/mcqa/mcqa-synergy/results
MODEL_NAME=meta-llama/Meta-Llama-3.1-8B-Instruct

echo "Data CSV : $DATA_CSV"
echo "Out dir  : $OUT_DIR"
echo "Model    : $MODEL_NAME"

python module1_synergy_TEN.py \
  --data_csv "$DATA_CSV" \
  --out_dir  "$OUT_DIR" \
  --model_name "$MODEL_NAME" \
  --device cuda:0 \
  --max_samples 1000 \
  --batch_size 8 \
  --dtype float16

echo "=== Module 1 finished at: $(date) ==="
echo "Result files in: $OUT_DIR"
ls -lh "$OUT_DIR"

