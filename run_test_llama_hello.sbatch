#!/bin/bash
#SBATCH --job-name=test_llama_hello        # Job name
#SBATCH --output=test_llama_hello_%j.out   # Stdout+stderr log
#SBATCH --partition=gpu                    # Partition name (edit if needed)
#SBATCH --gres=gpu:a40:4                   # 4x A40 GPUs (device_map=auto)
#SBATCH --cpus-per-task=4                  # CPU cores
#SBATCH --mem=64G                          # RAM
#SBATCH --time=01:00:00                    # Max runtime

echo "=== Running test_llama_hello job ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Host : $(hostname)"
echo "Time : $(date)"

echo "Visible GPUs:"
nvidia-smi || echo "nvidia-smi not available"

# ---- Activate your virtualenv (edit path if different) ----
source /scratch/ekata/pyenv/bin/activate

echo "Python path: $(which python)"
echo "Python ver : $(python --version)"

# ---- HF token (set this BEFORE sbatch, or uncomment + fill) ----
# export HF_TOKEN="hf_xxx_your_token_here"

# ---- Run the Python test ----
python test_llama_hello.py

echo "=== Job finished at: $(date) ==="
