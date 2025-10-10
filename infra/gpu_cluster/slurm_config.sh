#!/bin/bash
# SLURM configuration for GPU cluster

#SBATCH --job-name=dspy-eval
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Load modules
module load python/3.9
module load cuda/11.8
module load gcc/9.3.0

# Set environment variables
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Create virtual environment
if [ ! -d "venv" ]; then
    python -m venv venv
fi

source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Run evaluation
python apps/cli/eval.py \
    --project sharktank \
    --model gpt-4o-mini \
    --optimizer MIPROv2 \
    --num-samples 100 \
    --batch-size 8 \
    --output-dir experiments/runs

# Deactivate virtual environment
deactivate
