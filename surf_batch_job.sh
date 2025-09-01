#!/bin/bash
#SBATCH --job-name=moral_alignment_eval
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1  # Request 1 GPU for model inference
#SBATCH --output=moral_eval_%j.out
#SBATCH --error=moral_eval_%j.err

# Print job information
echo "Job started on $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Load required modules
module load Python/3.10
module load CUDA/11.8  # If using GPU

# Navigate to project directory
cd $HOME/Project06

# Activate virtual environment
source venv/bin/activate

# Set cache directories for models (SURF-specific)
export TRANSFORMERS_CACHE=$HOME/.cache/huggingface
export HF_HOME=$HOME/.cache/huggingface
export TORCH_HOME=$HOME/.cache/torch

# Run the experiment with specified profile
# You can change the profile and parameters as needed
python run_experiments.py \
    --profile lightweight \
    --sample-size 50 \
    --output-dir outputs/run_${SLURM_JOB_ID}

# Optional: Run enhanced pipeline with cross-evaluation
python run_enhanced.py \
    --profile lightweight \
    --sample-size 50 \
    --enable-cross-eval \
    --enable-conflict-resolution \
    --output-dir outputs/run_${SLURM_JOB_ID}

echo "Job completed on $(date)"