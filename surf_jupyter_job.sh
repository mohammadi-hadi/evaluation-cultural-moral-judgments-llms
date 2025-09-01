#!/bin/bash
#SBATCH --job-name=moral_alignment_jupyter
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu  # or 'normal' if no GPU needed
#SBATCH --gres=gpu:1     # Request 1 GPU (remove if not needed)
#SBATCH --output=jupyter_%j.out

# Load modules
module load Python/3.10

# Activate virtual environment
cd $HOME/Project06
source venv/bin/activate

# Get the hostname
hostname=$(hostname -s)
port=8888

echo "Starting Jupyter on ${hostname}:${port}"
echo "=========================================="
echo "To connect from your local machine:"
echo "1. Open a new terminal and run:"
echo "   ssh -L ${port}:${hostname}:${port} your-username@login.surf.nl"
echo "2. Open browser and go to:"
echo "   http://localhost:${port}"
echo "=========================================="

# Start Jupyter
jupyter notebook --no-browser --port=${port} --ip=${hostname}