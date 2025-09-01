#!/bin/bash
# Setup script for SURF storage at /data/storage_4_tb/moral-alignment-pipeline

echo "======================================"
echo "SURF Storage Setup"
echo "======================================"

# Set base path
BASE_PATH="/data/storage_4_tb/moral-alignment-pipeline"

# Create directory structure
echo "Creating directory structure at $BASE_PATH..."

mkdir -p $BASE_PATH/{models,outputs,data,temp,logs}
mkdir -p $BASE_PATH/outputs/{scores,traces,prompts,evaluations,visualizations,experiments,cross_evaluation,cache}
mkdir -p $BASE_PATH/outputs/scores/{logprob,direct,cot}
mkdir -p $BASE_PATH/outputs/traces/{raw,processed}
mkdir -p $BASE_PATH/outputs/prompts/{templates,filled}
mkdir -p $BASE_PATH/outputs/evaluations/{peer,human}
mkdir -p $BASE_PATH/outputs/visualizations/{figures,reports}
mkdir -p $BASE_PATH/outputs/experiments/{runs,logs}
mkdir -p $BASE_PATH/models/{transformers,torch,datasets}
mkdir -p $BASE_PATH/logs/{local,server,jobs}

echo "✅ Directory structure created"

# Clone or update repository
if [ ! -d "$BASE_PATH/code" ]; then
    echo "Cloning repository..."
    git clone https://github.com/mohammadi-hadi/moral-alignment-pipeline.git $BASE_PATH/code
else
    echo "Updating repository..."
    cd $BASE_PATH/code
    git pull origin main
fi

cd $BASE_PATH/code

# Create Python virtual environment
if [ ! -d "$BASE_PATH/venv" ]; then
    echo "Creating virtual environment..."
    module load Python/3.10
    python -m venv $BASE_PATH/venv
fi

# Activate environment and install packages
echo "Installing Python packages..."
source $BASE_PATH/venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Create environment configuration
echo "Creating environment configuration..."
cat > $BASE_PATH/code/.env << EOF
# Environment Configuration for SURF
MORAL_ALIGNMENT_ENV=server

# Storage paths
BASE_PATH=$BASE_PATH
MODEL_CACHE=$BASE_PATH/models
OUTPUTS_PATH=$BASE_PATH/outputs
DATA_PATH=$BASE_PATH/data
TEMP_PATH=$BASE_PATH/temp

# Model cache paths for HuggingFace
TRANSFORMERS_CACHE=$BASE_PATH/models/transformers
HF_HOME=$BASE_PATH/models
TORCH_HOME=$BASE_PATH/models/torch
HF_DATASETS_CACHE=$BASE_PATH/models/datasets

# API Keys (add your keys here)
OPENAI_API_KEY=your-key-here
GEMINI_API_KEY=your-key-here
ANTHROPIC_API_KEY=your-key-here
COHERE_API_KEY=your-key-here
MISTRAL_API_KEY=your-key-here
EOF

# Create launcher script
echo "Creating launcher scripts..."
cat > $BASE_PATH/launch_local_models.sh << 'EOF'
#!/bin/bash
# Run small models that fit in single GPU

cd /data/storage_4_tb/moral-alignment-pipeline/code
source /data/storage_4_tb/moral-alignment-pipeline/venv/bin/activate

export TRANSFORMERS_CACHE=/data/storage_4_tb/moral-alignment-pipeline/models
export HF_HOME=/data/storage_4_tb/moral-alignment-pipeline/models

python run_unified.py \
    --profile server_only \
    --models llama-3.2-11b-instruct gemma-2-9b-it \
    --sample-size 50
EOF

cat > $BASE_PATH/launch_large_models.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=moral_large
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=320G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --output=/data/storage_4_tb/moral-alignment-pipeline/logs/jobs/%j.out

cd /data/storage_4_tb/moral-alignment-pipeline/code
source /data/storage_4_tb/moral-alignment-pipeline/venv/bin/activate

export TRANSFORMERS_CACHE=/data/storage_4_tb/moral-alignment-pipeline/models
export HF_HOME=/data/storage_4_tb/moral-alignment-pipeline/models

python run_unified.py \
    --profile server_only \
    --models llama-3.3-70b-instruct mixtral-8x22b-instruct \
    --sample-size 100
EOF

chmod +x $BASE_PATH/launch_*.sh

# Create sync script
cat > $BASE_PATH/sync_with_local.sh << 'EOF'
#!/bin/bash
# Sync outputs with local machine

LOCAL_USER="your-local-username"
LOCAL_HOST="your-local-ip"
LOCAL_PATH="~/Documents/Project06/outputs"

# Sync outputs from SURF to local
rsync -avz --progress \
    /data/storage_4_tb/moral-alignment-pipeline/outputs/ \
    $LOCAL_USER@$LOCAL_HOST:$LOCAL_PATH/

echo "Sync completed"
EOF

chmod +x $BASE_PATH/sync_with_local.sh

# Check GPU availability
echo ""
echo "Checking GPU availability..."
module load CUDA/11.8
nvidia-smi

# Create test script
cat > $BASE_PATH/test_setup.py << 'EOF'
#!/usr/bin/env python3
import torch
import os
from pathlib import Path

print("="*60)
print("SURF Storage Test")
print("="*60)

# Check paths
base_path = Path("/data/storage_4_tb/moral-alignment-pipeline")
print(f"Base path exists: {base_path.exists()}")
print(f"Models cache: {base_path / 'models'}")
print(f"Outputs: {base_path / 'outputs'}")

# Check GPU
print(f"\nGPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")

# Check environment variables
print(f"\nTRANSFORMERS_CACHE: {os.getenv('TRANSFORMERS_CACHE', 'Not set')}")
print(f"HF_HOME: {os.getenv('HF_HOME', 'Not set')}")

print("\n✅ Setup test complete")
EOF

chmod +x $BASE_PATH/test_setup.py

echo ""
echo "======================================"
echo "✅ SURF Storage Setup Complete!"
echo "======================================"
echo ""
echo "Base path: $BASE_PATH"
echo "Code location: $BASE_PATH/code"
echo "Model cache: $BASE_PATH/models"
echo "Outputs: $BASE_PATH/outputs"
echo ""
echo "Next steps:"
echo "1. Add your API keys to $BASE_PATH/code/.env"
echo "2. Test setup: python $BASE_PATH/test_setup.py"
echo "3. Run small models: $BASE_PATH/launch_local_models.sh"
echo "4. Submit large models: sbatch $BASE_PATH/launch_large_models.sh"
echo "5. Sync with local: $BASE_PATH/sync_with_local.sh"