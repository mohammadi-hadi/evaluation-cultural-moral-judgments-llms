#!/bin/bash

# Server Setup Script for Moral Alignment Evaluation
# For 4xA100 GPU Server Deployment

echo "=================================================="
echo "MORAL ALIGNMENT EVALUATION - SERVER SETUP"
echo "=================================================="
echo ""

# Configuration
BASE_DIR="/data/storage_4_tb/moral-alignment-pipeline"
MODELS_DIR="$BASE_DIR/models"
DATA_DIR="$BASE_DIR/data"
OUTPUT_DIR="$BASE_DIR/outputs"

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}1. CHECKING ENVIRONMENT${NC}"
echo "------------------------"

# Check CUDA
if command -v nvidia-smi &> /dev/null; then
    echo -e "✅ CUDA available"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo -e "${RED}❌ CUDA not found - GPU required!${NC}"
    exit 1
fi

# Check Python
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    echo -e "✅ Python $PYTHON_VERSION found"
else
    echo -e "${RED}❌ Python3 not found${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}2. CREATING DIRECTORY STRUCTURE${NC}"
echo "---------------------------------"

# Create directories
echo "Creating directories at $BASE_DIR..."
mkdir -p "$BASE_DIR"
mkdir -p "$MODELS_DIR"
mkdir -p "$DATA_DIR"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$BASE_DIR/cache"
mkdir -p "$BASE_DIR/logs"

echo "✅ Directory structure created"

echo ""
echo -e "${GREEN}3. COPYING FILES${NC}"
echo "-----------------"

# Get current directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Copy Python files
echo "Copying Python modules..."
cp "$SCRIPT_DIR/server_model_runner.py" "$BASE_DIR/"
cp "$SCRIPT_DIR/download_models.py" "$BASE_DIR/"
cp "$SCRIPT_DIR/run_all_models.ipynb" "$BASE_DIR/"

# Copy data files
echo "Copying test datasets..."
cp "$SCRIPT_DIR/data/"*.csv "$DATA_DIR/" 2>/dev/null || echo "  No CSV files to copy"

# Copy requirements
cp "$SCRIPT_DIR/requirements.txt" "$BASE_DIR/"

echo "✅ Files copied to $BASE_DIR"

echo ""
echo -e "${GREEN}4. INSTALLING DEPENDENCIES${NC}"
echo "---------------------------"

# Create virtual environment (optional)
read -p "Create virtual environment? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Creating virtual environment..."
    python3 -m venv "$BASE_DIR/venv"
    source "$BASE_DIR/venv/bin/activate"
    echo "✅ Virtual environment created and activated"
fi

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r "$BASE_DIR/requirements.txt"

echo "✅ Dependencies installed"

echo ""
echo -e "${GREEN}5. DOWNLOADING MODELS (Optional)${NC}"
echo "---------------------------------"

read -p "Download models now? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    cd "$BASE_DIR"
    echo "Starting model download..."
    python download_models.py --priority CRITICAL
    echo "✅ Critical models downloaded"
    
    read -p "Download HIGH priority models? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python download_models.py --priority HIGH
        echo "✅ High priority models downloaded"
    fi
else
    echo "Skipping model download. Run later with:"
    echo "  cd $BASE_DIR"
    echo "  python download_models.py --priority CRITICAL"
fi

echo ""
echo -e "${GREEN}6. SETUP COMPLETE${NC}"
echo "------------------"

echo "✅ Server setup complete!"
echo ""
echo "Next steps:"
echo "1. To run evaluation with Python:"
echo "   cd $BASE_DIR"
echo "   python server_model_runner.py --models llama3.1-70b qwen2.5-32b --samples 100"
echo ""
echo "2. To run with Jupyter notebook:"
echo "   cd $BASE_DIR"
echo "   jupyter notebook run_all_models.ipynb"
echo ""
echo "3. To monitor GPU usage:"
echo "   watch -n 1 nvidia-smi"
echo ""
echo "4. View available models:"
echo "   cd $BASE_DIR"
echo "   python -c \"from server_model_runner import ServerModelRunner; r=ServerModelRunner(); print(r.get_available_models())\""
echo ""
echo "=================================================="