#!/bin/bash
# Quick Reference Commands for Running Moral Alignment Evaluation

echo "========================================"
echo "MORAL ALIGNMENT EVALUATION - QUICK START"
echo "========================================"
echo ""

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}1. SETUP COMMANDS${NC}"
echo "-------------------"
echo "# Install Ollama models (if not already installed):"
echo "ollama pull llama3.2:3b"
echo "ollama pull phi4:14b"
echo "ollama pull mistral:latest"
echo "ollama pull qwen2.5:7b"
echo "ollama pull gemma2:2b"
echo ""
echo "# Set OpenAI API key (for API models):"
echo "export OPENAI_API_KEY='your-api-key-here'"
echo ""

echo -e "${GREEN}2. CREATE TEST DATASETS${NC}"
echo "------------------------"
echo "# Create stratified test datasets:"
echo "python create_test_dataset.py"
echo ""

echo -e "${GREEN}3. RUN EVALUATIONS${NC}"
echo "-------------------"
echo "# Quick test (3 samples, local only):"
echo "python run_parallel_evaluation.py --samples 3 --no-api --no-server"
echo ""
echo "# Small test (100 samples):"
echo "python run_parallel_evaluation.py --samples 100"
echo ""
echo "# Medium test (1000 samples):"
echo "python run_parallel_evaluation.py --dataset sample_data/test_dataset_1000.csv --samples 1000"
echo ""
echo "# Full test (5000 samples) - WARNING: This takes hours:"
echo "python run_parallel_evaluation.py --dataset sample_data/test_dataset_5000.csv --samples 5000"
echo ""

echo -e "${GREEN}4. MONITOR PROGRESS${NC}"
echo "--------------------"
echo "# In a separate terminal, run:"
echo "python monitor_evaluation.py"
echo ""

echo -e "${GREEN}5. CHECK RESULTS${NC}"
echo "-----------------"
echo "# View latest results:"
echo "ls -la outputs/parallel_evaluation/run_*/"
echo ""
echo "# View combined results:"
echo "python -c \"import json; f=open('outputs/parallel_evaluation/run_*/combined_results.json'); print(json.dumps(json.load(f)['summary'], indent=2))\""
echo ""

echo -e "${GREEN}6. SERVER DEPLOYMENT${NC}"
echo "---------------------"
echo "# Copy to server:"
echo "scp -r server/ your-server:/data/storage_4_tb/moral-alignment-pipeline/"
echo "scp sample_data/test_dataset_5000.csv your-server:/data/storage_4_tb/moral-alignment-pipeline/data/"
echo ""
echo "# On server, run Jupyter notebook:"
echo "jupyter notebook server/run_all_models.ipynb"
echo ""

echo -e "${YELLOW}CURRENT STATUS:${NC}"
echo "---------------"
# Check Ollama status
if command -v ollama &> /dev/null; then
    echo -e "✅ Ollama installed"
    MODEL_COUNT=$(ollama list 2>/dev/null | wc -l)
    echo -e "   Models available: $((MODEL_COUNT-1))"
else
    echo -e "❌ Ollama not installed"
fi

# Check API key
if [ -n "$OPENAI_API_KEY" ]; then
    echo -e "✅ OpenAI API key configured"
else
    echo -e "❌ OpenAI API key not set"
fi

# Check test datasets
for size in 100 1000 2500 5000; do
    if [ -f "sample_data/test_dataset_${size}.csv" ]; then
        echo -e "✅ test_dataset_${size}.csv exists"
    else
        echo -e "⚠️  test_dataset_${size}.csv not found"
    fi
done

echo ""
echo "========================================"