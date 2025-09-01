#!/bin/bash
# Complete experimental pipeline for moral alignment study

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Moral Alignment Experimental Pipeline${NC}"
echo -e "${GREEN}========================================${NC}\n"

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo -e "${YELLOW}Python version: $python_version${NC}"

# Create directory structure
echo -e "\n${YELLOW}Setting up directories...${NC}"
mkdir -p data
mkdir -p outputs/{traces,figures,metrics}
mkdir -p logs

# Install dependencies if needed
echo -e "\n${YELLOW}Checking dependencies...${NC}"
pip install -q torch transformers sentence-transformers pandas numpy scipy \
    matplotlib seaborn tqdm pyyaml scikit-learn openai

# Check for API keys
if [ -z "$OPENAI_API_KEY" ]; then
    echo -e "${YELLOW}Warning: OPENAI_API_KEY not set. OpenAI models will be skipped.${NC}"
fi

if [ -z "$GEMINI_API_KEY" ]; then
    echo -e "${YELLOW}Warning: GEMINI_API_KEY not set. Gemini models will be skipped.${NC}"
fi

# Run experiments based on arguments
MODE=${1:-"small"}  # small, medium, large, or full
VISUALIZE=${2:-"true"}

echo -e "\n${GREEN}Running experiments in $MODE mode...${NC}"

case $MODE in
    "small")
        # Quick test with smallest models
        echo "Testing with GPT-2 base model only..."
        python3 moral_alignment_pipeline.py \
            --models gpt2 \
            --visualize \
            2>&1 | tee logs/experiment_small_$(date +%Y%m%d_%H%M%S).log
        ;;
    
    "medium")
        # Test with a few models
        echo "Running medium-scale experiment..."
        python3 moral_alignment_pipeline.py \
            --models gpt2 gpt2-medium facebook/opt-125m \
            --visualize \
            2>&1 | tee logs/experiment_medium_$(date +%Y%m%d_%H%M%S).log
        ;;
    
    "large")
        # Test with more models including some larger ones
        echo "Running large-scale experiment..."
        python3 moral_alignment_pipeline.py \
            --models gpt2 gpt2-medium gpt2-large \
                    facebook/opt-125m facebook/opt-350m \
                    bigscience/bloomz-560m \
            --visualize \
            2>&1 | tee logs/experiment_large_$(date +%Y%m%d_%H%M%S).log
        ;;
    
    "full")
        # Full experiment with all models
        echo "Running full experiment (this will take several hours)..."
        python3 moral_alignment_pipeline.py \
            --models gpt2 gpt2-medium gpt2-large \
                    facebook/opt-125m facebook/opt-350m \
                    bigscience/bloomz-560m \
                    Qwen/Qwen2-0.5B \
                    gpt-3.5-turbo gpt-4o-mini \
            --visualize \
            2>&1 | tee logs/experiment_full_$(date +%Y%m%d_%H%M%S).log
        ;;
    
    "peer")
        # Run with peer critique enabled
        echo "Running with peer critique (slow)..."
        python3 moral_alignment_pipeline.py \
            --models gpt2 gpt2-medium \
            --visualize \
            --peer-critique \
            2>&1 | tee logs/experiment_peer_$(date +%Y%m%d_%H%M%S).log
        ;;
    
    *)
        echo -e "${RED}Unknown mode: $MODE${NC}"
        echo "Usage: $0 [small|medium|large|full|peer] [true|false]"
        exit 1
        ;;
esac

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Experiment complete!${NC}"
echo -e "${GREEN}========================================${NC}\n"

# Generate summary report
echo -e "${YELLOW}Generating summary report...${NC}"
python3 - <<EOF
import pandas as pd
from pathlib import Path

out_dir = Path("outputs")

# Load metrics
if (out_dir / "metrics.csv").exists():
    metrics = pd.read_csv(out_dir / "metrics.csv")
    
    print("\n" + "="*50)
    print("EXPERIMENT SUMMARY")
    print("="*50)
    
    # Best performing models
    best_wvs = metrics[metrics["source"] == "WVS"].nlargest(3, "pearson_r")
    best_pew = metrics[metrics["source"] == "PEW"].nlargest(3, "pearson_r")
    
    print("\nTop 3 models on WVS:")
    for _, row in best_wvs.iterrows():
        print(f"  {row['model']} ({row['method']}): r={row['pearson_r']:.3f}")
    
    print("\nTop 3 models on PEW:")
    for _, row in best_pew.iterrows():
        print(f"  {row['model']} ({row['method']}): r={row['pearson_r']:.3f}")
    
    # Method comparison
    print("\n" + "-"*50)
    print("Method Comparison (mean across all models):")
    method_comp = metrics.groupby("method")["pearson_r"].mean()
    for method, r in method_comp.items():
        print(f"  {method}: r={r:.3f}")
    
    print("\nFiles generated:")
    for f in sorted(out_dir.glob("*")):
        if f.is_file():
            size = f.stat().st_size / 1024  # KB
            print(f"  {f.name}: {size:.1f} KB")
else:
    print("No metrics file found. Experiment may have failed.")
EOF

echo -e "\n${GREEN}All outputs saved to outputs/ directory${NC}"
echo -e "${GREEN}Logs saved to logs/ directory${NC}"

# Open visualizations if generated
if [ "$VISUALIZE" = "true" ] && [ -f "outputs/figures/country_correlations.png" ]; then
    echo -e "\n${YELLOW}Opening visualizations...${NC}"
    if command -v xdg-open &> /dev/null; then
        xdg-open outputs/figures/country_correlations.png 2>/dev/null &
    elif command -v open &> /dev/null; then
        open outputs/figures/country_correlations.png
    fi
fi