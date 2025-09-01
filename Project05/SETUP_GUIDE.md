# Setup Guide

This guide helps you get started with the Cultural Moral Judgments with LLMs project.

## Repository Structure

The repository has been reorganized with the following structure:

```
cultural-moral-judgments-llms/
├── README.md                    # Main documentation
├── requirements.txt             # Python dependencies
├── LICENSE                      # MIT License
├── .gitignore                  # Git ignore rules
├── SETUP_GUIDE.md              # This file
├── data/                       # Data directory
│   ├── raw/                    # Original survey data
│   └── processed/              # Processed data (generated)
├── src/                        # Source code modules
│   ├── __init__.py            # Package initialization
│   ├── data_processing.py     # Data loading and preprocessing
│   ├── model_evaluation.py    # Model evaluation functions
│   ├── visualization.py       # Plotting utilities
│   └── utils.py              # Helper functions
├── scripts/                    # Executable scripts
│   ├── run_all_models.py      # Main evaluation script
│   └── generate_plots.py      # Plot generation script
├── notebooks/                  # Jupyter notebooks
├── results/                    # Output directory
│   ├── model_outputs/         # Model predictions
│   └── figures/               # Generated plots
├── docs/                       # Documentation
│   └── methodology.md         # Detailed methodology
└── archive/                    # Original notebooks (archived)
    └── original_notebooks/
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

Place the following data files in `data/raw/`:
- `WVS_Moral.csv` - World Values Survey moral questions
- `Country_Codes_Names.csv` - Country code mappings
- `Pew Research Global Attitudes Project Spring 2013 Dataset for web.sav` - PEW survey data

### 3. Run Evaluation

To evaluate all models on both datasets:

```bash
python scripts/run_all_models.py
```

For specific models or datasets:

```bash
# Evaluate specific models
python scripts/run_all_models.py --models "google/gemma-2-9b-it" "meta-llama/Llama-3.3-70B-Instruct"

# Evaluate on specific dataset
python scripts/run_all_models.py --datasets wvs

# Debug mode (fewer samples)
python scripts/run_all_models.py --debug
```

### 4. Generate Plots

To generate plots from existing results:

```bash
python scripts/generate_plots.py
```

## Using the Source Code

### Data Processing

```python
from src import load_wvs_data, load_pew_data

# Load data
wvs_df = load_wvs_data('data/raw/WVS_Moral.csv', 'data/raw/Country_Codes_Names.csv')
pew_df = load_pew_data('data/raw/Pew Research Global Attitudes Project Spring 2013 Dataset for web.sav')
```

### Model Evaluation

```python
from src import evaluate_model

# Evaluate a model
results = evaluate_model(
    model_name="meta-llama/Llama-3.3-70B-Instruct",
    dataset="wvs",
    cultures=["United States", "China", "Germany"]
)
```

### Visualization

```python
from src import plot_correlation_scatter, create_summary_report

# Plot results
plot_correlation_scatter(results, "WVS", "Llama-3.3-70B", save_path="results/figures")

# Create summary
summary = create_summary_report(all_results, save_path="results")
```

## Available Models

The following models are recommended for evaluation:

**Smaller/Earlier Models:**
- gpt2, gpt2-medium, gpt2-large
- facebook/opt-1.3b, facebook/opt-2.7b
- bigscience/bloomz-560m, bigscience/bloomz-1b7
- Qwen/Qwen-1_8B

**Instruction-Tuned Models:**
- google/gemma-2-9b-it
- meta-llama/Llama-3.3-70B-Instruct
- meta-llama/Meta-Llama-3-8B-Instruct

## Notes

- GPU is recommended for faster evaluation
- Large models (>7B parameters) use 8-bit quantization by default
- Results are saved to `results/model_outputs/` as CSV files
- Plots are saved to `results/figures/`

## Troubleshooting

1. **Out of Memory**: Use `--no-cuda` flag or reduce batch size in the code
2. **Missing Data**: Ensure all required data files are in `data/raw/`
3. **Model Access**: Some models require Hugging Face authentication:
   ```bash
   huggingface-cli login
   ```

## Original Notebooks

The original Jupyter notebooks have been archived in `archive/original_notebooks/` for reference. 