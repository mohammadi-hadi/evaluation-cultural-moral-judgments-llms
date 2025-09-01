# Exploring Cultural Variations in Moral Judgments with Large Language Models

This repository contains the code and data for the paper "Exploring Cultural Variations in Moral Judgments with Large Language Models".

## Abstract

Large Language Models (LLMs) have shown strong performance across many tasks, but their ability to capture culturally diverse moral values remains unclear. In this paper, we examine whether LLMs can mirror variations in moral attitudes reported by two major cross-cultural surveys: the World Values Survey and the PEW Research Center's Global Attitudes Survey. We compare smaller, monolingual, and multilingual models (GPT-2, OPT, BLOOMZ, and Qwen) with more recent instruction-tuned models (GPT-4o, GPT-4o-mini, Gemma-2-9b-it, and Llama-3.3-70B-Instruct). Using log-probability-based moral justifiability scores, we correlate each model's outputs with survey data covering a broad set of ethical topics. Our results show that many earlier or smaller models often produce near-zero or negative correlations with human judgments. In contrast, advanced instruction-tuned models (including GPT-4o and GPT-4o-mini) achieve substantially higher positive correlations, suggesting they better reflect real-world moral attitudes. While scaling up model size and using instruction tuning can improve alignment with cross-cultural moral norms, challenges remain for certain topics and regions. We discuss these findings in relation to bias analysis, training data diversity, and strategies for improving the cultural sensitivity of LLMs.

## Repository Structure

```
cultural-moral-judgments-llms/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── LICENSE                      # License information
├── .gitignore                  # Git ignore file
├── data/                       # Survey data and processed datasets
│   ├── raw/                    # Original survey data files
│   └── processed/              # Processed data files
├── src/                        # Source code
│   ├── data_processing.py      # Data loading and preprocessing functions
│   ├── model_evaluation.py     # Model evaluation functions
│   ├── visualization.py        # Plotting and visualization utilities
│   └── utils.py               # Helper functions
├── notebooks/                  # Jupyter notebooks for experiments
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_evaluation.ipynb
│   └── 03_results_analysis.ipynb
├── scripts/                    # Standalone scripts
│   ├── run_all_models.py       # Run evaluation on all models
│   └── generate_plots.py       # Generate all plots
├── results/                    # Output files
│   ├── model_outputs/          # Model prediction results
│   └── figures/                # Generated plots and figures
└── docs/                       # Additional documentation
    └── methodology.md          # Detailed methodology description
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/cultural-moral-judgments-llms.git
cd cultural-moral-judgments-llms
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up Hugging Face authentication (if using gated models):
```bash
huggingface-cli login
```

## Data

This project uses two main survey datasets:

1. **World Values Survey (WVS) Wave 7**: Contains moral attitudes data from 55+ countries
2. **PEW Research Global Attitudes Survey (Spring 2013)**: Contains moral attitudes data from 40 countries

The data files should be placed in the `data/raw/` directory:
- `WVS_Cross-National_Wave_7_csv_v5_0.csv`
- `Pew Research Global Attitudes Project Spring 2013 Dataset for web.sav`

## Usage

### Quick Start

To evaluate all models on both WVS and PEW datasets:

```bash
python scripts/run_all_models.py
```

### Detailed Usage

1. **Data Preprocessing**:
```python
from src.data_processing import load_wvs_data, load_pew_data

wvs_df = load_wvs_data('data/raw/WVS_Moral.csv')
pew_df = load_pew_data('data/raw/Pew Research Global Attitudes Project Spring 2013 Dataset for web.sav')
```

2. **Model Evaluation**:
```python
from src.model_evaluation import evaluate_model

results = evaluate_model(
    model_name="meta-llama/Llama-3.3-70B-Instruct",
    dataset="wvs",
    cultures=["United States", "China", "Germany"],
    use_cuda=True
)
```

3. **Visualization**:
```python
from src.visualization import plot_correlation_results

plot_correlation_results(results, save_path="results/figures/")
```

## Models Evaluated

The following models were evaluated in our study:

### Smaller/Earlier Models:
- GPT-2 (various sizes)
- OPT (various sizes)
- BLOOMZ
- Qwen

### Instruction-Tuned Models:
- GPT-4o
- GPT-4o-mini
- Gemma-2-9b-it
- Llama-3.3-70B-Instruct

## Methodology

Our approach uses log-probability-based moral justifiability scores. For each moral topic and country, we:

1. Generate paired prompts comparing moral vs. non-moral framings
2. Calculate log-probability differences between these framings
3. Correlate model outputs with survey data
4. Analyze results across different models and cultural contexts

For detailed methodology, see [docs/methodology.md](docs/methodology.md).

## Results

Key findings:
- Instruction-tuned models show significantly higher correlations with human moral judgments
- Model size and instruction tuning both contribute to better cultural alignment
- Certain topics and regions remain challenging for all models

## Citation

If you use this code or data in your research, please cite:

```bibtex
@article{yourname2024cultural,
  title={Exploring Cultural Variations in Moral Judgments with Large Language Models},
  author={Your Name and Collaborators},
  journal={Journal Name},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or collaborations, please contact: [h.mohammadi@uu.nl]

## Acknowledgments

We thank the World Values Survey Association and PEW Research Center for making their data publicly available. 
