# Moral Alignment Pipeline - Complete Implementation

## 📚 Overview

This is a complete implementation of the research paper **"Exploring Cultural Variations in Moral Judgments with Large Language Models"**. The pipeline evaluates 30+ language models on cross-cultural moral judgments using World Values Survey (WVS) and PEW Research data.

### Key Features:
- ✅ Support for 30+ models (local and API-based)
- ✅ Dual elicitation: log-probability and direct scoring
- ✅ Chain-of-thought reasoning with 3-step protocol
- ✅ Reciprocal peer critique system
- ✅ Self-consistency calculation
- ✅ Comprehensive visualizations
- ✅ Easy deployment on any server

## 🚀 Quick Start

### Option 1: Jupyter Notebook (Recommended for Exploration)

1. **Open the notebook:**
   ```bash
   jupyter notebook moral_alignment_complete.ipynb
   ```

2. **Set API keys (if using API models):**
   ```python
   import os
   os.environ['OPENAI_API_KEY'] = 'your-key-here'
   os.environ['GEMINI_API_KEY'] = 'your-key-here'
   ```

3. **Run all cells** or step through interactively

### Option 2: Command Line Script (Recommended for Batch Processing)

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run with minimal profile (quick test):**
   ```bash
   python run_experiments.py --profile minimal --sample-size 20
   ```

3. **Run full evaluation:**
   ```bash
   python run_experiments.py --profile full
   ```

## 📁 Project Structure

```
Project06/
├── moral_alignment_complete.ipynb  # Main Jupyter notebook (all-in-one)
├── run_experiments.py              # Command-line runner
├── requirements.txt                # Python dependencies
├── models_config.yaml              # Model configurations
├── sample_data/                    # Survey data files
│   ├── WVS_Moral.csv              # World Values Survey data
│   ├── Country_Codes_Names.csv    # Country mappings
│   └── Pew Research *.sav         # PEW survey data
└── outputs/                        # Generated results (auto-created)
    ├── *_lp_scores.csv            # Log-probability scores
    ├── *_dir_scores.csv           # Direct scores
    ├── traces/                    # Reasoning traces
    ├── figures/                   # Visualizations
    └── all_metrics.csv            # Summary metrics
```

## 🔧 Installation

### Requirements:
- Python 3.8+
- CUDA 11.8+ (for GPU support)
- 24GB+ GPU memory (for large models)

### Step 1: Clone and Install
```bash
# Clone the repository
git clone <repository-url>
cd Project06

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Set Environment Variables
```bash
# For API models (optional)
export OPENAI_API_KEY="your-openai-api-key"
export GEMINI_API_KEY="your-gemini-api-key"

# Or create a .env file
echo "OPENAI_API_KEY=your-key" >> .env
echo "GEMINI_API_KEY=your-key" >> .env
```

## 🎯 Usage Examples

### 1. Test with Small Models
```python
# In Jupyter notebook
MODELS_TO_EVALUATE = ['gpt2', 'opt-125m']
SAMPLE_SIZE = 20  # Quick test
```

### 2. Evaluate Specific Models
```bash
python run_experiments.py --models gpt2 gpt2-medium opt-350m --sample-size 50
```

### 3. Run Full Paper Replication
```bash
python run_experiments.py --profile full --output-dir full_results
```

### 4. API Models Only
```bash
python run_experiments.py --profile api_only
```

### 5. Skip API Models (Local Only)
```bash
python run_experiments.py --profile standard --skip-api
```

## 📊 Available Models

### Small Pre-trained Models:
- GPT-2 (117M, 345M, 774M, 1.5B)
- OPT (125M, 350M, 1.3B, 2.7B)

### Multilingual Models:
- BLOOMZ (560M, 1.7B, 176B)
- Qwen (0.5B, 1.8B, 72B)

### Instruction-Tuned Models:
- Gemma-2-9B-IT
- Llama-3.3-70B-Instruct
- Llama-3-8B-Instruct

### API Models:
- GPT-3.5-turbo
- GPT-4o, GPT-4o-mini
- Gemini-1.5-Pro, Gemini-1.5-Flash

## 🌍 Deployment Options

### Google Colab
```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Upload notebook and data
# Run cells with Colab runtime
```

### AWS/Cloud Server
```bash
# SSH to server
ssh user@server

# Clone repository
git clone <repo-url>

# Install dependencies
pip install -r requirements.txt

# Run experiments
nohup python run_experiments.py --profile full &
```

### Docker Container
```dockerfile
FROM pytorch/pytorch:2.0.0-cuda11.8-cudnn8-runtime
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["python", "run_experiments.py", "--profile", "standard"]
```

## 📈 Outputs and Visualizations

### Generated Files:
- **Scores**: `outputs/*_lp_scores.csv`, `outputs/*_dir_scores.csv`
- **Traces**: `outputs/traces/*_traces.jsonl`
- **Metrics**: `outputs/all_metrics.csv`
- **Figures**: `outputs/figures/*.png`

### Visualizations:
1. **Correlation Comparison**: WVS vs PEW alignment
2. **Self-Consistency**: Model reasoning stability
3. **Country Heatmaps**: Geographic patterns
4. **Topic Difficulty**: Challenging moral topics
5. **Model Clustering**: Behavioral similarity

## ⚙️ Configuration

### Modify `models_config.yaml`:
```yaml
deployment_profiles:
  custom:
    models: ["gpt2", "gemma-2-9b-it", "gpt-4o-mini"]
    sample_size: 100
    batch_size: 8
```

### Use Custom Profile:
```bash
python run_experiments.py --profile custom
```

## 🐛 Troubleshooting

### Out of Memory:
- Use 8-bit quantization: `load_in_8bit: true`
- Reduce batch size
- Use smaller models first

### API Rate Limits:
- Add delays between API calls
- Use `--skip-api` for local models only

### Missing Data Files:
- Ensure `sample_data/` contains WVS and PEW files
- Check file paths in configuration

## 📝 Citation

If you use this implementation, please cite:
```bibtex
@article{mohammadi2024cultural,
  title={Exploring Cultural Variations in Moral Judgments with Large Language Models},
  author={Mohammadi, H. and others},
  year={2024}
}
```

## 📧 Support

For issues or questions:
- Create an issue on GitHub
- Email: h.mohammadi@uu.nl

## 📄 License

MIT License - See LICENSE file for details

---

**Note**: This implementation provides a complete, reproducible pipeline that can be easily deployed on any server with GPU access. All code is contained in the Jupyter notebook for maximum portability.