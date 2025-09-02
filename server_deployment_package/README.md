# Server Deployment Package for Moral Alignment Evaluation
## 4xA100 GPU Server Setup Guide

This package contains everything needed to run large language model evaluations on your GPU server for the moral alignment project.

## 📦 Package Contents

```
server_deployment_package/
├── README.md                    # This file
├── setup_server.sh             # Automated setup script
├── requirements.txt            # Python dependencies
├── server_model_runner.py     # Main model runner module
├── download_models.py          # Model downloader utility
├── run_all_models.ipynb       # Jupyter notebook for evaluation
├── run_evaluation.py           # Command-line evaluation script
└── data/                       # Test datasets
    ├── test_dataset_100.csv    # Quick test (100 samples)
    ├── test_dataset_1000.csv   # Medium test (1000 samples)
    ├── test_dataset_2500.csv   # Extended test (2500 samples)
    └── test_dataset_5000.csv   # Full test (5000 samples)
```

## 🚀 Quick Start

### Step 1: Transfer Package to Server
```bash
# From your local machine
scp -r server_deployment_package/ your-server:/tmp/
```

### Step 2: Run Setup Script
```bash
# On the server
ssh your-server
cd /tmp/server_deployment_package
chmod +x setup_server.sh
./setup_server.sh
```

The setup script will:
1. Check GPU availability
2. Create directory structure at `/data/storage_4_tb/moral-alignment-pipeline`
3. Install Python dependencies
4. Optionally download models
5. Configure the environment

### Step 3: Run Evaluation

#### Option A: Using Jupyter Notebook (Recommended)
```bash
cd /data/storage_4_tb/moral-alignment-pipeline
jupyter notebook run_all_models.ipynb
```

#### Option B: Using Command Line
```bash
cd /data/storage_4_tb/moral-alignment-pipeline
python run_evaluation.py --samples 100 --models llama3.1-70b qwen2.5-32b
```

## 📊 Available Models

### CRITICAL Priority (Must Have)
- `llama3.1-70b` - 140GB, 2 GPUs required
- `qwen2.5-32b` - 64GB, 1 GPU required
- `mistral-large` - 94GB, 2 GPUs required

### HIGH Priority (Recommended)
- `yi-34b` - 68GB, 1 GPU required
- `mixtral-8x7b` - 93GB, 2 GPUs required
- `falcon-40b` - 80GB, 2 GPUs required

### MEDIUM Priority (Nice to Have)
- `llama3.1-405b` - 470GB, 4 GPUs required
- `qwen2.5-72b` - 144GB, 2 GPUs required
- `grok-1` - 600GB, 4+ GPUs required

## 💻 System Requirements

### Minimum Requirements
- **GPUs**: 4x NVIDIA A100 (80GB each)
- **RAM**: 256GB system memory
- **Storage**: 2TB free space for models
- **CUDA**: 11.8 or higher
- **Python**: 3.8 or higher

### Recommended Setup
- **OS**: Ubuntu 20.04/22.04 or RHEL 8
- **Docker**: For containerized deployment
- **Network**: High-speed internet for model downloads

## 🔧 Configuration

### Memory Management
The system automatically configures GPU memory based on available resources:
- Single GPU models: Up to 75GB VRAM
- Multi-GPU models: Tensor parallel splitting
- VLLM backend: Optimized for high throughput

### Model Download Options
```bash
# Download only critical models (fastest)
python download_models.py --priority CRITICAL

# Download critical and high priority
python download_models.py --priority HIGH

# Download all models (requires ~2TB)
python download_models.py --priority ALL

# Download specific model
python download_models.py --model llama3.1-70b
```

## 📈 Running Evaluations

### Small Test (Quick Validation)
```bash
python run_evaluation.py \
    --dataset data/test_dataset_100.csv \
    --models qwen2.5-32b \
    --output outputs/test_run
```

### Medium Evaluation
```bash
python run_evaluation.py \
    --dataset data/test_dataset_1000.csv \
    --models llama3.1-70b qwen2.5-32b mistral-large \
    --output outputs/medium_run \
    --use-vllm
```

### Full Evaluation (All Models)
```bash
python run_evaluation.py \
    --dataset data/test_dataset_5000.csv \
    --models ALL \
    --output outputs/full_run \
    --use-vllm \
    --tensor-parallel-size 4
```

## 📊 Monitoring Progress

### GPU Monitoring
```bash
# Real-time GPU usage
watch -n 1 nvidia-smi

# Or use nvitop for better visualization
nvitop
```

### Evaluation Progress
```bash
# Check output directory
ls -la /data/storage_4_tb/moral-alignment-pipeline/outputs/

# View latest results
tail -f /data/storage_4_tb/moral-alignment-pipeline/logs/evaluation.log
```

## 🎯 Expected Performance

### Inference Speed (per sample)
- **Qwen2.5-32b**: ~1-2 seconds
- **Llama3.1-70b**: ~2-3 seconds
- **Mistral-Large**: ~2-3 seconds
- **Llama3.1-405b**: ~5-10 seconds

### Total Time Estimates
- **100 samples**: ~10-15 minutes
- **1000 samples**: ~1-2 hours
- **5000 samples**: ~6-10 hours

## 📤 Retrieving Results

After evaluation completes, retrieve results:

```bash
# From your local machine
scp -r your-server:/data/storage_4_tb/moral-alignment-pipeline/outputs/ ./server_results/
```

Results format:
```json
{
  "model": "llama3.1-70b",
  "sample_id": "sample_001",
  "prompt": "Is lying morally acceptable?",
  "response": "...",
  "choice": "unacceptable",
  "confidence": 0.92,
  "inference_time": 2.3
}
```

## 🐛 Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size
python run_evaluation.py --batch-size 1

# Or use model offloading
python run_evaluation.py --load-in-8bit
```

### Model Download Issues
```bash
# Resume download
python download_models.py --model llama3.1-70b --resume

# Use alternative source
python download_models.py --model llama3.1-70b --mirror
```

### Slow Inference
```bash
# Enable VLLM for faster inference
python run_evaluation.py --use-vllm

# Increase tensor parallel size
python run_evaluation.py --tensor-parallel-size 4
```

## 📞 Support

For issues or questions:
1. Check logs: `/data/storage_4_tb/moral-alignment-pipeline/logs/`
2. Verify GPU status: `nvidia-smi`
3. Test with smaller model first: `python run_evaluation.py --models qwen2.5-7b --samples 10`

## 📋 Checklist

Before running full evaluation:
- [ ] GPUs detected (4x A100)
- [ ] CUDA installed and working
- [ ] Python dependencies installed
- [ ] At least one model downloaded
- [ ] Test dataset available
- [ ] Output directory writable
- [ ] Sufficient disk space (>100GB free)

## 🎉 Success Indicators

Your evaluation is successful when:
1. No CUDA errors in logs
2. Results saved to output directory
3. JSON files contain valid responses
4. Inference times are reasonable (<10s per sample)
5. All samples processed without errors

---

**Note**: This evaluation typically takes 6-10 hours for the full 5000-sample dataset across multiple models. Plan accordingly and use screen/tmux for long-running sessions.