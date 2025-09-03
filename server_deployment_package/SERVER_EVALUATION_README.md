# Server Evaluation with 4xA100 GPUs

This package provides optimized evaluation scripts for running large language models on a server with 4xA100 GPUs.

## 🚀 Quick Start

### 1. Deploy to Server
```bash
# On your local machine
./deploy_server_evaluation.sh
```

### 2. Run on Server
```bash
# SSH to server and navigate to the evaluation directory
cd /data/storage_4_tb/moral-alignment-pipeline

# Check available models
python run_server_evaluation.py --list

# See time estimates
python run_server_evaluation.py --estimate

# Run full evaluation (5000 samples per model)
python run_server_evaluation.py

# Test with smaller sample size
python run_server_evaluation.py --samples 1000

# Run single model
python run_server_evaluation.py --model qwen2.5-32b
```

### 3. Monitor Progress
```bash
# Real-time monitoring (updates every 30s)
python live_progress_monitor.py --continuous

# Single progress check
python live_progress_monitor.py --once
```

## 🎯 Optimized Models

The server evaluation focuses on large models that benefit from multiple GPUs:

| Model | Size | GPUs | Priority | Notes |
|-------|------|------|----------|-------|
| qwen2.5-32b | 32B | 1-2 | CRITICAL | Excellent balance |
| qwq-32b | 32B | 1-2 | HIGH | Reasoning specialist |  
| llama3.3-70b | 70B | 2-4 | CRITICAL | Best open 70B |
| qwen2.5-72b | 72B | 2-4 | CRITICAL | Excellent cross-cultural |
| mixtral-8x7b | MoE | 2 | HIGH | MoE architecture |

## 🔧 GPU Utilization Strategy

- **32B models**: Use 1-2 GPUs with tensor parallelism
- **70B+ models**: Use 2-4 GPUs with tensor parallelism  
- **MoE models**: Optimized routing across multiple GPUs
- **Memory optimization**: 8-bit quantization for large models
- **Batch optimization**: Dynamic batch sizes based on model size

## 📊 Performance Expectations

Based on A100 80GB performance:

- **32B models**: ~3-5 samples/second
- **70B models**: ~2-3 samples/second  
- **Total time**: ~6-8 hours for all models (5000 samples each)

## 🎮 Background Execution

For long-running evaluations:

```bash
# Run in background with logging
nohup python run_server_evaluation.py > logs/server_evaluation_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Check process
ps aux | grep run_server_evaluation

# Monitor log file
tail -f logs/server_evaluation_*.log
```

## 🔍 Monitoring & Debugging

### Real-time Progress
```bash
# Live progress monitor
python live_progress_monitor.py --continuous --interval 30
```

### System Resources
```bash
# GPU usage
nvidia-smi -l 5

# System resources  
htop

# Disk space
df -h /data/storage_4_tb/
```

### Debug Mode
```bash
# Enable debug logging
export CUDA_LAUNCH_BLOCKING=1
python run_server_evaluation.py --model qwen2.5-32b --samples 100
```

## 📂 Output Structure

```
/data/storage_4_tb/moral-alignment-pipeline/
├── outputs/
│   └── server_results/
│       ├── qwen2.5-32b_results.json
│       ├── qwq-32b_results.json
│       └── ...
├── logs/
│   └── server_evaluation_*.log
└── cache/
    └── huggingface/
```

## ⚠️ Troubleshooting

### GPU Not Detected
```bash
# Check GPU status
nvidia-smi

# Verify CUDA environment
echo $CUDA_VISIBLE_DEVICES

# Test PyTorch CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
```

### Memory Issues
```bash
# Clear GPU memory
python -c "import torch; torch.cuda.empty_cache()"

# Reduce batch size in server_model_runner.py
# Lower gpu_memory_utilization from 0.9 to 0.8
```

### Model Loading Issues
```bash
# Check model availability
ls -la /data/storage_4_tb/moral-alignment-pipeline/models/

# Verify HuggingFace cache
ls -la /data/storage_4_tb/.cache/huggingface/
```

## 🎯 Integration with Local Evaluation

The server evaluation produces results in the same format as local evaluation for easy integration:

```bash
# Combine results after both complete
python combine_results.py --local outputs/local_results/ --server outputs/server_results/
```

## 🚀 Maximum Performance Tips

1. **Warm-up**: Let first model load completely before judging performance
2. **Monitoring**: Keep an eye on GPU utilization with `nvidia-smi`
3. **Storage**: Use fast SSD storage for model cache
4. **Network**: Ensure stable connection if downloading models
5. **Memory**: Close other applications to maximize available GPU memory

---

**Ready to harness the power of 4xA100 GPUs!** 🚀