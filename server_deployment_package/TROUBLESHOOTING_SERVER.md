# Server Troubleshooting Guide

Quick fixes for common server issues when running 4xA100 GPU evaluation.

## 🚨 Error: ModuleNotFoundError: nvidia_ml_py3

**Problem**: The server error shows `ModuleNotFoundError: No module named 'nvidia_ml_py3'`

**Quick Fix**:
```bash
# SSH to server
cd /data/storage_4_tb/moral-alignment-pipeline

# Run the automatic fix script
python fix_gpu_monitor.py

# This will:
# 1. Install nvidia-ml-py3
# 2. Test GPU detection 
# 3. Verify PyTorch CUDA
# 4. Show GPU count
```

## 🚨 Error: NameError: name 'logger' is not defined

**Problem**: GPU monitor has logger definition issues

**Quick Fix**:
```bash
# SSH to server and run the deployment script again
cd /path/to/server_deployment_package
./deploy_server_evaluation.sh

# This copies the fixed gpu_monitor.py file
```

## 🚨 Problem: Can't see GPU usage

**Problem**: nvidia-smi not showing GPU usage during evaluation

**Check Commands**:
```bash
# Check if GPUs are visible
nvidia-smi

# Check if processes are running
ps aux | grep python

# Check GPU processes specifically  
nvidia-smi pmon -s u

# Check CUDA environment
echo $CUDA_VISIBLE_DEVICES
```

## 🚨 Problem: Models not found

**Problem**: Server evaluation shows "Not found" for all models

**Solutions**:
```bash
# Check if models directory exists
ls -la /data/storage_4_tb/moral-alignment-pipeline/models/

# If models not there, check HuggingFace cache
ls -la /data/storage_4_tb/.cache/huggingface/

# Download models manually if needed
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('Qwen/Qwen2.5-32B-Instruct')"
```

## 🚨 Problem: Permission denied

**Problem**: Can't create directories or write files

**Quick Fix**:
```bash
# Fix permissions
sudo chown -R $USER:$USER /data/storage_4_tb/moral-alignment-pipeline/
chmod -R 755 /data/storage_4_tb/moral-alignment-pipeline/

# Create necessary directories
mkdir -p /data/storage_4_tb/moral-alignment-pipeline/outputs/server_results
mkdir -p /data/storage_4_tb/moral-alignment-pipeline/logs
```

## 🚨 Problem: Out of memory

**Problem**: GPU runs out of memory during evaluation

**Solutions**:
```bash
# Clear GPU memory
python -c "import torch; torch.cuda.empty_cache()"

# Check current GPU memory usage
nvidia-smi

# Reduce batch size in server_model_runner.py
# Look for gpu_memory_util and reduce from 0.9 to 0.8
```

## 📋 Complete Diagnostic Check

Run this comprehensive check if you're having issues:

```bash
cd /data/storage_4_tb/moral-alignment-pipeline

# 1. Fix dependencies
python fix_gpu_monitor.py

# 2. Run comprehensive diagnostics
python server_diagnostics.py

# 3. Check server setup
python run_server_evaluation.py --list

# 4. Test single model evaluation
python run_server_evaluation.py --model qwen2.5-32b --samples 100

# 5. Start monitoring
python server_live_monitor.py --once
```

## 🚨 Problem: GPUs Active but No Evaluation Detected

**Problem**: Server monitor shows GPUs are active (high usage/memory) but no evaluation processes detected

**Quick Diagnostic**:
```bash
# Run comprehensive server diagnostics
python server_diagnostics.py

# This will check:
# - What processes are using GPUs
# - Recent result files
# - Running Python processes
# - Network services
# - Memory usage details
```

**Common Causes**:
- Evaluation running with different process name
- Previous evaluation still running
- Other processes using GPUs (Jupyter, other models)
- Monitor looking in wrong directory for result files

## 🎯 Expected Working Output

When everything is working correctly, you should see:

```bash
$ python server_live_monitor.py --once

🚀 SERVER LIVE MONITOR - 5000 SAMPLES PER MODEL
======================================================================
📅 2025-09-03 12:30:45

🖥️  GPU STATUS - 4 A100 GPUs:
   GPU 0: 🔥 85% | 65.2GB/80.0GB (82%)
   GPU 1: 🔥 90% | 70.1GB/80.0GB (88%)
   GPU 2: 🟡 60% | 45.3GB/80.0GB (57%)
   GPU 3: 🟡 55% | 42.1GB/80.0GB (53%)

🔄 IN PROGRESS:
   • mixtral-8x22b: 2847/5000 (57.0%) [4GPU-4-GPU FIRST] - updated 2.3m ago

🎯 GPU PRIORITY STATUS:
   🔥 4-GPU Models: 0/2 completed, 1 in progress
   ⚡ 2-GPU Models: 0/5 completed, 0 in progress
```

## 🚀 Success Indicators

✅ **GPU Detection**: Should show 4 A100 GPUs
✅ **Model Loading**: Models load without memory errors
✅ **Progress Tracking**: Shows X/5000 samples completed
✅ **GPU Utilization**: High GPU usage (>80%) during evaluation
✅ **File Output**: Result files appearing in outputs/server_results/

## ⚠️ Red Flags

❌ **GPU Count 0**: GPU detection failing
❌ **No GPU Processes**: Models not using GPUs
❌ **Low GPU Usage**: (<10%) Models running on CPU
❌ **No Progress Files**: Results not being saved

---

**Need more help?** 
- Check the full logs: `tail -f logs/server_evaluation_*.log`
- Monitor system resources: `htop` and `nvidia-smi -l 1`