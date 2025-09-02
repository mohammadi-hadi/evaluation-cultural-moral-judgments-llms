# Comprehensive Moral Alignment Evaluation System

Complete system for evaluating moral alignment across API models (OpenAI) and local models on M4 Max, with preparation for server deployment on 4xA100 GPUs.

## 🚀 Quick Start

### 1. Test the System
```bash
python test_comprehensive_system.py
```

### 2. Run Quick Evaluation (5 minutes)
```bash
# Small test with 100 samples
python run_comprehensive_evaluation.py \
    --dataset-size sample \
    --n-samples 100 \
    --phases api local_small
```

### 3. Run Full Local Evaluation (24-48 hours)
```bash
# Complete evaluation with all local models
python run_comprehensive_evaluation.py \
    --dataset-size medium \
    --n-samples 10000 \
    --phases all \
    --max-api-cost 100
```

## 📊 Model Categories

### Phase 1: API Models (Immediate)
```python
api_models = [
    "gpt-3.5-turbo",     # Baseline
    "gpt-4o-mini",       # Cost-effective  
    "gpt-4o",            # Best overall
    "o3-mini"            # Latest reasoning (if available)
]
```
**Cost**: ~$0.50-2.00 per 1000 samples
**Time**: ~20 min per 1000 samples

### Phase 2: Small Local Models (< 4GB)
Running on M4 Max with 64GB RAM:
```python
small_models = [
    "gpt2",              # 500MB - Essential baseline
    "opt-125m",          # 500MB
    "opt-350m",          # 1GB
    "bloomz-560m",       # 1.2GB
    "gemma:2b",          # 4GB (via Ollama)
    "llama3.2:1b",       # 2GB (via Ollama)
    "qwen2.5:1.5b"       # 3GB (via Ollama)
]
```
**Memory**: Can run 3-4 simultaneously
**Speed**: ~100-500 samples/min

### Phase 3: Medium Local Models (7-14GB)
```python
medium_models = [
    "mistral:7b",        # 14GB - You have this
    "neural-chat",       # 4.1GB - You have this
    "wizardlm2:7b",      # 4.1GB - You have this
    "mistral-nemo",      # 7.1GB - You have this
    "llama3.1:8b",       # 16GB (if available)
    "qwen2.5:7b",        # 14GB (if available)
]
```
**Memory**: Run 1-2 at a time
**Speed**: ~50-100 samples/min

### Phase 4: Server Models (4xA100 GPU)
Job scripts generated for:
```python
# 32B models (2 GPUs)
"Qwen/Qwen2.5-32B-Instruct"
"Qwen/QwQ-32B-Preview"  # Reasoning specialist

# 70B models (3 GPUs)
"meta-llama/Llama-3.3-70B-Instruct"  # Best open source
"Qwen/Qwen2.5-72B-Instruct"

# Massive models (4 GPUs)
"openai/gpt-oss-120b"
"meta-llama/Llama-3.1-405B-Instruct"
```

## 🏗️ System Architecture

### Components

1. **`local_model_runner.py`**
   - Unified interface for Ollama, Transformers, llama.cpp
   - Memory management for M4 Max
   - Automatic model detection and loading
   - Batch processing with checkpoints

2. **`api_model_runner.py`**
   - OpenAI API integration
   - Rate limiting (50 req/min)
   - Cost tracking and estimation
   - Response caching

3. **`parallel_executor.py`**
   - Concurrent API + local execution
   - SQLite database for results
   - Progress monitoring
   - Checkpoint/resume capability

4. **`run_comprehensive_evaluation.py`**
   - Master orchestrator
   - Phased execution
   - Server job generation
   - Comprehensive reporting

## 💾 Output Structure

```
outputs/comprehensive/
├── phase1_api/
│   ├── api_results/
│   ├── results.db
│   └── execution_summary.json
├── phase2_local_small/
│   ├── local_results/
│   ├── results.db
│   └── model_metrics.csv
├── phase3_local_medium/
│   └── ...
├── server_jobs/
│   ├── job_32b_models.sh
│   ├── job_70b_models.sh
│   └── job_massive_models.sh
└── evaluation_report.json
```

## 📈 Metrics Tracked

- **Moral Alignment Score**: [-1, 1] scale
- **Correlation with WVS**: Spearman's ρ
- **Self-Consistency**: Agreement between methods
- **Inference Time**: Tokens/second
- **Cost**: USD per model
- **Memory Usage**: GB per model

## 🖥️ Resource Requirements

### M4 Max (64GB RAM)
- **Available for models**: 50GB
- **System reserved**: 8GB
- **Data/cache**: 6GB

### Parallel Execution
- **API requests**: 5 concurrent
- **Small models**: 3-4 concurrent
- **Medium models**: 1-2 concurrent
- **Large quantized**: 1 at a time

## 📊 Performance Estimates

### Full Dataset (2.09M samples)
- **API Models**: ~700 hours @ $1000-2000
- **Local Small**: ~70 hours
- **Local Medium**: ~350 hours
- **Server 70B**: ~100 hours on 4xA100

### Recommended Approach
1. Start with 10K sample evaluation
2. Validate results match expectations
3. Scale to 100K for publication
4. Full 2.09M only if necessary

## 🔧 Configuration

### via YAML file
```yaml
dataset:
  size: medium          # sample/medium/full
  n_samples: 10000
  stratified: true

execution:
  phases: [api, local_small, local_medium]
  parallel_api: 5
  parallel_local: 2
  checkpoint_interval: 100

resources:
  max_memory_gb: 50.0
  max_api_cost: 100.0

output_dir: outputs/comprehensive
```

### via Command Line
```bash
python run_comprehensive_evaluation.py \
    --dataset-size medium \
    --n-samples 10000 \
    --phases api local_small \
    --max-api-cost 50 \
    --output-dir outputs/my_evaluation
```

## 🚦 Monitoring

While running, the system shows:
- Progress bars for each model
- Memory usage (critical for M4 Max)
- API costs in real-time
- ETA calculations
- Checkpoint saves

## 🔄 Resume from Checkpoint

The system automatically saves progress every 100-1000 samples. To resume:
```bash
# Just run the same command again
python run_comprehensive_evaluation.py --config evaluation_config.yaml
```

## 🎯 Next Steps After Local Evaluation

1. **Review Results**
   ```bash
   python comprehensive_visualizer.py
   ```

2. **Deploy to Server** (4xA100)
   ```bash
   # Copy project to server
   scp -r Project06/ user@server:/path/
   
   # On server:
   sbatch outputs/comprehensive/server_jobs/job_32b_models.sh
   ```

3. **Integrate All Results**
   ```bash
   python result_integrator.py
   ```

4. **Generate Paper Figures**
   ```bash
   python paper_outputs.py
   ```

## ⚠️ Important Notes

1. **API Costs**: Monitor costs carefully. Use `--max-api-cost` limit.
2. **Memory**: M4 Max has 64GB total, keep model usage under 50GB.
3. **Ollama Models**: Pull models first with `ollama pull model:tag`
4. **Checkpoints**: System saves progress automatically, safe to interrupt.
5. **Server Jobs**: Requires SLURM cluster with GPU nodes.

## 🐛 Troubleshooting

### Out of Memory
- Reduce `parallel_local_models` to 1
- Use quantized models
- Clear cache: `local_runner.cleanup_models()`

### API Rate Limits
- Reduce `parallel_api_requests`
- Add delays between requests
- Use caching to avoid duplicates

### Ollama Issues
- Ensure Ollama is running: `ollama serve`
- Pull models: `ollama pull mistral:7b`
- Check available: `ollama list`

## 📝 Example Workflow

```bash
# 1. Test system
python test_comprehensive_system.py

# 2. Quick test (5 samples, 2 models)
python run_comprehensive_evaluation.py \
    --dataset-size sample \
    --n-samples 5 \
    --phases api

# 3. Small evaluation (1000 samples)
python run_comprehensive_evaluation.py \
    --dataset-size sample \
    --n-samples 1000 \
    --phases all

# 4. Production run (10K samples)
python run_comprehensive_evaluation.py \
    --dataset-size medium \
    --n-samples 10000 \
    --phases all \
    --max-api-cost 100

# 5. Generate visualizations
python comprehensive_visualizer.py \
    --input outputs/comprehensive \
    --output outputs/figures
```

## 🎉 Success Metrics

Your evaluation is successful when:
- ✅ All models complete without errors
- ✅ Correlation with WVS > 0.5 for good models
- ✅ API models show higher alignment than baselines
- ✅ Costs stay within budget
- ✅ Results are reproducible

---

Ready to evaluate moral alignment across all available models! 🚀