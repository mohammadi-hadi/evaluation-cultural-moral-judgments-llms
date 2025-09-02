# EXACT Server Instructions - Same Dataset as Local/API

## 🎯 **CRITICAL: Use Same Dataset**

I'm currently running **local + API evaluation** on the exact same 5000 samples from `test_dataset_5000.csv`. You must use the **EXACT same samples** for comparison.

### **Current Evaluation Status:**
- ✅ **Local Models**: Running (6 models: llama3.2:3b, phi4:14b, mistral, qwen2.5:7b, gemma2:2b, gpt-oss:20b)
- ✅ **API Models**: 3 batch jobs submitted (gpt-3.5-turbo, gpt-4o-mini, gpt-4o)
- 🔄 **Server Models**: Ready to run with exact same samples

## 🚀 **Transfer to Server:**

```bash
# Copy the deployment package + exact samples
scp -r server_deployment_package/ root@52.178.4.252:/tmp/
```

## 🖥️ **On Your SURF Server:**

### **1. Setup:**
```bash
ssh root@52.178.4.252
cd /tmp/server_deployment_package
chmod +x setup_server.sh
./setup_server.sh
```

### **2. Use Jupyter Notebook (Recommended):**

Access: https://jupyter4a1004tb.ocre-explainabl.src.surf-hosted.nl

**REPLACE the sample generation section with:**

```python
# EXACT SAMPLES - Same as local/API evaluation
import sys
sys.path.append('/data/storage_4_tb/moral-alignment-pipeline')
from load_exact_samples import load_exact_samples

# Load the EXACT same samples
samples = load_exact_samples()
print(f"Loaded {len(samples)} exact samples (same as local/API)")
```

### **3. Or Command Line:**
```bash
cd /data/storage_4_tb/moral-alignment-pipeline
python run_evaluation.py \
    --dataset server_samples.json \
    --models llama3.1-70b qwen2.5-32b mistral-large \
    --samples 5000 \
    --use-vllm \
    --tensor-parallel-size 4
```

## 📊 **Sample Data Details:**

**What I'm running locally:**
- **5000 samples** from real WVS (World Values Survey) data
- **64 countries** represented
- **13 moral questions** (Q176-Q188)
- **Stratified sampling** ensuring all countries/questions included

**Sample format:**
```json
{
  "id": "sample_0000_Q176",
  "prompt": "On a scale from 1-10, how would you rate: claiming government benefits you're not entitled to?...",
  "country": "20",
  "question": "Q176", 
  "human_response": 9
}
```

## ⚠️ **DISABLE Sample Generation in Notebook**

**Replace this section in the notebook:**
```python
# OLD (DELETE THIS):
if not dataset_path.exists():
    print("Creating sample dataset...")
    moral_questions = [...]
    # ... sample generation code
```

**With this:**
```python
# NEW (USE THIS):
from load_exact_samples import load_exact_samples
samples = load_exact_samples()
print(f"Using exact samples: {len(samples)} (same as local/API)")
```

## 🎯 **Result Comparison**

After server completes, we'll have:
- **Local Results**: 6 Ollama models × 5000 samples = 30,000 evaluations
- **API Results**: 3 OpenAI models × 5000 samples = 15,000 evaluations  
- **Server Results**: Multiple large models × 5000 samples = X evaluations
- **SAME DATASET**: All using identical WVS moral questions

## 📈 **Current Progress:**

**Local Evaluation**: ~16% complete (running in background)
**API Evaluation**: Batch jobs submitted (24h processing)
**Server Evaluation**: Ready for you to start

All three approaches will evaluate the **exact same 5000 moral scenarios** from the World Values Survey, ensuring perfect comparability!

The `server_samples.json` file contains the exact samples I'm using locally.