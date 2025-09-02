#!/usr/bin/env python3
"""
Full Dataset Evaluation Script for 2.09M Samples
Optimized for parallel execution with API and local models
"""

import os
import sys
import json
from pathlib import Path
from enhanced_parallel_executor import EnhancedParallelExecutor, EnhancedExecutionConfig

# Set OpenAI API key from environment
if 'OPENAI_API_KEY' not in os.environ:
    if os.path.exists('.env'):
        with open('.env') as f:
            for line in f:
                if 'OPENAI_API_KEY=' in line:
                    key = line.split('=', 1)[1].strip()
                    os.environ['OPENAI_API_KEY'] = key
                    break

print("="*70)
print("FULL DATASET MORAL ALIGNMENT EVALUATION")
print("="*70)
print("Dataset: World Values Survey (WVS)")
print("Total samples: 2,091,504")
print("="*70)

# Ask for confirmation given the scale
response = input("\n⚠️  WARNING: This will process 2.09M samples!\n" +
                "Estimated costs:\n" +
                "- API costs: $2000-4000\n" + 
                "- Time: 7-14 days with rate limits\n" +
                "- Storage: ~10GB\n\n" +
                "Do you want to proceed? (yes/no): ").strip().lower()

if response != 'yes':
    print("Evaluation cancelled.")
    sys.exit(0)

# Configuration for FULL dataset evaluation
config = EnhancedExecutionConfig(
    dataset_size='full',  # Use full dataset
    n_samples=2091504,    # All 2.09M samples
    
    # API models - use most cost-effective
    api_models=[
        'gpt-3.5-turbo',    # Cheapest, good baseline
        'gpt-4o-mini',      # Better quality, still affordable
        # 'gpt-4o',         # Uncomment only if budget allows ($$$)
    ],
    
    # Local models that work on M4 Max
    local_models=[
        # Small models that can run efficiently
        'mistral:latest',       # 7B model
        'wizardlm2:7b',        # 4.1GB model
        'mistral-nemo:latest', # 7.1GB model
        # Add more if available and working
    ],
    
    # Optimize for throughput
    parallel_api_requests=5,   # Max allowed by OpenAI
    parallel_local_models=2,   # Balance between speed and memory
    
    # Cost and resource management
    max_api_cost=5000.0,  # $5000 hard limit
    max_memory_gb=50.0,   # Leave room for system
    
    # Output and checkpointing
    output_dir='outputs/full_evaluation',
    checkpoint_interval=1000,  # Save every 1000 samples
    
    # Enhanced features for long-running evaluation
    auto_resume=True,     # Essential for multi-day run
    max_retries=10,       # More retries for reliability
)

# Calculate estimates
print("\n📊 Evaluation Estimates:")
print(f"- Samples per model: {config.n_samples:,}")
print(f"- Total evaluations: {config.n_samples * (len(config.api_models) + len(config.local_models)):,}")

# API cost estimates (rough)
api_cost_per_1k = {
    'gpt-3.5-turbo': 2.0,    # ~$2 per 1K samples
    'gpt-4o-mini': 3.0,      # ~$3 per 1K samples  
    'gpt-4o': 20.0,          # ~$20 per 1K samples
}

total_api_cost = 0
for model in config.api_models:
    cost = (config.n_samples / 1000) * api_cost_per_1k.get(model, 5.0)
    total_api_cost += cost
    print(f"- {model}: ~${cost:,.0f}")

print(f"\n💰 Total estimated API cost: ${total_api_cost:,.0f}")

# Time estimates
api_time_hours = (config.n_samples * len(config.api_models)) / 3600  # ~1 sec per sample with retries
local_time_hours = (config.n_samples * len(config.local_models)) / 7200  # ~0.5 sec per sample

print(f"\n⏱️  Time estimates:")
print(f"- API models: ~{api_time_hours/24:.1f} days")
print(f"- Local models: ~{local_time_hours/24:.1f} days")
print(f"- Total (parallel): ~{max(api_time_hours, local_time_hours)/24:.1f} days")

print("\n" + "="*70)
print("📍 Checkpoint system enabled - can safely interrupt and resume")
print("📊 Monitor progress at: http://localhost:8501")
print("📁 Results saved to: outputs/full_evaluation/")
print("="*70)

# Final confirmation
response = input("\n🚀 Ready to start. Type 'START' to begin: ").strip().upper()
if response != 'START':
    print("Evaluation cancelled.")
    sys.exit(0)

print("\n🎯 Starting full dataset evaluation...")
print("⚠️  Press Ctrl+C to pause (will auto-resume from checkpoint)")
print("-"*70 + "\n")

# Run the evaluation
try:
    executor = EnhancedParallelExecutor(config)
    executor.run_enhanced_parallel_execution()
except KeyboardInterrupt:
    print("\n\n⏸️  Evaluation paused. Run this script again to resume from checkpoint.")
    print(f"📍 Progress saved in: {config.output_dir}/results.db")
except Exception as e:
    print(f"\n❌ Error: {e}")
    print(f"📍 Progress saved. Run script again to resume from checkpoint.")
