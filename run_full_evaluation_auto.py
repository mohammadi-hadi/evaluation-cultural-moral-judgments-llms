#!/usr/bin/env python3
"""
Full Dataset Evaluation Script - Auto-start version
"""

import os
import sys
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

# Configuration for FULL dataset evaluation
config = EnhancedExecutionConfig(
    dataset_size='full',  # Use full dataset
    n_samples=2091504,    # All 2.09M samples
    
    # API models
    api_models=[
        'gpt-3.5-turbo',    # Cheapest
        'gpt-4o-mini',      # Better quality
    ],
    
    # Local models
    local_models=[
        'mistral:latest',
        'wizardlm2:7b',
        'mistral-nemo:latest',
    ],
    
    # Parallel settings
    parallel_api_requests=5,
    parallel_local_models=2,
    
    # Limits
    max_api_cost=5000.0,
    max_memory_gb=50.0,
    
    # Output
    output_dir='outputs/full_evaluation',
    checkpoint_interval=1000,
    
    # Features
    auto_resume=True,
    max_retries=10,
)

print("\n📊 Configuration:")
print(f"- Samples: {config.n_samples:,}")
print(f"- API Models: {', '.join(config.api_models)}")
print(f"- Local Models: {', '.join(config.local_models)}")
print(f"- Output: {config.output_dir}")
print(f"- Max API Cost: ${config.max_api_cost}")
print(f"- Checkpoint every: {config.checkpoint_interval} samples")

print("\n🚀 Starting full dataset evaluation...")
print("📊 Monitor at: http://localhost:8501")
print("⚠️  Press Ctrl+C to pause (auto-resumes from checkpoint)")
print("="*70 + "\n")

# Start monitoring dashboard
os.system("streamlit run monitoring_dashboard.py --server.port 8501 --server.headless true &")

# Run evaluation
try:
    executor = EnhancedParallelExecutor(config)
    executor.run_enhanced_parallel_execution()
    print("\n✅ Evaluation completed successfully!")
except KeyboardInterrupt:
    print("\n⏸️  Paused. Run again to resume from checkpoint.")
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("📍 Run again to resume from checkpoint.")
