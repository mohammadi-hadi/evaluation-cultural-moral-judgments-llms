#!/usr/bin/env python3
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

# Configuration for comprehensive evaluation
config = EnhancedExecutionConfig(
    dataset_size='sample',
    n_samples=1000,  # Start with 1000 samples
    
    # API models (immediate)
    api_models=[
        'gpt-3.5-turbo',    # Baseline
        'gpt-4o-mini',      # Cost-effective
    ],
    
    # Local models available on your system
    local_models=[
        'mistral:latest',       # You have this (7B)
        'mistral-nemo:latest',  # You have this (7.1GB)
        'wizardlm2:7b',        # You have this (4.1GB)
        # Skip neural-chat due to temperature issue
        # 'neural-chat:latest',  
    ],
    
    # Parallel execution settings
    parallel_api_requests=3,  # Moderate to avoid rate limits
    parallel_local_models=2,  # Run 2 local models at once
    
    # Cost and resource limits
    max_api_cost=10.0,  # $10 limit for safety
    max_memory_gb=50.0,  # Leave 14GB for system
    
    # Output and checkpointing
    output_dir='outputs/comprehensive',
    checkpoint_interval=100,  # Save every 100 samples
    
    # Enhanced features
    auto_resume=True,  # Resume from checkpoint if interrupted
    max_retries=5     # Retry failed requests
)

print("="*70)
print("COMPREHENSIVE MORAL ALIGNMENT EVALUATION")
print("="*70)
print(f"Dataset: {config.n_samples} samples")
print(f"API Models: {', '.join(config.api_models)}")
print(f"Local Models: {', '.join(config.local_models)}")
print(f"Max API Cost: ${config.max_api_cost}")
print(f"Output: {config.output_dir}")
print("="*70)
print("\n📊 Starting evaluation with real-time monitoring...")
print("💡 Dashboard available at: http://localhost:8501")
print("⚠️  Press Ctrl+C to pause (will resume from checkpoint)\n")

# Run the evaluation
executor = EnhancedParallelExecutor(config)
executor.run_enhanced_parallel_execution()
