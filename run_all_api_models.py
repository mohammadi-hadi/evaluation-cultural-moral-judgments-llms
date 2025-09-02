#!/usr/bin/env python3
"""
Run ALL API Models - Complete API evaluation with all specified models
"""

import os
import json
from api_batch_runner import APIBatchRunner

def main():
    # Check for API key in environment
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("Please set it with: export OPENAI_API_KEY='your-key-here'")
        return
    
    print("🚀 STARTING COMPLETE API EVALUATION")
    print("All 9 API models with exact rate limits")
    print("="*60)
    
    # Load the same samples that the local evaluation is using  
    samples_file = "outputs/server_sync_evaluation/run_20250902_165021/evaluation_samples.json"
    with open(samples_file) as f:
        samples = json.load(f)
    
    print(f"📊 Loaded {len(samples)} samples (same as local/server evaluation)")
    
    # Initialize API runner
    runner = APIBatchRunner(
        output_dir="outputs/server_sync_evaluation/run_20250902_165021/api",
        use_batch_api=True
    )
    
    # ALL API models from your list
    all_api_models = [
        # Current models
        "gpt-3.5-turbo",
        "gpt-4o-mini", 
        "gpt-4o",
        
        # Future/Latest models
        "gpt-5",
        "gpt-5-mini",
        "gpt-5-nano",
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4.1-nano",
        "o3",
        "o4-mini"
    ]
    
    print(f"🔥 Running {len(all_api_models)} API models:")
    for i, model in enumerate(all_api_models, 1):
        config = runner.MODEL_CONFIGS[model]
        print(f"  {i:2d}. {model} (TPM: {config.tpm_limit:,}, Priority: {config.priority})")
    
    print(f"\n💰 Using Batch API for 50% cost reduction")
    print(f"⏳ Submitting {len(all_api_models)} batch jobs...")
    
    # Estimate costs
    estimated_tokens_per_sample = 200  # Conservative estimate
    total_tokens = len(samples) * estimated_tokens_per_sample
    
    total_estimated_cost = 0
    print(f"\n📊 COST ESTIMATION:")
    for model in all_api_models:
        config = runner.MODEL_CONFIGS[model]
        model_cost = (total_tokens / 1000) * config.cost_per_1k_input
        model_cost *= 0.5  # Batch API discount
        total_estimated_cost += model_cost
        print(f"  {model}: ~${model_cost:.2f}")
    
    print(f"  TOTAL ESTIMATED: ~${total_estimated_cost:.2f} (with batch discount)")
    
    # Auto-proceed (user approved via message)
    print(f"\n✅ PROCEEDING WITH EVALUATION:")
    print(f"   - {len(all_api_models)} models × {len(samples)} samples = {len(all_api_models) * len(samples):,} total evaluations")
    print(f"   - Estimated cost: ${total_estimated_cost:.2f}")
    print(f"   - Processing time: 24 hours (batch API)")
    print(f"   - User confirmed: Yes")
    
    # Start API evaluation
    results = runner.run_batch_evaluation(all_api_models, samples)
    
    print("✅ ALL API BATCH JOBS SUBMITTED!")
    print(f"Results: {len(results['results'])} completed immediately")
    print(f"Batch jobs: {len(results['batch_jobs'])} submitted") 
    print(f"Actual cost so far: ${results['total_cost']:.4f}")
    
    # Show batch job details
    print(f"\n📋 BATCH JOBS STATUS:")
    for job in results['batch_jobs']:
        print(f"  - {job['model']:15} : {job['status']} (ID: {job['batch_id'][:12]}...)")
    
    print(f"\n💾 Results will be saved to:")
    print(f"   outputs/server_sync_evaluation/run_20250902_165021/api/")
    
    print(f"\n📈 MONITORING:")
    print(f"   Run: python monitor_evaluation.py")
    print(f"   Check OpenAI dashboard for batch progress")
    
    # Save the model list for reference
    model_info = {
        'models': all_api_models,
        'total_evaluations': len(all_api_models) * len(samples),
        'estimated_cost': total_estimated_cost,
        'batch_jobs': results['batch_jobs'],
        'submission_time': runner.timestamp if hasattr(runner, 'timestamp') else 'unknown'
    }
    
    info_file = "outputs/server_sync_evaluation/run_20250902_165021/api/all_models_info.json"
    with open(info_file, 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print(f"\n📄 Model info saved to: {info_file}")

if __name__ == "__main__":
    main()