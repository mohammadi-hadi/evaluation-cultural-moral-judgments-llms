#!/usr/bin/env python3
"""
Test script for comprehensive evaluation system
Quick test with minimal samples to verify everything works
"""

import os
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_system():
    """Test the comprehensive evaluation system"""
    
    print("="*70)
    print("COMPREHENSIVE EVALUATION SYSTEM TEST")
    print("="*70)
    
    # 1. Test imports
    print("\n1. Testing imports...")
    try:
        from local_model_runner import LocalModelRunner
        from api_model_runner import APIModelRunner
        from parallel_executor import ParallelExecutor, ExecutionConfig
        from run_comprehensive_evaluation import ComprehensiveEvaluator
        print("✅ All imports successful")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Please install requirements: pip install -r requirements.txt")
        return False
    
    # 2. Test local model runner
    print("\n2. Testing local model runner...")
    try:
        local_runner = LocalModelRunner(max_memory_gb=20)
        available = local_runner.list_available_models()
        
        print(f"✅ Local runner initialized")
        print(f"   Configured models: {len(available['configured'])}")
        print(f"   Ollama models: {len(available['ollama'])}")
        
        if available['ollama']:
            print(f"   Available Ollama models: {', '.join(available['ollama'][:3])}...")
    except Exception as e:
        print(f"⚠️  Local runner warning: {e}")
    
    # 3. Test API model runner
    print("\n3. Testing API model runner...")
    if os.getenv('OPENAI_API_KEY'):
        try:
            api_runner = APIModelRunner()
            print("✅ API runner initialized")
            
            # Estimate costs
            estimate = api_runner.estimate_cost("gpt-3.5-turbo", n_prompts=100)
            print(f"   Cost estimate for 100 prompts: ${estimate['estimated_cost']:.2f}")
        except Exception as e:
            print(f"❌ API runner error: {e}")
    else:
        print("⚠️  OpenAI API key not found - skipping API tests")
    
    # 4. Test parallel executor
    print("\n4. Testing parallel executor...")
    try:
        config = ExecutionConfig(
            dataset_size="sample",
            n_samples=10,  # Very small for testing
            api_models=[],
            local_models=[],
            output_dir="outputs/test"
        )
        
        executor = ParallelExecutor(config)
        print("✅ Parallel executor initialized")
        print(f"   Database created: {executor.db_path}")
    except Exception as e:
        print(f"❌ Parallel executor error: {e}")
        return False
    
    # 5. Quick integration test
    print("\n5. Running quick integration test...")
    print("   This will test with 5 samples on available models")
    
    response = input("\nRun quick test? (y/n): ").strip().lower()
    
    if response == 'y':
        try:
            # Get one API model and one local model
            api_models = ["gpt-3.5-turbo"] if os.getenv('OPENAI_API_KEY') else []
            
            local_models = []
            if available['ollama']:
                local_models = [available['ollama'][0]]  # Use first available
            elif available['configured']:
                local_models = ["gpt2"]  # Fallback to GPT-2
            
            print(f"\n   Testing with:")
            print(f"   - API models: {api_models}")
            print(f"   - Local models: {local_models}")
            
            config = ExecutionConfig(
                dataset_size="sample",
                n_samples=5,
                api_models=api_models,
                local_models=local_models,
                parallel_api_requests=1,
                parallel_local_models=1,
                max_api_cost=1.0,
                output_dir="outputs/quick_test"
            )
            
            executor = ParallelExecutor(config)
            executor.run_parallel_execution()
            
            print("\n✅ Integration test complete!")
            print(f"   Results saved to: outputs/quick_test/")
            
        except Exception as e:
            print(f"❌ Integration test failed: {e}")
            return False
    
    print("\n" + "="*70)
    print("SYSTEM TEST COMPLETE")
    print("="*70)
    
    print("\nTo run full evaluation:")
    print("  python run_comprehensive_evaluation.py --dataset-size sample --n-samples 100")
    
    print("\nTo run with specific models:")
    print("  python run_comprehensive_evaluation.py --phases api local_small")
    
    return True


if __name__ == "__main__":
    success = test_system()
    sys.exit(0 if success else 1)