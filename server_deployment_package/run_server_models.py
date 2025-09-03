#!/usr/bin/env python3
"""
Standalone Server Model Runner for Large Models (32B+) on 4xA100 GPUs
Optimized for running only large models with maximum GPU utilization
"""

import os
import sys
import json
import time
import torch
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Import optimized modules
from server_model_runner import ServerModelRunner
from download_models import ModelDownloader
from gpu_monitor import GPUMonitor
from batch_processor import BatchProcessor
from load_exact_samples import load_exact_samples

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_directories(base_dir: str = "/data/storage_4_tb/moral-alignment-pipeline") -> Dict[str, Path]:
    """Setup directory structure"""
    base_dir = Path(base_dir)
    dirs = {
        'base': base_dir,
        'data': base_dir / "data",
        'models': base_dir / "models", 
        'output': base_dir / "outputs",
        'results': base_dir / "outputs" / "server_results"
    }
    
    # Create directories
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Base directory: {dirs['base']}")
    logger.info(f"Results directory: {dirs['results']}")
    
    return dirs

def check_gpu_status() -> Dict[str, Any]:
    """Check and report GPU status"""
    if not torch.cuda.is_available():
        logger.error("❌ No GPUs available! This script requires GPUs.")
        sys.exit(1)
    
    n_gpus = torch.cuda.device_count()
    total_memory = 0
    gpu_info = []
    
    for i in range(n_gpus):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / (1024**3)
        total_memory += memory_gb
        gpu_info.append({
            'id': i,
            'name': props.name,
            'memory_gb': memory_gb
        })
        logger.info(f"GPU {i}: {props.name} - {memory_gb:.1f}GB")
    
    logger.info(f"Total GPU Memory: {total_memory:.1f}GB")
    
    return {
        'count': n_gpus,
        'total_memory': total_memory,
        'gpus': gpu_info
    }

def get_server_models() -> List[str]:
    """Get list of large models for server evaluation (32B+) - FROM ACTUAL DOWNLOADED MODELS"""
    return [
        "qwen2.5-32b",     # 32B model - needs 2 GPUs
        "qwq-32b",         # 32B model - needs 2 GPUs  
        "llama3.3-70b",    # 70B model - needs 4 GPUs  
        "qwen2.5-72b",     # 72B model - needs 4 GPUs
        "gpt-oss-120b",    # 120B model - needs 4 GPUs
    ]

def run_server_evaluation(base_dir: str = None, max_samples: int = 5000) -> List[Dict[str, Any]]:
    """Run server evaluation on large models only"""
    
    # Setup
    dirs = setup_directories(base_dir)
    gpu_info = check_gpu_status()
    
    print("\n" + "="*80)
    print("🚀 SERVER MODEL EVALUATION - LARGE MODELS ONLY")
    print("="*80)
    print("🎯 STRATEGY: Run only LARGE models on server (32B+)")
    print("🔧 Small models handled by local M4 Max evaluation")
    print(f"⚡ GPU Setup: {gpu_info['count']} GPUs, {gpu_info['total_memory']:.1f}GB total")
    
    # Initialize runner
    logger.info("Initializing ServerModelRunner...")
    runner = ServerModelRunner(
        base_dir=str(dirs['base']),
        use_vllm=True,
        tensor_parallel_size=4
    )
    
    # Get available models
    available_models = runner.get_available_models()
    logger.info(f"Available models on disk: {len(available_models)}")
    
    # Get server models (large only)
    server_models = get_server_models()
    models_to_evaluate = []
    
    print(f"\n📋 LARGE MODELS FOR SERVER EVALUATION:")
    for model_name in server_models:
        if model_name in available_models:
            models_to_evaluate.append(model_name)
            model_config = runner.MODEL_CONFIGS.get(model_name, {})
            size_gb = getattr(model_config, 'size_gb', 'unknown')
            gpu_config = runner.get_optimal_gpu_config(model_name)
            tensor_parallel = gpu_config.get('tensor_parallel', 1)
            print(f"  ✅ {model_name} ({size_gb}GB) → {tensor_parallel} GPUs")
        else:
            print(f"  ⏭️ {model_name} (not downloaded)")
    
    if not models_to_evaluate:
        logger.error("❌ No large models available for evaluation!")
        return []
    
    print(f"\n📊 Ready to evaluate {len(models_to_evaluate)} large models")
    
    # Load samples (EXACT same as local/API evaluation)
    print("\n🎯 Loading EXACT samples (same as local/API evaluation)")
    samples = load_exact_samples()
    
    if not samples:
        logger.error("❌ Failed to load samples!")
        return []
    
    # Use subset if requested
    eval_samples = samples[:max_samples]
    print(f"✅ Loaded {len(eval_samples)} EXACT samples")
    
    # Run evaluation
    all_results = []
    total_start_time = time.time()
    
    for model_name in models_to_evaluate:
        print(f"\n{'='*60}")
        print(f"🚀 EVALUATING: {model_name}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            # Use optimized evaluation method
            results = runner.evaluate_model_complete(model_name, eval_samples)
            all_results.extend(results)
            
            # Calculate statistics
            total_time = time.time() - start_time
            successful = sum(1 for r in results if r.get('success', False))
            
            print(f"\n✅ EVALUATION COMPLETE: {model_name}")
            print(f"   📊 Total samples: {len(results)}")
            print(f"   ✅ Successful: {successful} ({successful/len(results)*100:.1f}%)")
            print(f"   ⏱️  Total time: {total_time:.1f}s")
            print(f"   🚀 Speed: {len(results)/total_time:.1f} samples/sec")
            
            # Save individual model results
            output_file = dirs['results'] / f"{model_name}_results.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"   💾 Saved: {output_file}")
            
        except Exception as e:
            logger.error(f"❌ Error evaluating {model_name}: {e}")
            
            # Create error results
            error_results = []
            for i, sample in enumerate(eval_samples):
                error_result = {
                    'model': model_name,
                    'sample_id': sample.get('id', f'sample_{i}'),
                    'error': str(e),
                    'success': False,
                    'response': '',
                    'inference_time': 0,
                    'timestamp': datetime.now().isoformat()
                }
                error_results.append(error_result)
            all_results.extend(error_results)
    
    # Final statistics
    total_time = time.time() - total_start_time
    successful_results = sum(1 for r in all_results if r.get('success', False))
    
    print(f"\n🎉 SERVER EVALUATION COMPLETE!")
    print("="*80)
    print(f"   🚀 Models processed: {len(models_to_evaluate)}")
    print(f"   📊 Total results: {len(all_results):,}")
    print(f"   ✅ Successful: {successful_results:,} ({successful_results/len(all_results)*100:.1f}%)")
    print(f"   ⏱️  Total time: {total_time/60:.1f} minutes")
    print(f"   🎯 Average speed: {len(all_results)/total_time:.1f} samples/sec")
    print(f"   ⚡ GPU utilization: MAXIMIZED")
    
    # Save combined results for integration
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Standardized format for integration
    integration_file = dirs['output'] / f"server_results_for_integration_{timestamp}.json"
    standardized_results = []
    
    for result in all_results:
        if result.get('success', False):
            standardized_result = {
                'model': result['model'],
                'sample_id': result.get('sample_id', ''),
                'response': result.get('response', ''),
                'choice': result.get('choice', 'unknown'),
                'inference_time': result.get('inference_time', 0),
                'success': result.get('success', False),
                'timestamp': result.get('timestamp', timestamp),
                'evaluation_type': 'server'
            }
            standardized_results.append(standardized_result)
    
    with open(integration_file, 'w') as f:
        json.dump(standardized_results, f, indent=2)
    
    # Create metadata
    metadata = {
        'evaluation_type': 'server',
        'timestamp': timestamp,
        'total_samples': len(eval_samples),
        'total_models': len(models_to_evaluate),
        'total_successful_results': len(standardized_results),
        'models_evaluated': models_to_evaluate,
        'dataset_info': {
            'same_samples_as_local_api': True,
            'sample_count': len(eval_samples),
            'countries': 64,
            'moral_questions': 13,
            'source': 'World Values Survey'
        },
        'gpu_setup': {
            'gpu_count': gpu_info['count'],
            'gpu_type': '4x A100',
            'total_memory_gb': gpu_info['total_memory']
        }
    }
    
    metadata_file = dirs['output'] / f"server_metadata_for_integration_{timestamp}.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n📁 INTEGRATION FILES READY:")
    print(f"   📊 Results: {integration_file}")
    print(f"   📋 Metadata: {metadata_file}")
    print(f"   🔗 Compatible with local evaluation results")
    
    return all_results

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Server Model Evaluation for Large Models")
    parser.add_argument("--base-dir", type=str, 
                       default="/data/storage_4_tb/moral-alignment-pipeline",
                       help="Base directory for models and outputs")
    parser.add_argument("--max-samples", type=int, default=5000,
                       help="Maximum number of samples to evaluate")
    parser.add_argument("--models", nargs="+", 
                       help="Specific models to evaluate (default: all large models)")
    
    args = parser.parse_args()
    
    try:
        results = run_server_evaluation(
            base_dir=args.base_dir,
            max_samples=args.max_samples
        )
        
        if results:
            print(f"\n🎯 EVALUATION SUCCESS!")
            print(f"Generated {len(results)} results for large models")
            print(f"Ready for integration with local results")
        else:
            print(f"\n❌ EVALUATION FAILED!")
            print(f"No results generated")
            
    except KeyboardInterrupt:
        print(f"\n⚠️ Evaluation interrupted by user")
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        raise

if __name__ == "__main__":
    main()