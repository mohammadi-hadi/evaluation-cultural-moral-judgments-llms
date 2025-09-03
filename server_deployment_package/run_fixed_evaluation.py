#!/usr/bin/env python3
"""
Re-run server evaluation with fixed quantization configuration
"""

import os
import sys
import json
import time
import torch
import logging
from pathlib import Path
from datetime import datetime
from tqdm.auto import tqdm

# Import our fixed modules
from server_model_runner import ServerModelRunner
from load_exact_samples import load_exact_samples

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_fixed_evaluation():
    """Run server evaluation with fixed configuration"""
    
    print("🚀 RUNNING FIXED SERVER EVALUATION")
    print("="*60)
    
    # Configuration
    BASE_DIR = Path("/data/storage_4_tb/moral-alignment-pipeline")
    OUTPUT_DIR = BASE_DIR / "outputs" / "server_results"
    
    # Load the same 5000 samples as before
    print("📊 Loading exact same 5000 samples...")
    samples = load_exact_samples()
    print(f"✅ Loaded {len(samples)} samples")
    
    # Initialize server model runner with fixed configuration
    runner = ServerModelRunner(
        base_dir=str(BASE_DIR),
        use_vllm=True,
        tensor_parallel_size=4
    )
    
    # Models to evaluate (same as before but with fixed config)
    models_to_evaluate = [
        "qwen2.5-72b",    # 72B model - was failing with int8
        "llama3.3-70b",   # 70B model - was failing with int8
        "gpt-oss-120b",   # 120B model - was failing with mxfp4
    ]
    
    print(f"\n🎯 MODELS TO EVALUATE WITH FIXED CONFIGURATION:")
    for i, model in enumerate(models_to_evaluate, 1):
        print(f"  {i}. {model} → Fixed quantization + fallback")
    
    all_results = []
    evaluation_start = time.time()
    
    for i, model_name in enumerate(models_to_evaluate, 1):
        print(f"\n{'='*60}")
        print(f"🚀 EVALUATING MODEL {i}/{len(models_to_evaluate)}: {model_name}")
        print(f"{'='*60}")
        
        model_start = time.time()
        model_results = []
        
        try:
            # Load model with fixed configuration
            print(f"📥 Loading {model_name} with fixed configuration...")
            runner.load_model(model_name)
            load_time = time.time() - model_start
            print(f"✅ Model loaded in {load_time:.1f}s")
            
            # Evaluate all samples
            print(f"🔄 Processing {len(samples)} samples...")
            
            for sample_idx, sample in enumerate(tqdm(samples, desc=f"Evaluating {model_name}")):
                sample_start = time.time()
                
                try:
                    # Get the prompt
                    prompt = sample.get('prompt', '')
                    if not prompt:
                        raise ValueError("No prompt found in sample")
                    
                    # Generate response
                    if hasattr(runner.loaded_model, 'generate'):
                        # VLLM backend
                        from vllm import SamplingParams
                        sampling_params = SamplingParams(
                            temperature=0.7,
                            top_p=0.9,
                            max_tokens=100,
                            stop=None
                        )
                        outputs = runner.loaded_model.generate([prompt], sampling_params)
                        response = outputs[0].outputs[0].text if outputs else ""
                    else:
                        # Transformers backend
                        inputs = runner.tokenizer(prompt, return_tensors="pt")
                        with torch.no_grad():
                            outputs = runner.loaded_model.generate(
                                inputs.input_ids,
                                max_length=inputs.input_ids.shape[1] + 100,
                                temperature=0.7,
                                do_sample=True,
                                pad_token_id=runner.tokenizer.eos_token_id
                            )
                        response = runner.tokenizer.decode(outputs[0], skip_special_tokens=True)
                        # Remove the original prompt from response
                        response = response[len(prompt):].strip()
                    
                    inference_time = time.time() - sample_start
                    
                    # Create result
                    result = {
                        'model': model_name,
                        'sample_id': sample.get('id', f'sample_{sample_idx}'),
                        'success': True,
                        'response': response,
                        'inference_time': inference_time,
                        'timestamp': datetime.now().isoformat(),
                        'evaluation_type': 'server_fixed',
                        'backend_used': 'VLLM' if hasattr(runner.loaded_model, 'generate') else 'Transformers'
                    }
                    
                    model_results.append(result)
                    all_results.append(result)
                    
                except Exception as sample_error:
                    # Create error result
                    error_result = {
                        'model': model_name,
                        'sample_id': sample.get('id', f'sample_{sample_idx}'),
                        'success': False,
                        'response': '',
                        'inference_time': 0,
                        'error': str(sample_error),
                        'timestamp': datetime.now().isoformat(),
                        'evaluation_type': 'server_fixed',
                        'backend_used': 'ERROR'
                    }
                    
                    model_results.append(error_result)
                    all_results.append(error_result)
                    
                    if sample_idx % 100 == 0:  # Log errors periodically
                        logger.warning(f"Sample {sample_idx} failed: {sample_error}")
            
            # Calculate model statistics
            model_time = time.time() - model_start
            successful_samples = sum(1 for r in model_results if r.get('success', False))
            success_rate = successful_samples / len(model_results)
            avg_inference_time = sum(r.get('inference_time', 0) for r in model_results if r.get('success', False))
            avg_inference_time = avg_inference_time / successful_samples if successful_samples > 0 else 0
            
            print(f"\n✅ MODEL {i} COMPLETE: {model_name}")
            print(f"   📊 Total samples: {len(model_results):,}")
            print(f"   ✅ Successful: {successful_samples:,} ({success_rate:.1%})")
            print(f"   ❌ Failed: {len(model_results) - successful_samples:,}")
            print(f"   ⏱️  Total time: {model_time:.1f}s ({model_time/60:.1f} min)")
            print(f"   🚀 Avg inference: {avg_inference_time:.3f}s")
            print(f"   🔧 Backend: {model_results[0].get('backend_used', 'Unknown')}")
            
            # Save individual model results
            output_file = OUTPUT_DIR / f"{model_name}_results_fixed.json"
            with open(output_file, 'w') as f:
                json.dump(model_results, f, indent=2)
            print(f"   💾 Saved to: {output_file}")
            
            # Unload model to free memory
            runner.unload_model()
            
        except Exception as model_error:
            print(f"❌ MODEL {i} FAILED: {model_name}")
            print(f"   Error: {model_error}")
            
            # Create error results for all samples
            error_results = []
            for sample_idx, sample in enumerate(samples):
                error_result = {
                    'model': model_name,
                    'sample_id': sample.get('id', f'sample_{sample_idx}'),
                    'success': False,
                    'response': '',
                    'inference_time': 0,
                    'error': str(model_error),
                    'timestamp': datetime.now().isoformat(),
                    'evaluation_type': 'server_fixed',
                    'backend_used': 'ERROR'
                }
                error_results.append(error_result)
                all_results.append(error_result)
            
            model_results = error_results
            
            # Still save the error results
            output_file = OUTPUT_DIR / f"{model_name}_results_fixed.json"
            with open(output_file, 'w') as f:
                json.dump(error_results, f, indent=2)
            print(f"   💾 Error results saved to: {output_file}")
    
    # Final summary
    total_time = time.time() - evaluation_start
    successful_results = sum(1 for r in all_results if r.get('success', False))
    total_results = len(all_results)
    overall_success_rate = successful_results / total_results if total_results > 0 else 0
    
    print(f"\n🎉 FIXED EVALUATION COMPLETE!")
    print("="*60)
    print(f"📊 FINAL RESULTS:")
    print(f"   🎯 Models evaluated: {len(models_to_evaluate)}")
    print(f"   📊 Total results: {total_results:,}")
    print(f"   ✅ Successful: {successful_results:,} ({overall_success_rate:.1%})")
    print(f"   ❌ Failed: {total_results - successful_results:,}")
    print(f"   ⏱️  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    
    # Save combined results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    combined_file = OUTPUT_DIR / f"server_evaluation_fixed_{timestamp}.json"
    
    with open(combined_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"   💾 Combined results: {combined_file}")
    
    # Create summary report
    summary = {
        'evaluation_type': 'server_fixed',
        'timestamp': timestamp,
        'configuration_fixes': [
            'Fixed quantization methods (int8 → bitsandbytes, int4 → awq)',
            'Fixed dtype issues (mxfp4 → bfloat16)',
            'Added fallback to Transformers when VLLM fails',
            'Added retry without quantization',
        ],
        'models_evaluated': models_to_evaluate,
        'total_results': total_results,
        'successful_results': successful_results,
        'overall_success_rate': overall_success_rate,
        'total_time_minutes': total_time / 60,
        'samples_per_model': len(samples),
        'fixes_applied': True,
        'ready_for_integration': successful_results > 0
    }
    
    summary_file = OUTPUT_DIR / f"evaluation_summary_fixed_{timestamp}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"   📄 Summary report: {summary_file}")
    
    if overall_success_rate > 0:
        print(f"\n🎯 SUCCESS! Fixed configuration is working!")
        print(f"   Ready for full integration with API/Local results")
    else:
        print(f"\n❌ All models still failing. Need further investigation.")
    
    return all_results, summary

if __name__ == "__main__":
    try:
        results, summary = run_fixed_evaluation()
        
        # Exit with success if we have any successful results
        if summary['successful_results'] > 0:
            print(f"\n✅ Evaluation completed with {summary['successful_results']} successful results")
            sys.exit(0)
        else:
            print(f"\n❌ No successful results obtained")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️  Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        sys.exit(1)