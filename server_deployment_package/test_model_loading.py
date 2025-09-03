#!/usr/bin/env python3
"""
Test script to verify model loading with fixed configuration
"""

import os
import sys
import json
import time
import torch
import logging
from pathlib import Path

# Import our fixed server model runner
from server_model_runner import ServerModelRunner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model_loading():
    """Test loading models with fixed configuration"""
    
    print("🧪 TESTING MODEL LOADING WITH FIXED CONFIGURATION")
    print("="*60)
    
    # Initialize runner
    base_dir = "/data/storage_4_tb/moral-alignment-pipeline"
    runner = ServerModelRunner(
        base_dir=base_dir,
        use_vllm=True,
        tensor_parallel_size=4
    )
    
    # Test models that were failing
    test_models = [
        "qwen2.5-72b",    # Was failing with int8 quantization
        "llama3.3-70b",   # Was failing with int8 quantization  
        "gpt-oss-120b",   # Was failing with mxfp4 quantization
    ]
    
    results = {}
    
    for model_name in test_models:
        print(f"\n🔧 Testing {model_name}...")
        print("-" * 40)
        
        try:
            start_time = time.time()
            
            # Try to load the model
            runner.load_model(model_name)
            
            load_time = time.time() - start_time
            
            # Test a simple inference
            test_prompt = "Hello, this is a test."
            
            if hasattr(runner.loaded_model, 'generate'):
                # VLLM model
                from vllm import SamplingParams
                sampling_params = SamplingParams(
                    temperature=0.7,
                    top_p=0.9,
                    max_tokens=50
                )
                outputs = runner.loaded_model.generate([test_prompt], sampling_params)
                response = outputs[0].outputs[0].text if outputs else ""
            else:
                # Transformers model
                inputs = runner.tokenizer(test_prompt, return_tensors="pt")
                with torch.no_grad():
                    outputs = runner.loaded_model.generate(
                        inputs.input_ids,
                        max_length=inputs.input_ids.shape[1] + 50,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=runner.tokenizer.eos_token_id
                    )
                response = runner.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            inference_time = time.time() - start_time - load_time
            
            results[model_name] = {
                'status': 'SUCCESS',
                'load_time': load_time,
                'inference_time': inference_time,
                'response': response[:100] + "..." if len(response) > 100 else response,
                'backend': 'VLLM' if hasattr(runner.loaded_model, 'generate') else 'Transformers'
            }
            
            print(f"✅ {model_name} loaded successfully!")
            print(f"   ⏱️  Load time: {load_time:.1f}s")
            print(f"   🚀 Inference time: {inference_time:.1f}s") 
            print(f"   🔧 Backend: {results[model_name]['backend']}")
            print(f"   📝 Response: {response[:50]}...")
            
            # Unload model to free memory for next test
            runner.unload_model()
            
        except Exception as e:
            results[model_name] = {
                'status': 'FAILED',
                'error': str(e),
                'load_time': 0,
                'inference_time': 0,
                'response': '',
                'backend': 'NONE'
            }
            
            print(f"❌ {model_name} failed: {e}")
            
            # Try to unload anyway
            try:
                runner.unload_model()
            except:
                pass
    
    # Print summary
    print(f"\n📊 TEST RESULTS SUMMARY")
    print("="*60)
    
    successful = sum(1 for r in results.values() if r['status'] == 'SUCCESS')
    total = len(results)
    
    print(f"✅ Successful: {successful}/{total}")
    print(f"❌ Failed: {total - successful}/{total}")
    
    for model_name, result in results.items():
        status_icon = "✅" if result['status'] == 'SUCCESS' else "❌"
        print(f"{status_icon} {model_name}: {result['status']} ({result['backend']})")
    
    # Save results
    results_file = Path("test_model_loading_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📁 Results saved to: {results_file}")
    
    return results

if __name__ == "__main__":
    try:
        results = test_model_loading()
        
        # Exit with appropriate code
        failed_count = sum(1 for r in results.values() if r['status'] == 'FAILED')
        sys.exit(failed_count)
        
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        sys.exit(1)