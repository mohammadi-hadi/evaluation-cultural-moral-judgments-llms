#!/usr/bin/env python3
"""
Local Model Evaluation Script for M4 Max
Provides IDENTICAL output format to server evaluation for seamless integration
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import subprocess
import requests
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LocalModelEvaluator:
    """Evaluate small models locally with IDENTICAL output format to server"""
    
    def __init__(self, base_dir: str = None, ollama_url: str = "http://localhost:11434"):
        """Initialize local evaluator"""
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.ollama_url = ollama_url
        self.output_dir = self.base_dir / "outputs" / "local_results"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Models that run efficiently on M4 Max (64GB RAM) - ACTUAL DOWNLOADED SMALL MODELS
        self.local_models = {
            # Ultra-small models (1-4B parameters)
            "gpt2": {"size_gb": 1, "priority": "HIGH", "notes": "Baseline model"},
            "llama3.2:1b": {"size_gb": 2, "priority": "HIGH", "notes": "Ultra-fast small model"},
            "llama3.2:3b": {"size_gb": 6, "priority": "HIGH", "notes": "Efficient small model"},
            "gemma3:4b": {"size_gb": 8, "priority": "HIGH", "notes": "Latest Gemma generation"},
            "phi3:3.8b": {"size_gb": 8, "priority": "HIGH", "notes": "Microsoft's efficient model"},
            "phi-3.5-mini": {"size_gb": 8, "priority": "HIGH", "notes": "Phi 3.5 mini"},
            
            # Small models (7-9B parameters)
            "mistral:7b": {"size_gb": 14, "priority": "HIGH", "notes": "Moved from server for efficiency"}, 
            "mistral-7b": {"size_gb": 14, "priority": "HIGH", "notes": "Alternative mistral"},
            "qwen2.5:7b": {"size_gb": 14, "priority": "HIGH", "notes": "High-performance Chinese model"},
            "qwen2.5-7b": {"size_gb": 14, "priority": "HIGH", "notes": "Alternative qwen"},
            "gemma:7b": {"size_gb": 14, "priority": "HIGH", "notes": "Google's instruction-tuned"},
            "llama3.1:8b": {"size_gb": 16, "priority": "HIGH", "notes": "Well-balanced, 128K context"},
            "llama3.1-8b": {"size_gb": 16, "priority": "HIGH", "notes": "Alternative llama3.1"},
            "llama3:8b": {"size_gb": 16, "priority": "HIGH", "notes": "Original Llama 3 8B"},
            "gemma2:9b": {"size_gb": 18, "priority": "HIGH", "notes": "Improved Gemma 2"},
            "gemma-9b": {"size_gb": 18, "priority": "HIGH", "notes": "Alternative gemma2"}
        }
        
        logger.info(f"LocalModelEvaluator initialized")
        logger.info(f"  Base dir: {self.base_dir}")
        logger.info(f"  Ollama URL: {self.ollama_url}")
        logger.info(f"  Local models: {list(self.local_models.keys())}")
    
    def check_ollama_status(self) -> bool:
        """Check if Ollama is running"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def get_available_models(self) -> List[str]:
        """Get list of available local models"""
        if not self.check_ollama_status():
            logger.error("❌ Ollama is not running. Please start Ollama first.")
            return []
        
        try:
            response = requests.get(f"{self.ollama_url}/api/tags")
            data = response.json()
            available = [model['name'].replace(':latest', '') for model in data['models']]
            
            # Filter to only our target models
            local_available = []
            for model in self.local_models.keys():
                if model in available or model.replace(':latest', '') in available:
                    local_available.append(model)
            
            logger.info(f"Available local models: {local_available}")
            return local_available
        except Exception as e:
            logger.error(f"Error checking available models: {e}")
            return []
    
    def generate_response(self, model: str, prompt: str) -> Dict[str, Any]:
        """Generate response from local model with IDENTICAL format to server"""
        start_time = time.time()
        
        try:
            # Ollama API call
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "top_p": 0.9,
                        "max_tokens": 512
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                inference_time = time.time() - start_time
                
                return {
                    "success": True,
                    "response": data.get("response", ""),
                    "inference_time": inference_time,
                    "timestamp": datetime.now().isoformat(),
                    "evaluation_type": "local"
                }
            else:
                return {
                    "success": False,
                    "response": "",
                    "inference_time": time.time() - start_time,
                    "timestamp": datetime.now().isoformat(),
                    "evaluation_type": "local",
                    "error": f"HTTP {response.status_code}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "response": "",
                "inference_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat(),
                "evaluation_type": "local",
                "error": str(e)
            }
    
    def extract_moral_choice(self, response_text: str) -> str:
        """Extract moral choice - IDENTICAL to server extraction"""
        if not response_text:
            return 'unknown'
        
        response_lower = str(response_text).lower()
        
        # Look for clear indicators
        if 'acceptable' in response_lower and 'unacceptable' not in response_lower:
            return 'acceptable'
        elif 'unacceptable' in response_lower and 'acceptable' not in response_lower:
            return 'unacceptable'
        elif 'yes' in response_lower and 'no' not in response_lower:
            return 'acceptable'
        elif 'no' in response_lower and 'yes' not in response_lower:
            return 'unacceptable'
        else:
            return 'unknown'
    
    def evaluate_model_complete(self, model_name: str, samples: List[Dict]) -> List[Dict]:
        """Evaluate model on all samples with IDENTICAL output format to server"""
        logger.info(f"🚀 Starting local evaluation of {model_name}")
        logger.info(f"   📊 Total samples: {len(samples)}")
        logger.info(f"   🖥️  Running on M4 Max with Ollama")
        
        all_results = []
        
        # Process samples with progress bar
        with tqdm(total=len(samples), desc=f"Evaluating {model_name}", unit="samples") as pbar:
            for i, sample in enumerate(samples):
                try:
                    # Generate response
                    result = self.generate_response(model_name, sample['prompt'])
                    
                    # Create IDENTICAL result format to server
                    final_result = {
                        'model': model_name,
                        'sample_id': sample.get('id', f'sample_{i}'),
                        'success': result['success'],
                        'response': result['response'],
                        'choice': self.extract_moral_choice(result['response']) if result['success'] else 'unknown',
                        'inference_time': result['inference_time'],
                        'timestamp': result['timestamp'],
                        'evaluation_type': 'local'  # Mark as local for tracking
                    }
                    
                    # Add error if present
                    if not result['success'] and 'error' in result:
                        final_result['error'] = result['error']
                    
                    all_results.append(final_result)
                    pbar.update(1)
                    
                except Exception as e:
                    # Error result with IDENTICAL format
                    error_result = {
                        'model': model_name,
                        'sample_id': sample.get('id', f'sample_{i}'),
                        'error': str(e),
                        'success': False,
                        'response': '',
                        'choice': 'unknown',
                        'inference_time': 0,
                        'timestamp': datetime.now().isoformat(),
                        'evaluation_type': 'local'
                    }
                    all_results.append(error_result)
                    pbar.update(1)
        
        # Calculate statistics
        successful = sum(1 for r in all_results if r.get('success', False))
        success_rate = successful / len(all_results) if all_results else 0
        
        logger.info(f"✅ Local evaluation complete: {model_name}")
        logger.info(f"   📊 Total samples: {len(all_results)}")
        logger.info(f"   ✅ Successful: {successful} ({success_rate:.1%})")
        logger.info(f"   🖥️  Evaluation type: local")
        
        return all_results
    
    def run_all_local_models(self, samples: List[Dict]) -> List[Dict]:
        """Run evaluation on all available local models"""
        available_models = self.get_available_models()
        
        if not available_models:
            logger.error("❌ No models available locally")
            return []
        
        logger.info(f"🚀 Starting evaluation of {len(available_models)} local models")
        logger.info(f"   📊 Samples per model: {len(samples)}")
        logger.info(f"   🖥️  Running on M4 Max")
        
        all_results = []
        start_time = time.time()
        
        for model_name in available_models:
            try:
                results = self.evaluate_model_complete(model_name, samples)
                all_results.extend(results)
                
                # Save individual model results (IDENTICAL format to server)
                output_file = self.output_dir / f"{model_name}_results_local.json"
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2)
                logger.info(f"   💾 Saved to: {output_file}")
                
            except Exception as e:
                logger.error(f"❌ Failed to evaluate {model_name}: {e}")
        
        total_time = time.time() - start_time
        successful_results = sum(1 for r in all_results if r.get('success', False))
        
        logger.info(f"\n🎉 LOCAL EVALUATION COMPLETE!")
        logger.info(f"   📊 Total results: {len(all_results):,}")
        logger.info(f"   ✅ Successful results: {successful_results:,} ({successful_results/len(all_results)*100:.1f}%)")
        logger.info(f"   ⏱️  Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        logger.info(f"   🖥️  Average speed: {len(all_results)/total_time:.1f} samples/sec")
        
        return all_results
    
    def save_results_for_integration(self, results: List[Dict], timestamp: str = None):
        """Save results in IDENTICAL format to server for seamless integration"""
        if timestamp is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save in IDENTICAL format to server
        integration_file = self.output_dir / f"local_results_for_integration_{timestamp}.json"
        with open(integration_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create metadata file (IDENTICAL to server format)
        successful_results = sum(1 for r in results if r.get('success', False))
        unique_models = list(set(r['model'] for r in results))
        unique_samples = list(set(r['sample_id'] for r in results))
        
        metadata = {
            'evaluation_type': 'local',
            'timestamp': timestamp,
            'total_samples': len(unique_samples),
            'total_models': len(unique_models),
            'total_successful_results': successful_results,
            'models_evaluated': unique_models,
            'dataset_info': {
                'same_samples_as_server_api': True,
                'sample_count': len(unique_samples),
                'countries': 64,
                'moral_questions': 13,
                'source': 'World Values Survey'
            },
            'system_setup': {
                'hardware': 'M4 Max',
                'memory': '64GB',
                'evaluation_engine': 'Ollama'
            },
            'output_files': {
                'standardized_results': str(integration_file),
                'evaluation_type': 'local'
            }
        }
        
        metadata_file = self.output_dir / f"local_metadata_for_integration_{timestamp}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✅ Integration files saved:")
        logger.info(f"   📊 Results: {integration_file}")
        logger.info(f"   📋 Metadata: {metadata_file}")
        logger.info(f"   🔗 Ready for seamless integration with server results")
        
        return str(integration_file), str(metadata_file)

def load_exact_samples():
    """Load the exact same 5000 samples used by server evaluation"""
    # This should match the server's load_exact_samples function
    try:
        # Try to import from the same location as server
        sys.path.append('/Users/hadimohammadi/Documents/Project06/server_deployment_package')
        from load_exact_samples import load_exact_samples as server_load_samples
        return server_load_samples()
    except ImportError:
        logger.error("❌ Cannot import server's load_exact_samples function")
        logger.info("   📋 Please ensure samples are identical to server evaluation")
        return []

def main():
    """Main function for local evaluation"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Local Model Evaluation for M4 Max")
    parser.add_argument("--models", nargs="+", 
                       help="Specific models to evaluate (default: all available)")
    parser.add_argument("--samples", type=int, default=5000,
                       help="Number of samples to evaluate (default: 5000)")
    parser.add_argument("--output-dir", type=str,
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    print("🖥️  LOCAL MODEL EVALUATION - M4 MAX")
    print("=" * 60)
    print("🎯 Evaluating small models locally for optimal resource usage")
    print("🔗 Output format identical to server evaluation")
    
    # Initialize evaluator
    evaluator = LocalModelEvaluator(base_dir=args.output_dir)
    
    # Check Ollama status
    if not evaluator.check_ollama_status():
        logger.error("❌ Ollama is not running. Please start Ollama first:")
        logger.info("   brew install ollama")
        logger.info("   ollama serve")
        return
    
    # Load samples (IDENTICAL to server)
    print("\n🎯 Loading EXACT samples (same as server evaluation)")
    samples = load_exact_samples()
    
    if not samples:
        logger.error("❌ Failed to load samples")
        return
    
    # Use subset if requested
    if args.samples < len(samples):
        samples = samples[:args.samples]
    
    print(f"✅ Loaded {len(samples)} EXACT samples")
    
    # Run evaluation
    results = evaluator.run_all_local_models(samples)
    
    if results:
        # Save for integration
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_file, metadata_file = evaluator.save_results_for_integration(results, timestamp)
        
        print(f"\n🎉 LOCAL EVALUATION SUCCESS!")
        print(f"   📊 Generated {len(results)} results")
        print(f"   🔗 Ready for integration with server results")
        print(f"   📁 Files: {result_file}")
        print(f"          {metadata_file}")
    else:
        logger.error("❌ No results generated")

if __name__ == "__main__":
    main()