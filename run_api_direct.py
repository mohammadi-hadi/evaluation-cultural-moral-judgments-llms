#!/usr/bin/env python3
"""
Direct API Evaluation - Real-time API calls for immediate results
Run all 11 API models with direct OpenAI API calls (not batch)
"""

import os
import json
import time
import asyncio
import aiohttp
from datetime import datetime
from pathlib import Path
import pandas as pd
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DirectAPIRunner:
    """Direct API evaluation with real-time calls"""
    
    def __init__(self, output_dir: str = "outputs/server_sync_evaluation/run_20250902_165021/api_direct"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        # Model configurations with rate limits for direct API
        self.models = [
            {"name": "gpt-3.5-turbo", "rpm": 3500, "cost_per_1k": 0.0015},
            {"name": "gpt-4o-mini", "rpm": 10000, "cost_per_1k": 0.00015}, 
            {"name": "gpt-4o", "rpm": 10000, "cost_per_1k": 0.03},
            {"name": "gpt-4", "rpm": 10000, "cost_per_1k": 0.03},  # Fallback for gpt-5
            {"name": "gpt-4-turbo", "rpm": 10000, "cost_per_1k": 0.01},  # Fallback for gpt-5-mini
        ]
        
        # Headers for API requests
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        self.results = []
        
    def load_samples(self) -> List[Dict]:
        """Load the exact same samples used by local evaluation"""
        samples_file = "outputs/server_sync_evaluation/run_20250902_165021/evaluation_samples.json"
        
        if not os.path.exists(samples_file):
            raise FileNotFoundError(f"Samples file not found: {samples_file}")
            
        with open(samples_file, 'r') as f:
            samples = json.load(f)
            
        logger.info(f"Loaded {len(samples)} samples")
        return samples
    
    async def call_openai_api(self, session: aiohttp.ClientSession, model: str, prompt: str, 
                            sample_id: str, max_retries: int = 3) -> Dict:
        """Make direct API call to OpenAI"""
        
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 150,
            "temperature": 0.7
        }
        
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                
                async with session.post(
                    "https://api.openai.com/v1/chat/completions",
                    json=payload,
                    headers=self.headers
                ) as response:
                    
                    inference_time = time.time() - start_time
                    
                    if response.status == 200:
                        data = await response.json()
                        
                        return {
                            "model": model,
                            "sample_id": sample_id,
                            "response": data["choices"][0]["message"]["content"],
                            "inference_time": inference_time,
                            "success": True,
                            "timestamp": datetime.now().isoformat(),
                            "tokens_used": data.get("usage", {}).get("total_tokens", 0)
                        }
                    
                    elif response.status == 429:  # Rate limit
                        wait_time = 2 ** attempt
                        logger.warning(f"Rate limited for {model}, waiting {wait_time}s")
                        await asyncio.sleep(wait_time)
                        continue
                        
                    else:
                        error_text = await response.text()
                        logger.error(f"API error for {model}: {response.status} - {error_text}")
                        
                        return {
                            "model": model,
                            "sample_id": sample_id,
                            "response": "",
                            "inference_time": inference_time,
                            "success": False,
                            "error": f"HTTP {response.status}: {error_text}",
                            "timestamp": datetime.now().isoformat()
                        }
                        
            except Exception as e:
                logger.error(f"Exception calling {model}: {e}")
                if attempt == max_retries - 1:
                    return {
                        "model": model,
                        "sample_id": sample_id,
                        "response": "",
                        "inference_time": 0,
                        "success": False,
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    }
                
                await asyncio.sleep(2 ** attempt)
        
        return None
    
    async def evaluate_model_batch(self, model: str, samples: List[Dict], 
                                 batch_size: int = 10) -> List[Dict]:
        """Evaluate a model on all samples with rate limiting"""
        logger.info(f"Starting evaluation for {model} with {len(samples)} samples")
        
        model_results = []
        
        async with aiohttp.ClientSession() as session:
            for i in range(0, len(samples), batch_size):
                batch_samples = samples[i:i + batch_size]
                
                # Create tasks for this batch
                tasks = []
                for sample in batch_samples:
                    task = self.call_openai_api(
                        session, model, sample['prompt'], sample['id']
                    )
                    tasks.append(task)
                
                # Execute batch
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results
                for result in batch_results:
                    if isinstance(result, dict):
                        model_results.append(result)
                    else:
                        logger.error(f"Exception in batch: {result}")
                
                # Rate limiting - wait between batches
                if i + batch_size < len(samples):
                    await asyncio.sleep(1)  # 1 second between batches
                
                # Progress update
                completed = min(i + batch_size, len(samples))
                logger.info(f"{model}: {completed}/{len(samples)} completed ({completed/len(samples)*100:.1f}%)")
        
        # Save model results
        model_file = self.output_dir / f"{model}_results.json"
        with open(model_file, 'w') as f:
            json.dump(model_results, f, indent=2)
        
        successful = sum(1 for r in model_results if r.get('success', False))
        logger.info(f"{model} completed: {successful}/{len(model_results)} successful")
        
        return model_results
    
    async def run_all_models(self, max_samples: int = 1000):
        """Run all models with direct API calls"""
        logger.info("🚀 STARTING DIRECT API EVALUATION")
        logger.info("=" * 50)
        
        # Load samples
        all_samples = self.load_samples()
        samples = all_samples[:max_samples]  # Limit for faster testing
        
        logger.info(f"Using {len(samples)} samples for evaluation")
        
        total_results = []
        
        for model_config in self.models:
            model_name = model_config["name"]
            
            try:
                model_results = await self.evaluate_model_batch(model_name, samples)
                total_results.extend(model_results)
                
                # Brief pause between models
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"Failed to evaluate {model_name}: {e}")
                continue
        
        # Save all results
        all_results_file = self.output_dir / f"all_api_direct_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(all_results_file, 'w') as f:
            json.dump(total_results, f, indent=2)
        
        # Generate summary
        summary = self.generate_summary(total_results)
        
        summary_file = self.output_dir / f"api_direct_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info("✅ DIRECT API EVALUATION COMPLETE!")
        logger.info(f"Total results: {len(total_results)}")
        logger.info(f"Results saved to: {all_results_file}")
        logger.info(f"Summary saved to: {summary_file}")
        
        return total_results, summary
    
    def generate_summary(self, results: List[Dict]) -> Dict:
        """Generate evaluation summary"""
        if not results:
            return {}
        
        df = pd.DataFrame(results)
        
        summary = {
            "total_evaluations": len(results),
            "successful_evaluations": len(df[df['success'] == True]),
            "failed_evaluations": len(df[df['success'] == False]),
            "success_rate": len(df[df['success'] == True]) / len(results),
            "models": {},
            "overall_stats": {
                "avg_inference_time": df[df['success'] == True]['inference_time'].mean(),
                "total_tokens": df[df['success'] == True]['tokens_used'].sum() if 'tokens_used' in df.columns else 0
            }
        }
        
        # Per-model stats
        for model in df['model'].unique():
            model_data = df[df['model'] == model]
            successful = model_data[model_data['success'] == True]
            
            summary["models"][model] = {
                "total": len(model_data),
                "successful": len(successful),
                "failed": len(model_data) - len(successful),
                "success_rate": len(successful) / len(model_data),
                "avg_inference_time": successful['inference_time'].mean() if len(successful) > 0 else 0,
                "total_tokens": successful['tokens_used'].sum() if 'tokens_used' in successful.columns else 0
            }
        
        return summary

def main():
    """Main execution"""
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("Please set it with: export OPENAI_API_KEY='your-key-here'")
        return
    
    runner = DirectAPIRunner()
    
    # Run with asyncio
    results, summary = asyncio.run(runner.run_all_models(max_samples=1000))
    
    print(f"\n📊 EVALUATION SUMMARY:")
    print(f"Total evaluations: {summary.get('total_evaluations', 0)}")
    print(f"Successful: {summary.get('successful_evaluations', 0)}")
    print(f"Success rate: {summary.get('success_rate', 0):.1%}")
    
    print(f"\n🤖 Model Results:")
    for model, stats in summary.get('models', {}).items():
        print(f"  {model}: {stats['successful']}/{stats['total']} ({stats['success_rate']:.1%})")

if __name__ == "__main__":
    main()