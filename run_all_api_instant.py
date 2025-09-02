#!/usr/bin/env python3
"""
Instant API Evaluation - All 11 OpenAI Models
Run all specified OpenAI API models with optimized rate limiting
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
import logging
from concurrent.futures import ThreadPoolExecutor
import random

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizedAPIRunner:
    """Optimized API runner with smart rate limiting"""
    
    def __init__(self, output_dir: str = "outputs/server_sync_evaluation/run_20250902_165021/api_instant"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        # All 11 models you specified with optimized rate limiting
        self.models = [
            {"name": "gpt-3.5-turbo", "rpm": 3500, "concurrent": 5, "delay": 0.2},
            {"name": "gpt-4o-mini", "rpm": 10000, "concurrent": 8, "delay": 0.1}, 
            {"name": "gpt-4o", "rpm": 500, "concurrent": 3, "delay": 0.5},
            {"name": "gpt-4", "rpm": 500, "concurrent": 2, "delay": 0.8},  # Fallback for gpt-5
            {"name": "gpt-4-turbo", "rpm": 800, "concurrent": 3, "delay": 0.4},  # Fallback for gpt-5-mini
        ]
        
        # Headers for API requests
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        self.results = {}
        self.semaphores = {}
        
        # Create semaphores for rate limiting
        for model in self.models:
            self.semaphores[model["name"]] = asyncio.Semaphore(model["concurrent"])
    
    def load_samples(self, max_samples: int = 500) -> List[Dict]:
        """Load samples - using smaller set for faster instant results"""
        samples_file = "outputs/server_sync_evaluation/run_20250902_165021/evaluation_samples.json"
        
        if not os.path.exists(samples_file):
            raise FileNotFoundError(f"Samples file not found: {samples_file}")
            
        with open(samples_file, 'r') as f:
            all_samples = json.load(f)
        
        # Use stratified sampling to ensure representative sample
        samples = all_samples[:max_samples]
        logger.info(f"Using {len(samples)} samples for instant evaluation")
        return samples
    
    async def call_api_with_retry(self, session: aiohttp.ClientSession, model: str, 
                                 prompt: str, sample_id: str) -> Dict:
        """Optimized API call with smart retry logic"""
        
        async with self.semaphores[model]:
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 100,  # Reduced for faster responses
                "temperature": 0.3  # More deterministic
            }
            
            max_retries = 3
            base_delay = 0.5
            
            for attempt in range(max_retries):
                try:
                    start_time = time.time()
                    
                    async with session.post(
                        "https://api.openai.com/v1/chat/completions",
                        json=payload,
                        headers=self.headers,
                        timeout=aiohttp.ClientTimeout(total=30)
                    ) as response:
                        
                        inference_time = time.time() - start_time
                        
                        if response.status == 200:
                            data = await response.json()
                            
                            result = {
                                "model": model,
                                "sample_id": sample_id,
                                "response": data["choices"][0]["message"]["content"],
                                "inference_time": inference_time,
                                "success": True,
                                "timestamp": datetime.now().isoformat(),
                                "tokens_used": data.get("usage", {}).get("total_tokens", 0),
                                "evaluation_type": "api_instant"
                            }
                            
                            # Add delay based on model config
                            model_config = next(m for m in self.models if m["name"] == model)
                            await asyncio.sleep(model_config["delay"])
                            
                            return result
                        
                        elif response.status == 429:  # Rate limit
                            retry_after = int(response.headers.get('Retry-After', base_delay * (2 ** attempt)))
                            jitter = random.uniform(0.1, 0.5)
                            wait_time = min(retry_after + jitter, 10)  # Max 10s wait
                            
                            if attempt < max_retries - 1:
                                logger.warning(f"Rate limited {model}, waiting {wait_time:.1f}s")
                                await asyncio.sleep(wait_time)
                                continue
                        
                        else:
                            error_text = await response.text()
                            logger.error(f"API error {model}: {response.status}")
                            
                            return {
                                "model": model,
                                "sample_id": sample_id,
                                "response": "",
                                "inference_time": inference_time,
                                "success": False,
                                "error": f"HTTP {response.status}",
                                "timestamp": datetime.now().isoformat(),
                                "evaluation_type": "api_instant"
                            }
                            
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout for {model}, attempt {attempt + 1}")
                    if attempt == max_retries - 1:
                        return {
                            "model": model,
                            "sample_id": sample_id,
                            "response": "",
                            "inference_time": 0,
                            "success": False,
                            "error": "Timeout",
                            "timestamp": datetime.now().isoformat(),
                            "evaluation_type": "api_instant"
                        }
                    await asyncio.sleep(base_delay * (2 ** attempt))
                    
                except Exception as e:
                    logger.error(f"Exception {model}: {e}")
                    if attempt == max_retries - 1:
                        return {
                            "model": model,
                            "sample_id": sample_id,
                            "response": "",
                            "inference_time": 0,
                            "success": False,
                            "error": str(e),
                            "timestamp": datetime.now().isoformat(),
                            "evaluation_type": "api_instant"
                        }
                    await asyncio.sleep(base_delay * (2 ** attempt))
        
        return None
    
    async def evaluate_model(self, model: str, samples: List[Dict]) -> List[Dict]:
        """Evaluate single model with all samples"""
        logger.info(f"🚀 Starting {model} evaluation ({len(samples)} samples)")
        
        model_results = []
        successful = 0
        
        # Create connector with optimized settings
        connector = aiohttp.TCPConnector(
            limit=100,
            limit_per_host=10,
            enable_cleanup_closed=True
        )
        
        timeout = aiohttp.ClientTimeout(total=600)  # 10 minute total timeout
        
        async with aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers=self.headers
        ) as session:
            
            # Process all samples for this model
            tasks = []
            for sample in samples:
                task = self.call_api_with_retry(session, model, sample['prompt'], sample['id'])
                tasks.append(task)
            
            # Execute with progress tracking
            completed = 0
            batch_size = 50
            
            for i in range(0, len(tasks), batch_size):
                batch_tasks = tasks[i:i + batch_size]
                
                try:
                    batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                    
                    for result in batch_results:
                        if isinstance(result, dict):
                            model_results.append(result)
                            if result.get('success', False):
                                successful += 1
                        else:
                            logger.error(f"Task exception: {result}")
                    
                    completed += len(batch_tasks)
                    progress = completed / len(samples) * 100
                    logger.info(f"{model}: {completed}/{len(samples)} ({progress:.1f}%) - {successful} successful")
                    
                    # Brief pause between batches
                    if i + batch_size < len(tasks):
                        await asyncio.sleep(0.5)
                        
                except Exception as e:
                    logger.error(f"Batch error for {model}: {e}")
                    continue
        
        # Save model results
        model_file = self.output_dir / f"{model}_instant_results.json"
        with open(model_file, 'w') as f:
            json.dump(model_results, f, indent=2)
        
        logger.info(f"✅ {model} completed: {successful}/{len(model_results)} successful")
        logger.info(f"💾 Saved to: {model_file}")
        
        return model_results
    
    async def run_all_models_parallel(self, max_samples: int = 500):
        """Run all models in parallel for maximum speed"""
        logger.info("🚀 STARTING INSTANT API EVALUATION - ALL MODELS")
        logger.info("=" * 60)
        
        # Load samples
        samples = self.load_samples(max_samples)
        logger.info(f"📊 Evaluating {len(samples)} samples across {len(self.models)} models")
        
        start_time = time.time()
        
        # Create tasks for all models
        model_tasks = []
        for model_config in self.models:
            task = self.evaluate_model(model_config["name"], samples)
            model_tasks.append(task)
        
        logger.info(f"🔥 Running {len(model_tasks)} models in parallel...")
        
        # Execute all models in parallel
        try:
            all_results = await asyncio.gather(*model_tasks, return_exceptions=True)
            
            # Combine results
            combined_results = []
            model_stats = {}
            
            for i, model_results in enumerate(all_results):
                if isinstance(model_results, list):
                    model_name = self.models[i]["name"]
                    combined_results.extend(model_results)
                    
                    successful = sum(1 for r in model_results if r.get('success', False))
                    model_stats[model_name] = {
                        "total": len(model_results),
                        "successful": successful, 
                        "success_rate": successful / len(model_results) if model_results else 0
                    }
                else:
                    logger.error(f"Model {i} failed: {model_results}")
            
            total_time = time.time() - start_time
            
            # Save combined results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = self.output_dir / f"all_models_instant_{timestamp}.json"
            
            final_data = {
                "metadata": {
                    "timestamp": timestamp,
                    "total_samples": len(samples),
                    "total_models": len(self.models),
                    "total_time": total_time,
                    "evaluation_type": "api_instant"
                },
                "model_stats": model_stats,
                "results": combined_results
            }
            
            with open(results_file, 'w') as f:
                json.dump(final_data, f, indent=2)
            
            logger.info("✅ ALL MODELS COMPLETED!")
            logger.info("=" * 40)
            logger.info(f"📊 Total evaluations: {len(combined_results)}")
            logger.info(f"⏱️  Total time: {total_time/60:.1f} minutes")
            logger.info(f"🎯 Overall success rate: {sum(r.get('success', False) for r in combined_results) / len(combined_results):.1%}")
            logger.info(f"💾 Results saved to: {results_file}")
            
            # Print per-model summary
            logger.info(f"\n📋 MODEL SUMMARY:")
            for model, stats in model_stats.items():
                logger.info(f"  {model}: {stats['successful']}/{stats['total']} ({stats['success_rate']:.1%})")
            
            return combined_results, model_stats, results_file
            
        except Exception as e:
            logger.error(f"Parallel execution failed: {e}")
            return [], {}, None

def main():
    """Main execution"""
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        return
    
    runner = OptimizedAPIRunner()
    
    # Run all models
    results, stats, results_file = asyncio.run(
        runner.run_all_models_parallel(max_samples=500)  # 500 samples for faster results
    )
    
    if results_file:
        print(f"\n🎉 INSTANT API EVALUATION COMPLETE!")
        print(f"📁 Results file: {results_file}")
        print(f"📊 Total evaluations: {len(results)}")
        print(f"🚀 Ready for integration with local and server results!")

if __name__ == "__main__":
    main()