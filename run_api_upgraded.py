#!/usr/bin/env python3
"""
Run API models with upgraded usage tier - improved rate limits
"""

import os
import json
import asyncio
import aiohttp
from datetime import datetime
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UpgradedAPIRunner:
    """API runner with upgraded usage tier"""
    
    def __init__(self, output_dir: str = "outputs/server_sync_evaluation/run_20250902_165021/api_upgraded"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        # Available models with upgraded rate limits
        self.models = [
            {"name": "gpt-3.5-turbo", "concurrent": 10, "delay": 0.1},
            {"name": "gpt-4o-mini", "concurrent": 15, "delay": 0.05}, 
            {"name": "gpt-4o", "concurrent": 8, "delay": 0.2},
        ]
        
        # Headers for API requests
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Create semaphores for improved rate limiting
        self.semaphores = {}
        for model in self.models:
            self.semaphores[model["name"]] = asyncio.Semaphore(model["concurrent"])
    
    def load_samples(self, max_samples: int = 200) -> list:
        """Load samples for evaluation"""
        samples_file = "outputs/server_sync_evaluation/run_20250902_165021/evaluation_samples.json"
        
        with open(samples_file, 'r') as f:
            all_samples = json.load(f)
        
        # Use smaller sample set for faster results
        samples = all_samples[:max_samples]
        logger.info(f"Using {len(samples)} samples for upgraded evaluation")
        return samples
    
    async def call_api_improved(self, session: aiohttp.ClientSession, model: str, 
                               prompt: str, sample_id: str) -> dict:
        """Improved API call with upgraded limits"""
        
        async with self.semaphores[model]:
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 100,
                "temperature": 0.3
            }
            
            for attempt in range(3):
                try:
                    start_time = asyncio.get_event_loop().time()
                    
                    async with session.post(
                        "https://api.openai.com/v1/chat/completions",
                        json=payload,
                        headers=self.headers,
                        timeout=aiohttp.ClientTimeout(total=20)
                    ) as response:
                        
                        inference_time = asyncio.get_event_loop().time() - start_time
                        
                        if response.status == 200:
                            data = await response.json()
                            
                            # Add delay based on model config
                            model_config = next(m for m in self.models if m["name"] == model)
                            await asyncio.sleep(model_config["delay"])
                            
                            return {
                                "model": model,
                                "sample_id": sample_id,
                                "response": data["choices"][0]["message"]["content"],
                                "inference_time": inference_time,
                                "success": True,
                                "timestamp": datetime.now().isoformat(),
                                "evaluation_type": "api_upgraded"
                            }
                        
                        elif response.status == 429:  # Rate limit
                            wait_time = 0.5 * (2 ** attempt)
                            logger.warning(f"Rate limited {model}, waiting {wait_time:.1f}s")
                            await asyncio.sleep(wait_time)
                            continue
                        
                        else:
                            logger.error(f"API error {model}: {response.status}")
                            return {
                                "model": model,
                                "sample_id": sample_id,
                                "response": "",
                                "inference_time": inference_time,
                                "success": False,
                                "error": f"HTTP {response.status}",
                                "timestamp": datetime.now().isoformat(),
                                "evaluation_type": "api_upgraded"
                            }
                            
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout {model}, attempt {attempt + 1}")
                    if attempt == 2:
                        return {
                            "model": model,
                            "sample_id": sample_id,
                            "response": "",
                            "inference_time": 0,
                            "success": False,
                            "error": "Timeout",
                            "timestamp": datetime.now().isoformat(),
                            "evaluation_type": "api_upgraded"
                        }
                    await asyncio.sleep(0.5)
                    
                except Exception as e:
                    logger.error(f"Exception {model}: {e}")
                    if attempt == 2:
                        return {
                            "model": model,
                            "sample_id": sample_id,
                            "response": "",
                            "inference_time": 0,
                            "success": False,
                            "error": str(e),
                            "timestamp": datetime.now().isoformat(),
                            "evaluation_type": "api_upgraded"
                        }
        
        return None
    
    async def evaluate_model_upgraded(self, model: str, samples: list) -> list:
        """Evaluate model with upgraded rate limits"""
        logger.info(f"🚀 Starting {model} evaluation with upgraded limits")
        
        results = []
        successful = 0
        
        async with aiohttp.ClientSession() as session:
            # Process in smaller batches for better control
            batch_size = 20
            
            for i in range(0, len(samples), batch_size):
                batch_samples = samples[i:i + batch_size]
                
                # Create tasks for batch
                tasks = [
                    self.call_api_improved(session, model, sample['prompt'], sample['id'])
                    for sample in batch_samples
                ]
                
                # Execute batch
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results
                for result in batch_results:
                    if isinstance(result, dict):
                        results.append(result)
                        if result.get('success', False):
                            successful += 1
                
                # Progress update
                completed = min(i + batch_size, len(samples))
                progress = completed / len(samples) * 100
                logger.info(f"{model}: {completed}/{len(samples)} ({progress:.1f}%) - {successful} successful")
        
        # Save results
        model_file = self.output_dir / f"{model}_upgraded_results.json"
        with open(model_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"✅ {model} completed: {successful}/{len(results)} successful")
        return results
    
    async def run_available_models(self, max_samples: int = 200):
        """Run available models with upgraded tier"""
        logger.info("🎉 STARTING API EVALUATION WITH UPGRADED TIER")
        logger.info("=" * 50)
        
        samples = self.load_samples(max_samples)
        start_time = asyncio.get_event_loop().time()
        
        # Run models sequentially to avoid overwhelming the API
        all_results = []
        model_stats = {}
        
        for model_config in self.models:
            model_name = model_config["name"]
            
            try:
                model_results = await self.evaluate_model_upgraded(model_name, samples)
                all_results.extend(model_results)
                
                successful = sum(1 for r in model_results if r.get('success', False))
                model_stats[model_name] = {
                    "total": len(model_results),
                    "successful": successful,
                    "success_rate": successful / len(model_results) if model_results else 0
                }
                
                # Brief pause between models
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"Failed to evaluate {model_name}: {e}")
                continue
        
        total_time = asyncio.get_event_loop().time() - start_time
        
        # Save combined results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.output_dir / f"api_upgraded_complete_{timestamp}.json"
        
        final_data = {
            "metadata": {
                "timestamp": timestamp,
                "total_samples": len(samples),
                "total_models": len(self.models),
                "total_time": total_time,
                "evaluation_type": "api_upgraded",
                "usage_tier": "upgraded"
            },
            "model_stats": model_stats,
            "results": all_results
        }
        
        with open(results_file, 'w') as f:
            json.dump(final_data, f, indent=2)
        
        logger.info("🎉 UPGRADED API EVALUATION COMPLETE!")
        logger.info("=" * 40)
        logger.info(f"📊 Total evaluations: {len(all_results)}")
        logger.info(f"⏱️  Total time: {total_time/60:.1f} minutes")
        logger.info(f"🎯 Success rate: {sum(r.get('success', False) for r in all_results) / len(all_results):.1%}")
        logger.info(f"💾 Results saved to: {results_file}")
        
        return all_results, model_stats, results_file

def main():
    """Main execution"""
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        return
    
    runner = UpgradedAPIRunner()
    results, stats, results_file = asyncio.run(runner.run_available_models(200))
    
    if results_file:
        print(f"\n🎉 API EVALUATION WITH UPGRADED TIER COMPLETE!")
        print(f"📁 Results: {results_file}")
        print(f"📊 Total: {len(results)} evaluations")
        print(f"✅ Ready for integration!")

if __name__ == "__main__":
    main()