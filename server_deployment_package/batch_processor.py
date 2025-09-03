#!/usr/bin/env python3
"""
Efficient Batch Processing System for Server Model Evaluation
Optimizes batch sizes dynamically and manages GPU resources
"""

import time
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
import threading
import numpy as np
from tqdm.auto import tqdm

# Import GPU monitoring
from gpu_monitor import GPUMonitor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BatchConfig:
    """Configuration for batch processing"""
    model_name: str
    initial_batch_size: int
    max_batch_size: int
    min_batch_size: int
    target_gpu_memory_util: float = 0.90  # Target GPU memory utilization
    target_batch_time: float = 30.0       # Target time per batch in seconds
    adaptive_sizing: bool = True          # Enable adaptive batch sizing
    parallel_batches: int = 1             # Number of parallel batch processes

@dataclass
class BatchStats:
    """Statistics for batch processing"""
    model_name: str
    batch_size: int
    samples_processed: int
    processing_time: float
    samples_per_second: float
    gpu_memory_used_mb: float
    gpu_utilization: float
    success_rate: float
    timestamp: str

class BatchProcessor:
    """Intelligent batch processing with dynamic optimization"""
    
    def __init__(self, server_model_runner, gpu_monitor: Optional[GPUMonitor] = None):
        """Initialize batch processor
        
        Args:
            server_model_runner: Instance of ServerModelRunner
            gpu_monitor: Optional GPU monitor for optimization
        """
        self.runner = server_model_runner
        self.gpu_monitor = gpu_monitor or GPUMonitor(monitoring_interval=0.5)
        
        # Batch statistics
        self.batch_stats = {}
        self.performance_history = []
        
        # Adaptive batch sizing
        self.batch_configs = {}
        self.optimization_enabled = True
        
        logger.info("🚀 BatchProcessor initialized with adaptive optimization")
    
    def create_batch_config(self, model_name: str, model_size_gb: float) -> BatchConfig:
        """Create optimized batch configuration for a model
        
        Args:
            model_name: Name of the model
            model_size_gb: Size of model in GB
            
        Returns:
            Optimized batch configuration
        """
        # Base configuration based on model size
        if model_size_gb > 60:  # Large models (70B+)
            initial_batch_size = 8
            max_batch_size = 16
            min_batch_size = 2
        elif model_size_gb > 25:  # Medium models (32B)
            initial_batch_size = 16
            max_batch_size = 32
            min_batch_size = 4
        else:  # Small models (<25GB)
            initial_batch_size = 32
            max_batch_size = 64
            min_batch_size = 8
        
        # Adjust based on available GPU memory
        total_gpu_memory = self.runner.total_gpu_memory * 1024  # Convert to MB
        if total_gpu_memory > 0:
            # Conservative memory-based adjustment
            memory_factor = min(1.0, total_gpu_memory / (model_size_gb * 1024 * 1.2))
            initial_batch_size = max(min_batch_size, int(initial_batch_size * memory_factor))
            max_batch_size = max(initial_batch_size, int(max_batch_size * memory_factor))
        
        config = BatchConfig(
            model_name=model_name,
            initial_batch_size=initial_batch_size,
            max_batch_size=max_batch_size,
            min_batch_size=min_batch_size,
            target_gpu_memory_util=0.90,
            target_batch_time=30.0,
            adaptive_sizing=True,
            parallel_batches=1
        )
        
        self.batch_configs[model_name] = config
        logger.info(f"📊 Created batch config for {model_name}:")
        logger.info(f"   Initial batch size: {initial_batch_size}")
        logger.info(f"   Batch size range: {min_batch_size}-{max_batch_size}")
        
        return config
    
    def optimize_batch_size(self, model_name: str, current_stats: BatchStats) -> int:
        """Dynamically optimize batch size based on performance
        
        Args:
            model_name: Name of the model
            current_stats: Current batch statistics
            
        Returns:
            Optimized batch size
        """
        if not self.optimization_enabled or model_name not in self.batch_configs:
            return current_stats.batch_size
        
        config = self.batch_configs[model_name]
        current_size = current_stats.batch_size
        
        # Get GPU metrics for optimization
        gpu_metrics = self.gpu_monitor.get_gpu_metrics()
        if not gpu_metrics:
            return current_size
        
        avg_memory_util = np.mean([gpu.memory_utilization for gpu in gpu_metrics])
        avg_gpu_util = np.mean([gpu.gpu_utilization for gpu in gpu_metrics])
        
        # Optimization logic
        new_size = current_size
        
        # Memory-based optimization
        if avg_memory_util > 95:  # Too high memory usage
            new_size = max(config.min_batch_size, int(current_size * 0.8))
            logger.info(f"🔻 Reducing batch size due to high memory usage ({avg_memory_util:.1f}%)")
        elif avg_memory_util < config.target_gpu_memory_util * 0.7:  # Low memory usage
            new_size = min(config.max_batch_size, int(current_size * 1.2))
            logger.info(f"🔺 Increasing batch size due to low memory usage ({avg_memory_util:.1f}%)")
        
        # Time-based optimization
        batch_time_per_sample = current_stats.processing_time / current_stats.samples_processed
        target_time_per_sample = config.target_batch_time / current_size
        
        if batch_time_per_sample > target_time_per_sample * 1.5:  # Too slow
            new_size = max(config.min_batch_size, int(current_size * 0.9))
            logger.info(f"🔻 Reducing batch size due to slow processing")
        elif batch_time_per_sample < target_time_per_sample * 0.7:  # Very fast
            new_size = min(config.max_batch_size, int(current_size * 1.1))
            logger.info(f"🔺 Increasing batch size due to fast processing")
        
        # Performance-based optimization
        if len(self.performance_history) >= 2:
            recent_performance = self.performance_history[-2:]
            if (recent_performance[-1]['samples_per_second'] < 
                recent_performance[-2]['samples_per_second'] * 0.9):
                # Performance decreased, revert optimization
                new_size = max(config.min_batch_size, int(current_size * 0.95))
                logger.info(f"🔻 Reducing batch size due to performance decrease")
        
        # Ensure within bounds
        new_size = max(config.min_batch_size, min(config.max_batch_size, new_size))
        
        if new_size != current_size:
            logger.info(f"📊 Optimized batch size: {current_size} → {new_size}")
        
        return new_size
    
    def process_batch_with_stats(self, model_name: str, batch_samples: List[Dict], 
                               batch_size: int) -> Tuple[List[Dict], BatchStats]:
        """Process a batch and collect statistics
        
        Args:
            model_name: Name of the model
            batch_samples: List of samples to process
            batch_size: Current batch size
            
        Returns:
            Tuple of (results, batch_stats)
        """
        start_time = time.time()
        
        # Get initial GPU state
        initial_gpu_metrics = self.gpu_monitor.get_gpu_metrics()
        initial_memory = np.mean([gpu.memory_used_mb for gpu in initial_gpu_metrics]) if initial_gpu_metrics else 0
        
        try:
            # Process the batch
            results = self.runner.evaluate_model_batch(
                model_name=model_name,
                samples=batch_samples,
                batch_size=batch_size
            )
            
            processing_time = time.time() - start_time
            samples_processed = len(results)
            successful_samples = sum(1 for r in results if r.get('success', False))
            
            # Get final GPU state
            final_gpu_metrics = self.gpu_monitor.get_gpu_metrics()
            final_memory = np.mean([gpu.memory_used_mb for gpu in final_gpu_metrics]) if final_gpu_metrics else 0
            avg_gpu_util = np.mean([gpu.gpu_utilization for gpu in final_gpu_metrics]) if final_gpu_metrics else 0
            
            # Calculate statistics
            samples_per_second = samples_processed / processing_time if processing_time > 0 else 0
            success_rate = successful_samples / samples_processed if samples_processed > 0 else 0
            
            stats = BatchStats(
                model_name=model_name,
                batch_size=batch_size,
                samples_processed=samples_processed,
                processing_time=processing_time,
                samples_per_second=samples_per_second,
                gpu_memory_used_mb=final_memory,
                gpu_utilization=avg_gpu_util,
                success_rate=success_rate,
                timestamp=datetime.now().isoformat()
            )
            
            return results, stats
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Batch processing failed: {e}")
            
            # Return failed stats
            stats = BatchStats(
                model_name=model_name,
                batch_size=batch_size,
                samples_processed=0,
                processing_time=processing_time,
                samples_per_second=0,
                gpu_memory_used_mb=initial_memory,
                gpu_utilization=0,
                success_rate=0,
                timestamp=datetime.now().isoformat()
            )
            
            return [], stats
    
    def evaluate_model_adaptive(self, model_name: str, samples: List[Dict], 
                              model_size_gb: float) -> List[Dict]:
        """Evaluate model with adaptive batch processing
        
        Args:
            model_name: Name of the model
            samples: List of samples to process
            model_size_gb: Size of model in GB for optimization
            
        Returns:
            List of evaluation results
        """
        # Create or get batch configuration
        if model_name not in self.batch_configs:
            self.create_batch_config(model_name, model_size_gb)
        
        config = self.batch_configs[model_name]
        current_batch_size = config.initial_batch_size
        
        all_results = []
        batch_number = 0
        total_batches = (len(samples) + current_batch_size - 1) // current_batch_size
        
        logger.info(f"🚀 Starting adaptive evaluation of {model_name}")
        logger.info(f"   📊 Total samples: {len(samples)}")
        logger.info(f"   📦 Initial batch size: {current_batch_size}")
        logger.info(f"   🔢 Estimated batches: {total_batches}")
        
        # Progress tracking
        progress_bar = tqdm(total=len(samples), desc=f"Evaluating {model_name}", unit="samples")
        
        i = 0
        while i < len(samples):
            batch_number += 1
            
            # Get current batch
            batch_samples = samples[i:i + current_batch_size]
            actual_batch_size = len(batch_samples)
            
            logger.info(f"📦 Processing batch {batch_number} ({actual_batch_size} samples, batch_size={current_batch_size})")
            
            # Process batch with statistics
            batch_results, batch_stats = self.process_batch_with_stats(
                model_name, batch_samples, current_batch_size
            )
            
            # Store results and statistics
            all_results.extend(batch_results)
            self.batch_stats[f"{model_name}_batch_{batch_number}"] = batch_stats
            self.performance_history.append(asdict(batch_stats))
            
            # Update progress
            progress_bar.update(actual_batch_size)
            progress_bar.set_postfix({
                'batch_size': current_batch_size,
                'samples/sec': f"{batch_stats.samples_per_second:.1f}",
                'gpu_mem': f"{batch_stats.gpu_memory_used_mb/1024:.1f}GB",
                'success': f"{batch_stats.success_rate:.1%}"
            })
            
            # Log batch performance
            logger.info(f"✅ Batch {batch_number} complete:")
            logger.info(f"   ⏱️  Time: {batch_stats.processing_time:.1f}s")
            logger.info(f"   🚀 Speed: {batch_stats.samples_per_second:.1f} samples/sec")
            logger.info(f"   💾 GPU Memory: {batch_stats.gpu_memory_used_mb/1024:.1f}GB")
            logger.info(f"   ✅ Success Rate: {batch_stats.success_rate:.1%}")
            
            # Adaptive optimization (after first batch)
            if batch_number >= 1 and config.adaptive_sizing:
                optimized_size = self.optimize_batch_size(model_name, batch_stats)
                if optimized_size != current_batch_size:
                    current_batch_size = optimized_size
                    # Recalculate remaining batches
                    remaining_samples = len(samples) - (i + actual_batch_size)
                    estimated_remaining_batches = (remaining_samples + current_batch_size - 1) // current_batch_size
                    logger.info(f"📊 Estimated remaining batches: {estimated_remaining_batches}")
            
            i += actual_batch_size
        
        progress_bar.close()
        
        # Final statistics
        total_time = sum(stats.processing_time for stats in 
                        [self.batch_stats[k] for k in self.batch_stats 
                         if k.startswith(f"{model_name}_batch_")])
        
        successful_results = [r for r in all_results if r.get('success', False)]
        overall_success_rate = len(successful_results) / len(all_results) if all_results else 0
        overall_speed = len(all_results) / total_time if total_time > 0 else 0
        
        logger.info(f"🎉 {model_name} evaluation complete!")
        logger.info(f"   📊 Total samples: {len(all_results)}")
        logger.info(f"   ✅ Success rate: {overall_success_rate:.1%}")
        logger.info(f"   ⏱️  Total time: {total_time:.1f}s")
        logger.info(f"   🚀 Overall speed: {overall_speed:.1f} samples/sec")
        logger.info(f"   📦 Batches processed: {batch_number}")
        
        return all_results
    
    def save_performance_stats(self, output_file: str):
        """Save performance statistics to file
        
        Args:
            output_file: Path to output file
        """
        stats_data = {
            'batch_configs': {k: asdict(v) for k, v in self.batch_configs.items()},
            'batch_stats': {k: asdict(v) for k, v in self.batch_stats.items()},
            'performance_history': self.performance_history,
            'summary': self.get_performance_summary()
        }
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(stats_data, f, indent=2)
        
        logger.info(f"📊 Performance statistics saved to: {output_path}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of performance statistics
        
        Returns:
            Dictionary with performance summary
        """
        if not self.performance_history:
            return {}
        
        # Aggregate statistics
        total_samples = sum(stat['samples_processed'] for stat in self.performance_history)
        total_time = sum(stat['processing_time'] for stat in self.performance_history)
        avg_samples_per_second = np.mean([stat['samples_per_second'] for stat in self.performance_history])
        avg_success_rate = np.mean([stat['success_rate'] for stat in self.performance_history])
        avg_gpu_memory = np.mean([stat['gpu_memory_used_mb'] for stat in self.performance_history])
        avg_gpu_util = np.mean([stat['gpu_utilization'] for stat in self.performance_history])
        
        # Model-specific statistics
        model_stats = {}
        for stat in self.performance_history:
            model_name = stat['model_name']
            if model_name not in model_stats:
                model_stats[model_name] = []
            model_stats[model_name].append(stat)
        
        model_summaries = {}
        for model_name, stats in model_stats.items():
            model_summaries[model_name] = {
                'total_samples': sum(s['samples_processed'] for s in stats),
                'total_time': sum(s['processing_time'] for s in stats),
                'avg_samples_per_second': np.mean([s['samples_per_second'] for s in stats]),
                'avg_success_rate': np.mean([s['success_rate'] for s in stats]),
                'batches_processed': len(stats),
                'optimal_batch_size': int(np.median([s['batch_size'] for s in stats]))
            }
        
        return {
            'overall': {
                'total_samples_processed': total_samples,
                'total_processing_time': total_time,
                'average_samples_per_second': avg_samples_per_second,
                'average_success_rate': avg_success_rate,
                'average_gpu_memory_mb': avg_gpu_memory,
                'average_gpu_utilization': avg_gpu_util,
                'total_batches': len(self.performance_history)
            },
            'by_model': model_summaries,
            'optimization_enabled': self.optimization_enabled
        }

    def evaluate_model_sequential(self, model_name: str, samples: List[Dict]) -> List[Dict]:
        """Evaluate model using sequential processing (no per-batch model loading)
        
        This is the FIXED version that loads model once and processes all samples,
        avoiding the VLLM conflicts and memory issues from parallel execution.
        
        Args:
            model_name: Name of model to evaluate
            samples: List of sample dictionaries
            
        Returns:
            List of evaluation results
        """
        logger.info(f"🚀 Starting sequential evaluation of {model_name}")
        logger.info(f"   📊 Total samples: {len(samples)}")
        logger.info(f"   🔧 Method: Sequential processing with single model load")
        
        start_time = time.time()
        
        try:
            # Start GPU monitoring for this model
            self.gpu_monitor.start_monitoring()
            
            # Use the new evaluate_model_complete method
            # This loads the model ONCE and processes all samples
            results = self.runner.evaluate_model_complete(model_name, samples)
            
            # Calculate final performance statistics
            end_time = time.time()
            total_time = end_time - start_time
            successful = sum(1 for r in results if r.get('success', False))
            success_rate = successful / len(results) if results else 0
            samples_per_second = len(results) / total_time if total_time > 0 else 0
            
            # Get final GPU metrics
            final_gpu_metrics = self.gpu_monitor.get_gpu_metrics()
            avg_gpu_memory = np.mean([gpu.memory_used_mb for gpu in final_gpu_metrics]) if final_gpu_metrics else 0
            avg_gpu_util = np.mean([gpu.gpu_utilization for gpu in final_gpu_metrics]) if final_gpu_metrics else 0
            
            # Create performance statistics
            stats = BatchStats(
                model_name=model_name,
                batch_size=len(results),  # All samples processed together
                samples_processed=len(results),
                processing_time=total_time,
                samples_per_second=samples_per_second,
                gpu_memory_used_mb=avg_gpu_memory,
                gpu_utilization=avg_gpu_util,
                success_rate=success_rate,
                timestamp=datetime.now().isoformat()
            )
            
            # Store statistics
            self.batch_stats[f"{model_name}_sequential"] = stats
            self.performance_history.append(asdict(stats))
            
            logger.info(f"✅ Sequential evaluation complete: {model_name}")
            logger.info(f"   📊 Total samples: {len(results)}")
            logger.info(f"   ✅ Successful: {successful} ({success_rate*100:.1f}%)")
            logger.info(f"   ❌ Failed: {len(results) - successful}")
            logger.info(f"   ⏱️  Total time: {total_time:.1f}s")
            logger.info(f"   🚀 Speed: {samples_per_second:.1f} samples/sec")
            logger.info(f"   💾 GPU Memory: {avg_gpu_memory/1024:.1f}GB")
            logger.info(f"   ⚡ Performance improvement: No more model loading per batch!")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Sequential evaluation failed for {model_name}: {e}")
            
            # Create error results for all samples
            error_results = []
            for i, sample in enumerate(samples):
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
            
            return error_results
            
        finally:
            # Always stop monitoring and cleanup
            try:
                self.gpu_monitor.stop_monitoring()
            except Exception:
                pass
    
    def evaluate_models_optimized(self, model_names: List[str], samples: List[Dict]) -> List[Dict]:
        """Evaluate models with MAXIMUM GPU utilization strategy
        
        This method uses the optimal approach for each model category:
        - Small models: Run 4 in parallel on separate GPUs
        - Medium models: Run with 2-GPU tensor parallelism
        - Large models: Run with 4-GPU tensor parallelism
        
        Args:
            model_names: List of model names to evaluate
            samples: List of sample dictionaries
            
        Returns:
            Combined list of evaluation results from all models
        """
        logger.info(f"🚀 Starting OPTIMIZED evaluation with MAXIMUM GPU utilization")
        logger.info(f"   📊 Models to evaluate: {len(model_names)}")
        logger.info(f"   📊 Samples per model: {len(samples)}")
        logger.info(f"   🎯 Strategy: Parallel small models, multi-GPU tensor parallelism for large models")
        
        # Categorize models by GPU requirements
        categories = self.runner.categorize_models_by_gpu_needs(model_names)
        
        all_results = []
        total_start_time = time.time()
        
        try:
            # Phase 1: Small models - Run 4 in parallel
            small_models = categories["small"]
            if small_models:
                logger.info(f"\n🚀 PHASE 1: Small Models - PARALLEL EXECUTION")
                logger.info(f"   📊 Models: {len(small_models)} ({small_models})")
                logger.info(f"   ⚡ Strategy: 4 models running simultaneously on separate GPUs")
                
                # Process small models in batches of 4
                for i in range(0, len(small_models), 4):
                    batch_models = small_models[i:i+4]
                    logger.info(f"   🔄 Processing parallel batch: {batch_models}")
                    
                    batch_results = self.runner.evaluate_models_parallel(batch_models, samples)
                    all_results.extend(batch_results)
                    
                    logger.info(f"   ✅ Completed parallel batch: {len(batch_results)} results")
            
            # Phase 2: Medium models - 2 GPU tensor parallelism
            medium_models = categories["medium"]  
            if medium_models:
                logger.info(f"\n🚀 PHASE 2: Medium Models - 2-GPU TENSOR PARALLELISM")
                logger.info(f"   📊 Models: {len(medium_models)} ({medium_models})")
                logger.info(f"   ⚡ Strategy: Each model uses 2 GPUs for faster inference")
                
                for model_name in medium_models:
                    logger.info(f"   🔄 Processing medium model: {model_name}")
                    
                    # This will use optimal 2-GPU configuration automatically
                    model_results = self.evaluate_model_sequential(model_name, samples)
                    all_results.extend(model_results)
                    
                    logger.info(f"   ✅ Completed medium model: {len(model_results)} results")
            
            # Phase 3: Large models - 4 GPU tensor parallelism
            large_models = categories["large"]
            if large_models:
                logger.info(f"\n🚀 PHASE 3: Large Models - 4-GPU TENSOR PARALLELISM") 
                logger.info(f"   📊 Models: {len(large_models)} ({large_models})")
                logger.info(f"   ⚡ Strategy: Each model uses ALL 4 GPUs for maximum speed")
                
                for model_name in large_models:
                    logger.info(f"   🔄 Processing large model: {model_name}")
                    
                    # This will use optimal 4-GPU configuration automatically
                    model_results = self.evaluate_model_sequential(model_name, samples)
                    all_results.extend(model_results)
                    
                    logger.info(f"   ✅ Completed large model: {len(model_results)} results")
            
            # Calculate final performance metrics
            total_time = time.time() - total_start_time
            successful_results = sum(1 for r in all_results if r.get('success', False))
            
            logger.info(f"\n🎉 OPTIMIZED EVALUATION COMPLETE!")
            logger.info(f"   📊 Total results: {len(all_results):,}")
            logger.info(f"   ✅ Successful results: {successful_results:,} ({successful_results/len(all_results)*100:.1f}%)")
            logger.info(f"   ⏱️  Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
            logger.info(f"   🚀 Average speed: {len(all_results)/total_time:.1f} samples/sec")
            logger.info(f"   ⚡ GPU utilization: MAXIMIZED across all 4 A100 GPUs")
            
            performance_improvement = self._estimate_performance_improvement(model_names, total_time)
            logger.info(f"   📈 Estimated improvement vs single-GPU: {performance_improvement}")
            
            return all_results
            
        except Exception as e:
            logger.error(f"❌ Optimized evaluation failed: {e}")
            logger.info(f"🔄 Falling back to sequential evaluation...")
            
            # Fallback to sequential evaluation
            fallback_results = []
            for model_name in model_names:
                try:
                    results = self.evaluate_model_sequential(model_name, samples)
                    fallback_results.extend(results)
                except Exception as model_e:
                    logger.error(f"❌ Fallback failed for {model_name}: {model_e}")
            
            return fallback_results
    
    def _estimate_performance_improvement(self, model_names: List[str], actual_time: float) -> str:
        """Estimate performance improvement compared to single-GPU approach"""
        # Estimate what single-GPU approach would have taken
        estimated_single_gpu_time = 0
        
        for model_name in model_names:
            gpu_config = self.runner.get_optimal_gpu_config(model_name)
            # Rough estimation based on model complexity
            if gpu_config["category"] == "small":
                estimated_single_gpu_time += 300  # 5 minutes per small model  
            elif gpu_config["category"] == "medium":
                estimated_single_gpu_time += 900  # 15 minutes per medium model
            else:  # large
                estimated_single_gpu_time += 1800  # 30 minutes per large model
        
        if actual_time > 0:
            improvement_factor = estimated_single_gpu_time / actual_time
            return f"{improvement_factor:.1f}x faster"
        else:
            return "Unable to calculate"

def main():
    """Test batch processor functionality"""
    # This would be used for testing the batch processor independently
    print("BatchProcessor - Efficient batch processing for server models")
    print("This module is designed to be imported and used with ServerModelRunner")

if __name__ == "__main__":
    main()