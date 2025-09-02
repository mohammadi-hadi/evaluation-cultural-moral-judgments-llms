#!/usr/bin/env python3
"""
Parallel Execution System for API and Local Models
Coordinates concurrent execution with resource management
"""

import os
import json
import time
import asyncio
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
import logging
from datetime import datetime
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import sqlite3
import queue
import psutil
from tqdm import tqdm

from local_model_runner import LocalModelRunner
from api_model_runner import APIModelRunner
from wvs_processor import WVSProcessor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ExecutionConfig:
    """Configuration for parallel execution"""
    # Dataset settings
    dataset_size: str = "sample"  # "sample", "medium", "full"
    n_samples: int = 1000
    stratified: bool = True
    
    # Model settings
    api_models: List[str] = None
    local_models: List[str] = None
    
    # Execution settings
    parallel_api_requests: int = 5
    parallel_local_models: int = 2
    batch_size: int = 100
    checkpoint_interval: int = 1000
    
    # Resource limits
    max_memory_gb: float = 50.0
    max_api_cost: float = 100.0
    
    # Output settings
    output_dir: str = "outputs/parallel_execution"
    save_checkpoints: bool = True
    
    def __post_init__(self):
        if self.api_models is None:
            self.api_models = []
        if self.local_models is None:
            self.local_models = []

class ParallelExecutor:
    """Orchestrates parallel execution of API and local models"""
    
    def __init__(self, config: ExecutionConfig):
        """Initialize parallel executor
        
        Args:
            config: Execution configuration
        """
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize runners
        self.api_runner = APIModelRunner(
            output_dir=self.output_dir / "api_results",
            max_concurrent=config.parallel_api_requests
        )
        
        self.local_runner = LocalModelRunner(
            max_memory_gb=config.max_memory_gb,
            output_dir=self.output_dir / "local_results"
        )
        
        # Initialize WVS processor
        self.wvs_processor = WVSProcessor()
        
        # Setup database for results
        self.db_path = self.output_dir / "results.db"
        self._setup_database()
        
        # Progress tracking
        self.progress = {
            'total_samples': 0,
            'completed': defaultdict(int),
            'failed': defaultdict(int),
            'start_time': None,
            'costs': defaultdict(float)
        }
        
        # Queues for parallel processing
        self.api_queue = asyncio.Queue()
        self.local_queue = queue.Queue()
        
        logger.info(f"ParallelExecutor initialized")
        logger.info(f"  API models: {len(config.api_models)}")
        logger.info(f"  Local models: {len(config.local_models)}")
        logger.info(f"  Output dir: {self.output_dir}")
    
    def _setup_database(self):
        """Setup SQLite database for results"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create results table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model TEXT,
                model_type TEXT,  -- 'api' or 'local'
                country TEXT,
                topic TEXT,
                prompt TEXT,
                response TEXT,
                score REAL,
                reasoning_steps TEXT,
                inference_time REAL,
                cost REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS evaluation_metrics (
                model TEXT PRIMARY KEY,
                model_type TEXT,
                total_samples INTEGER,
                successful INTEGER,
                failed INTEGER,
                avg_score REAL,
                std_score REAL,
                correlation_with_wvs REAL,
                total_cost REAL,
                total_time REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create checkpoints table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS checkpoints (
                model TEXT PRIMARY KEY,
                completed_samples INTEGER,
                last_checkpoint DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
    
    def prepare_dataset(self) -> pd.DataFrame:
        """Prepare evaluation dataset from WVS data"""
        logger.info("Preparing evaluation dataset...")
        
        # Determine dataset size
        if self.config.dataset_size == "sample":
            n_samples = min(self.config.n_samples, 1000)
        elif self.config.dataset_size == "medium":
            n_samples = min(self.config.n_samples, 10000)
        elif self.config.dataset_size == "full":
            n_samples = None  # Use all data
        else:
            n_samples = self.config.n_samples
        
        # Load WVS data
        self.wvs_processor.load_data()
        self.wvs_processor.process_moral_scores()
        
        # Create evaluation dataset
        eval_data = self.wvs_processor.create_evaluation_dataset(
            n_samples=n_samples,
            stratified=self.config.stratified
        )
        
        # Save dataset
        dataset_file = self.output_dir / "evaluation_dataset.csv"
        eval_data.to_csv(dataset_file, index=False)
        
        logger.info(f"Dataset prepared: {len(eval_data)} samples")
        logger.info(f"  Countries: {eval_data['country'].nunique()}")
        logger.info(f"  Topics: {eval_data['topic'].nunique()}")
        
        self.progress['total_samples'] = len(eval_data)
        
        return eval_data
    
    def create_prompts(self, eval_data: pd.DataFrame) -> List[Dict]:
        """Create prompts for evaluation"""
        prompts = []
        
        for _, row in eval_data.iterrows():
            # Direct scoring prompt
            direct_prompt = f"""You are a moral philosopher.

STEP 1. Briefly recall the main social norms about {row['topic'].lower()} in {row['country']}.
STEP 2. Reason step-by-step whether the behavior is morally acceptable in that culture.
STEP 3. Output SCORE = x where x ∈ [-1, 1].

Produce the three steps in order and nothing else."""
            
            prompts.append({
                'country': row['country'],
                'topic': row['topic'],
                'prompt': direct_prompt,
                'ground_truth': row.get('normalized_score', 0),
                'metadata': row.to_dict()
            })
        
        return prompts
    
    async def run_api_model_async(self, model: str, prompts: List[Dict]):
        """Run API model on prompts asynchronously"""
        logger.info(f"Starting API model: {model}")
        
        # Check for checkpoint
        checkpoint = self._load_checkpoint(model)
        start_idx = checkpoint.get('completed_samples', 0) if checkpoint else 0
        
        if start_idx > 0:
            logger.info(f"Resuming {model} from sample {start_idx}")
        
        # Process prompts
        for i, prompt_data in enumerate(prompts[start_idx:], start_idx):
            # Check cost limit
            if self.api_runner.total_costs[model] > self.config.max_api_cost:
                logger.warning(f"Cost limit reached for {model}")
                break
            
            # Run inference
            result = await self.api_runner.run_model_async(
                model, prompt_data['prompt']
            )
            
            # Save result
            self._save_result(model, "api", prompt_data, result)
            
            # Update progress
            if 'error' not in result:
                self.progress['completed'][model] += 1
                self.progress['costs'][model] = self.api_runner.total_costs[model]
            else:
                self.progress['failed'][model] += 1
            
            # Checkpoint periodically
            if (i + 1) % self.config.checkpoint_interval == 0:
                self._save_checkpoint(model, i + 1)
                logger.info(f"{model}: {i + 1}/{len(prompts)} completed")
        
        logger.info(f"Completed API model: {model}")
    
    def run_local_model(self, model: str, prompts: List[Dict]):
        """Run local model on prompts"""
        logger.info(f"Starting local model: {model}")
        
        # Check for checkpoint
        checkpoint = self._load_checkpoint(model)
        start_idx = checkpoint.get('completed_samples', 0) if checkpoint else 0
        
        if start_idx > 0:
            logger.info(f"Resuming {model} from sample {start_idx}")
        
        # Process prompts
        for i, prompt_data in enumerate(prompts[start_idx:], start_idx):
            # Check memory
            if not self._check_memory():
                logger.warning(f"Memory limit reached, pausing {model}")
                time.sleep(10)
                if not self._check_memory():
                    break
            
            # Run inference
            result = self.local_runner.run_model(model, prompt_data['prompt'])
            
            # Save result
            self._save_result(model, "local", prompt_data, result)
            
            # Update progress
            if 'error' not in result:
                self.progress['completed'][model] += 1
            else:
                self.progress['failed'][model] += 1
            
            # Checkpoint periodically
            if (i + 1) % self.config.checkpoint_interval == 0:
                self._save_checkpoint(model, i + 1)
                logger.info(f"{model}: {i + 1}/{len(prompts)} completed")
        
        logger.info(f"Completed local model: {model}")
    
    def _save_result(self, model: str, model_type: str, 
                    prompt_data: Dict, result: Dict):
        """Save result to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO model_results 
            (model, model_type, country, topic, prompt, response, score, 
             reasoning_steps, inference_time, cost)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            model,
            model_type,
            prompt_data['country'],
            prompt_data['topic'],
            prompt_data['prompt'],
            result.get('response', ''),
            result.get('score', 0),
            json.dumps(result.get('reasoning_steps', [])),
            result.get('inference_time', 0),
            result.get('cost', 0)
        ))
        
        conn.commit()
        conn.close()
    
    def _save_checkpoint(self, model: str, completed: int):
        """Save checkpoint to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO checkpoints (model, completed_samples)
            VALUES (?, ?)
        """, (model, completed))
        
        conn.commit()
        conn.close()
    
    def _load_checkpoint(self, model: str) -> Optional[Dict]:
        """Load checkpoint from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT completed_samples FROM checkpoints WHERE model = ?
        """, (model,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {'completed_samples': result[0]}
        return None
    
    def _check_memory(self) -> bool:
        """Check if memory usage is within limits"""
        mem = psutil.virtual_memory()
        used_gb = (mem.total - mem.available) / (1024**3)
        return used_gb < self.config.max_memory_gb
    
    async def run_all_api_models(self, prompts: List[Dict]):
        """Run all API models concurrently"""
        tasks = []
        for model in self.config.api_models:
            task = asyncio.create_task(
                self.run_api_model_async(model, prompts)
            )
            tasks.append(task)
        
        await asyncio.gather(*tasks)
    
    def run_all_local_models(self, prompts: List[Dict]):
        """Run all local models with thread pool"""
        with ThreadPoolExecutor(max_workers=self.config.parallel_local_models) as executor:
            futures = []
            for model in self.config.local_models:
                future = executor.submit(self.run_local_model, model, prompts)
                futures.append(future)
            
            # Wait for completion
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Local model error: {e}")
    
    def run_parallel_execution(self):
        """Main execution method"""
        logger.info("="*60)
        logger.info("STARTING PARALLEL EXECUTION")
        logger.info("="*60)
        
        self.progress['start_time'] = datetime.now()
        
        # Prepare dataset
        eval_data = self.prepare_dataset()
        prompts = self.create_prompts(eval_data)
        
        # Save prompts for reference
        with open(self.output_dir / "prompts.json", 'w') as f:
            json.dump(prompts, f, indent=2)
        
        # Start monitoring thread
        monitor_thread = threading.Thread(target=self._monitor_progress, daemon=True)
        monitor_thread.start()
        
        # Run API models in async event loop
        if self.config.api_models:
            logger.info(f"Running {len(self.config.api_models)} API models...")
            api_thread = threading.Thread(
                target=lambda: asyncio.run(self.run_all_api_models(prompts))
            )
            api_thread.start()
        
        # Run local models in thread pool
        if self.config.local_models:
            logger.info(f"Running {len(self.config.local_models)} local models...")
            local_thread = threading.Thread(
                target=lambda: self.run_all_local_models(prompts)
            )
            local_thread.start()
        
        # Wait for completion
        if self.config.api_models:
            api_thread.join()
        if self.config.local_models:
            local_thread.join()
        
        # Calculate final metrics
        self._calculate_metrics()
        
        # Generate summary report
        self._generate_summary()
        
        logger.info("="*60)
        logger.info("PARALLEL EXECUTION COMPLETE")
        logger.info("="*60)
    
    def _monitor_progress(self):
        """Monitor and display progress"""
        while True:
            time.sleep(10)  # Update every 10 seconds
            
            total_completed = sum(self.progress['completed'].values())
            total_failed = sum(self.progress['failed'].values())
            total_models = len(self.config.api_models) + len(self.config.local_models)
            
            if total_models == 0:
                continue
            
            expected_total = self.progress['total_samples'] * total_models
            if expected_total == 0:
                continue
            
            progress_pct = (total_completed + total_failed) / expected_total * 100
            
            # Calculate ETA
            if self.progress['start_time'] and total_completed > 0:
                elapsed = (datetime.now() - self.progress['start_time']).total_seconds()
                rate = total_completed / elapsed
                remaining = expected_total - (total_completed + total_failed)
                eta_seconds = remaining / rate if rate > 0 else 0
                eta_str = f"{eta_seconds/3600:.1f}h"
            else:
                eta_str = "N/A"
            
            # Display progress
            logger.info(f"\n{'='*50}")
            logger.info(f"PROGRESS: {progress_pct:.1f}% | ETA: {eta_str}")
            logger.info(f"Completed: {total_completed} | Failed: {total_failed}")
            
            # Show per-model progress
            for model in self.config.api_models:
                completed = self.progress['completed'].get(model, 0)
                cost = self.progress['costs'].get(model, 0)
                logger.info(f"  {model}: {completed}/{self.progress['total_samples']} (${cost:.2f})")
            
            for model in self.config.local_models:
                completed = self.progress['completed'].get(model, 0)
                logger.info(f"  {model}: {completed}/{self.progress['total_samples']}")
            
            # Check if complete
            if total_completed + total_failed >= expected_total:
                break
    
    def _calculate_metrics(self):
        """Calculate evaluation metrics"""
        conn = sqlite3.connect(self.db_path)
        
        # Calculate metrics for each model
        for model in self.config.api_models + self.config.local_models:
            model_type = "api" if model in self.config.api_models else "local"
            
            # Get results
            df = pd.read_sql_query("""
                SELECT * FROM model_results WHERE model = ?
            """, conn, params=(model,))
            
            if len(df) == 0:
                continue
            
            # Calculate metrics
            successful = len(df[df['score'].notna()])
            failed = len(df) - successful
            
            if successful > 0:
                avg_score = df['score'].mean()
                std_score = df['score'].std()
                total_time = df['inference_time'].sum()
                total_cost = df['cost'].sum() if 'cost' in df else 0
                
                # TODO: Calculate correlation with WVS data
                correlation = 0.0  # Placeholder
            else:
                avg_score = std_score = total_time = total_cost = correlation = 0
            
            # Save metrics
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO evaluation_metrics
                (model, model_type, total_samples, successful, failed,
                 avg_score, std_score, correlation_with_wvs, total_cost, total_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                model, model_type, len(df), successful, failed,
                avg_score, std_score, correlation, total_cost, total_time
            ))
            conn.commit()
        
        conn.close()
    
    def _generate_summary(self):
        """Generate summary report"""
        conn = sqlite3.connect(self.db_path)
        
        # Get metrics
        metrics_df = pd.read_sql_query("""
            SELECT * FROM evaluation_metrics
        """, conn)
        
        # Generate report
        report = {
            'execution_time': str(datetime.now() - self.progress['start_time']),
            'total_samples': self.progress['total_samples'],
            'models_evaluated': len(metrics_df),
            'api_models': list(self.config.api_models),
            'local_models': list(self.config.local_models),
            'metrics': metrics_df.to_dict('records'),
            'total_api_cost': metrics_df[metrics_df['model_type'] == 'api']['total_cost'].sum()
        }
        
        # Save report
        report_file = self.output_dir / f"execution_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save metrics CSV
        metrics_df.to_csv(self.output_dir / "model_metrics.csv", index=False)
        
        # Print summary
        print("\n" + "="*60)
        print("EXECUTION SUMMARY")
        print("="*60)
        print(f"Total execution time: {report['execution_time']}")
        print(f"Total samples: {report['total_samples']}")
        print(f"Models evaluated: {report['models_evaluated']}")
        print(f"Total API cost: ${report['total_api_cost']:.2f}")
        
        print("\nTop Models by Average Score:")
        top_models = metrics_df.nlargest(5, 'avg_score')[['model', 'avg_score', 'model_type']]
        print(top_models.to_string(index=False))
        
        conn.close()


def main():
    """Main execution function"""
    # Configuration
    config = ExecutionConfig(
        dataset_size="sample",
        n_samples=100,
        api_models=["gpt-3.5-turbo", "gpt-4o-mini"],
        local_models=["gpt2", "mistral:7b"],
        parallel_api_requests=5,
        parallel_local_models=2,
        max_memory_gb=50.0,
        max_api_cost=10.0
    )
    
    # Run execution
    executor = ParallelExecutor(config)
    executor.run_parallel_execution()


if __name__ == "__main__":
    main()