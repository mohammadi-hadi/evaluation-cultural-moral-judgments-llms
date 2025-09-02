#!/usr/bin/env python3
"""
Enhanced Parallel Execution System with Robust Error Recovery
Handles internet disconnections, automatic retries, and real-time monitoring
"""

import os
import json
import time
import asyncio
import threading
import socket
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
from tqdm.asyncio import tqdm as async_tqdm
from tqdm import tqdm
import signal
import sys
from collections import defaultdict

from local_model_runner import LocalModelRunner
from api_model_runner import APIModelRunner
from wvs_processor import WVSProcessor

# Setup logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class EnhancedExecutionConfig:
    """Enhanced configuration with error recovery settings"""
    # Dataset settings
    dataset_size: str = "sample"
    n_samples: int = 1000
    stratified: bool = True
    
    # Model settings
    api_models: List[str] = None
    local_models: List[str] = None
    
    # Execution settings
    parallel_api_requests: int = 5
    parallel_local_models: int = 2
    batch_size: int = 100
    checkpoint_interval: int = 100
    
    # Resource limits
    max_memory_gb: float = 50.0
    max_api_cost: float = 100.0
    
    # Error recovery settings
    max_retries: int = 5
    retry_delay: int = 10  # seconds
    internet_check_interval: int = 30  # seconds
    auto_resume: bool = True
    
    # Output settings
    output_dir: str = "outputs/comprehensive"
    save_checkpoints: bool = True
    
    def __post_init__(self):
        if self.api_models is None:
            self.api_models = []
        if self.local_models is None:
            self.local_models = []

class InternetMonitor:
    """Monitor internet connectivity"""
    
    def __init__(self, check_interval: int = 30):
        self.check_interval = check_interval
        self.is_connected = True
        self.last_check = time.time()
        self.monitor_thread = None
        self.stop_monitoring = False
    
    def check_connection(self) -> bool:
        """Check if internet is available"""
        try:
            # Try multiple hosts for robustness
            hosts = [
                ("8.8.8.8", 53),  # Google DNS
                ("1.1.1.1", 53),  # Cloudflare DNS
                ("api.openai.com", 443)  # OpenAI API
            ]
            
            for host, port in hosts:
                try:
                    socket.create_connection((host, port), timeout=3)
                    return True
                except:
                    continue
            
            return False
        except:
            return False
    
    def start_monitoring(self):
        """Start background monitoring thread"""
        self.stop_monitoring = False
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while not self.stop_monitoring:
            current_connection = self.check_connection()
            
            if current_connection != self.is_connected:
                if current_connection:
                    logger.info("✅ Internet connection restored")
                else:
                    logger.warning("❌ Internet connection lost")
            
            self.is_connected = current_connection
            self.last_check = time.time()
            time.sleep(self.check_interval)
    
    def stop(self):
        """Stop monitoring"""
        self.stop_monitoring = True
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)

class EnhancedParallelExecutor:
    """Enhanced orchestrator with robust error handling"""
    
    def __init__(self, config: EnhancedExecutionConfig):
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
        
        # Setup database
        self.db_path = self.output_dir / "results.db"
        self._setup_database()
        
        # Internet monitoring
        self.internet_monitor = InternetMonitor(config.internet_check_interval)
        
        # Progress tracking with more detail
        self.progress = {
            'total_samples': 0,
            'completed': defaultdict(int),
            'failed': defaultdict(int),
            'retries': defaultdict(int),
            'start_time': None,
            'costs': defaultdict(float),
            'last_checkpoint': defaultdict(lambda: None)
        }
        
        # Error tracking
        self.error_log = []
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.shutdown_requested = False
        
        logger.info(f"Enhanced Parallel Executor initialized")
        logger.info(f"  Auto-resume: {config.auto_resume}")
        logger.info(f"  Max retries: {config.max_retries}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info("Shutdown requested, saving progress...")
        self.shutdown_requested = True
        self._save_all_checkpoints()
        self.internet_monitor.stop()
        sys.exit(0)
    
    def _setup_database(self):
        """Setup SQLite database with enhanced schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Enhanced results table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model TEXT,
                model_type TEXT,
                country TEXT,
                topic TEXT,
                prompt TEXT,
                response TEXT,
                score REAL,
                reasoning_steps TEXT,
                inference_time REAL,
                cost REAL,
                retry_count INTEGER DEFAULT 0,
                error_message TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Error log table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS error_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model TEXT,
                error_type TEXT,
                error_message TEXT,
                retry_count INTEGER,
                resolved BOOLEAN DEFAULT 0,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Session info table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS session_info (
                session_id TEXT PRIMARY KEY,
                start_time DATETIME,
                last_update DATETIME,
                config TEXT,
                status TEXT
            )
        """)
        
        # Existing tables (metrics, checkpoints)
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
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS checkpoints (
                model TEXT PRIMARY KEY,
                completed_samples INTEGER,
                last_checkpoint DATETIME DEFAULT CURRENT_TIMESTAMP,
                retry_count INTEGER DEFAULT 0
            )
        """)
        
        conn.commit()
        conn.close()
    
    def _log_error(self, model: str, error_type: str, error_message: str, retry_count: int = 0):
        """Log error to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO error_log (model, error_type, error_message, retry_count)
            VALUES (?, ?, ?, ?)
        """, (model, error_type, str(error_message), retry_count))
        
        conn.commit()
        conn.close()
        
        # Also keep in memory for quick access
        self.error_log.append({
            'model': model,
            'error_type': error_type,
            'error_message': str(error_message),
            'retry_count': retry_count,
            'timestamp': datetime.now()
        })
    
    async def run_api_model_with_retry(self, model: str, prompts: List[Dict]):
        """Run API model with automatic retry and internet monitoring"""
        logger.info(f"🚀 Starting API model: {model}")
        
        # Check for checkpoint
        checkpoint = self._load_checkpoint(model)
        start_idx = checkpoint.get('completed_samples', 0) if checkpoint else 0
        
        if start_idx > 0:
            logger.info(f"📂 Resuming {model} from sample {start_idx}/{len(prompts)}")
        
        # Progress bar
        pbar = async_tqdm(
            total=len(prompts) - start_idx,
            desc=f"API: {model}",
            position=self.config.api_models.index(model) if model in self.config.api_models else 0,
            leave=True
        )
        
        consecutive_failures = 0
        
        for i, prompt_data in enumerate(prompts[start_idx:], start_idx):
            if self.shutdown_requested:
                break
            
            # Check cost limit
            if self.api_runner.total_costs[model] > self.config.max_api_cost:
                logger.warning(f"💰 Cost limit reached for {model}")
                break
            
            # Retry loop
            retry_count = 0
            success = False
            
            while retry_count < self.config.max_retries and not success:
                try:
                    # Check internet connection
                    if not self.internet_monitor.is_connected:
                        logger.warning(f"⏸️  Waiting for internet connection...")
                        while not self.internet_monitor.is_connected:
                            await asyncio.sleep(5)
                            if self.shutdown_requested:
                                break
                        logger.info(f"✅ Internet restored, resuming {model}")
                    
                    # Run inference
                    result = await self.api_runner.run_model_async(
                        model, prompt_data['prompt'], max_retries=3
                    )
                    
                    # Check for error in result
                    if 'error' in result:
                        raise Exception(result['error'])
                    
                    # Save successful result
                    self._save_result(model, "api", prompt_data, result, retry_count)
                    
                    # Update progress
                    self.progress['completed'][model] += 1
                    self.progress['costs'][model] = self.api_runner.total_costs[model]
                    
                    success = True
                    consecutive_failures = 0
                    pbar.update(1)
                    
                except Exception as e:
                    retry_count += 1
                    consecutive_failures += 1
                    
                    error_type = "API" if "API" in str(e) else "Network"
                    self._log_error(model, error_type, str(e), retry_count)
                    
                    if retry_count < self.config.max_retries:
                        wait_time = self.config.retry_delay * (2 ** min(retry_count - 1, 3))
                        logger.warning(f"⚠️  {model} error (retry {retry_count}/{self.config.max_retries}): {e}")
                        logger.info(f"⏳ Waiting {wait_time}s before retry...")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"❌ {model} failed after {retry_count} retries: {e}")
                        self.progress['failed'][model] += 1
                        
                        # Save failed result
                        failed_result = {'error': str(e), 'retry_count': retry_count}
                        self._save_result(model, "api", prompt_data, failed_result, retry_count)
            
            # Checkpoint periodically
            if (i + 1) % self.config.checkpoint_interval == 0:
                self._save_checkpoint(model, i + 1)
                logger.info(f"💾 {model}: Checkpoint saved at {i + 1}/{len(prompts)}")
            
            # If too many consecutive failures, pause
            if consecutive_failures >= 10:
                logger.error(f"🛑 {model}: Too many consecutive failures, pausing for 60s")
                await asyncio.sleep(60)
                consecutive_failures = 0
        
        pbar.close()
        logger.info(f"✅ Completed API model: {model}")
    
    def run_local_model_with_retry(self, model: str, prompts: List[Dict]):
        """Run local model with retry and memory monitoring"""
        logger.info(f"🚀 Starting local model: {model}")
        
        # Check for checkpoint
        checkpoint = self._load_checkpoint(model)
        start_idx = checkpoint.get('completed_samples', 0) if checkpoint else 0
        
        if start_idx > 0:
            logger.info(f"📂 Resuming {model} from sample {start_idx}/{len(prompts)}")
        
        # Progress bar
        pbar = tqdm(
            total=len(prompts) - start_idx,
            desc=f"Local: {model}",
            position=len(self.config.api_models) + self.config.local_models.index(model) 
                     if model in self.config.local_models else 0,
            leave=True
        )
        
        consecutive_failures = 0
        
        for i, prompt_data in enumerate(prompts[start_idx:], start_idx):
            if self.shutdown_requested:
                break
            
            # Check memory
            if not self._check_memory():
                logger.warning(f"⚠️  Memory limit reached, cleaning up...")
                self.local_runner.cleanup_models()
                time.sleep(10)
                if not self._check_memory():
                    logger.error(f"❌ Cannot free enough memory for {model}")
                    break
            
            # Retry loop
            retry_count = 0
            success = False
            
            while retry_count < self.config.max_retries and not success:
                try:
                    # Run inference
                    result = self.local_runner.run_model(model, prompt_data['prompt'])
                    
                    # Check for error
                    if 'error' in result:
                        raise Exception(result['error'])
                    
                    # Save successful result
                    self._save_result(model, "local", prompt_data, result, retry_count)
                    
                    # Update progress
                    self.progress['completed'][model] += 1
                    
                    success = True
                    consecutive_failures = 0
                    pbar.update(1)
                    
                except Exception as e:
                    retry_count += 1
                    consecutive_failures += 1
                    
                    error_type = "Memory" if "memory" in str(e).lower() else "Model"
                    self._log_error(model, error_type, str(e), retry_count)
                    
                    if retry_count < self.config.max_retries:
                        wait_time = self.config.retry_delay * retry_count
                        logger.warning(f"⚠️  {model} error (retry {retry_count}/{self.config.max_retries}): {e}")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"❌ {model} failed after {retry_count} retries")
                        self.progress['failed'][model] += 1
                        
                        # Save failed result
                        failed_result = {'error': str(e), 'retry_count': retry_count}
                        self._save_result(model, "local", prompt_data, failed_result, retry_count)
            
            # Checkpoint periodically
            if (i + 1) % self.config.checkpoint_interval == 0:
                self._save_checkpoint(model, i + 1)
                logger.info(f"💾 {model}: Checkpoint saved at {i + 1}/{len(prompts)}")
            
            # If too many failures, clean up and retry
            if consecutive_failures >= 5:
                logger.warning(f"🧹 {model}: Cleaning up after failures")
                self.local_runner.cleanup_models()
                time.sleep(30)
                consecutive_failures = 0
        
        pbar.close()
        logger.info(f"✅ Completed local model: {model}")
    
    def _save_result(self, model: str, model_type: str, 
                    prompt_data: Dict, result: Dict, retry_count: int = 0):
        """Save result to database with retry count"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO model_results 
            (model, model_type, country, topic, prompt, response, score, 
             reasoning_steps, inference_time, cost, retry_count, error_message)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            model,
            model_type,
            prompt_data['country'],
            prompt_data['topic'],
            prompt_data['prompt'],
            result.get('response', ''),
            result.get('score', None),
            json.dumps(result.get('reasoning_steps', [])),
            result.get('inference_time', 0),
            result.get('cost', 0),
            retry_count,
            result.get('error', None)
        ))
        
        conn.commit()
        conn.close()
    
    def _save_checkpoint(self, model: str, completed: int):
        """Save checkpoint with retry information"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        retries = self.progress['retries'].get(model, 0)
        
        cursor.execute("""
            INSERT OR REPLACE INTO checkpoints (model, completed_samples, retry_count)
            VALUES (?, ?, ?)
        """, (model, completed, retries))
        
        conn.commit()
        conn.close()
        
        self.progress['last_checkpoint'][model] = datetime.now()
    
    def _save_all_checkpoints(self):
        """Save checkpoints for all models"""
        for model in self.config.api_models + self.config.local_models:
            if self.progress['completed'][model] > 0:
                self._save_checkpoint(model, self.progress['completed'][model])
    
    def _load_checkpoint(self, model: str) -> Optional[Dict]:
        """Load checkpoint from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT completed_samples, retry_count FROM checkpoints WHERE model = ?
        """, (model,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                'completed_samples': result[0],
                'retry_count': result[1]
            }
        return None
    
    def _check_memory(self) -> bool:
        """Check if memory usage is within limits"""
        mem = psutil.virtual_memory()
        used_gb = (mem.total - mem.available) / (1024**3)
        return used_gb < self.config.max_memory_gb
    
    def prepare_dataset(self) -> pd.DataFrame:
        """Prepare evaluation dataset"""
        logger.info("📊 Preparing evaluation dataset...")
        
        # Determine dataset size
        if self.config.dataset_size == "sample":
            n_samples = min(self.config.n_samples, 1000)
        elif self.config.dataset_size == "medium":
            n_samples = min(self.config.n_samples, 10000)
        elif self.config.dataset_size == "full":
            n_samples = self.config.n_samples if self.config.n_samples else None
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
        
        logger.info(f"✅ Dataset prepared: {len(eval_data)} samples")
        logger.info(f"   Countries: {eval_data['country'].nunique()}")
        logger.info(f"   Topics: {eval_data['topic'].nunique()}")
        
        self.progress['total_samples'] = len(eval_data)
        
        return eval_data
    
    def create_prompts(self, eval_data: pd.DataFrame) -> List[Dict]:
        """Create prompts for evaluation"""
        prompts = []
        
        for _, row in eval_data.iterrows():
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
    
    async def run_all_api_models(self, prompts: List[Dict]):
        """Run all API models concurrently with retry"""
        tasks = []
        for model in self.config.api_models:
            task = asyncio.create_task(
                self.run_api_model_with_retry(model, prompts)
            )
            tasks.append(task)
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    def run_all_local_models(self, prompts: List[Dict]):
        """Run all local models with thread pool"""
        with ThreadPoolExecutor(max_workers=self.config.parallel_local_models) as executor:
            futures = []
            for model in self.config.local_models:
                future = executor.submit(
                    self.run_local_model_with_retry, model, prompts
                )
                futures.append(future)
            
            # Wait for completion
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Local model thread error: {e}")
    
    def run_enhanced_parallel_execution(self):
        """Main execution with monitoring and recovery"""
        logger.info("="*70)
        logger.info("🚀 STARTING ENHANCED PARALLEL EXECUTION")
        logger.info("="*70)
        
        self.progress['start_time'] = datetime.now()
        
        # Start internet monitoring
        self.internet_monitor.start_monitoring()
        logger.info("🌐 Internet monitoring started")
        
        # Save session info
        self._save_session_info("running")
        
        # Prepare dataset
        eval_data = self.prepare_dataset()
        prompts = self.create_prompts(eval_data)
        
        # Save prompts
        with open(self.output_dir / "prompts.json", 'w') as f:
            json.dump(prompts, f, indent=2)
        
        # Print execution plan
        logger.info("\n📋 Execution Plan:")
        logger.info(f"   Samples: {len(prompts)}")
        logger.info(f"   API Models: {self.config.api_models}")
        logger.info(f"   Local Models: {self.config.local_models}")
        logger.info(f"   Max retries: {self.config.max_retries}")
        logger.info(f"   Auto-resume: {self.config.auto_resume}")
        
        # Start monitoring thread
        monitor_thread = threading.Thread(
            target=self._enhanced_monitor_progress, 
            daemon=True
        )
        monitor_thread.start()
        
        # Run API models in async event loop
        api_thread = None
        if self.config.api_models:
            logger.info(f"\n☁️  Starting {len(self.config.api_models)} API models...")
            api_thread = threading.Thread(
                target=lambda: asyncio.run(self.run_all_api_models(prompts))
            )
            api_thread.start()
        
        # Run local models in thread pool
        local_thread = None
        if self.config.local_models:
            logger.info(f"\n💻 Starting {len(self.config.local_models)} local models...")
            local_thread = threading.Thread(
                target=lambda: self.run_all_local_models(prompts)
            )
            local_thread.start()
        
        # Wait for completion
        if api_thread:
            api_thread.join()
        if local_thread:
            local_thread.join()
        
        # Stop internet monitoring
        self.internet_monitor.stop()
        
        # Calculate final metrics
        self._calculate_metrics()
        
        # Generate summary report
        self._generate_enhanced_summary()
        
        # Update session status
        self._save_session_info("completed")
        
        logger.info("="*70)
        logger.info("✅ ENHANCED PARALLEL EXECUTION COMPLETE")
        logger.info("="*70)
    
    def _save_session_info(self, status: str):
        """Save session information"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        cursor.execute("""
            INSERT OR REPLACE INTO session_info 
            (session_id, start_time, last_update, config, status)
            VALUES (?, ?, ?, ?, ?)
        """, (
            session_id,
            self.progress.get('start_time', datetime.now()),
            datetime.now(),
            json.dumps(asdict(self.config)),
            status
        ))
        
        conn.commit()
        conn.close()
    
    def _enhanced_monitor_progress(self):
        """Enhanced progress monitoring with detailed stats"""
        while True:
            time.sleep(10)
            
            # Calculate overall progress
            total_completed = sum(self.progress['completed'].values())
            total_failed = sum(self.progress['failed'].values())
            total_retries = sum(self.progress['retries'].values())
            total_models = len(self.config.api_models) + len(self.config.local_models)
            
            if total_models == 0:
                continue
            
            expected_total = self.progress['total_samples'] * total_models
            if expected_total == 0:
                continue
            
            progress_pct = (total_completed + total_failed) / expected_total * 100
            
            # Calculate ETA
            elapsed = (datetime.now() - self.progress['start_time']).total_seconds()
            if total_completed > 0:
                rate = total_completed / elapsed
                remaining = expected_total - (total_completed + total_failed)
                eta_seconds = remaining / rate if rate > 0 else 0
                eta_str = f"{eta_seconds/3600:.1f}h"
            else:
                eta_str = "calculating..."
            
            # Clear screen and display progress
            os.system('clear' if os.name == 'posix' else 'cls')
            
            print("="*70)
            print("📊 EVALUATION PROGRESS MONITOR")
            print("="*70)
            print(f"Overall: {progress_pct:.1f}% | ETA: {eta_str}")
            print(f"Completed: {total_completed:,} | Failed: {total_failed:,} | Retries: {total_retries:,}")
            print(f"Internet: {'✅ Connected' if self.internet_monitor.is_connected else '❌ Disconnected'}")
            
            # Memory and system stats
            mem = psutil.virtual_memory()
            cpu = psutil.cpu_percent(interval=1)
            print(f"Memory: {mem.percent:.1f}% | CPU: {cpu:.1f}%")
            
            print("\n" + "-"*70)
            print("MODEL STATUS:")
            print("-"*70)
            
            # API models
            if self.config.api_models:
                print("\n☁️  API Models:")
                for model in self.config.api_models:
                    completed = self.progress['completed'].get(model, 0)
                    failed = self.progress['failed'].get(model, 0)
                    cost = self.progress['costs'].get(model, 0)
                    pct = (completed + failed) / self.progress['total_samples'] * 100 if self.progress['total_samples'] > 0 else 0
                    
                    status = "✅" if completed == self.progress['total_samples'] else "🔄"
                    print(f"  {status} {model}: {completed}/{self.progress['total_samples']} ({pct:.1f}%) | ${cost:.2f}")
            
            # Local models
            if self.config.local_models:
                print("\n💻 Local Models:")
                for model in self.config.local_models:
                    completed = self.progress['completed'].get(model, 0)
                    failed = self.progress['failed'].get(model, 0)
                    pct = (completed + failed) / self.progress['total_samples'] * 100 if self.progress['total_samples'] > 0 else 0
                    
                    status = "✅" if completed == self.progress['total_samples'] else "🔄"
                    print(f"  {status} {model}: {completed}/{self.progress['total_samples']} ({pct:.1f}%)")
            
            # Recent errors
            if self.error_log:
                print("\n⚠️  Recent Errors:")
                for error in self.error_log[-3:]:
                    print(f"  • {error['model']}: {error['error_type']} - {error['error_message'][:50]}...")
            
            print("\n" + "="*70)
            print("Press Ctrl+C to save progress and exit gracefully")
            
            # Check if complete
            if total_completed + total_failed >= expected_total:
                break
    
    def _calculate_metrics(self):
        """Calculate evaluation metrics"""
        conn = sqlite3.connect(self.db_path)
        
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
                total_cost = df['cost'].sum()
                correlation = 0.0  # TODO: Calculate with WVS
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
    
    def _generate_enhanced_summary(self):
        """Generate enhanced summary report"""
        conn = sqlite3.connect(self.db_path)
        
        # Get metrics
        metrics_df = pd.read_sql_query("SELECT * FROM evaluation_metrics", conn)
        
        # Get errors
        errors_df = pd.read_sql_query("SELECT * FROM error_log", conn)
        
        # Generate report
        elapsed = datetime.now() - self.progress['start_time']
        
        report = {
            'execution_summary': {
                'start_time': self.progress['start_time'].isoformat(),
                'end_time': datetime.now().isoformat(),
                'total_duration': str(elapsed),
                'total_samples': self.progress['total_samples'],
                'internet_disconnections': len([e for e in self.error_log if 'Network' in e.get('error_type', '')])
            },
            'models_evaluated': {
                'api': list(self.config.api_models),
                'local': list(self.config.local_models)
            },
            'metrics': metrics_df.to_dict('records'),
            'error_summary': {
                'total_errors': len(errors_df),
                'by_type': errors_df['error_type'].value_counts().to_dict() if not errors_df.empty else {},
                'total_retries': sum(self.progress['retries'].values())
            },
            'costs': {
                'total_api_cost': metrics_df[metrics_df['model_type'] == 'api']['total_cost'].sum(),
                'by_model': dict(self.progress['costs'])
            }
        }
        
        # Save report
        report_file = self.output_dir / f"enhanced_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save metrics CSV
        metrics_df.to_csv(self.output_dir / "model_metrics.csv", index=False)
        
        # Print summary
        print("\n" + "="*70)
        print("📊 EXECUTION SUMMARY")
        print("="*70)
        print(f"Duration: {elapsed}")
        print(f"Total samples: {report['execution_summary']['total_samples']:,}")
        print(f"Models evaluated: {len(metrics_df)}")
        print(f"Total API cost: ${report['costs']['total_api_cost']:.2f}")
        print(f"Total errors: {report['error_summary']['total_errors']}")
        print(f"Total retries: {report['error_summary']['total_retries']}")
        
        if not metrics_df.empty:
            print("\n🏆 Top Models by Average Score:")
            top_models = metrics_df.nlargest(5, 'avg_score')[['model', 'avg_score', 'model_type']]
            for _, row in top_models.iterrows():
                print(f"  {row['model']}: {row['avg_score']:.3f} ({row['model_type']})")
        
        print(f"\n📁 Full report saved to: {report_file}")
        
        conn.close()


def main():
    """Main entry point"""
    # Configuration
    config = EnhancedExecutionConfig(
        dataset_size="sample",
        n_samples=100,
        api_models=["gpt-3.5-turbo", "gpt-4o-mini"],
        local_models=["mistral:7b", "neural-chat:latest"],
        parallel_api_requests=5,
        parallel_local_models=2,
        max_memory_gb=50.0,
        max_api_cost=10.0,
        max_retries=5,
        auto_resume=True
    )
    
    # Run enhanced execution
    executor = EnhancedParallelExecutor(config)
    executor.run_enhanced_parallel_execution()


if __name__ == "__main__":
    main()