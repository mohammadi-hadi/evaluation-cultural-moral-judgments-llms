#!/usr/bin/env python3
"""
Unified Runner for Moral Alignment Pipeline
Automatically routes models between local (M4 Max) and server (SURF 4xA100)
"""

import argparse
import asyncio
import sys
import os
from pathlib import Path
from typing import List, Dict, Optional
import yaml
import logging
from datetime import datetime

# Add environment manager
from environment_manager import get_environment_manager
from data_storage import DataStorageManager
from prompts_manager import PromptsManager
from cross_evaluation import CrossEvaluator
from conflict_resolver import ConflictResolver

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class UnifiedPipeline:
    """Unified pipeline that handles both local and server execution"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize unified pipeline"""
        self.env_manager = get_environment_manager()
        self.env_manager.print_status()
        
        # Initialize storage with proper paths
        self.storage = DataStorageManager(
            base_dir=self.env_manager.outputs_path,
            compress=True
        )
        
        # Initialize other components
        self.prompts_manager = PromptsManager(
            output_dir=self.env_manager.outputs_path / "prompts"
        )
        
        # Load models configuration
        with open('models_config.yaml', 'r') as f:
            self.models_config = yaml.safe_load(f)
        
        self.cross_evaluator = CrossEvaluator(
            models_config=self.models_config,
            output_dir=self.env_manager.outputs_path / "cross_evaluation"
        )
        
        self.conflict_resolver = ConflictResolver()
        
        # Track which models run where
        self.model_locations = {}
    
    def determine_model_execution(self, models: List[str]) -> Dict[str, List[str]]:
        """Determine which models run locally vs on server"""
        local_models = []
        server_models = []
        
        for model in models:
            if self.env_manager.should_run_locally(model):
                local_models.append(model)
                self.model_locations[model] = 'local'
            else:
                server_models.append(model)
                self.model_locations[model] = 'server'
        
        return {
            'local': local_models,
            'server': server_models
        }
    
    def run_local_models(self, models: List[str], sample_size: int = None):
        """Run models locally on M4 Max"""
        if not models:
            return {}
        
        logger.info(f"Running {len(models)} models locally on M4 Max")
        logger.info(f"Models: {models}")
        
        results = {}
        
        for model in models:
            logger.info(f"\n{'='*60}")
            logger.info(f"Evaluating: {model} (LOCAL)")
            logger.info(f"{'='*60}")
            
            # Get model configuration
            model_config = self.env_manager.get_model_config(model)
            
            # Check if it's an API model
            is_api_model = model in [m['name'] for m in self.models_config.get('api_models', [])]
            
            if is_api_model:
                logger.info(f"Using API for {model}")
                # API models can run from anywhere
                results[model] = self._run_api_model(model, model_config, sample_size)
            else:
                logger.info(f"Loading local model {model}")
                # Local model inference
                results[model] = self._run_local_model(model, model_config, sample_size)
        
        return results
    
    def submit_server_jobs(self, models: List[str], sample_size: int = None):
        """Submit jobs to SURF server for large models"""
        if not models:
            return []
        
        logger.info(f"Submitting {len(models)} models to SURF server")
        logger.info(f"Models: {models}")
        
        job_ids = []
        
        for model in models:
            logger.info(f"\nSubmitting job for: {model}")
            
            # Create job script
            job_script = self.env_manager.create_surf_job(model, 'default')
            
            # Submit job (or simulate if not on SURF)
            if self.env_manager.environment == 'server':
                import subprocess
                result = subprocess.run(['sbatch', str(job_script)], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    job_id = result.stdout.strip().split()[-1]
                    job_ids.append(job_id)
                    logger.info(f"Submitted job {job_id} for {model}")
                else:
                    logger.error(f"Failed to submit job for {model}: {result.stderr}")
            else:
                logger.info(f"Would submit job: {job_script}")
                logger.info("(Not on SURF, skipping actual submission)")
                job_ids.append(f"simulated_{model}")
        
        return job_ids
    
    def _run_api_model(self, model_name: str, config: Dict, sample_size: int = None):
        """Run API-based model (OpenAI, Anthropic, etc.)"""
        # This is a placeholder - integrate with your actual API calling code
        logger.info(f"Running API model: {model_name}")
        
        # Check for API key
        api_provider = None
        for provider in ['openai', 'anthropic', 'google', 'cohere', 'mistral']:
            if provider in model_name.lower() or \
               (provider == 'openai' and 'gpt' in model_name.lower()) or \
               (provider == 'anthropic' and 'claude' in model_name.lower()) or \
               (provider == 'google' and 'gemini' in model_name.lower()):
                api_provider = provider
                break
        
        if api_provider:
            key_name = f"{api_provider.upper()}_API_KEY"
            if not os.getenv(key_name):
                logger.warning(f"No API key found for {api_provider} ({key_name})")
                return None
        
        # Placeholder for actual API call
        return {
            'model': model_name,
            'status': 'completed',
            'location': 'local_api',
            'sample_size': sample_size
        }
    
    def _run_local_model(self, model_name: str, config: Dict, sample_size: int = None):
        """Run local model on M4 Max"""
        logger.info(f"Running local model: {model_name}")
        
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            # Use MPS for M4 Max
            device = config['device']
            logger.info(f"Using device: {device}")
            
            # Load model with proper cache directory
            logger.info(f"Loading from cache: {config['cache_dir']}")
            
            tokenizer = AutoTokenizer.from_pretrained(
                config['model_name'],
                cache_dir=config['cache_dir']
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                config['model_name'],
                cache_dir=config['cache_dir'],
                torch_dtype=torch.float16 if device != 'cpu' else torch.float32,
                device_map='auto' if device == 'cuda' else None
            )
            
            if device == 'mps':
                model = model.to('mps')
            
            logger.info(f"Model loaded successfully on {device}")
            
            # Placeholder for actual evaluation
            return {
                'model': model_name,
                'status': 'completed',
                'location': 'local',
                'device': device,
                'sample_size': sample_size
            }
            
        except Exception as e:
            logger.error(f"Error running local model {model_name}: {e}")
            return {
                'model': model_name,
                'status': 'failed',
                'error': str(e)
            }
    
    def sync_results(self):
        """Sync results between local and server"""
        logger.info("Syncing results between local and server...")
        
        if self.env_manager.config['sync']['auto_sync']:
            self.env_manager.sync_outputs('both')
            logger.info("Sync completed")
        else:
            logger.info("Auto-sync disabled in configuration")
    
    def run_pipeline(self, models: List[str], sample_size: int = None,
                    enable_cross_eval: bool = True,
                    enable_conflict_resolution: bool = True):
        """Run complete pipeline with automatic routing"""
        
        logger.info("\n" + "="*60)
        logger.info("Starting Unified Pipeline")
        logger.info("="*60)
        
        # Start experiment run
        run_id = self.storage.start_experiment_run(
            models=models,
            config={
                'environment': self.env_manager.environment,
                'execution_mode': self.env_manager.config['execution']['mode'],
                'sample_size': sample_size,
                'enable_cross_eval': enable_cross_eval,
                'enable_conflict_resolution': enable_conflict_resolution,
                'model_locations': self.model_locations
            }
        )
        
        logger.info(f"Experiment ID: {run_id}")
        
        # Determine execution locations
        execution_plan = self.determine_model_execution(models)
        
        logger.info("\n" + "="*60)
        logger.info("Execution Plan")
        logger.info("="*60)
        logger.info(f"Local models ({len(execution_plan['local'])}): {execution_plan['local']}")
        logger.info(f"Server models ({len(execution_plan['server'])}): {execution_plan['server']}")
        
        # Run local models
        local_results = {}
        if execution_plan['local']:
            logger.info("\n" + "="*60)
            logger.info("Running Local Models")
            logger.info("="*60)
            local_results = self.run_local_models(execution_plan['local'], sample_size)
        
        # Submit server jobs
        server_jobs = []
        if execution_plan['server']:
            logger.info("\n" + "="*60)
            logger.info("Submitting Server Jobs")
            logger.info("="*60)
            server_jobs = self.submit_server_jobs(execution_plan['server'], sample_size)
            
            if server_jobs:
                logger.info(f"Submitted {len(server_jobs)} jobs to server")
                logger.info("Run 'squeue -u $USER' on SURF to check status")
        
        # Save execution plan
        plan_file = self.env_manager.outputs_path / f"execution_plan_{run_id}.yaml"
        with open(plan_file, 'w') as f:
            yaml.dump({
                'run_id': run_id,
                'timestamp': datetime.now().isoformat(),
                'environment': self.env_manager.environment,
                'execution_plan': execution_plan,
                'local_results': local_results,
                'server_jobs': server_jobs,
                'model_locations': self.model_locations
            }, f)
        
        logger.info(f"\nExecution plan saved to: {plan_file}")
        
        # Complete run
        self.storage.complete_experiment_run(run_id)
        
        # Sync results if configured
        if self.env_manager.config['sync']['auto_sync']:
            self.sync_results()
        
        return {
            'run_id': run_id,
            'local_results': local_results,
            'server_jobs': server_jobs,
            'execution_plan': execution_plan
        }


def main():
    parser = argparse.ArgumentParser(
        description="Unified runner for local (M4 Max) and server (SURF) execution"
    )
    
    parser.add_argument(
        '--models',
        nargs='+',
        help='List of models to evaluate'
    )
    
    parser.add_argument(
        '--profile',
        choices=['local_only', 'server_only', 'hybrid_optimal', 'dev_test', 'full_evaluation'],
        default='hybrid_optimal',
        help='Execution profile'
    )
    
    parser.add_argument(
        '--sample-size',
        type=int,
        help='Number of samples to evaluate'
    )
    
    parser.add_argument(
        '--force-local',
        action='store_true',
        help='Force all models to run locally'
    )
    
    parser.add_argument(
        '--force-server',
        action='store_true',
        help='Force all models to run on server'
    )
    
    parser.add_argument(
        '--sync',
        action='store_true',
        help='Sync results between local and server'
    )
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run test mode'
    )
    
    args = parser.parse_args()
    
    # Override execution mode if forced
    if args.force_local:
        os.environ['MORAL_ALIGNMENT_ENV'] = 'local'
    elif args.force_server:
        os.environ['MORAL_ALIGNMENT_ENV'] = 'server'
    
    # Initialize pipeline
    pipeline = UnifiedPipeline()
    
    # Handle sync-only mode
    if args.sync:
        pipeline.sync_results()
        return 0
    
    # Handle test mode
    if args.test:
        test_models = ['gpt2', 'gpt-4o-mini']
        logger.info("Running test mode with minimal models")
        results = pipeline.run_pipeline(
            models=test_models,
            sample_size=5
        )
        logger.info(f"Test completed: {results['run_id']}")
        return 0
    
    # Load profile if specified
    if args.profile and not args.models:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        profile_config = config['deployment_profiles'].get(args.profile, {})
        
        if args.profile in ['local_only', 'hybrid_optimal']:
            models = profile_config.get('local_models', [])
            if args.profile == 'hybrid_optimal':
                models.extend(profile_config.get('server_models', []))
        elif args.profile == 'server_only':
            models = profile_config.get('server_models', [])
        else:
            models = profile_config.get('models', [])
        
        sample_size = args.sample_size or profile_config.get('sample_size', 20)
    else:
        models = args.models or ['gpt2', 'gpt-4o-mini']
        sample_size = args.sample_size or 20
    
    # Run pipeline
    logger.info(f"Running pipeline with {len(models)} models")
    results = pipeline.run_pipeline(
        models=models,
        sample_size=sample_size
    )
    
    logger.info("\n" + "="*60)
    logger.info("Pipeline Complete")
    logger.info("="*60)
    logger.info(f"Run ID: {results['run_id']}")
    logger.info(f"Local models completed: {len(results['local_results'])}")
    logger.info(f"Server jobs submitted: {len(results['server_jobs'])}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())