#!/usr/bin/env python3
"""
Comprehensive Moral Alignment Evaluation
Master script for running full-scale evaluation with all models
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from collections import defaultdict
import yaml

from parallel_executor import ParallelExecutor, ExecutionConfig
from local_model_runner import LocalModelRunner
from api_model_runner import APIModelRunner

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Model configurations from your list
MODEL_CONFIGS = {
    "api_models": {
        "immediate": [
            "gpt-3.5-turbo",      # Good baseline
            "gpt-4o-mini",        # Cost-effective
            "gpt-4o",             # Best overall
            # "o3-mini"           # If available
        ]
    },
    
    "local_models": {
        "small": [  # < 4GB - Can run multiple simultaneously
            "gpt2",                    # openai-community/gpt2
            "opt-125m",                # facebook/opt-125m
            "opt-350m",                # facebook/opt-350m
            "bloomz-560m",             # bigscience/bloomz-560m
            "llama3.2:1b",            # Via Ollama (if available)
            "gemma:2b",               # Via Ollama (if available)
            "qwen2.5:1.5b",           # Via Ollama (if available)
        ],
        
        "medium": [  # 7-14GB - Run one at a time
            "mistral:7b",             # Via Ollama (you have this)
            "neural-chat:latest",     # Via Ollama (you have this)
            "wizardlm2:7b",          # Via Ollama (you have this)
            "mistral-nemo:latest",    # Via Ollama (you have this)
            "deepseek-r1:latest",     # Via Ollama (you have this)
            # Add more as needed
        ],
        
        "quantized_large": [  # 4-bit quantized versions
            # "gpt-oss:20b-q4",       # If available via Ollama
            # "qwen2.5:14b-q4",       # If available via Ollama
            # "phi:4-q4",             # If available via Ollama
        ]
    },
    
    "server_models": {  # For 4xA100 GPU server
        "32b": [
            "Qwen/Qwen2.5-32B-Instruct",
            "Qwen/QwQ-32B-Preview",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        ],
        "70b": [
            "meta-llama/Llama-3.3-70B-Instruct",
            "meta-llama/Llama-3.1-70B-Instruct",
            "Qwen/Qwen2.5-72B-Instruct",
        ],
        "massive": [
            "openai/gpt-oss-120b",
            "mistralai/Mixtral-8x22B-Instruct-v0.1",
            "meta-llama/Llama-3.1-405B-Instruct",
        ]
    }
}

class ComprehensiveEvaluator:
    """Master evaluator for comprehensive moral alignment testing"""
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize evaluator
        
        Args:
            config_file: Path to YAML configuration file
        """
        self.config = self._load_config(config_file)
        self.output_dir = Path(self.config.get('output_dir', 'outputs/comprehensive'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Track execution phases
        self.phases_completed = []
        self.current_phase = None
        
        logger.info("ComprehensiveEvaluator initialized")
        logger.info(f"Output directory: {self.output_dir}")
    
    def _load_config(self, config_file: Optional[str]) -> Dict:
        """Load configuration from file or use defaults"""
        if config_file and Path(config_file).exists():
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)
        
        # Default configuration
        return {
            'dataset': {
                'size': 'sample',  # 'sample', 'medium', 'full'
                'n_samples': 1000,
                'stratified': True
            },
            'execution': {
                'phases': ['api', 'local_small', 'local_medium'],
                'parallel_api': 5,
                'parallel_local': 2,
                'batch_size': 100,
                'checkpoint_interval': 100
            },
            'resources': {
                'max_memory_gb': 50.0,
                'max_api_cost': 100.0
            },
            'output_dir': 'outputs/comprehensive'
        }
    
    def run_phase1_api_models(self):
        """Phase 1: Run API models (immediate execution)"""
        logger.info("="*60)
        logger.info("PHASE 1: API MODELS")
        logger.info("="*60)
        
        self.current_phase = "api"
        
        # Get API models
        api_models = MODEL_CONFIGS['api_models']['immediate']
        
        # Check API key
        if not os.getenv('OPENAI_API_KEY'):
            logger.error("OpenAI API key not found. Skipping API models.")
            return
        
        # Create configuration
        config = ExecutionConfig(
            dataset_size=self.config['dataset']['size'],
            n_samples=self.config['dataset']['n_samples'],
            api_models=api_models,
            local_models=[],  # No local models in this phase
            parallel_api_requests=self.config['execution']['parallel_api'],
            max_api_cost=self.config['resources']['max_api_cost'],
            output_dir=self.output_dir / "phase1_api"
        )
        
        # Run execution
        executor = ParallelExecutor(config)
        executor.run_parallel_execution()
        
        self.phases_completed.append("api")
        logger.info("Phase 1 complete: API models evaluated")
    
    def run_phase2_local_small(self):
        """Phase 2: Run small local models (< 4GB)"""
        logger.info("="*60)
        logger.info("PHASE 2: SMALL LOCAL MODELS")
        logger.info("="*60)
        
        self.current_phase = "local_small"
        
        # Get small models that are available
        runner = LocalModelRunner()
        available = runner.list_available_models()
        
        # Filter to only available models
        small_models = []
        for model in MODEL_CONFIGS['local_models']['small']:
            if model in available['configured']:
                small_models.append(model)
            elif any(model in m for m in available['ollama']):
                small_models.append(model)
        
        if not small_models:
            logger.warning("No small local models available")
            return
        
        logger.info(f"Running {len(small_models)} small models: {small_models}")
        
        # Create configuration
        config = ExecutionConfig(
            dataset_size=self.config['dataset']['size'],
            n_samples=self.config['dataset']['n_samples'],
            api_models=[],
            local_models=small_models,
            parallel_local_models=3,  # Can run multiple small models
            max_memory_gb=30.0,  # Leave room for system
            output_dir=self.output_dir / "phase2_local_small"
        )
        
        # Run execution
        executor = ParallelExecutor(config)
        executor.run_parallel_execution()
        
        self.phases_completed.append("local_small")
        logger.info("Phase 2 complete: Small local models evaluated")
    
    def run_phase3_local_medium(self):
        """Phase 3: Run medium local models (7-14GB)"""
        logger.info("="*60)
        logger.info("PHASE 3: MEDIUM LOCAL MODELS")
        logger.info("="*60)
        
        self.current_phase = "local_medium"
        
        # Get medium models that are available
        runner = LocalModelRunner()
        available = runner.list_available_models()
        
        # Filter to only available models
        medium_models = []
        for model in MODEL_CONFIGS['local_models']['medium']:
            if model in available['ollama']:
                medium_models.append(model)
        
        if not medium_models:
            logger.warning("No medium local models available")
            return
        
        logger.info(f"Running {len(medium_models)} medium models: {medium_models}")
        
        # Create configuration
        config = ExecutionConfig(
            dataset_size=self.config['dataset']['size'],
            n_samples=self.config['dataset']['n_samples'],
            api_models=[],
            local_models=medium_models,
            parallel_local_models=1,  # Run one at a time due to memory
            max_memory_gb=50.0,
            output_dir=self.output_dir / "phase3_local_medium"
        )
        
        # Run execution
        executor = ParallelExecutor(config)
        executor.run_parallel_execution()
        
        self.phases_completed.append("local_medium")
        logger.info("Phase 3 complete: Medium local models evaluated")
    
    def generate_server_jobs(self):
        """Generate job scripts for server models (4xA100)"""
        logger.info("="*60)
        logger.info("GENERATING SERVER JOB SCRIPTS")
        logger.info("="*60)
        
        jobs_dir = self.output_dir / "server_jobs"
        jobs_dir.mkdir(exist_ok=True)
        
        # Generate job for 32B models
        self._generate_job_script(
            jobs_dir / "job_32b_models.sh",
            MODEL_CONFIGS['server_models']['32b'],
            gpus=2,
            memory="160G",
            time="12:00:00"
        )
        
        # Generate job for 70B models
        self._generate_job_script(
            jobs_dir / "job_70b_models.sh",
            MODEL_CONFIGS['server_models']['70b'],
            gpus=3,
            memory="240G",
            time="24:00:00"
        )
        
        # Generate job for massive models
        self._generate_job_script(
            jobs_dir / "job_massive_models.sh",
            MODEL_CONFIGS['server_models']['massive'],
            gpus=4,
            memory="320G",
            time="48:00:00"
        )
        
        logger.info(f"Server job scripts generated in {jobs_dir}")
        logger.info("To run on server:")
        logger.info("  1. Copy this entire project to server")
        logger.info("  2. Install requirements: pip install -r requirements.txt")
        logger.info("  3. Submit jobs: sbatch job_32b_models.sh")
    
    def _generate_job_script(self, output_file: Path, models: List[str], 
                            gpus: int, memory: str, time: str):
        """Generate SLURM job script for server execution"""
        script = f"""#!/bin/bash
#SBATCH --job-name=moral_alignment
#SBATCH --time={time}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem={memory}
#SBATCH --partition=gpu
#SBATCH --gres=gpu:{gpus}
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err

# Setup environment
module load Python/3.10
module load CUDA/11.8

# Activate virtual environment
source venv/bin/activate

# Set cache directories
export TRANSFORMERS_CACHE=/data/storage_4_tb/moral-alignment/cache
export HF_HOME=/data/storage_4_tb/moral-alignment/cache

# Run models
"""
        
        for model in models:
            script += f"""
echo "Running {model}"
python server_model_runner.py \\
    --model "{model}" \\
    --dataset-size medium \\
    --n-samples 10000 \\
    --output-dir outputs/server/{model.replace('/', '_')}

"""
        
        with open(output_file, 'w') as f:
            f.write(script)
        
        # Make executable
        output_file.chmod(0o755)
        logger.info(f"Generated: {output_file}")
    
    def run_comprehensive_evaluation(self):
        """Run complete evaluation pipeline"""
        logger.info("="*70)
        logger.info("COMPREHENSIVE MORAL ALIGNMENT EVALUATION")
        logger.info("="*70)
        
        start_time = datetime.now()
        
        # Phase 1: API Models
        if 'api' in self.config['execution']['phases']:
            try:
                self.run_phase1_api_models()
            except Exception as e:
                logger.error(f"Phase 1 failed: {e}")
        
        # Phase 2: Small Local Models
        if 'local_small' in self.config['execution']['phases']:
            try:
                self.run_phase2_local_small()
            except Exception as e:
                logger.error(f"Phase 2 failed: {e}")
        
        # Phase 3: Medium Local Models
        if 'local_medium' in self.config['execution']['phases']:
            try:
                self.run_phase3_local_medium()
            except Exception as e:
                logger.error(f"Phase 3 failed: {e}")
        
        # Generate server job scripts
        self.generate_server_jobs()
        
        # Generate final report
        self._generate_final_report(start_time)
        
        logger.info("="*70)
        logger.info("EVALUATION COMPLETE")
        logger.info("="*70)
    
    def _generate_final_report(self, start_time: datetime):
        """Generate final evaluation report"""
        elapsed = datetime.now() - start_time
        
        report = {
            'execution_summary': {
                'start_time': start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'total_duration': str(elapsed),
                'phases_completed': self.phases_completed
            },
            'configuration': self.config,
            'models_evaluated': {
                'api': MODEL_CONFIGS['api_models']['immediate'] if 'api' in self.phases_completed else [],
                'local_small': [],  # Fill from actual execution
                'local_medium': [],  # Fill from actual execution
                'server_pending': list(MODEL_CONFIGS['server_models'].keys())
            },
            'next_steps': [
                "1. Review results in outputs/comprehensive/",
                "2. Run visualization: python comprehensive_visualizer.py",
                "3. Deploy server jobs for large models",
                "4. Integrate all results: python result_integrator.py"
            ]
        }
        
        report_file = self.output_dir / f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Final report saved to: {report_file}")
        
        # Print summary
        print("\n" + "="*70)
        print("EVALUATION SUMMARY")
        print("="*70)
        print(f"Duration: {elapsed}")
        print(f"Phases completed: {', '.join(self.phases_completed)}")
        print(f"Output directory: {self.output_dir}")
        print("\nNext steps:")
        for step in report['next_steps']:
            print(f"  {step}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Run comprehensive moral alignment evaluation")
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--phases', nargs='+', 
                       choices=['api', 'local_small', 'local_medium', 'all'],
                       default=['all'],
                       help='Phases to run')
    parser.add_argument('--dataset-size', 
                       choices=['sample', 'medium', 'full'],
                       default='sample',
                       help='Dataset size')
    parser.add_argument('--n-samples', type=int, default=1000,
                       help='Number of samples (if not full)')
    parser.add_argument('--max-api-cost', type=float, default=100.0,
                       help='Maximum API cost in USD')
    parser.add_argument('--output-dir', type=str, 
                       default='outputs/comprehensive',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Create config from arguments
    if not args.config:
        config = {
            'dataset': {
                'size': args.dataset_size,
                'n_samples': args.n_samples,
                'stratified': True
            },
            'execution': {
                'phases': ['api', 'local_small', 'local_medium'] if 'all' in args.phases else args.phases,
                'parallel_api': 5,
                'parallel_local': 2,
                'batch_size': 100,
                'checkpoint_interval': 100
            },
            'resources': {
                'max_memory_gb': 50.0,
                'max_api_cost': args.max_api_cost
            },
            'output_dir': args.output_dir
        }
        
        # Save config
        config_file = Path(args.output_dir) / 'evaluation_config.yaml'
        config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(config_file, 'w') as f:
            yaml.dump(config, f)
        
        evaluator = ComprehensiveEvaluator()
        evaluator.config = config
        evaluator.output_dir = Path(args.output_dir)
    else:
        evaluator = ComprehensiveEvaluator(args.config)
    
    # Run evaluation
    evaluator.run_comprehensive_evaluation()


if __name__ == "__main__":
    main()