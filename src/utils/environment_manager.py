#!/usr/bin/env python3
"""
Environment Manager for Flexible Local/Server Execution
Handles automatic routing between M4 Max (local) and SURF 4xA100 (server)
"""

import os
import sys
import yaml
import torch
import platform
import subprocess
import socket
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
import psutil
from datetime import datetime

logger = logging.getLogger(__name__)

class EnvironmentManager:
    """Manages execution environment and model routing"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize environment manager with configuration"""
        self.config = self._load_config(config_path)
        self.environment = self._detect_environment()
        self.setup_paths()
        self.setup_cache()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        config_file = Path(config_path)
        if not config_file.exists():
            # Try relative to script location
            config_file = Path(__file__).parent / config_path
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)
        else:
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return self._get_default_config()
    
    def _detect_environment(self) -> str:
        """Detect whether running on local M4 Max or SURF server"""
        
        # Check for explicit environment variable
        if os.getenv('MORAL_ALIGNMENT_ENV'):
            return os.getenv('MORAL_ALIGNMENT_ENV')
        
        # Auto-detect based on various indicators
        hostname = socket.gethostname().lower()
        
        # Check for SURF/SLURM environment
        if 'SLURM_JOB_ID' in os.environ or 'surf' in hostname:
            logger.info("Detected SURF server environment")
            return 'server'
        
        # Check for macOS (M4 Max)
        if platform.system() == 'Darwin':
            # Check if it's Apple Silicon
            if platform.processor() == 'arm' or 'apple' in platform.processor().lower():
                logger.info("Detected Apple Silicon Mac (M4 Max)")
                return 'local'
        
        # Check for specific storage path that indicates SURF
        if Path('/data/storage_4_tb').exists():
            logger.info("Detected SURF storage path")
            return 'server'
        
        # Default to local
        logger.info("Defaulting to local environment")
        return 'local'
    
    def setup_paths(self):
        """Setup paths based on environment"""
        env_config = self.config['storage'][self.environment]
        
        # Expand paths and create directories
        self.base_path = Path(os.path.expanduser(env_config['base_path']))
        self.model_cache = Path(os.path.expanduser(env_config['model_cache']))
        self.outputs_path = Path(os.path.expanduser(env_config['outputs']))
        self.data_path = Path(os.path.expanduser(env_config['data']))
        self.temp_path = Path(os.path.expanduser(env_config['temp']))
        
        # Create directories if they don't exist
        for path in [self.base_path, self.model_cache, self.outputs_path, 
                     self.data_path, self.temp_path]:
            path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Base path: {self.base_path}")
        logger.info(f"Model cache: {self.model_cache}")
        logger.info(f"Outputs: {self.outputs_path}")
    
    def setup_cache(self):
        """Setup model cache directories for HuggingFace"""
        # Set environment variables for HuggingFace
        os.environ['TRANSFORMERS_CACHE'] = str(self.model_cache)
        os.environ['HF_HOME'] = str(self.model_cache)
        os.environ['TORCH_HOME'] = str(self.model_cache / 'torch')
        os.environ['HF_DATASETS_CACHE'] = str(self.model_cache / 'datasets')
        
        # Create cache subdirectories
        (self.model_cache / 'torch').mkdir(exist_ok=True)
        (self.model_cache / 'datasets').mkdir(exist_ok=True)
        
        logger.info(f"Set TRANSFORMERS_CACHE to: {self.model_cache}")
    
    def get_device(self) -> str:
        """Get appropriate device for current environment"""
        device_override = self.config['execution'].get('device_override')
        
        if device_override:
            return device_override
        
        if self.environment == 'local':
            # M4 Max - use Metal Performance Shaders
            if torch.backends.mps.is_available() and self.config['resources']['local'].get('use_mps', True):
                return 'mps'
            else:
                return 'cpu'
        else:
            # SURF server - use CUDA
            if torch.cuda.is_available():
                return 'cuda'
            else:
                logger.warning("CUDA not available on server, falling back to CPU")
                return 'cpu'
    
    def should_run_locally(self, model_name: str, model_size_gb: Optional[float] = None) -> bool:
        """Determine if a model should run locally or on server"""
        
        # Check execution mode
        exec_mode = self.config['execution']['mode']
        
        if exec_mode == 'local':
            return True
        elif exec_mode == 'server':
            return False
        elif exec_mode == 'hybrid':
            # Check model routing configuration
            local_models = self.config['model_routing']['local_models']
            server_models = self.config['model_routing']['server_models']
            
            # Explicit routing
            if model_name in local_models:
                return True
            if model_name in server_models:
                return False
            
            # Auto-routing based on size
            if self.config['model_routing']['auto_routing']['enabled'] and model_size_gb:
                threshold = self.config['model_routing']['auto_routing']['size_threshold_gb']
                return model_size_gb < threshold
            
            # Check memory usage
            if self.config['model_routing']['auto_routing']['enabled']:
                memory_percent = psutil.virtual_memory().percent
                memory_threshold = self.config['model_routing']['auto_routing']['memory_threshold_percent']
                if memory_percent > memory_threshold:
                    logger.warning(f"Memory usage {memory_percent}% > {memory_threshold}%, routing to server")
                    return False
        
        # Default based on environment
        return self.environment == 'local'
    
    def get_model_config(self, model_name: str) -> Dict:
        """Get configuration for a specific model"""
        # Determine where to run
        run_locally = self.should_run_locally(model_name)
        
        # Get resource config
        if run_locally:
            resources = self.config['resources']['local']
            device = 'mps' if self.environment == 'local' else 'cuda'
        else:
            resources = self.config['resources']['server']
            device = 'cuda'
        
        return {
            'model_name': model_name,
            'device': device,
            'batch_size': resources['batch_size'],
            'cache_dir': str(self.model_cache),
            'output_dir': str(self.outputs_path),
            'run_locally': run_locally,
            'enable_quantization': resources.get('enable_quantization', False),
            'max_memory': resources.get('max_memory_gb', 32) * 1024**3  # Convert to bytes
        }
    
    def sync_outputs(self, direction: str = 'both'):
        """Sync outputs between local and server"""
        if not self.config['sync']['rsync']['enabled']:
            return
        
        rsync_config = self.config['sync']['rsync']
        
        if direction in ['to_server', 'both']:
            # Sync local outputs to server
            local_outputs = self.config['storage']['local']['outputs']
            remote_outputs = f"{rsync_config['remote_user']}@{rsync_config['remote_host']}:{rsync_config['remote_path']}/outputs/"
            
            cmd = [
                'rsync', '-avz', '--progress',
                '-e', f"ssh -i {rsync_config['ssh_key']}",
                local_outputs + '/',
                remote_outputs
            ]
            
            logger.info(f"Syncing outputs to server: {' '.join(cmd)}")
            subprocess.run(cmd)
        
        if direction in ['from_server', 'both']:
            # Sync server outputs to local
            remote_outputs = f"{rsync_config['remote_user']}@{rsync_config['remote_host']}:{rsync_config['remote_path']}/outputs/"
            local_outputs = self.config['storage']['local']['outputs']
            
            cmd = [
                'rsync', '-avz', '--progress',
                '-e', f"ssh -i {rsync_config['ssh_key']}",
                remote_outputs,
                local_outputs + '/'
            ]
            
            logger.info(f"Syncing outputs from server: {' '.join(cmd)}")
            subprocess.run(cmd)
    
    def get_execution_command(self, model_name: str, profile: str = 'default') -> List[str]:
        """Get command to execute model based on environment"""
        
        if self.should_run_locally(model_name):
            # Run locally
            return [
                sys.executable,
                'run_experiments.py',
                '--models', model_name,
                '--profile', profile,
                '--output-dir', str(self.outputs_path)
            ]
        else:
            # Submit to SURF queue
            job_script = self.create_surf_job(model_name, profile)
            return ['sbatch', str(job_script)]
    
    def create_surf_job(self, model_name: str, profile: str) -> Path:
        """Create SURF job script for specific model"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        job_file = self.temp_path / f"job_{model_name}_{timestamp}.sh"
        
        # Determine resources based on model
        if model_name in ['llama-3.3-70b-instruct', 'llama-3.2-90b-instruct', 'bloom-176b']:
            gpus = 4  # Use all 4 A100s
            memory = "320G"
            time = "12:00:00"
        elif model_name in ['gemma-2-27b-it', 'mistral-large-2', 'qwen-72b']:
            gpus = 2
            memory = "160G"
            time = "08:00:00"
        else:
            gpus = 1
            memory = "80G"
            time = "04:00:00"
        
        job_content = f"""#!/bin/bash
#SBATCH --job-name=moral_{model_name}
#SBATCH --time={time}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem={memory}
#SBATCH --partition=gpu
#SBATCH --gres=gpu:{gpus}
#SBATCH --output={self.outputs_path}/logs/job_%j.out
#SBATCH --error={self.outputs_path}/logs/job_%j.err

# Setup environment
module load Python/3.10
module load CUDA/11.8

cd {self.base_path}
source venv/bin/activate

# Set cache paths
export TRANSFORMERS_CACHE={self.model_cache}
export HF_HOME={self.model_cache}
export TORCH_HOME={self.model_cache}/torch

# Run experiment
python run_experiments.py \\
    --models {model_name} \\
    --profile {profile} \\
    --output-dir {self.outputs_path}/run_$SLURM_JOB_ID

echo "Job completed for {model_name}"
"""
        
        with open(job_file, 'w') as f:
            f.write(job_content)
        
        return job_file
    
    def _get_default_config(self) -> Dict:
        """Get default configuration if config.yaml is missing"""
        return {
            'execution': {'mode': 'hybrid', 'auto_detect': True},
            'storage': {
                'local': {
                    'base_path': '~/Documents/Project06',
                    'model_cache': '~/.cache/huggingface',
                    'outputs': '~/Documents/Project06/outputs',
                    'data': '~/Documents/Project06/sample_data',
                    'temp': '~/Documents/Project06/temp'
                },
                'server': {
                    'base_path': '/data/storage_4_tb/moral-alignment-pipeline',
                    'model_cache': '/data/storage_4_tb/moral-alignment-pipeline/models',
                    'outputs': '/data/storage_4_tb/moral-alignment-pipeline/outputs',
                    'data': '/data/storage_4_tb/moral-alignment-pipeline/data',
                    'temp': '/data/storage_4_tb/moral-alignment-pipeline/temp'
                }
            },
            'model_routing': {
                'local_models': ['gpt2', 'gpt-4o', 'claude-3.5-sonnet'],
                'server_models': ['llama-3.3-70b-instruct'],
                'auto_routing': {'enabled': True, 'size_threshold_gb': 8}
            },
            'resources': {
                'local': {'max_memory_gb': 50, 'use_mps': True, 'batch_size': 4},
                'server': {'max_memory_per_gpu_gb': 40, 'num_gpus': 4, 'batch_size': 16}
            }
        }
    
    def print_status(self):
        """Print current environment status"""
        print("="*60)
        print("Environment Manager Status")
        print("="*60)
        print(f"Environment: {self.environment.upper()}")
        print(f"Device: {self.get_device()}")
        print(f"Base Path: {self.base_path}")
        print(f"Model Cache: {self.model_cache}")
        print(f"Outputs: {self.outputs_path}")
        print(f"Execution Mode: {self.config['execution']['mode']}")
        
        if self.environment == 'local':
            print(f"System: {platform.system()} {platform.machine()}")
            print(f"Memory: {psutil.virtual_memory().total / 1e9:.1f} GB")
        else:
            if torch.cuda.is_available():
                print(f"GPUs: {torch.cuda.device_count()} x {torch.cuda.get_device_name(0)}")
        
        print("="*60)


# Singleton instance
_env_manager = None

def get_environment_manager() -> EnvironmentManager:
    """Get or create singleton environment manager"""
    global _env_manager
    if _env_manager is None:
        _env_manager = EnvironmentManager()
    return _env_manager


if __name__ == "__main__":
    # Test environment detection and setup
    manager = get_environment_manager()
    manager.print_status()
    
    # Test model routing
    test_models = ['gpt2', 'llama-3.3-70b-instruct', 'gpt-4o']
    print("\nModel Routing Tests:")
    for model in test_models:
        run_local = manager.should_run_locally(model)
        print(f"{model}: {'LOCAL' if run_local else 'SERVER'}")
    
    # Test configuration
    print("\nModel Configuration for 'gpt2':")
    config = manager.get_model_config('gpt2')
    for key, value in config.items():
        print(f"  {key}: {value}")