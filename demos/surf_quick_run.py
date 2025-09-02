#!/usr/bin/env python3
"""
Quick runner script optimized for SURF environment
Handles resource management and parallel processing
"""

import os
import sys
import argparse
import torch
import logging
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'surf_run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def check_surf_environment():
    """Check and report SURF environment settings"""
    logger.info("="*60)
    logger.info("SURF Environment Check")
    logger.info("="*60)
    
    # Check SLURM environment
    if 'SLURM_JOB_ID' in os.environ:
        logger.info(f"Running in SLURM job: {os.environ['SLURM_JOB_ID']}")
        logger.info(f"Node: {os.environ.get('SLURM_NODELIST', 'unknown')}")
        logger.info(f"CPUs: {os.environ.get('SLURM_CPUS_PER_TASK', 'unknown')}")
    else:
        logger.info("Not running in SLURM environment")
    
    # Check GPU availability
    if torch.cuda.is_available():
        logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        logger.info("No GPU available - using CPU")
    
    # Check memory
    import psutil
    mem = psutil.virtual_memory()
    logger.info(f"Available RAM: {mem.available / 1e9:.1f} GB / {mem.total / 1e9:.1f} GB")
    
    # Check cache directories
    cache_dir = Path.home() / '.cache' / 'huggingface'
    if cache_dir.exists():
        logger.info(f"HuggingFace cache: {cache_dir}")
    else:
        cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created HuggingFace cache: {cache_dir}")
    
    logger.info("="*60)

def run_minimal_test():
    """Run minimal test to verify setup"""
    logger.info("Running minimal test...")
    
    try:
        # Test imports
        from prompts_manager import PromptsManager
        from data_storage import DataStorageManager
        
        # Initialize components
        pm = PromptsManager()
        storage = DataStorageManager()
        
        # Create test prompts
        test_prompts = pm.create_logprob_prompts(
            country="Netherlands",  # Appropriate for SURF :)
            topic="euthanasia",
            model="gpt-4o"
        )
        
        logger.info(f"✅ Created {len(test_prompts)} test prompts")
        logger.info("✅ Minimal test passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="SURF-optimized runner for Moral Alignment Pipeline")
    parser.add_argument('--test', action='store_true', help='Run minimal test only')
    parser.add_argument('--profile', default='lightweight', help='Model profile to use')
    parser.add_argument('--sample-size', type=int, default=20, help='Number of samples')
    parser.add_argument('--use-gpu', action='store_true', help='Force GPU usage')
    parser.add_argument('--output-dir', default='outputs', help='Output directory')
    
    args = parser.parse_args()
    
    # Check environment
    check_surf_environment()
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Check API keys
    api_keys = {
        'OPENAI_API_KEY': 'OpenAI',
        'GEMINI_API_KEY': 'Google Gemini',
        'ANTHROPIC_API_KEY': 'Anthropic Claude',
        'COHERE_API_KEY': 'Cohere',
        'MISTRAL_API_KEY': 'Mistral'
    }
    
    logger.info("\nAPI Keys Status:")
    available_apis = []
    for key, name in api_keys.items():
        if os.getenv(key) and os.getenv(key) != 'your-key-here':
            logger.info(f"  ✅ {name} API key found")
            available_apis.append(name)
        else:
            logger.info(f"  ❌ {name} API key not set")
    
    if not available_apis:
        logger.warning("No API keys found! Only local models will work.")
    
    # Run test if requested
    if args.test:
        success = run_minimal_test()
        sys.exit(0 if success else 1)
    
    # Set device
    if args.use_gpu and torch.cuda.is_available():
        device = "cuda"
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        logger.info("Using CPU")
    
    # Run main pipeline
    logger.info(f"\nStarting pipeline with profile: {args.profile}")
    logger.info(f"Sample size: {args.sample_size}")
    logger.info(f"Output directory: {args.output_dir}")
    
    try:
        # Import and run the main pipeline
        if Path("run_experiments.py").exists():
            import subprocess
            cmd = [
                sys.executable, "run_experiments.py",
                "--profile", args.profile,
                "--sample-size", str(args.sample_size),
                "--output-dir", args.output_dir
            ]
            
            logger.info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("✅ Pipeline completed successfully!")
            else:
                logger.error(f"Pipeline failed with code {result.returncode}")
                logger.error(result.stderr)
        else:
            logger.error("run_experiments.py not found!")
            
    except Exception as e:
        logger.error(f"Error running pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()