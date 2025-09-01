#!/usr/bin/env python3
"""
Run Moral Alignment Experiments
Complete pipeline for evaluating LLMs on cross-cultural moral judgments

Usage:
    python run_experiments.py --models gpt2 opt-125m --sample-size 20
    python run_experiments.py --profile minimal
    python run_experiments.py --profile full --skip-api
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import yaml
import torch
import pandas as pd
import numpy as np

# Add project directory to path
sys.path.append(str(Path(__file__).parent))

def setup_environment():
    """Setup environment variables and check dependencies"""
    
    print("="*60)
    print("Moral Alignment Experiment Pipeline")
    print("="*60)
    
    # Check PyTorch and CUDA
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Check for API keys
    api_keys = {
        'OPENAI_API_KEY': 'OpenAI',
        'GEMINI_API_KEY': 'Google Gemini',
        'ANTHROPIC_API_KEY': 'Anthropic'
    }
    
    print("\nAPI Keys:")
    for key, name in api_keys.items():
        if os.getenv(key):
            print(f"  ✓ {name} API key found")
        else:
            print(f"  ✗ {name} API key not found (set {key} environment variable)")
    
    return True


def load_config(config_path: Path) -> dict:
    """Load model configuration from YAML file"""
    
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        return None
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_models_from_profile(config: dict, profile: str) -> List[str]:
    """Get list of models based on deployment profile"""
    
    if 'deployment_profiles' not in config:
        return []
    
    profiles = config['deployment_profiles']
    
    if profile not in profiles:
        print(f"Error: Profile '{profile}' not found in config")
        return []
    
    profile_config = profiles[profile]
    
    if profile_config['models'] == 'all':
        # Get all models from config
        models = []
        for category in ['gpt2_family', 'opt_family', 'multilingual', 'qwen_family', 
                        'instruction_tuned', 'falcon_family', 'other_models', 'api_models']:
            if category in config:
                models.extend([m['name'] for m in config[category]])
        return models
    else:
        return profile_config['models']


def run_experiment(args):
    """Main experiment runner"""
    
    # Setup environment
    if not setup_environment():
        return 1
    
    # Load configuration
    config_path = Path(args.config)
    config = load_config(config_path)
    
    if not config:
        return 1
    
    # Determine models to evaluate
    if args.profile:
        models = get_models_from_profile(config, args.profile)
        print(f"\nUsing profile '{args.profile}' with {len(models)} models")
    elif args.models:
        models = args.models
        print(f"\nEvaluating specified models: {models}")
    else:
        # Default to minimal profile
        models = get_models_from_profile(config, 'minimal')
        print(f"\nUsing default 'minimal' profile with {len(models)} models")
    
    # Filter out API models if requested
    if args.skip_api:
        api_models = {m['name'] for m in config.get('api_models', [])}
        models = [m for m in models if m not in api_models]
        print(f"Skipping API models. {len(models)} models remaining.")
    
    if not models:
        print("Error: No models to evaluate")
        return 1
    
    # Import the notebook code
    print("\nImporting evaluation pipeline...")
    
    try:
        # Import from the notebook cells (we'll use exec for simplicity)
        notebook_path = Path(__file__).parent / 'moral_alignment_complete.ipynb'
        
        if notebook_path.exists():
            import nbformat
            from nbconvert import PythonExporter
            
            # Convert notebook to Python
            with open(notebook_path, 'r') as f:
                notebook = nbformat.read(f, as_version=4)
            
            exporter = PythonExporter()
            source, _ = exporter.from_notebook_node(notebook)
            
            # Execute the notebook code
            exec_globals = {}
            exec(source, exec_globals)
            
            # Use the functions from the notebook
            evaluate_model = exec_globals.get('evaluate_model')
            load_wvs_data = exec_globals.get('load_wvs_data')
            load_pew_data = exec_globals.get('load_pew_data')
            
        else:
            print(f"Error: Notebook not found at {notebook_path}")
            print("Please ensure moral_alignment_complete.ipynb is in the same directory")
            return 1
            
    except ImportError as e:
        print(f"Error importing dependencies: {e}")
        print("Please install nbformat and nbconvert: pip install nbformat nbconvert")
        return 1
    except Exception as e:
        print(f"Error loading notebook: {e}")
        # Fallback: import from separate modules if they exist
        try:
            from moral_alignment_complete import evaluate_model, load_wvs_data, load_pew_data
        except ImportError:
            print("Could not import evaluation functions")
            return 1
    
    # Set up directories
    base_dir = Path(args.output_dir)
    base_dir.mkdir(exist_ok=True, parents=True)
    
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory not found at {data_dir}")
        return 1
    
    # Load survey data
    print("\nLoading survey data...")
    try:
        wvs_df = load_wvs_data(data_dir)
        pew_df = load_pew_data(data_dir)
        all_survey_data = pd.concat([wvs_df, pew_df], ignore_index=True)
        print(f"Loaded {len(all_survey_data)} country-topic pairs")
    except Exception as e:
        print(f"Error loading data: {e}")
        return 1
    
    # Initialize embedder
    print("\nLoading sentence embedder...")
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Prepare model configurations
    all_model_configs = {}
    for category in ['gpt2_family', 'opt_family', 'multilingual', 'qwen_family', 
                    'instruction_tuned', 'falcon_family', 'other_models', 'api_models']:
        if category in config:
            for model_config in config[category]:
                all_model_configs[model_config['name']] = model_config
    
    # Run evaluation
    print(f"\n{'='*60}")
    print("Starting Evaluation")
    print(f"{'='*60}")
    print(f"Models: {models}")
    print(f"Sample size: {args.sample_size if args.sample_size else 'Full dataset'}")
    
    all_results = {}
    all_metrics = []
    
    for model_name in models:
        if model_name not in all_model_configs:
            print(f"Warning: Configuration not found for {model_name}. Skipping.")
            continue
        
        # Evaluate model
        result = evaluate_model(
            model_name=model_name,
            model_config=all_model_configs[model_name],
            survey_data=all_survey_data,
            embedder=embedder,
            sample_size=args.sample_size
        )
        
        if result:
            all_results[model_name] = result
            all_metrics.extend(result['metrics'].to_dict('records'))
            
            # Save intermediate results
            out_dir = base_dir / 'outputs'
            out_dir.mkdir(exist_ok=True)
            
            result['lp_scores'].to_csv(out_dir / f"{model_name}_lp_scores.csv", index=False)
            result['dir_scores'].to_csv(out_dir / f"{model_name}_dir_scores.csv", index=False)
            
            trace_dir = out_dir / 'traces'
            trace_dir.mkdir(exist_ok=True)
            result['traces'].to_json(trace_dir / f"{model_name}_traces.jsonl", 
                                    orient='records', lines=True)
    
    # Save final results
    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        metrics_df.to_csv(base_dir / 'all_metrics.csv', index=False)
        
        print("\n" + "="*60)
        print("EXPERIMENT COMPLETE")
        print("="*60)
        print("\nFinal Results:")
        print(metrics_df.to_string())
        
        # Save summary
        summary = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'models_evaluated': list(all_results.keys()),
            'n_models': len(all_results),
            'sample_size': args.sample_size,
            'output_directory': str(base_dir)
        }
        
        with open(base_dir / 'experiment_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nResults saved to: {base_dir}")
    else:
        print("\nNo results generated")
        return 1
    
    return 0


def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(
        description="Run moral alignment experiments on LLMs",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Model selection
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument(
        '--models', 
        nargs='+',
        help='List of model names to evaluate'
    )
    model_group.add_argument(
        '--profile',
        choices=['minimal', 'standard', 'full', 'api_only'],
        help='Use predefined deployment profile'
    )
    
    # Data and output
    parser.add_argument(
        '--data-dir',
        default='sample_data',
        help='Directory containing WVS and PEW data files'
    )
    parser.add_argument(
        '--output-dir',
        default='experiment_results',
        help='Directory for output files'
    )
    
    # Experiment settings
    parser.add_argument(
        '--sample-size',
        type=int,
        help='Number of country-topic pairs to evaluate (default: all)'
    )
    parser.add_argument(
        '--config',
        default='models_config.yaml',
        help='Path to models configuration file'
    )
    
    # Options
    parser.add_argument(
        '--skip-api',
        action='store_true',
        help='Skip API-based models (OpenAI, Gemini)'
    )
    parser.add_argument(
        '--skip-peer-critique',
        action='store_true',
        help='Skip reciprocal peer critique phase'
    )
    parser.add_argument(
        '--no-visualizations',
        action='store_true',
        help='Skip generating visualization plots'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run experiment
    try:
        sys.exit(run_experiment(args))
    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()