#!/usr/bin/env python3
"""
Enhanced Moral Alignment Pipeline Runner
Quick launcher for the complete pipeline with all enhanced features
"""

import argparse
import asyncio
import sys
from pathlib import Path
import yaml
import pandas as pd
from datetime import datetime

# Import enhanced components
from prompts_manager import PromptsManager
from cross_evaluation import CrossEvaluator
from data_storage import DataStorageManager
from conflict_resolver import ConflictResolver

def main():
    parser = argparse.ArgumentParser(
        description="Run Enhanced Moral Alignment Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--profile',
        default='lightweight',
        choices=['minimal', 'lightweight', 'standard', 'cutting_edge_2024', 'api_only_2024', 'hybrid', 'full'],
        help='Deployment profile to use'
    )
    
    parser.add_argument(
        '--sample-size',
        type=int,
        default=20,
        help='Number of country-topic pairs to evaluate'
    )
    
    parser.add_argument(
        '--enable-cross-eval',
        action='store_true',
        default=True,
        help='Enable LLM cross-evaluation'
    )
    
    parser.add_argument(
        '--enable-conflict-resolution',
        action='store_true',
        default=True,
        help='Enable automatic conflict resolution'
    )
    
    parser.add_argument(
        '--launch-dashboard',
        action='store_true',
        help='Launch human evaluation dashboard after completion'
    )
    
    parser.add_argument(
        '--output-dir',
        default='outputs',
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("Enhanced Moral Alignment Pipeline")
    print("="*60)
    print(f"Profile: {args.profile}")
    print(f"Sample Size: {args.sample_size}")
    print(f"Cross-Evaluation: {args.enable_cross_eval}")
    print(f"Conflict Resolution: {args.enable_conflict_resolution}")
    print(f"Output Directory: {args.output_dir}")
    print("="*60)
    
    # Load configuration
    config_path = Path("models_config.yaml")
    if not config_path.exists():
        print("Error: models_config.yaml not found")
        return 1
    
    with open(config_path, 'r') as f:
        models_config = yaml.safe_load(f)
    
    # Get models from profile
    profile_config = models_config['deployment_profiles'].get(args.profile, {})
    if not profile_config:
        print(f"Error: Profile '{args.profile}' not found")
        return 1
    
    models = profile_config.get('models', [])
    if models == 'all':
        # Get all models
        models = []
        for category in ['instruction_tuned', 'api_models']:
            if category in models_config:
                models.extend([m['name'] for m in models_config[category]])
    
    print(f"\nModels to evaluate: {models[:5]}..." if len(models) > 5 else f"\nModels to evaluate: {models}")
    
    # Initialize enhanced components
    print("\nInitializing components...")
    pm = PromptsManager(output_dir=Path(args.output_dir) / "prompts")
    storage = DataStorageManager(base_dir=Path(args.output_dir))
    evaluator = CrossEvaluator(models_config=models_config, output_dir=Path(args.output_dir) / "cross_evaluation")
    resolver = ConflictResolver()
    
    # Start experiment run
    run_id = storage.start_experiment_run(
        models=models,
        config={
            'profile': args.profile,
            'sample_size': args.sample_size,
            'enable_cross_eval': args.enable_cross_eval,
            'enable_conflict_resolution': args.enable_conflict_resolution,
            'timestamp': datetime.now().isoformat()
        }
    )
    
    print(f"\nStarted experiment run: {run_id}")
    
    # Placeholder for main evaluation
    print("\nNote: Main evaluation pipeline would run here.")
    print("This launcher demonstrates the enhanced component integration.")
    
    # Example workflow
    print("\nEnhanced Workflow:")
    print("1. ✅ PromptsManager initialized - Managing all prompts")
    print("2. ✅ DataStorageManager initialized - Tracking experiment")
    print("3. ✅ CrossEvaluator ready - For peer evaluation")
    print("4. ✅ ConflictResolver ready - For disagreement detection")
    
    # Save example data
    print("\nSaving example data...")
    
    # Example prompt creation
    example_prompts = pm.create_logprob_prompts(
        country="United States",
        topic="abortion",
        model=models[0] if models else "gpt-4o"
    )
    print(f"  - Created {len(example_prompts)} prompts")
    
    # Save session
    session_files = pm.save_session_prompts(run_id)
    print(f"  - Saved prompts to {session_files[0]}")
    
    # Complete run
    storage.complete_experiment_run(run_id)
    print(f"  - Experiment run completed")
    
    # Launch dashboard if requested
    if args.launch_dashboard:
        print("\nLaunching human evaluation dashboard...")
        import subprocess
        subprocess.Popen(["streamlit", "run", "human_dashboard.py"])
        print("Dashboard launched at http://localhost:8501")
    
    print("\n" + "="*60)
    print("Pipeline Setup Complete!")
    print("="*60)
    print("\nNext Steps:")
    print("1. Run full evaluation with: python run_experiments.py --profile", args.profile)
    print("2. Review results in:", args.output_dir)
    if not args.launch_dashboard:
        print("3. Launch dashboard with: streamlit run human_dashboard.py")
    print("\n✅ All enhanced components are ready for use!")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())