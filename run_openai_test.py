#!/usr/bin/env python3
"""
OpenAI Test Runner for Moral Alignment Pipeline
Comprehensive testing script for OpenAI models with visualization
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import json
from typing import Dict, List

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Local imports
from env_loader import get_env_loader
from wvs_processor import WVSProcessor
from moral_alignment_tester import MoralAlignmentTester
from validation_suite import ValidationSuite
from paper_outputs import PaperOutputGenerator
from moral_visualization import MoralVisualizationEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main execution function"""
    
    parser = argparse.ArgumentParser(
        description="Test OpenAI models for moral alignment evaluation"
    )
    
    parser.add_argument(
        '--sample-size',
        type=int,
        default=5,
        help='Number of test scenarios to run (default: 5)'
    )
    
    parser.add_argument(
        '--model',
        default='gpt-3.5-turbo',
        help='OpenAI model to test (default: gpt-3.5-turbo)'
    )
    
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Run quick test with minimal samples'
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("OpenAI Model Testing Pipeline")
    logger.info("=" * 60)
    
    # Step 1: Check environment
    logger.info("\n📋 Step 1: Checking Environment")
    env_loader = get_env_loader()
    env_info = env_loader.get_environment_info()
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        logger.error("❌ OpenAI API key not configured!")
        logger.info("Please add your OpenAI API key to the .env file")
        return 1
    
    logger.info(f"✅ OpenAI API configured")
    
    # Step 2: Initialize components
    logger.info("\n🔧 Step 2: Initializing Components")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(f"outputs/openai_test_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize WVS processor
    wvs = WVSProcessor()
    wvs.load_data()
    wvs.process_moral_scores()
    
    # Initialize tester
    tester = MoralAlignmentTester(output_dir=output_dir)
    
    # Step 3: Prepare test data
    logger.info("\n📊 Step 3: Preparing Test Data")
    
    if args.quick_test:
        sample_size = 3
    else:
        sample_size = args.sample_size
    
    eval_data = wvs.create_evaluation_dataset(
        n_samples=sample_size,
        topics=wvs.KEY_TOPICS[:5],  # Only 5 topics for quick test
        stratified=True
    )
    
    logger.info(f"Created {len(eval_data)} test samples")
    logger.info(f"Countries: {eval_data['country'].nunique()}")
    logger.info(f"Topics: {eval_data['topic'].nunique()}")
    
    # Step 4: Run tests with rate limiting
    logger.info(f"\n🧪 Step 4: Testing {args.model}")
    logger.info("⏳ This will take a few minutes due to rate limiting...")
    
    results = {
        'scores': [],
        'metrics': {},
        'model': args.model,
        'timestamp': timestamp
    }
    
    for idx, row in eval_data.iterrows():
        print(f"\r  Processing {idx+1}/{len(eval_data)}...", end="", flush=True)
        
        try:
            # Test with log-probability method
            lp_result = tester._test_logprob_scoring(args.model, row)
            if lp_result:
                lp_result['ground_truth'] = row['normalized_score']
                results['scores'].append(lp_result)
            
            time.sleep(0.5)  # Rate limit
            
            # Test with direct scoring method
            dir_result = tester._test_direct_scoring(args.model, row)
            if dir_result:
                dir_result['ground_truth'] = row['normalized_score']
                results['scores'].append(dir_result)
            
            time.sleep(0.5)  # Rate limit
            
        except Exception as e:
            logger.warning(f"\nError processing sample {idx}: {e}")
            continue
    
    print()  # New line after progress
    logger.info(f"✅ Processed {len(results['scores'])} scores")
    
    # Step 5: Calculate metrics
    logger.info("\n📈 Step 5: Calculating Metrics")
    
    if results['scores']:
        df = pd.DataFrame(results['scores'])
        
        for method in ['logprob', 'direct']:
            method_df = df[df['method'] == method]
            if len(method_df) > 1:
                corr = method_df[['ground_truth', 'model_score']].corr().iloc[0, 1]
                mae = (method_df['model_score'] - method_df['ground_truth']).abs().mean()
                
                results['metrics'][f'correlation_{method}'] = float(corr)
                results['metrics'][f'mae_{method}'] = float(mae)
                results['metrics'][f'n_{method}'] = len(method_df)
                
                print(f"\n{method.upper()} Method:")
                print(f"  Correlation: {corr:.3f}")
                print(f"  MAE: {mae:.3f}")
                print(f"  Samples: {len(method_df)}")
    
    # Step 6: Save results
    logger.info("\n💾 Step 6: Saving Results")
    
    results_file = output_dir / f"{args.model}_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Create comprehensive results
    comprehensive = {
        'model_results': {args.model: results},
        'run_metadata': {
            'timestamp': timestamp,
            'n_samples': sample_size,
            'models_tested': [args.model]
        }
    }
    
    comp_file = output_dir / "comprehensive_results.json"
    with open(comp_file, 'w') as f:
        json.dump(comprehensive, f, indent=2, default=str)
    
    # Step 7: Generate paper outputs and visualizations
    logger.info("\n📄 Step 7: Generating Paper Outputs and Visualizations")
    
    paper_gen = PaperOutputGenerator(
        results_dir=output_dir,
        output_dir=output_dir / "paper"
    )
    
    outputs = paper_gen.generate_all_outputs()
    logger.info(f"Generated {len(outputs)} paper outputs")
    
    # Generate visualizations
    if results['scores']:
        viz = MoralVisualizationEngine(output_dir=output_dir / "figures")
        viz_df = pd.DataFrame(results['scores'])
        viz_df['model'] = args.model
        
        plots = viz.create_all_plots(results={args.model: results}, df=viz_df)
        logger.info(f"Generated {len(plots)} visualizations")
    
    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("✅ Testing Complete!")
    logger.info("=" * 60)
    logger.info(f"Model tested: {args.model}")
    logger.info(f"Total tests: {len(results['scores'])}")
    
    for key, value in results['metrics'].items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.3f}")
        else:
            logger.info(f"  {key}: {value}")
    
    logger.info(f"\n📁 All outputs saved to: {output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())