#!/usr/bin/env python3
"""
OpenAI Test Runner for Moral Alignment Pipeline
Comprehensive testing script for OpenAI models with visualization
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Local imports
from env_loader import get_env_loader
from test_openai_models import OpenAIModelTester
from visualization_engine import VisualizationEngine
from output_generator import OutputGenerator

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
        '--models',
        nargs='+',
        default=None,
        help='Specific models to test (default: all OpenAI models)'
    )
    
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Run quick test with minimal samples'
    )
    
    parser.add_argument(
        '--visualize',
        action='store_true',
        default=True,
        help='Generate visualizations after testing'
    )
    
    parser.add_argument(
        '--export',
        action='store_true',
        default=True,
        help='Export results in multiple formats'
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("OpenAI Model Testing Pipeline")
    logger.info("=" * 60)
    
    # Step 1: Check environment
    logger.info("\n📋 Step 1: Checking Environment")
    env_loader = get_env_loader()
    env_info = env_loader.get_environment_info()
    
    if not env_info['has_openai']:
        logger.error("❌ OpenAI API key not configured!")
        logger.info("Please add your OpenAI API key to the .env file")
        return 1
    
    logger.info(f"✅ OpenAI API configured")
    logger.info(f"Available API models: {len(env_info['api_models'])}")
    
    # Step 2: Run tests
    logger.info("\n🧪 Step 2: Testing OpenAI Models")
    
    if args.quick_test:
        sample_size = 2
        logger.info("Running quick test with 2 samples")
    else:
        sample_size = args.sample_size
    
    tester = OpenAIModelTester(sample_size=sample_size)
    
    # Override models if specified
    if args.models:
        tester.models = args.models
        logger.info(f"Testing specific models: {args.models}")
    
    # Run comprehensive test
    test_summary = tester.run_comprehensive_test()
    
    if test_summary.get('error'):
        logger.error(f"Testing failed: {test_summary['error']}")
        return 1
    
    logger.info(f"✅ Tested {test_summary['models_tested']} models successfully")
    
    # Step 3: Analyze results
    logger.info("\n📊 Step 3: Analyzing Results")
    
    # Get agreement analysis
    agreement = tester.analyze_model_agreement()
    
    # Generate test report
    report = tester.generate_test_report()
    
    # Step 4: Generate visualizations
    if args.visualize:
        logger.info("\n📈 Step 4: Generating Visualizations")
        
        viz_engine = VisualizationEngine()
        
        # Convert results to DataFrame for visualization
        all_results = []
        for model, results in tester.results.items():
            all_results.extend(results)
        
        if all_results:
            df = pd.DataFrame(all_results)
            
            # Create visualizations
            plots_created = []
            
            # Performance comparison
            plot_path = viz_engine.plot_model_performance_comparison(df)
            if plot_path:
                plots_created.append(plot_path)
                logger.info(f"Created performance comparison: {plot_path}")
            
            # Agreement heatmap
            if agreement:
                plot_path = viz_engine.plot_model_agreement_heatmap(agreement)
                if plot_path:
                    plots_created.append(plot_path)
                    logger.info(f"Created agreement heatmap: {plot_path}")
            
            # Response patterns
            plot_path = viz_engine.plot_response_patterns(df)
            if plot_path:
                plots_created.append(plot_path)
                logger.info(f"Created response patterns: {plot_path}")
            
            # Cost analysis
            plot_path = viz_engine.plot_cost_analysis(tester.models, sample_size * 200)
            if plot_path:
                plots_created.append(plot_path)
                logger.info(f"Created cost analysis: {plot_path}")
            
            # Dashboard
            plot_path = viz_engine.create_dashboard(df)
            if plot_path:
                plots_created.append(plot_path)
                logger.info(f"Created dashboard: {plot_path}")
            
            logger.info(f"✅ Generated {len(plots_created)} visualizations")
    
    # Step 5: Export results
    if args.export:
        logger.info("\n💾 Step 5: Exporting Results")
        
        output_gen = OutputGenerator()
        
        # Save test results first
        results_file = output_gen.results_dir / f"openai_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump({
                'summary': test_summary,
                'agreement': agreement,
                'results': all_results if 'all_results' in locals() else []
            }, f, indent=2, default=str)
        
        # Generate all outputs
        outputs = output_gen.generate_all_outputs()
        
        logger.info(f"✅ Exported {len(outputs)} output formats")
    
    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("✅ Testing Complete!")
    logger.info("=" * 60)
    logger.info(f"Models tested: {', '.join(test_summary['models_tested'])}")
    logger.info(f"Total tests: {test_summary['total_tests']}")
    logger.info(f"Success rate: {test_summary['successful_tests']}/{test_summary['total_tests']}")
    logger.info(f"Total time: {test_summary['total_time']:.2f}s")
    logger.info(f"Estimated cost: ${test_summary['estimated_cost']:.2f}")
    
    # Print cost breakdown
    logger.info("\n💰 Cost Breakdown:")
    for model in test_summary['models_tested']:
        cost_info = env_loader.estimate_costs(model, sample_size)
        logger.info(f"  {model}: ${cost_info['estimated_cost_usd']:.2f}")
    
    logger.info("\n📁 Output Locations:")
    logger.info(f"  Results: outputs/")
    logger.info(f"  Plots: outputs/plots/")
    logger.info(f"  Tables: outputs/tables/")
    logger.info(f"  Reports: outputs/reports/")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())