#!/usr/bin/env python3
"""
Full Evaluation Runner for Moral Alignment Pipeline
Orchestrates complete testing, validation, and output generation
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

# Local imports
from env_loader import get_env_loader
from wvs_processor import WVSProcessor
from moral_alignment_tester import MoralAlignmentTester
from validation_suite import ValidationSuite
from paper_outputs import PaperOutputGenerator
from visualization_engine import VisualizationEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FullEvaluationRunner:
    """Orchestrates the complete moral alignment evaluation pipeline"""
    
    def __init__(self, output_base: str = "outputs"):
        """Initialize evaluation runner
        
        Args:
            output_base: Base directory for all outputs
        """
        self.output_base = Path(output_base)
        self.output_base.mkdir(exist_ok=True)
        
        # Create timestamp for this run
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = self.output_base / f"run_{self.timestamp}"
        self.run_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.env_loader = get_env_loader()
        self.wvs_processor = WVSProcessor()
        self.alignment_tester = MoralAlignmentTester(output_dir=self.run_dir / "alignment_tests")
        self.validator = ValidationSuite(output_dir=self.run_dir / "validation")
        self.paper_generator = PaperOutputGenerator(
            results_dir=self.run_dir / "alignment_tests",
            output_dir=self.run_dir / "paper"
        )
        self.viz_engine = VisualizationEngine(output_dir=self.run_dir / "visualizations")
        
        # Store results
        self.results = {
            'run_metadata': {
                'timestamp': self.timestamp,
                'run_dir': str(self.run_dir)
            }
        }
    
    def check_prerequisites(self) -> bool:
        """Check if all prerequisites are met
        
        Returns:
            True if ready to run
        """
        logger.info("Checking prerequisites...")
        
        # Check for API keys
        env_info = self.env_loader.get_environment_info()
        
        if not env_info['available_apis']:
            logger.error("No API keys configured. Please add API keys to .env file")
            return False
        
        logger.info(f"Available APIs: {list(env_info['available_apis'].keys())}")
        logger.info(f"API models available: {len(env_info['api_models'])}")
        
        # Check for WVS data
        wvs_file = Path("sample_data/WVS_Moral.csv")
        if not wvs_file.exists():
            logger.error(f"WVS data not found at {wvs_file}")
            return False
        
        logger.info("✅ All prerequisites met")
        return True
    
    def prepare_data(self, n_samples: int) -> pd.DataFrame:
        """Prepare evaluation data
        
        Args:
            n_samples: Number of samples to prepare
            
        Returns:
            Evaluation dataset
        """
        logger.info(f"Preparing {n_samples} evaluation samples...")
        
        # Load and process WVS data
        self.wvs_processor.load_data()
        self.wvs_processor.process_moral_scores()
        
        # Create evaluation dataset
        eval_data = self.wvs_processor.create_evaluation_dataset(
            n_samples=n_samples,
            topics=self.wvs_processor.KEY_TOPICS,
            stratified=True
        )
        
        # Save evaluation data
        eval_file = self.run_dir / "evaluation_data.csv"
        eval_data.to_csv(eval_file, index=False)
        logger.info(f"Saved evaluation data to {eval_file}")
        
        # Calculate human baseline
        baseline = self.wvs_processor.calculate_human_baseline()
        baseline_file = self.run_dir / "human_baseline.json"
        with open(baseline_file, 'w') as f:
            json.dump(baseline, f, indent=2, default=str)
        
        self.results['data_preparation'] = {
            'n_samples': len(eval_data),
            'countries': eval_data['country'].nunique(),
            'topics': eval_data['topic'].nunique(),
            'baseline_file': str(baseline_file)
        }
        
        return eval_data
    
    def run_model_evaluation(self, 
                           models: List[str],
                           n_samples: int) -> Dict:
        """Run model evaluation
        
        Args:
            models: List of models to evaluate
            n_samples: Number of samples
            
        Returns:
            Evaluation results
        """
        logger.info(f"Running evaluation for {len(models)} models...")
        
        # Run comprehensive test
        test_results = self.alignment_tester.run_comprehensive_test(
            models=models,
            n_samples=n_samples
        )
        
        self.results['model_evaluation'] = test_results
        
        # Save results
        results_file = self.run_dir / "alignment_tests" / "comprehensive_results.json"
        with open(results_file, 'w') as f:
            json.dump(test_results, f, indent=2, default=str)
        
        return test_results
    
    def run_validation(self, test_results: Dict) -> Dict:
        """Run validation suite
        
        Args:
            test_results: Results from model evaluation
            
        Returns:
            Validation results
        """
        logger.info("Running validation suite...")
        
        all_validations = {
            'model_validations': {},
            'cross_model_agreement': {},
            'human_alignment': {}
        }
        
        # Validate each model
        if 'model_results' in test_results:
            for model, results in test_results['model_results'].items():
                validation = self.validator.validate_model_results(results, model)
                all_validations['model_validations'][model] = validation
            
            # Cross-model agreement
            all_validations['cross_model_agreement'] = self.validator.validate_cross_model_agreement(
                test_results['model_results']
            )
        
        # Human alignment (load baseline)
        baseline_file = self.run_dir / "human_baseline.json"
        if baseline_file.exists():
            with open(baseline_file, 'r') as f:
                baseline = json.load(f)
            
            for model, results in test_results.get('model_results', {}).items():
                alignment = self.validator.validate_human_alignment(results, baseline)
                all_validations['human_alignment'][model] = alignment
        
        # Save validation results
        self.validator.save_validation_results(all_validations)
        self.results['validation'] = all_validations
        
        return all_validations
    
    def generate_visualizations(self, test_results: Dict) -> List[str]:
        """Generate visualizations
        
        Args:
            test_results: Results from model evaluation
            
        Returns:
            List of generated visualization paths
        """
        logger.info("Generating visualizations...")
        
        viz_paths = []
        
        # Convert results to DataFrame for visualization
        all_scores = []
        if 'model_results' in test_results:
            for model, results in test_results['model_results'].items():
                if 'scores' in results:
                    scores_df = pd.DataFrame(results['scores'])
                    scores_df['model'] = model
                    all_scores.append(scores_df)
        
        if all_scores:
            combined_df = pd.concat(all_scores, ignore_index=True)
            
            # Generate visualizations
            viz_paths.append(self.viz_engine.plot_model_performance_comparison(combined_df))
            viz_paths.append(self.viz_engine.plot_response_patterns(combined_df))
            viz_paths.append(self.viz_engine.create_dashboard(combined_df))
            
            # Cost analysis
            models = list(test_results['model_results'].keys())
            viz_paths.append(self.viz_engine.plot_cost_analysis(models, len(combined_df)))
        
        self.results['visualizations'] = viz_paths
        logger.info(f"Generated {len(viz_paths)} visualizations")
        
        return viz_paths
    
    def generate_paper_outputs(self) -> Dict:
        """Generate paper-ready outputs
        
        Returns:
            Dictionary of generated outputs
        """
        logger.info("Generating paper outputs...")
        
        outputs = self.paper_generator.generate_all_outputs()
        self.results['paper_outputs'] = outputs
        
        return outputs
    
    def run_full_pipeline(self, 
                         mode: str = "quick",
                         models: Optional[List[str]] = None,
                         generate_paper: bool = True) -> Dict:
        """Run complete evaluation pipeline
        
        Args:
            mode: Evaluation mode (quick/standard/full)
            models: Specific models to test
            generate_paper: Whether to generate paper outputs
            
        Returns:
            Complete results
        """
        logger.info("=" * 60)
        logger.info(f"Starting Full Evaluation Pipeline - Mode: {mode}")
        logger.info("=" * 60)
        
        # Check prerequisites
        if not self.check_prerequisites():
            logger.error("Prerequisites not met. Exiting.")
            return {}
        
        # Determine sample size based on mode
        sample_sizes = {
            'quick': 100,
            'standard': 500,
            'comprehensive': 1000,
            'full': 5000
        }
        n_samples = sample_sizes.get(mode, 100)
        
        # Determine models
        if models is None:
            # Use available API models
            env_info = self.env_loader.get_environment_info()
            models = env_info['api_models'][:3]  # Top 3 for testing
        
        logger.info(f"Configuration:")
        logger.info(f"  Mode: {mode}")
        logger.info(f"  Samples: {n_samples}")
        logger.info(f"  Models: {models}")
        
        # Step 1: Prepare data
        logger.info("\n📊 Step 1: Preparing Data")
        eval_data = self.prepare_data(n_samples)
        
        # Step 2: Run model evaluation
        logger.info("\n🧪 Step 2: Running Model Evaluation")
        test_results = self.run_model_evaluation(models, n_samples)
        
        # Step 3: Run validation
        logger.info("\n✅ Step 3: Running Validation")
        validation_results = self.run_validation(test_results)
        
        # Step 4: Generate visualizations
        logger.info("\n📈 Step 4: Generating Visualizations")
        viz_paths = self.generate_visualizations(test_results)
        
        # Step 5: Generate paper outputs
        if generate_paper:
            logger.info("\n📝 Step 5: Generating Paper Outputs")
            paper_outputs = self.generate_paper_outputs()
        
        # Save complete results
        results_file = self.run_dir / "complete_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Generate summary report
        self._generate_summary_report()
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ Full Evaluation Pipeline Complete!")
        logger.info("=" * 60)
        logger.info(f"Results saved to: {self.run_dir}")
        
        return self.results
    
    def _generate_summary_report(self):
        """Generate summary report of the run"""
        
        report = []
        report.append("# Moral Alignment Evaluation Summary")
        report.append(f"\nRun ID: {self.timestamp}")
        report.append(f"Output Directory: {self.run_dir}")
        
        # Data summary
        if 'data_preparation' in self.results:
            data = self.results['data_preparation']
            report.append("\n## Data Preparation")
            report.append(f"- Samples: {data['n_samples']}")
            report.append(f"- Countries: {data['countries']}")
            report.append(f"- Topics: {data['topics']}")
        
        # Model evaluation summary
        if 'model_evaluation' in self.results:
            eval_results = self.results['model_evaluation']
            if 'comparative_analysis' in eval_results:
                analysis = eval_results['comparative_analysis']
                
                report.append("\n## Model Performance")
                if 'model_rankings' in analysis:
                    report.append("\n### Rankings (by correlation)")
                    for i, (model, corr) in enumerate(analysis['model_rankings'].items(), 1):
                        report.append(f"{i}. {model}: {corr:.3f}")
                
                if 'best_model' in analysis:
                    report.append(f"\n**Best Model**: {analysis['best_model']}")
        
        # Validation summary
        if 'validation' in self.results:
            validation = self.results['validation']
            if 'cross_model_agreement' in validation:
                cma = validation['cross_model_agreement']
                if 'consensus_metrics' in cma:
                    cm = cma['consensus_metrics']
                    report.append("\n## Validation Results")
                    report.append(f"- Mean Consensus: {cm.get('mean_consensus', 0):.3f}")
                    report.append(f"- Mean Disagreement: {cm.get('mean_disagreement', 0):.3f}")
        
        # Outputs generated
        report.append("\n## Outputs Generated")
        if 'visualizations' in self.results:
            report.append(f"- Visualizations: {len(self.results['visualizations'])}")
        if 'paper_outputs' in self.results:
            report.append(f"- Paper Outputs: {len(self.results['paper_outputs'])}")
        
        # Save report
        report_str = "\n".join(report)
        report_file = self.run_dir / "summary_report.md"
        report_file.write_text(report_str)
        
        # Print to console
        print("\n" + report_str)


def main():
    """Main execution function"""
    
    parser = argparse.ArgumentParser(
        description="Run full moral alignment evaluation pipeline"
    )
    
    parser.add_argument(
        '--mode',
        choices=['quick', 'standard', 'comprehensive', 'full'],
        default='quick',
        help='Evaluation mode determining sample size'
    )
    
    parser.add_argument(
        '--models',
        nargs='+',
        default=None,
        help='Specific models to evaluate'
    )
    
    parser.add_argument(
        '--no-paper',
        action='store_true',
        help='Skip paper output generation'
    )
    
    parser.add_argument(
        '--output-dir',
        default='outputs',
        help='Base output directory'
    )
    
    args = parser.parse_args()
    
    # Create runner
    runner = FullEvaluationRunner(output_base=args.output_dir)
    
    # Run pipeline
    results = runner.run_full_pipeline(
        mode=args.mode,
        models=args.models,
        generate_paper=not args.no_paper
    )
    
    if results:
        logger.info("\n✅ Success! Evaluation complete.")
        return 0
    else:
        logger.error("\n❌ Evaluation failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())