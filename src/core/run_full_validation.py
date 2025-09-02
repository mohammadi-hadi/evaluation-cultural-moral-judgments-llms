#!/usr/bin/env python3
"""
Full Validation Pipeline with LLM Judge and Human Review
Implements the complete methodology from the paper:
1. Dual elicitation (log-prob and direct scoring)
2. Reciprocal model critique (peer review)
3. Human arbitration for conflicts
"""

import os
import json
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

# Local imports
from env_loader import get_env_loader
from wvs_processor import WVSProcessor
from moral_alignment_tester import MoralAlignmentTester
from model_judge import ModelJudge, ReasoningTrace
from validation_suite import ValidationSuite
from paper_outputs import PaperOutputGenerator
from moral_visualization import MoralVisualizationEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FullValidationPipeline:
    """Complete validation pipeline matching paper methodology"""
    
    def __init__(self, output_dir: str = "outputs/full_validation"):
        """Initialize validation pipeline
        
        Args:
            output_dir: Directory for all outputs
        """
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = Path(output_dir) / f"run_{self.timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.env_loader = get_env_loader()
        self.wvs = WVSProcessor()
        self.tester = MoralAlignmentTester(output_dir=self.output_dir / "alignment_tests")
        self.judge = None  # Will be initialized with API keys
        self.validator = ValidationSuite(output_dir=self.output_dir / "validation")
        
        # Store results
        self.results = {
            'metadata': {
                'timestamp': self.timestamp,
                'output_dir': str(self.output_dir)
            }
        }
    
    def run_pipeline(self,
                    models: List[str] = None,
                    n_samples: int = 10,
                    run_peer_review: bool = True,
                    save_for_human_review: bool = True) -> Dict:
        """Run complete validation pipeline
        
        Args:
            models: List of models to test (None = use available)
            n_samples: Number of samples to test
            run_peer_review: Whether to run reciprocal critique
            save_for_human_review: Whether to save conflicts for human review
            
        Returns:
            Complete validation results
        """
        logger.info("=" * 60)
        logger.info("FULL VALIDATION PIPELINE")
        logger.info("=" * 60)
        
        # Step 1: Check environment and models
        logger.info("\n📋 Step 1: Checking Environment")
        env_info = self.env_loader.get_environment_info()
        
        if models is None:
            # Use available OpenAI models for now
            if env_info['has_openai']:
                models = ['gpt-3.5-turbo', 'gpt-4o-mini']  # Start with these
            else:
                logger.error("No API keys configured")
                return {}
        
        logger.info(f"Models to test: {models}")
        self.results['models'] = models
        
        # Step 2: Prepare evaluation data
        logger.info("\n📊 Step 2: Preparing Evaluation Data")
        self.wvs.load_data()
        self.wvs.process_moral_scores()
        
        eval_data = self.wvs.create_evaluation_dataset(
            n_samples=n_samples,
            topics=self.wvs.KEY_TOPICS[:5],  # Focus on key topics
            stratified=True
        )
        
        eval_data.to_csv(self.output_dir / "evaluation_data.csv", index=False)
        logger.info(f"Created {len(eval_data)} evaluation samples")
        logger.info(f"Countries: {eval_data['country'].nunique()}")
        logger.info(f"Topics: {eval_data['topic'].nunique()}")
        
        self.results['evaluation_data'] = {
            'n_samples': len(eval_data),
            'countries': eval_data['country'].nunique(),
            'topics': eval_data['topic'].nunique()
        }
        
        # Step 3: Run dual elicitation
        logger.info("\n🧪 Step 3: Running Dual Elicitation")
        logger.info("(Both log-probability and direct scoring)")
        
        test_results = self.tester.run_full_validation_pipeline(
            models=models,
            n_samples=n_samples
        )
        
        self.results['test_results'] = test_results
        
        # Step 4: Detect conflicts
        logger.info("\n⚔️ Step 4: Detecting Conflicts")
        
        if 'model_results' in test_results:
            conflicts = self.tester.detect_conflicts(test_results['model_results'])
            logger.info(f"Found {len(conflicts)} conflicts (threshold={self.tester.conflict_threshold})")
            
            if conflicts:
                # Save detailed conflict analysis
                conflict_df = pd.DataFrame(conflicts)
                conflict_df.to_csv(self.output_dir / "conflicts.csv", index=False)
                
                # Group by severity
                conflict_df['severity'] = pd.cut(
                    conflict_df['difference'],
                    bins=[0.4, 0.6, 0.8, 1.0, 2.0],
                    labels=['medium', 'high', 'critical', 'extreme']
                )
                
                self.results['conflicts'] = {
                    'total': len(conflicts),
                    'by_severity': conflict_df['severity'].value_counts().to_dict(),
                    'sample': conflicts[:5]  # Include sample
                }
        
        # Step 5: Run reciprocal critique (if requested)
        if run_peer_review and self.tester.reasoning_traces:
            logger.info("\n👥 Step 5: Running Reciprocal Model Critique")
            
            # Initialize judge with API keys
            api_keys = {}
            if os.getenv('OPENAI_API_KEY'):
                api_keys['openai'] = os.getenv('OPENAI_API_KEY')
            
            self.judge = ModelJudge(api_keys=api_keys)
            
            # Run reciprocal critique
            critique_df = self.judge.run_reciprocal_critique(
                models=models,
                reasoning_traces=self.tester.reasoning_traces[:20],  # Limit for demo
                sample_size=5  # Small sample for testing
            )
            
            # Calculate peer-agreement rates
            agreement_rates = self.judge.calculate_peer_agreement_rates(critique_df)
            logger.info(f"Peer-agreement rates: {agreement_rates}")
            
            # Identify contentious cases
            contentious = self.judge.identify_contentious_cases(critique_df)
            logger.info(f"Contentious cases needing human review: {len(contentious)}")
            
            # Save peer review results
            self.judge.save_critique_results(
                critique_df,
                agreement_rates,
                contentious,
                output_dir=self.output_dir / "peer_review"
            )
            
            self.results['peer_review'] = {
                'n_critiques': len(critique_df),
                'agreement_rates': agreement_rates,
                'n_contentious': len(contentious)
            }
        
        # Step 6: Prepare for human review
        if save_for_human_review and 'conflicts' in self.results:
            logger.info("\n👤 Step 6: Preparing for Human Review")
            
            # Create human review dataset
            human_review_data = self._prepare_human_review_data(
                conflicts[:50],  # Limit to 50 for manageable review
                test_results.get('model_results', {})
            )
            
            human_review_file = self.output_dir / "human_review_cases.json"
            with open(human_review_file, 'w') as f:
                json.dump(human_review_data, f, indent=2)
            
            logger.info(f"Saved {len(human_review_data['cases'])} cases for human review")
            logger.info(f"Review file: {human_review_file}")
            
            self.results['human_review'] = {
                'n_cases': len(human_review_data['cases']),
                'file': str(human_review_file),
                'instruction': "Run 'streamlit run human_dashboard.py' to review"
            }
        
        # Step 7: Calculate final metrics
        logger.info("\n📈 Step 7: Calculating Final Metrics")
        
        if 'model_results' in test_results:
            # Survey alignment (correlations)
            correlations = {}
            for model, results in test_results['model_results'].items():
                if 'metrics' in results:
                    correlations[model] = {
                        'logprob': results['metrics'].get('correlation_logprob', 0),
                        'direct': results['metrics'].get('correlation_direct', 0)
                    }
            
            self.results['survey_alignment'] = correlations
            
            # Self-consistency (from reasoning traces)
            if self.tester.reasoning_traces:
                # Group traces by model and calculate consistency
                from collections import defaultdict
                traces_by_model = defaultdict(list)
                for trace in self.tester.reasoning_traces:
                    traces_by_model[trace.model].append(trace.final_score)
                
                consistency = {}
                for model, scores in traces_by_model.items():
                    if len(scores) > 1:
                        consistency[model] = 1 - np.std(scores)  # Simple consistency metric
                
                self.results['self_consistency'] = consistency
        
        # Step 8: Generate visualizations
        logger.info("\n🎨 Step 8: Generating Visualizations")
        
        viz = MoralVisualizationEngine(output_dir=self.output_dir / "figures")
        
        # Prepare data for visualization
        if 'model_results' in test_results:
            all_scores = []
            for model, results in test_results['model_results'].items():
                if 'scores' in results:
                    for score in results['scores']:
                        score['model'] = model
                        all_scores.append(score)
            
            if all_scores:
                viz_df = pd.DataFrame(all_scores)
                plots = viz.create_all_plots(
                    results=test_results['model_results'],
                    df=viz_df
                )
                logger.info(f"Generated {len(plots)} visualizations")
                self.results['visualizations'] = plots
        
        # Step 9: Generate paper outputs
        logger.info("\n📄 Step 9: Generating Paper Outputs")
        
        paper_gen = PaperOutputGenerator(
            results_dir=self.output_dir,
            output_dir=self.output_dir / "paper"
        )
        
        # Save results in format expected by paper generator
        for model in models:
            if model in test_results.get('model_results', {}):
                model_file = self.output_dir / f"{model}_results.json"
                with open(model_file, 'w') as f:
                    json.dump(test_results['model_results'][model], f, indent=2)
        
        paper_outputs = paper_gen.generate_all_outputs()
        self.results['paper_outputs'] = paper_outputs
        
        # Save complete results
        results_file = self.output_dir / "complete_validation_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Generate summary report
        self._generate_summary_report()
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ VALIDATION PIPELINE COMPLETE!")
        logger.info("=" * 60)
        logger.info(f"Results saved to: {self.output_dir}")
        
        return self.results
    
    def _prepare_human_review_data(self, conflicts: List[Dict], 
                                  model_results: Dict) -> Dict:
        """Prepare conflicts for human review with full context
        
        Args:
            conflicts: List of conflict cases
            model_results: Full model results for context
            
        Returns:
            Formatted data for human review
        """
        review_cases = []
        
        for conflict in conflicts:
            # Get full reasoning for both models
            reasoning1 = conflict.get('reasoning1', '')
            reasoning2 = conflict.get('reasoning2', '')
            
            # Try to get more detailed reasoning from traces
            for trace in self.tester.reasoning_traces:
                if (trace.model == conflict['model1'] and 
                    trace.country == conflict['country'] and 
                    trace.topic == conflict['topic']):
                    reasoning1 = trace.raw_response
                elif (trace.model == conflict['model2'] and 
                      trace.country == conflict['country'] and 
                      trace.topic == conflict['topic']):
                    reasoning2 = trace.raw_response
            
            review_case = {
                'case_id': conflict['conflict_id'],
                'country': conflict['country'],
                'topic': conflict['topic'],
                'model_a': {
                    'name': conflict['model1'],
                    'score': conflict['score1'],
                    'reasoning': reasoning1
                },
                'model_b': {
                    'name': conflict['model2'],
                    'score': conflict['score2'],
                    'reasoning': reasoning2
                },
                'difference': conflict['difference'],
                'question': f"Which model better reflects how people in {conflict['country']} view {conflict['topic']}?"
            }
            
            review_cases.append(review_case)
        
        return {
            'metadata': {
                'n_cases': len(review_cases),
                'timestamp': self.timestamp,
                'instructions': "Please evaluate which model's reasoning better reflects the cultural norms of the specified country."
            },
            'cases': review_cases
        }
    
    def _generate_summary_report(self):
        """Generate markdown summary report"""
        
        report = []
        report.append("# Full Validation Pipeline Report")
        report.append(f"\nRun ID: {self.timestamp}")
        report.append(f"Output Directory: {self.output_dir}")
        
        # Models tested
        report.append("\n## Models Tested")
        for model in self.results.get('models', []):
            report.append(f"- {model}")
        
        # Evaluation data
        if 'evaluation_data' in self.results:
            data = self.results['evaluation_data']
            report.append("\n## Evaluation Data")
            report.append(f"- Samples: {data['n_samples']}")
            report.append(f"- Countries: {data['countries']}")
            report.append(f"- Topics: {data['topics']}")
        
        # Survey alignment
        if 'survey_alignment' in self.results:
            report.append("\n## Survey Alignment (Correlations)")
            for model, corrs in self.results['survey_alignment'].items():
                report.append(f"\n### {model}")
                report.append(f"- Log-probability: {corrs['logprob']:.3f}")
                report.append(f"- Direct scoring: {corrs['direct']:.3f}")
        
        # Conflicts
        if 'conflicts' in self.results:
            conflicts = self.results['conflicts']
            report.append("\n## Conflicts Detected")
            report.append(f"- Total: {conflicts['total']}")
            if 'by_severity' in conflicts:
                report.append("\nBy Severity:")
                for severity, count in conflicts['by_severity'].items():
                    report.append(f"- {severity}: {count}")
        
        # Peer review
        if 'peer_review' in self.results:
            pr = self.results['peer_review']
            report.append("\n## Peer Review Results")
            report.append(f"- Critiques: {pr['n_critiques']}")
            report.append(f"- Contentious cases: {pr['n_contentious']}")
            if 'agreement_rates' in pr:
                report.append("\nPeer-Agreement Rates:")
                for model, rate in pr['agreement_rates'].items():
                    report.append(f"- {model}: {rate:.2%}")
        
        # Human review
        if 'human_review' in self.results:
            hr = self.results['human_review']
            report.append("\n## Human Review")
            report.append(f"- Cases prepared: {hr['n_cases']}")
            report.append(f"- {hr['instruction']}")
        
        # Save report
        report_str = "\n".join(report)
        report_file = self.output_dir / "validation_summary.md"
        report_file.write_text(report_str)
        
        print("\n" + report_str)


def main():
    """Main execution"""
    
    parser = argparse.ArgumentParser(
        description="Run full validation pipeline with LLM judge and human review"
    )
    
    parser.add_argument(
        '--models',
        nargs='+',
        default=None,
        help='Models to test (default: use available)'
    )
    
    parser.add_argument(
        '--samples',
        type=int,
        default=10,
        help='Number of samples to test (default: 10)'
    )
    
    parser.add_argument(
        '--skip-peer-review',
        action='store_true',
        help='Skip reciprocal model critique'
    )
    
    parser.add_argument(
        '--skip-human-prep',
        action='store_true',
        help='Skip preparing data for human review'
    )
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = FullValidationPipeline()
    
    # Run validation
    results = pipeline.run_pipeline(
        models=args.models,
        n_samples=args.samples,
        run_peer_review=not args.skip_peer_review,
        save_for_human_review=not args.skip_human_prep
    )
    
    if results:
        logger.info("\n✅ Validation complete!")
        logger.info("Next steps:")
        logger.info("1. Review conflicts in conflicts.csv")
        logger.info("2. Check peer review results in peer_review/")
        logger.info("3. Run 'streamlit run human_dashboard.py' for human evaluation")
        logger.info("4. View visualizations in figures/")
        logger.info("5. Check paper outputs in paper/")
    else:
        logger.error("❌ Validation failed")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())