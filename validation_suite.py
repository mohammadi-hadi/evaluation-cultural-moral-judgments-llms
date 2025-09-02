#!/usr/bin/env python3
"""
Validation Suite for Moral Alignment Pipeline
Multi-level validation including consistency, agreement, and statistical tests
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from scipy import stats
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import cohen_kappa_score, mean_absolute_error, mean_squared_error
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ValidationSuite:
    """Comprehensive validation for moral alignment results"""
    
    def __init__(self, output_dir: str = "outputs/validation"):
        """Initialize validation suite
        
        Args:
            output_dir: Directory for validation outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.validation_results = {}
        
    def validate_model_results(self, 
                              results: Dict,
                              model_name: str) -> Dict:
        """Validate results from a single model
        
        Args:
            results: Model results dictionary
            model_name: Name of the model
            
        Returns:
            Validation results
        """
        logger.info(f"Validating results for {model_name}")
        
        validation = {
            'model': model_name,
            'timestamp': datetime.now().isoformat(),
            'internal_consistency': {},
            'statistical_validity': {},
            'data_quality': {}
        }
        
        # Convert scores to DataFrame for analysis
        if 'scores' in results:
            scores_df = pd.DataFrame(results['scores'])
            
            # Internal consistency checks
            validation['internal_consistency'] = self._check_internal_consistency(scores_df)
            
            # Statistical validity
            validation['statistical_validity'] = self._check_statistical_validity(scores_df)
            
            # Data quality checks
            validation['data_quality'] = self._check_data_quality(scores_df)
        
        return validation
    
    def _check_internal_consistency(self, scores_df: pd.DataFrame) -> Dict:
        """Check internal consistency of model responses
        
        Args:
            scores_df: DataFrame with model scores
            
        Returns:
            Consistency metrics
        """
        consistency = {}
        
        # Check score distribution
        if 'model_score' in scores_df:
            scores = scores_df['model_score'].dropna()
            
            consistency['score_stats'] = {
                'mean': float(scores.mean()),
                'std': float(scores.std()),
                'min': float(scores.min()),
                'max': float(scores.max()),
                'median': float(scores.median())
            }
            
            # Check if scores are within valid range [-1, 1]
            out_of_range = ((scores < -1) | (scores > 1)).sum()
            consistency['scores_in_range'] = out_of_range == 0
            consistency['out_of_range_count'] = int(out_of_range)
            
            # Check for score clustering (potential bias)
            hist, bins = np.histogram(scores, bins=10)
            consistency['score_distribution'] = {
                'histogram': hist.tolist(),
                'bins': bins.tolist(),
                'skewness': float(stats.skew(scores)),
                'kurtosis': float(stats.kurtosis(scores))
            }
        
        # Check consistency by method
        if 'method' in scores_df:
            for method in scores_df['method'].unique():
                method_scores = scores_df[scores_df['method'] == method]['model_score'].dropna()
                consistency[f'{method}_variance'] = float(method_scores.var())
        
        # Check consistency by topic
        if 'topic' in scores_df:
            topic_consistency = {}
            for topic in scores_df['topic'].unique():
                topic_scores = scores_df[scores_df['topic'] == topic]['model_score'].dropna()
                if len(topic_scores) > 1:
                    topic_consistency[topic] = {
                        'std': float(topic_scores.std()),
                        'range': float(topic_scores.max() - topic_scores.min())
                    }
            consistency['topic_consistency'] = topic_consistency
        
        return consistency
    
    def _check_statistical_validity(self, scores_df: pd.DataFrame) -> Dict:
        """Check statistical validity of results
        
        Args:
            scores_df: DataFrame with model scores
            
        Returns:
            Statistical validity metrics
        """
        validity = {}
        
        if 'model_score' in scores_df and 'ground_truth' in scores_df:
            # Remove NaN values
            valid_data = scores_df[['model_score', 'ground_truth']].dropna()
            
            if len(valid_data) > 2:
                model_scores = valid_data['model_score']
                ground_truth = valid_data['ground_truth']
                
                # Correlation metrics
                pearson_r, pearson_p = pearsonr(model_scores, ground_truth)
                spearman_r, spearman_p = spearmanr(model_scores, ground_truth)
                kendall_tau, kendall_p = kendalltau(model_scores, ground_truth)
                
                validity['correlations'] = {
                    'pearson': {'r': float(pearson_r), 'p_value': float(pearson_p)},
                    'spearman': {'r': float(spearman_r), 'p_value': float(spearman_p)},
                    'kendall': {'tau': float(kendall_tau), 'p_value': float(kendall_p)}
                }
                
                # Error metrics
                validity['errors'] = {
                    'mae': float(mean_absolute_error(ground_truth, model_scores)),
                    'mse': float(mean_squared_error(ground_truth, model_scores)),
                    'rmse': float(np.sqrt(mean_squared_error(ground_truth, model_scores)))
                }
                
                # Bias detection
                residuals = model_scores - ground_truth
                validity['bias'] = {
                    'mean_bias': float(residuals.mean()),
                    'bias_std': float(residuals.std()),
                    'systematic_bias': abs(residuals.mean()) > 0.1  # Threshold for concern
                }
                
                # Statistical tests
                # Test if residuals are normally distributed
                shapiro_stat, shapiro_p = stats.shapiro(residuals[:min(5000, len(residuals))])
                validity['normality_test'] = {
                    'shapiro_stat': float(shapiro_stat),
                    'p_value': float(shapiro_p),
                    'is_normal': shapiro_p > 0.05
                }
        
        return validity
    
    def _check_data_quality(self, scores_df: pd.DataFrame) -> Dict:
        """Check data quality metrics
        
        Args:
            scores_df: DataFrame with model scores
            
        Returns:
            Data quality metrics
        """
        quality = {
            'total_samples': len(scores_df),
            'missing_values': {},
            'completeness': {}
        }
        
        # Check for missing values
        for col in scores_df.columns:
            missing_count = scores_df[col].isna().sum()
            if missing_count > 0:
                quality['missing_values'][col] = {
                    'count': int(missing_count),
                    'percentage': float(missing_count / len(scores_df) * 100)
                }
        
        # Completeness by category
        if 'country' in scores_df:
            quality['completeness']['countries'] = {
                'unique': int(scores_df['country'].nunique()),
                'coverage': list(scores_df['country'].unique())[:10]  # First 10 for brevity
            }
        
        if 'topic' in scores_df:
            quality['completeness']['topics'] = {
                'unique': int(scores_df['topic'].nunique()),
                'coverage': list(scores_df['topic'].unique())
            }
        
        # Check for duplicates
        quality['duplicates'] = {
            'count': int(scores_df.duplicated().sum()),
            'percentage': float(scores_df.duplicated().sum() / len(scores_df) * 100)
        }
        
        return quality
    
    def validate_cross_model_agreement(self, 
                                      all_results: Dict[str, Dict]) -> Dict:
        """Validate agreement between different models
        
        Args:
            all_results: Dictionary of results from all models
            
        Returns:
            Cross-model agreement metrics
        """
        logger.info("Validating cross-model agreement")
        
        agreement = {
            'pairwise_correlations': {},
            'consensus_metrics': {},
            'disagreement_analysis': {}
        }
        
        # Extract scores from all models
        model_scores = {}
        for model_name, results in all_results.items():
            if 'scores' in results:
                scores_df = pd.DataFrame(results['scores'])
                if 'model_score' in scores_df:
                    # Create unique identifier for each sample
                    scores_df['sample_id'] = scores_df['country'].astype(str) + '_' + scores_df['topic'].astype(str)
                    model_scores[model_name] = scores_df[['sample_id', 'model_score']].set_index('sample_id')
        
        if len(model_scores) > 1:
            # Merge all model scores
            merged_scores = pd.concat(model_scores, axis=1)
            merged_scores.columns = model_scores.keys()
            
            # Pairwise correlations
            models = list(model_scores.keys())
            for i, model1 in enumerate(models):
                for model2 in models[i+1:]:
                    valid_pairs = merged_scores[[model1, model2]].dropna()
                    if len(valid_pairs) > 2:
                        corr, p_value = pearsonr(valid_pairs[model1], valid_pairs[model2])
                        agreement['pairwise_correlations'][f"{model1}_vs_{model2}"] = {
                            'correlation': float(corr),
                            'p_value': float(p_value),
                            'n_samples': len(valid_pairs)
                        }
            
            # Consensus metrics
            valid_samples = merged_scores.dropna()
            if len(valid_samples) > 0:
                # Calculate mean and std across models for each sample
                consensus_scores = valid_samples.mean(axis=1)
                consensus_std = valid_samples.std(axis=1)
                
                agreement['consensus_metrics'] = {
                    'mean_consensus': float(consensus_scores.mean()),
                    'std_consensus': float(consensus_scores.std()),
                    'mean_disagreement': float(consensus_std.mean()),
                    'max_disagreement': float(consensus_std.max())
                }
                
                # Identify samples with high disagreement
                high_disagreement_threshold = consensus_std.quantile(0.9)
                high_disagreement_samples = consensus_std[consensus_std > high_disagreement_threshold]
                
                agreement['disagreement_analysis'] = {
                    'threshold': float(high_disagreement_threshold),
                    'n_high_disagreement': len(high_disagreement_samples),
                    'samples': high_disagreement_samples.index.tolist()[:10]  # First 10
                }
        
        return agreement
    
    def validate_human_alignment(self, 
                                model_results: Dict,
                                human_baseline: Dict) -> Dict:
        """Validate alignment with human survey data
        
        Args:
            model_results: Model results
            human_baseline: Human baseline statistics
            
        Returns:
            Human alignment metrics
        """
        logger.info("Validating human alignment")
        
        alignment = {
            'overall_alignment': {},
            'topic_alignment': {},
            'country_alignment': {},
            'regional_alignment': {}
        }
        
        if 'scores' in model_results:
            scores_df = pd.DataFrame(model_results['scores'])
            
            # Overall alignment
            if 'model_score' in scores_df and 'ground_truth' in scores_df:
                valid_scores = scores_df[['model_score', 'ground_truth']].dropna()
                if len(valid_scores) > 0:
                    corr, p_value = pearsonr(valid_scores['model_score'], valid_scores['ground_truth'])
                    alignment['overall_alignment'] = {
                        'correlation': float(corr),
                        'p_value': float(p_value),
                        'n_samples': len(valid_scores),
                        'significant': p_value < 0.05
                    }
            
            # Topic-wise alignment
            if 'topic' in scores_df and 'by_topic' in human_baseline:
                topic_correlations = {}
                for topic in scores_df['topic'].unique():
                    topic_data = scores_df[scores_df['topic'] == topic]
                    if len(topic_data) > 2 and topic in human_baseline['by_topic']:
                        model_mean = topic_data['model_score'].mean()
                        human_mean = human_baseline['by_topic'][topic]['mean']
                        topic_correlations[topic] = {
                            'model_mean': float(model_mean),
                            'human_mean': float(human_mean),
                            'difference': float(abs(model_mean - human_mean))
                        }
                alignment['topic_alignment'] = topic_correlations
            
            # Country-wise alignment
            if 'country' in scores_df and 'by_country' in human_baseline:
                country_correlations = {}
                for country in scores_df['country'].unique()[:10]:  # Top 10 countries
                    country_data = scores_df[scores_df['country'] == country]
                    if len(country_data) > 2 and country in human_baseline['by_country']:
                        valid_data = country_data[['model_score', 'ground_truth']].dropna()
                        if len(valid_data) > 2:
                            corr, _ = pearsonr(valid_data['model_score'], valid_data['ground_truth'])
                            country_correlations[country] = float(corr)
                alignment['country_alignment'] = country_correlations
        
        return alignment
    
    def perform_statistical_tests(self, 
                                 model1_results: Dict,
                                 model2_results: Dict) -> Dict:
        """Perform statistical tests between two models
        
        Args:
            model1_results: Results from first model
            model2_results: Results from second model
            
        Returns:
            Statistical test results
        """
        tests = {}
        
        # Extract scores
        if 'scores' in model1_results and 'scores' in model2_results:
            scores1_df = pd.DataFrame(model1_results['scores'])
            scores2_df = pd.DataFrame(model2_results['scores'])
            
            # Match samples
            if 'country' in scores1_df and 'topic' in scores1_df:
                scores1_df['sample_id'] = scores1_df['country'].astype(str) + '_' + scores1_df['topic'].astype(str)
                scores2_df['sample_id'] = scores2_df['country'].astype(str) + '_' + scores2_df['topic'].astype(str)
                
                merged = pd.merge(
                    scores1_df[['sample_id', 'model_score', 'ground_truth']],
                    scores2_df[['sample_id', 'model_score']],
                    on='sample_id',
                    suffixes=('_1', '_2')
                )
                
                if len(merged) > 0:
                    # Paired t-test on errors
                    errors1 = abs(merged['model_score_1'] - merged['ground_truth'])
                    errors2 = abs(merged['model_score_2'] - merged['ground_truth'])
                    
                    t_stat, p_value = stats.ttest_rel(errors1, errors2)
                    tests['paired_t_test'] = {
                        't_statistic': float(t_stat),
                        'p_value': float(p_value),
                        'significant': p_value < 0.05,
                        'better_model': 'model1' if errors1.mean() < errors2.mean() else 'model2'
                    }
                    
                    # Wilcoxon signed-rank test (non-parametric alternative)
                    w_stat, w_p_value = stats.wilcoxon(errors1, errors2)
                    tests['wilcoxon_test'] = {
                        'statistic': float(w_stat),
                        'p_value': float(w_p_value),
                        'significant': w_p_value < 0.05
                    }
                    
                    # Effect size (Cohen's d)
                    pooled_std = np.sqrt((errors1.std()**2 + errors2.std()**2) / 2)
                    cohens_d = (errors1.mean() - errors2.mean()) / pooled_std if pooled_std > 0 else 0
                    tests['effect_size'] = {
                        'cohens_d': float(cohens_d),
                        'interpretation': self._interpret_cohens_d(cohens_d)
                    }
        
        return tests
    
    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size
        
        Args:
            d: Cohen's d value
            
        Returns:
            Interpretation string
        """
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def generate_validation_report(self, 
                                  all_validations: Dict) -> str:
        """Generate comprehensive validation report
        
        Args:
            all_validations: All validation results
            
        Returns:
            Markdown report
        """
        report = []
        report.append("# Validation Report")
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Model-specific validations
        if 'model_validations' in all_validations:
            report.append("\n## Model Validations")
            
            for model, validation in all_validations['model_validations'].items():
                report.append(f"\n### {model}")
                
                # Internal consistency
                if 'internal_consistency' in validation:
                    ic = validation['internal_consistency']
                    if 'score_stats' in ic:
                        stats = ic['score_stats']
                        report.append(f"\n**Score Statistics:**")
                        report.append(f"- Mean: {stats['mean']:.3f}")
                        report.append(f"- Std: {stats['std']:.3f}")
                        report.append(f"- Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
                
                # Statistical validity
                if 'statistical_validity' in validation:
                    sv = validation['statistical_validity']
                    if 'correlations' in sv:
                        corr = sv['correlations']
                        report.append(f"\n**Correlations with Ground Truth:**")
                        report.append(f"- Pearson r: {corr['pearson']['r']:.3f} (p={corr['pearson']['p_value']:.4f})")
                        report.append(f"- Spearman ρ: {corr['spearman']['r']:.3f}")
                    
                    if 'errors' in sv:
                        errors = sv['errors']
                        report.append(f"\n**Error Metrics:**")
                        report.append(f"- MAE: {errors['mae']:.3f}")
                        report.append(f"- RMSE: {errors['rmse']:.3f}")
        
        # Cross-model agreement
        if 'cross_model_agreement' in all_validations:
            report.append("\n## Cross-Model Agreement")
            cma = all_validations['cross_model_agreement']
            
            if 'pairwise_correlations' in cma:
                report.append("\n**Pairwise Correlations:**")
                for pair, metrics in cma['pairwise_correlations'].items():
                    report.append(f"- {pair}: r={metrics['correlation']:.3f}")
        
        # Human alignment
        if 'human_alignment' in all_validations:
            report.append("\n## Human Alignment")
            ha = all_validations['human_alignment']
            
            if 'overall_alignment' in ha:
                oa = ha['overall_alignment']
                report.append(f"\n**Overall Alignment:**")
                report.append(f"- Correlation: {oa['correlation']:.3f}")
                report.append(f"- Significant: {oa['significant']}")
        
        return "\n".join(report)
    
    def save_validation_results(self, all_validations: Dict):
        """Save all validation results
        
        Args:
            all_validations: All validation results
        """
        # Save JSON
        json_file = self.output_dir / f"validation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_file, 'w') as f:
            json.dump(all_validations, f, indent=2, default=str)
        logger.info(f"Saved validation results to {json_file}")
        
        # Generate and save report
        report = self.generate_validation_report(all_validations)
        report_file = self.output_dir / f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        report_file.write_text(report)
        logger.info(f"Saved validation report to {report_file}")


def main():
    """Test validation suite"""
    logging.basicConfig(level=logging.INFO)
    
    validator = ValidationSuite()
    
    # Example validation (would normally load actual results)
    logger.info("Validation suite ready for use")
    
    print("\n✅ Validation suite initialized")
    print("Ready to validate model results")
    
    return validator


if __name__ == "__main__":
    main()