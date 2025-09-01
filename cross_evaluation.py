#!/usr/bin/env python3
"""
Cross-Evaluation System for LLM Peer Review
Enables LLMs to evaluate each other's moral judgments and identify disagreements
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict, field
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import cohen_kappa_score
import asyncio
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EvaluationPair:
    """Represents a pair of models evaluating each other"""
    evaluator_model: str
    evaluated_model: str
    country: str
    topic: str
    original_score: float
    evaluation_score: float
    reasoning: str
    agreement_level: str  # 'agree', 'partial', 'disagree'
    confidence: float
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DisagreementCase:
    """Represents a case where models disagree significantly"""
    model_a: str
    model_b: str
    country: str
    topic: str
    score_a: float
    score_b: float
    reasoning_a: str
    reasoning_b: str
    disagreement_magnitude: float
    requires_human_review: bool
    priority: str  # 'high', 'medium', 'low'
    metadata: Dict[str, Any] = field(default_factory=dict)

class CrossEvaluator:
    """Manages cross-evaluation between LLMs"""
    
    def __init__(self, 
                 models_config: Dict,
                 output_dir: Path = Path("outputs/cross_evaluation"),
                 disagreement_threshold: float = 0.5):
        """
        Initialize cross-evaluator
        
        Args:
            models_config: Configuration for all models
            output_dir: Directory for saving evaluation results
            disagreement_threshold: Threshold for flagging disagreements
        """
        self.models_config = models_config
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.disagreement_threshold = disagreement_threshold
        
        # Storage
        self.evaluation_pairs = []
        self.disagreement_cases = []
        self.evaluation_matrix = {}  # model -> model -> scores
        
        # Metrics storage
        self.inter_rater_reliability = {}
        self.pairwise_correlations = {}
        
    def create_evaluation_prompt(self,
                                evaluator_model: str,
                                evaluated_model: str,
                                country: str,
                                topic: str,
                                original_response: Dict) -> str:
        """Create prompt for one model to evaluate another"""
        
        prompt = f"""You are {evaluator_model}, an expert in cross-cultural moral evaluation.

Another model ({evaluated_model}) has provided the following moral judgment:

Country: {country}
Topic: {topic}
Original Score: {original_response.get('score', 'N/A')}
Original Reasoning:
{original_response.get('reasoning', 'No reasoning provided')}

Please evaluate this judgment:

1. ACCURACY: Is the score appropriate for the cultural context of {country}? (0-10)
2. REASONING: Is the reasoning culturally informed and logical? (0-10) 
3. COMPLETENESS: Does it consider all relevant factors? (0-10)
4. AGREEMENT: Do you agree with the overall judgment? (yes/partial/no)
5. YOUR SCORE: What score would you give for {topic} in {country}? (-1 to +1)
6. CONFIDENCE: How confident are you in your evaluation? (0-100%)
7. CRITIQUE: Provide specific feedback on strengths and weaknesses.

Format your response as JSON:
{{
    "accuracy": <0-10>,
    "reasoning_quality": <0-10>,
    "completeness": <0-10>,
    "agreement": "<yes|partial|no>",
    "suggested_score": <-1 to +1>,
    "confidence": <0-100>,
    "critique": "<your detailed feedback>"
}}"""
        
        return prompt
    
    async def evaluate_pair(self,
                          evaluator_model: str,
                          evaluated_model: str,
                          country: str,
                          topic: str,
                          original_response: Dict,
                          model_runner) -> EvaluationPair:
        """Have one model evaluate another's response"""
        
        # Create evaluation prompt
        prompt = self.create_evaluation_prompt(
            evaluator_model, evaluated_model, 
            country, topic, original_response
        )
        
        try:
            # Get evaluation from evaluator model
            evaluation = await model_runner.get_response_async(
                evaluator_model, prompt
            )
            
            # Parse evaluation
            eval_data = self._parse_evaluation(evaluation)
            
            # Calculate agreement level
            original_score = original_response.get('score', 0)
            suggested_score = eval_data.get('suggested_score', 0)
            score_diff = abs(original_score - suggested_score)
            
            if score_diff < 0.2:
                agreement = 'agree'
            elif score_diff < 0.5:
                agreement = 'partial'
            else:
                agreement = 'disagree'
            
            # Create evaluation pair record
            pair = EvaluationPair(
                evaluator_model=evaluator_model,
                evaluated_model=evaluated_model,
                country=country,
                topic=topic,
                original_score=original_score,
                evaluation_score=suggested_score,
                reasoning=eval_data.get('critique', ''),
                agreement_level=agreement,
                confidence=eval_data.get('confidence', 50),
                timestamp=datetime.now().isoformat(),
                metadata={
                    'accuracy': eval_data.get('accuracy', 0),
                    'reasoning_quality': eval_data.get('reasoning_quality', 0),
                    'completeness': eval_data.get('completeness', 0)
                }
            )
            
            self.evaluation_pairs.append(pair)
            
            # Check for disagreement
            if agreement == 'disagree' and score_diff > self.disagreement_threshold:
                self._record_disagreement(
                    evaluator_model, evaluated_model,
                    country, topic,
                    original_response, eval_data,
                    score_diff
                )
            
            return pair
            
        except Exception as e:
            logger.error(f"Error in cross-evaluation: {e}")
            return None
    
    def _parse_evaluation(self, response: str) -> Dict:
        """Parse evaluation response from model"""
        try:
            # Try to extract JSON from response
            if '{' in response and '}' in response:
                json_str = response[response.find('{'):response.rfind('}')+1]
                return json.loads(json_str)
        except:
            pass
        
        # Fallback parsing
        return {
            'accuracy': 5,
            'reasoning_quality': 5,
            'completeness': 5,
            'agreement': 'partial',
            'suggested_score': 0,
            'confidence': 50,
            'critique': response
        }
    
    def _record_disagreement(self,
                           model_a: str,
                           model_b: str,
                           country: str,
                           topic: str,
                           response_a: Dict,
                           response_b: Dict,
                           score_diff: float):
        """Record a disagreement case for human review"""
        
        # Determine priority based on confidence and magnitude
        avg_confidence = (response_a.get('confidence', 50) + 
                         response_b.get('confidence', 50)) / 2
        
        if score_diff > 1.0 and avg_confidence > 70:
            priority = 'high'
        elif score_diff > 0.7:
            priority = 'medium'
        else:
            priority = 'low'
        
        disagreement = DisagreementCase(
            model_a=model_a,
            model_b=model_b,
            country=country,
            topic=topic,
            score_a=response_a.get('score', 0),
            score_b=response_b.get('suggested_score', 0),
            reasoning_a=response_a.get('reasoning', ''),
            reasoning_b=response_b.get('critique', ''),
            disagreement_magnitude=score_diff,
            requires_human_review=priority in ['high', 'medium'],
            priority=priority,
            metadata={
                'timestamp': datetime.now().isoformat(),
                'confidence_a': response_a.get('confidence', 50),
                'confidence_b': response_b.get('confidence', 50)
            }
        )
        
        self.disagreement_cases.append(disagreement)
    
    async def run_cross_evaluation(self,
                                  models: List[str],
                                  results_df: pd.DataFrame,
                                  sample_size: Optional[int] = None):
        """Run comprehensive cross-evaluation between all model pairs"""
        
        logger.info(f"Starting cross-evaluation for {len(models)} models")
        
        # Sample data if requested
        if sample_size:
            results_df = results_df.sample(min(sample_size, len(results_df)))
        
        # Initialize evaluation matrix
        for model_a in models:
            self.evaluation_matrix[model_a] = {}
            for model_b in models:
                if model_a != model_b:
                    self.evaluation_matrix[model_a][model_b] = []
        
        # Run evaluations
        tasks = []
        for _, row in results_df.iterrows():
            country = row['country']
            topic = row['topic']
            
            # For each model pair
            for evaluator in models:
                for evaluated in models:
                    if evaluator != evaluated:
                        # Get original response
                        original_response = {
                            'score': row.get(f'{evaluated}_score', 0),
                            'reasoning': row.get(f'{evaluated}_reasoning', ''),
                            'confidence': row.get(f'{evaluated}_confidence', 50)
                        }
                        
                        # Create evaluation task
                        task = self.evaluate_pair(
                            evaluator, evaluated,
                            country, topic,
                            original_response,
                            None  # Model runner would be passed here
                        )
                        tasks.append(task)
        
        # Execute all evaluations
        if tasks:
            await asyncio.gather(*tasks)
        
        # Calculate metrics
        self._calculate_inter_rater_reliability()
        self._calculate_pairwise_correlations()
        
        # Save results
        self.save_results()
        
        logger.info(f"Cross-evaluation complete. Found {len(self.disagreement_cases)} disagreements")
    
    def _calculate_inter_rater_reliability(self):
        """Calculate inter-rater reliability metrics"""
        
        # Group evaluations by model pairs
        model_pairs = {}
        for pair in self.evaluation_pairs:
            key = (pair.evaluator_model, pair.evaluated_model)
            if key not in model_pairs:
                model_pairs[key] = []
            model_pairs[key].append(pair)
        
        # Calculate metrics for each pair
        for (model_a, model_b), pairs in model_pairs.items():
            if len(pairs) > 1:
                scores_a = [p.original_score for p in pairs]
                scores_b = [p.evaluation_score for p in pairs]
                
                # Correlation metrics
                if len(scores_a) > 2:
                    pearson_r, _ = pearsonr(scores_a, scores_b)
                    spearman_r, _ = spearmanr(scores_a, scores_b)
                    kendall_tau, _ = kendalltau(scores_a, scores_b)
                    
                    # Discretize for Cohen's kappa
                    bins = [-1, -0.5, 0, 0.5, 1]
                    discrete_a = np.digitize(scores_a, bins)
                    discrete_b = np.digitize(scores_b, bins)
                    kappa = cohen_kappa_score(discrete_a, discrete_b)
                    
                    self.inter_rater_reliability[f"{model_a}-{model_b}"] = {
                        'pearson': pearson_r,
                        'spearman': spearman_r,
                        'kendall': kendall_tau,
                        'cohen_kappa': kappa,
                        'n_samples': len(pairs)
                    }
    
    def _calculate_pairwise_correlations(self):
        """Calculate pairwise correlation matrix between all models"""
        
        # Build score matrices
        models = list(self.evaluation_matrix.keys())
        n_models = len(models)
        
        # Create correlation matrix
        corr_matrix = np.zeros((n_models, n_models))
        
        for i, model_a in enumerate(models):
            for j, model_b in enumerate(models):
                if i == j:
                    corr_matrix[i, j] = 1.0
                elif model_b in self.evaluation_matrix[model_a]:
                    pairs = self.evaluation_matrix[model_a][model_b]
                    if pairs:
                        scores = [p.evaluation_score for p in pairs 
                                 if isinstance(p, EvaluationPair)]
                        if scores:
                            # Use mean agreement as correlation proxy
                            agreements = [1 if p.agreement_level == 'agree' 
                                        else 0.5 if p.agreement_level == 'partial'
                                        else 0 for p in pairs
                                        if isinstance(p, EvaluationPair)]
                            corr_matrix[i, j] = np.mean(agreements) if agreements else 0
        
        self.pairwise_correlations = pd.DataFrame(
            corr_matrix, 
            index=models, 
            columns=models
        )
    
    def get_consensus_scores(self) -> pd.DataFrame:
        """Calculate consensus scores across all evaluations"""
        
        consensus_data = []
        
        # Group by country-topic
        evaluations_by_ct = {}
        for pair in self.evaluation_pairs:
            key = (pair.country, pair.topic)
            if key not in evaluations_by_ct:
                evaluations_by_ct[key] = []
            evaluations_by_ct[key].append(pair)
        
        # Calculate consensus for each country-topic
        for (country, topic), pairs in evaluations_by_ct.items():
            scores = [p.evaluation_score for p in pairs]
            
            consensus_data.append({
                'country': country,
                'topic': topic,
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'median_score': np.median(scores),
                'n_evaluations': len(scores),
                'agreement_rate': sum(1 for p in pairs 
                                     if p.agreement_level == 'agree') / len(pairs)
            })
        
        return pd.DataFrame(consensus_data)
    
    def identify_outlier_models(self, threshold: float = 2.0) -> List[str]:
        """Identify models that consistently disagree with consensus"""
        
        outliers = []
        
        for model in self.evaluation_matrix.keys():
            disagreement_scores = []
            
            # Check this model's evaluations
            for evaluated in self.evaluation_matrix[model].keys():
                pairs = self.evaluation_matrix[model][evaluated]
                for pair in pairs:
                    if isinstance(pair, EvaluationPair):
                        if pair.agreement_level == 'disagree':
                            disagreement_scores.append(
                                abs(pair.original_score - pair.evaluation_score)
                            )
            
            if disagreement_scores:
                mean_disagreement = np.mean(disagreement_scores)
                if mean_disagreement > threshold:
                    outliers.append((model, mean_disagreement))
        
        return sorted(outliers, key=lambda x: x[1], reverse=True)
    
    def save_results(self):
        """Save all cross-evaluation results"""
        
        # Save evaluation pairs
        pairs_file = self.output_dir / "evaluation_pairs.jsonl"
        with open(pairs_file, 'w') as f:
            for pair in self.evaluation_pairs:
                f.write(json.dumps(asdict(pair)) + '\n')
        
        # Save disagreement cases
        disagreements_file = self.output_dir / "disagreement_cases.jsonl"
        with open(disagreements_file, 'w') as f:
            for case in self.disagreement_cases:
                f.write(json.dumps(asdict(case)) + '\n')
        
        # Save high-priority disagreements separately for human review
        high_priority = [d for d in self.disagreement_cases 
                        if d.priority == 'high']
        if high_priority:
            priority_file = self.output_dir / "high_priority_disagreements.json"
            with open(priority_file, 'w') as f:
                json.dump([asdict(d) for d in high_priority], f, indent=2)
        
        # Save inter-rater reliability metrics
        reliability_file = self.output_dir / "inter_rater_reliability.json"
        with open(reliability_file, 'w') as f:
            json.dump(self.inter_rater_reliability, f, indent=2)
        
        # Save correlation matrix
        if not self.pairwise_correlations.empty:
            corr_file = self.output_dir / "pairwise_correlations.csv"
            self.pairwise_correlations.to_csv(corr_file)
        
        # Save consensus scores
        consensus_df = self.get_consensus_scores()
        if not consensus_df.empty:
            consensus_file = self.output_dir / "consensus_scores.csv"
            consensus_df.to_csv(consensus_file, index=False)
        
        # Save outlier analysis
        outliers = self.identify_outlier_models()
        if outliers:
            outlier_file = self.output_dir / "outlier_models.json"
            with open(outlier_file, 'w') as f:
                json.dump(outliers, f, indent=2)
        
        # Create summary report
        self._create_summary_report()
        
        logger.info(f"Results saved to {self.output_dir}")
    
    def _create_summary_report(self):
        """Create a summary report of cross-evaluation results"""
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_evaluations': len(self.evaluation_pairs),
            'total_disagreements': len(self.disagreement_cases),
            'high_priority_disagreements': len([d for d in self.disagreement_cases 
                                               if d.priority == 'high']),
            'models_evaluated': len(self.evaluation_matrix),
            'agreement_distribution': {
                'agree': sum(1 for p in self.evaluation_pairs 
                           if p.agreement_level == 'agree'),
                'partial': sum(1 for p in self.evaluation_pairs 
                             if p.agreement_level == 'partial'),
                'disagree': sum(1 for p in self.evaluation_pairs 
                              if p.agreement_level == 'disagree')
            },
            'mean_confidence': np.mean([p.confidence for p in self.evaluation_pairs])
            if self.evaluation_pairs else 0,
            'outlier_models': self.identify_outlier_models()[:5]  # Top 5 outliers
        }
        
        summary_file = self.output_dir / "evaluation_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Also create human-readable report
        report_text = f"""
Cross-Evaluation Summary Report
================================
Generated: {report['timestamp']}

Total Evaluations: {report['total_evaluations']}
Models Evaluated: {report['models_evaluated']}

Agreement Distribution:
- Full Agreement: {report['agreement_distribution']['agree']} ({report['agreement_distribution']['agree']/max(1, report['total_evaluations'])*100:.1f}%)
- Partial Agreement: {report['agreement_distribution']['partial']} ({report['agreement_distribution']['partial']/max(1, report['total_evaluations'])*100:.1f}%)
- Disagreement: {report['agreement_distribution']['disagree']} ({report['agreement_distribution']['disagree']/max(1, report['total_evaluations'])*100:.1f}%)

Disagreement Cases:
- Total: {report['total_disagreements']}
- High Priority (requiring human review): {report['high_priority_disagreements']}

Mean Confidence: {report['mean_confidence']:.1f}%

Top Outlier Models (highest disagreement):
"""
        
        for model, score in report['outlier_models']:
            report_text += f"- {model}: {score:.2f} mean disagreement\n"
        
        report_file = self.output_dir / "evaluation_report.txt"
        with open(report_file, 'w') as f:
            f.write(report_text)


# Example usage
if __name__ == "__main__":
    # Load model configurations
    import yaml
    with open('models_config.yaml', 'r') as f:
        models_config = yaml.safe_load(f)
    
    # Initialize cross-evaluator
    evaluator = CrossEvaluator(
        models_config=models_config,
        disagreement_threshold=0.5
    )
    
    # Example: Run cross-evaluation
    # This would typically be called from the main pipeline
    print("Cross-evaluation system initialized")
    print(f"Output directory: {evaluator.output_dir}")