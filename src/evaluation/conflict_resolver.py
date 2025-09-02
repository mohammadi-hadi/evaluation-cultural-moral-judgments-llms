#!/usr/bin/env python3
"""
Conflict Resolution System for Model Disagreements
Advanced detection and resolution strategies for moral judgment conflicts
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import json
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Conflict:
    """Represents a conflict between model judgments"""
    conflict_id: str
    models: List[str]
    country: str
    topic: str
    scores: List[float]
    reasonings: List[str]
    conflict_type: str  # 'binary', 'gradient', 'multimodal'
    severity: str  # 'low', 'medium', 'high', 'critical'
    resolution_status: str  # 'unresolved', 'auto_resolved', 'human_resolved'
    resolution_method: Optional[str] = None
    resolved_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class ConflictResolver:
    """Advanced conflict detection and resolution system"""
    
    def __init__(self, 
                 threshold_binary: float = 1.0,
                 threshold_gradient: float = 0.5,
                 threshold_multimodal: float = 0.3):
        """
        Initialize conflict resolver
        
        Args:
            threshold_binary: Threshold for binary conflicts (opposite signs)
            threshold_gradient: Threshold for gradient conflicts  
            threshold_multimodal: Threshold for multimodal conflicts
        """
        self.threshold_binary = threshold_binary
        self.threshold_gradient = threshold_gradient
        self.threshold_multimodal = threshold_multimodal
        
        self.conflicts = []
        self.resolution_strategies = {
            'consensus': self._resolve_by_consensus,
            'weighted_average': self._resolve_by_weighted_average,
            'confidence_weighted': self._resolve_by_confidence,
            'expertise_weighted': self._resolve_by_expertise,
            'cultural_context': self._resolve_by_cultural_context,
            'outlier_removal': self._resolve_by_outlier_removal,
            'clustering': self._resolve_by_clustering
        }
        
    def detect_conflicts(self, 
                        results_df: pd.DataFrame,
                        models: List[str]) -> List[Conflict]:
        """
        Detect conflicts in model judgments
        
        Args:
            results_df: DataFrame with model results
            models: List of model names to compare
            
        Returns:
            List of detected conflicts
        """
        conflicts = []
        
        # Group by country-topic pairs
        for (country, topic), group in results_df.groupby(['country', 'topic']):
            scores = []
            reasonings = []
            
            for model in models:
                score_col = f"{model}_score"
                reasoning_col = f"{model}_reasoning"
                
                if score_col in group.columns:
                    score = group[score_col].values[0] if len(group) > 0 else None
                    reasoning = group[reasoning_col].values[0] if reasoning_col in group.columns else ""
                    
                    if score is not None:
                        scores.append(score)
                        reasonings.append(reasoning)
            
            if len(scores) >= 2:
                conflict = self._analyze_conflict(
                    models[:len(scores)], 
                    country, 
                    topic, 
                    scores, 
                    reasonings
                )
                
                if conflict:
                    conflicts.append(conflict)
        
        self.conflicts = conflicts
        return conflicts
    
    def _analyze_conflict(self,
                         models: List[str],
                         country: str,
                         topic: str,
                         scores: List[float],
                         reasonings: List[str]) -> Optional[Conflict]:
        """Analyze potential conflict between model scores"""
        
        scores_array = np.array(scores)
        
        # Check for binary conflict (opposite signs)
        if self._is_binary_conflict(scores_array):
            return Conflict(
                conflict_id=self._generate_conflict_id(country, topic),
                models=models,
                country=country,
                topic=topic,
                scores=scores,
                reasonings=reasonings,
                conflict_type='binary',
                severity=self._calculate_severity(scores_array, 'binary'),
                resolution_status='unresolved',
                metadata={
                    'positive_models': [m for m, s in zip(models, scores) if s > 0],
                    'negative_models': [m for m, s in zip(models, scores) if s < 0]
                }
            )
        
        # Check for gradient conflict (large spread)
        elif self._is_gradient_conflict(scores_array):
            return Conflict(
                conflict_id=self._generate_conflict_id(country, topic),
                models=models,
                country=country,
                topic=topic,
                scores=scores,
                reasonings=reasonings,
                conflict_type='gradient',
                severity=self._calculate_severity(scores_array, 'gradient'),
                resolution_status='unresolved',
                metadata={
                    'score_range': float(np.ptp(scores_array)),
                    'std_dev': float(np.std(scores_array))
                }
            )
        
        # Check for multimodal conflict (clustering)
        elif self._is_multimodal_conflict(scores_array):
            clusters = self._identify_clusters(scores_array)
            return Conflict(
                conflict_id=self._generate_conflict_id(country, topic),
                models=models,
                country=country,
                topic=topic,
                scores=scores,
                reasonings=reasonings,
                conflict_type='multimodal',
                severity=self._calculate_severity(scores_array, 'multimodal'),
                resolution_status='unresolved',
                metadata={
                    'clusters': clusters,
                    'n_clusters': len(set(clusters))
                }
            )
        
        return None
    
    def _is_binary_conflict(self, scores: np.ndarray) -> bool:
        """Check if scores show binary conflict (opposite signs)"""
        has_positive = np.any(scores > 0.1)
        has_negative = np.any(scores < -0.1)
        
        if has_positive and has_negative:
            # Check if the difference is significant
            max_positive = np.max(scores[scores > 0])
            min_negative = np.min(scores[scores < 0])
            return (max_positive - min_negative) >= self.threshold_binary
        
        return False
    
    def _is_gradient_conflict(self, scores: np.ndarray) -> bool:
        """Check if scores show gradient conflict (large spread)"""
        score_range = np.ptp(scores)  # Peak to peak range
        return score_range >= self.threshold_gradient
    
    def _is_multimodal_conflict(self, scores: np.ndarray) -> bool:
        """Check if scores show multimodal distribution (clustering)"""
        if len(scores) < 3:
            return False
        
        # Use simple clustering to detect multimodality
        scores_reshaped = scores.reshape(-1, 1)
        
        # Try to find 2 clusters
        if len(scores) >= 2:
            kmeans = KMeans(n_clusters=min(2, len(scores)), random_state=42)
            labels = kmeans.fit_predict(scores_reshaped)
            
            # Check if clusters are well-separated
            cluster_centers = []
            for label in set(labels):
                cluster_scores = scores[labels == label]
                cluster_centers.append(np.mean(cluster_scores))
            
            if len(cluster_centers) >= 2:
                cluster_distance = np.abs(cluster_centers[0] - cluster_centers[1])
                return cluster_distance >= self.threshold_multimodal
        
        return False
    
    def _identify_clusters(self, scores: np.ndarray) -> List[int]:
        """Identify cluster assignments for scores"""
        if len(scores) < 2:
            return [0] * len(scores)
        
        scores_reshaped = scores.reshape(-1, 1)
        kmeans = KMeans(n_clusters=min(2, len(scores)), random_state=42)
        labels = kmeans.fit_predict(scores_reshaped)
        
        return labels.tolist()
    
    def _calculate_severity(self, scores: np.ndarray, conflict_type: str) -> str:
        """Calculate conflict severity"""
        if conflict_type == 'binary':
            # For binary conflicts, severity based on extremity
            max_abs = np.max(np.abs(scores))
            if max_abs > 0.8:
                return 'critical'
            elif max_abs > 0.6:
                return 'high'
            elif max_abs > 0.4:
                return 'medium'
            else:
                return 'low'
        
        elif conflict_type == 'gradient':
            # For gradient conflicts, severity based on spread
            score_range = np.ptp(scores)
            if score_range > 1.5:
                return 'critical'
            elif score_range > 1.0:
                return 'high'
            elif score_range > 0.7:
                return 'medium'
            else:
                return 'low'
        
        elif conflict_type == 'multimodal':
            # For multimodal conflicts, severity based on cluster separation
            clusters = self._identify_clusters(scores)
            cluster_means = []
            for label in set(clusters):
                cluster_scores = scores[np.array(clusters) == label]
                cluster_means.append(np.mean(cluster_scores))
            
            if len(cluster_means) >= 2:
                separation = np.abs(cluster_means[0] - cluster_means[1])
                if separation > 1.0:
                    return 'critical'
                elif separation > 0.7:
                    return 'high'
                elif separation > 0.5:
                    return 'medium'
            
            return 'low'
        
        return 'low'
    
    def _generate_conflict_id(self, country: str, topic: str) -> str:
        """Generate unique conflict ID"""
        import hashlib
        content = f"{country}_{topic}_{datetime.now().isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def resolve_conflicts(self, 
                         conflicts: List[Conflict],
                         strategy: str = 'consensus',
                         confidence_scores: Optional[Dict[str, float]] = None) -> List[Conflict]:
        """
        Resolve conflicts using specified strategy
        
        Args:
            conflicts: List of conflicts to resolve
            strategy: Resolution strategy to use
            confidence_scores: Optional confidence scores for models
            
        Returns:
            List of conflicts with resolution attempts
        """
        if strategy not in self.resolution_strategies:
            logger.warning(f"Unknown strategy {strategy}, using consensus")
            strategy = 'consensus'
        
        resolved_conflicts = []
        resolution_func = self.resolution_strategies[strategy]
        
        for conflict in conflicts:
            resolved = resolution_func(conflict, confidence_scores)
            resolved_conflicts.append(resolved)
        
        return resolved_conflicts
    
    def _resolve_by_consensus(self, 
                            conflict: Conflict,
                            confidence_scores: Optional[Dict[str, float]] = None) -> Conflict:
        """Resolve by finding consensus (median)"""
        scores = np.array(conflict.scores)
        resolved_score = float(np.median(scores))
        
        conflict.resolution_method = 'consensus_median'
        conflict.resolved_score = resolved_score
        conflict.resolution_status = 'auto_resolved'
        
        return conflict
    
    def _resolve_by_weighted_average(self,
                                    conflict: Conflict,
                                    confidence_scores: Optional[Dict[str, float]] = None) -> Conflict:
        """Resolve by weighted average"""
        scores = np.array(conflict.scores)
        
        if confidence_scores:
            weights = [confidence_scores.get(model, 1.0) for model in conflict.models]
            weights = np.array(weights) / np.sum(weights)
            resolved_score = float(np.average(scores, weights=weights))
        else:
            resolved_score = float(np.mean(scores))
        
        conflict.resolution_method = 'weighted_average'
        conflict.resolved_score = resolved_score
        conflict.resolution_status = 'auto_resolved'
        
        return conflict
    
    def _resolve_by_confidence(self,
                              conflict: Conflict,
                              confidence_scores: Optional[Dict[str, float]] = None) -> Conflict:
        """Resolve by selecting highest confidence model"""
        if not confidence_scores:
            return self._resolve_by_consensus(conflict, confidence_scores)
        
        confidences = [confidence_scores.get(model, 0) for model in conflict.models]
        max_conf_idx = np.argmax(confidences)
        resolved_score = conflict.scores[max_conf_idx]
        
        conflict.resolution_method = f'highest_confidence_{conflict.models[max_conf_idx]}'
        conflict.resolved_score = resolved_score
        conflict.resolution_status = 'auto_resolved'
        
        return conflict
    
    def _resolve_by_expertise(self,
                             conflict: Conflict,
                             confidence_scores: Optional[Dict[str, float]] = None) -> Conflict:
        """Resolve by model expertise in specific domain"""
        # This would require domain-specific expertise scores
        # For now, use model size/capability as proxy
        
        expertise_scores = {
            'gpt-4o': 0.95,
            'claude-3.5-sonnet': 0.94,
            'gemini-1.5-pro': 0.93,
            'gpt-4-turbo': 0.92,
            'llama-3.3-70b-instruct': 0.85,
            'mistral-large-2': 0.84
        }
        
        model_expertise = [expertise_scores.get(model, 0.5) for model in conflict.models]
        weights = np.array(model_expertise) / np.sum(model_expertise)
        resolved_score = float(np.average(conflict.scores, weights=weights))
        
        conflict.resolution_method = 'expertise_weighted'
        conflict.resolved_score = resolved_score
        conflict.resolution_status = 'auto_resolved'
        
        return conflict
    
    def _resolve_by_cultural_context(self,
                                    conflict: Conflict,
                                    confidence_scores: Optional[Dict[str, float]] = None) -> Conflict:
        """Resolve based on cultural context knowledge"""
        # This would require cultural expertise data
        # For demonstration, use regional model preferences
        
        regional_preferences = {
            'United States': {'gpt-4o': 1.2, 'claude-3.5-sonnet': 1.1},
            'China': {'qwen-72b': 1.3, 'gemini-1.5-pro': 1.0},
            'Germany': {'mistral-large-2': 1.1, 'claude-3.5-sonnet': 1.2},
            'Japan': {'gpt-4o': 1.1, 'gemini-1.5-pro': 1.2}
        }
        
        country_prefs = regional_preferences.get(conflict.country, {})
        weights = [country_prefs.get(model, 1.0) for model in conflict.models]
        weights = np.array(weights) / np.sum(weights)
        
        resolved_score = float(np.average(conflict.scores, weights=weights))
        
        conflict.resolution_method = 'cultural_context_weighted'
        conflict.resolved_score = resolved_score
        conflict.resolution_status = 'auto_resolved'
        
        return conflict
    
    def _resolve_by_outlier_removal(self,
                                   conflict: Conflict,
                                   confidence_scores: Optional[Dict[str, float]] = None) -> Conflict:
        """Resolve by removing outliers"""
        scores = np.array(conflict.scores)
        
        # Use IQR method for outlier detection
        Q1 = np.percentile(scores, 25)
        Q3 = np.percentile(scores, 75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        filtered_scores = scores[(scores >= lower_bound) & (scores <= upper_bound)]
        
        if len(filtered_scores) > 0:
            resolved_score = float(np.mean(filtered_scores))
        else:
            resolved_score = float(np.median(scores))
        
        conflict.resolution_method = 'outlier_removal'
        conflict.resolved_score = resolved_score
        conflict.resolution_status = 'auto_resolved'
        conflict.metadata['outliers_removed'] = len(scores) - len(filtered_scores)
        
        return conflict
    
    def _resolve_by_clustering(self,
                              conflict: Conflict,
                              confidence_scores: Optional[Dict[str, float]] = None) -> Conflict:
        """Resolve by identifying main cluster"""
        scores = np.array(conflict.scores)
        
        if len(scores) < 3:
            return self._resolve_by_consensus(conflict, confidence_scores)
        
        # Use DBSCAN for robust clustering
        scores_reshaped = scores.reshape(-1, 1)
        clustering = DBSCAN(eps=0.3, min_samples=2).fit(scores_reshaped)
        labels = clustering.labels_
        
        # Find largest cluster (excluding noise points labeled as -1)
        unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
        
        if len(unique_labels) > 0:
            main_cluster_label = unique_labels[np.argmax(counts)]
            main_cluster_scores = scores[labels == main_cluster_label]
            resolved_score = float(np.mean(main_cluster_scores))
        else:
            resolved_score = float(np.median(scores))
        
        conflict.resolution_method = 'main_cluster_consensus'
        conflict.resolved_score = resolved_score
        conflict.resolution_status = 'auto_resolved'
        conflict.metadata['n_clusters'] = len(unique_labels)
        
        return conflict
    
    def analyze_conflict_patterns(self, conflicts: List[Conflict]) -> Dict:
        """Analyze patterns in conflicts"""
        
        if not conflicts:
            return {}
        
        analysis = {
            'total_conflicts': len(conflicts),
            'by_type': {},
            'by_severity': {},
            'by_country': {},
            'by_topic': {},
            'resolution_success_rate': 0,
            'common_disagreement_pairs': []
        }
        
        # Count by type
        for conflict in conflicts:
            conflict_type = conflict.conflict_type
            if conflict_type not in analysis['by_type']:
                analysis['by_type'][conflict_type] = 0
            analysis['by_type'][conflict_type] += 1
            
            # Count by severity
            severity = conflict.severity
            if severity not in analysis['by_severity']:
                analysis['by_severity'][severity] = 0
            analysis['by_severity'][severity] += 1
            
            # Count by country
            country = conflict.country
            if country not in analysis['by_country']:
                analysis['by_country'][country] = 0
            analysis['by_country'][country] += 1
            
            # Count by topic
            topic = conflict.topic
            if topic not in analysis['by_topic']:
                analysis['by_topic'][topic] = 0
            analysis['by_topic'][topic] += 1
        
        # Calculate resolution success rate
        resolved = sum(1 for c in conflicts if c.resolution_status != 'unresolved')
        analysis['resolution_success_rate'] = resolved / len(conflicts) if conflicts else 0
        
        # Find common disagreement pairs
        model_pairs = {}
        for conflict in conflicts:
            for i, model1 in enumerate(conflict.models):
                for model2 in conflict.models[i+1:]:
                    pair = tuple(sorted([model1, model2]))
                    if pair not in model_pairs:
                        model_pairs[pair] = 0
                    model_pairs[pair] += 1
        
        # Sort by frequency
        analysis['common_disagreement_pairs'] = sorted(
            model_pairs.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]
        
        return analysis
    
    def export_conflicts(self, 
                        conflicts: List[Conflict],
                        output_path: Path):
        """Export conflicts to file"""
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to serializable format
        conflicts_data = []
        for conflict in conflicts:
            conflict_dict = {
                'conflict_id': conflict.conflict_id,
                'models': conflict.models,
                'country': conflict.country,
                'topic': conflict.topic,
                'scores': conflict.scores,
                'conflict_type': conflict.conflict_type,
                'severity': conflict.severity,
                'resolution_status': conflict.resolution_status,
                'resolution_method': conflict.resolution_method,
                'resolved_score': conflict.resolved_score,
                'metadata': conflict.metadata
            }
            conflicts_data.append(conflict_dict)
        
        # Save as JSON
        with open(output_path, 'w') as f:
            json.dump(conflicts_data, f, indent=2)
        
        # Also save analysis
        analysis = self.analyze_conflict_patterns(conflicts)
        analysis_path = output_path.with_suffix('.analysis.json')
        
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        logger.info(f"Exported {len(conflicts)} conflicts to {output_path}")
        
        return output_path, analysis_path


# Example usage
if __name__ == "__main__":
    # Initialize resolver
    resolver = ConflictResolver()
    
    # Example conflicts
    example_conflict = Conflict(
        conflict_id="test_001",
        models=['gpt-4o', 'claude-3.5-sonnet', 'gemini-1.5-pro'],
        country='United States',
        topic='abortion',
        scores=[0.3, -0.4, 0.1],
        reasonings=['Complex issue...', 'Controversial...', 'Depends on...'],
        conflict_type='binary',
        severity='high',
        resolution_status='unresolved'
    )
    
    # Resolve conflict
    resolved = resolver._resolve_by_consensus(example_conflict)
    print(f"Resolved score: {resolved.resolved_score}")
    
    # Analyze patterns
    analysis = resolver.analyze_conflict_patterns([example_conflict])
    print(f"Conflict analysis: {analysis}")
    
    print("Conflict resolver initialized successfully")