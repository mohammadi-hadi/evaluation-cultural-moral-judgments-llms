#!/usr/bin/env python3
"""
Analyze Quality of Local Model Results
Comprehensive assessment of response quality and patterns
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import re
from typing import Dict, List, Tuple

class LocalResultsAnalyzer:
    """Analyze quality of local model results"""
    
    def __init__(self, results_file: str):
        self.results_file = Path(results_file)
        self.results = self.load_results()
        self.df = pd.DataFrame(self.results)
        
    def load_results(self) -> List[Dict]:
        """Load local results"""
        with open(self.results_file, 'r') as f:
            return json.load(f)
    
    def analyze_response_patterns(self) -> Dict:
        """Analyze response patterns and quality issues"""
        print("🔍 ANALYZING LOCAL MODEL RESPONSE QUALITY")
        print("=" * 50)
        
        analysis = {
            'refusal_patterns': self.analyze_refusals(),
            'response_quality': self.analyze_response_structure(),
            'moral_choice_extraction': self.analyze_choice_extraction(),
            'model_comparison': self.compare_models(),
            'problematic_samples': self.identify_problematic_responses()
        }
        
        return analysis
    
    def analyze_refusals(self) -> Dict:
        """Analyze refusal patterns across models"""
        print("\n📋 REFUSAL PATTERN ANALYSIS")
        print("-" * 30)
        
        refusal_keywords = [
            "cannot provide", "can't provide", "i cannot", "unable to",
            "not appropriate", "don't feel comfortable", "cannot assist",
            "is there anything else", "something else", "help you with"
        ]
        
        refusal_stats = {}
        
        for model in self.df['model'].unique():
            model_data = self.df[self.df['model'] == model]
            
            refusals = 0
            null_choices = 0
            total = len(model_data)
            
            for _, row in model_data.iterrows():
                response = str(row['response']).lower()
                
                # Check for refusal patterns
                is_refusal = any(keyword in response for keyword in refusal_keywords)
                if is_refusal:
                    refusals += 1
                
                # Check for null choices
                if pd.isna(row['choice']) or row['choice'] is None:
                    null_choices += 1
            
            refusal_stats[model] = {
                'total_responses': total,
                'refusals': refusals,
                'null_choices': null_choices,
                'refusal_rate': refusals / total,
                'null_choice_rate': null_choices / total,
                'effective_responses': total - refusals - null_choices,
                'effective_rate': (total - refusals - null_choices) / total
            }
            
            print(f"\n{model}:")
            print(f"  Total responses: {total:,}")
            print(f"  Refusals: {refusals:,} ({refusals/total:.1%})")
            print(f"  Null choices: {null_choices:,} ({null_choices/total:.1%})")
            print(f"  Effective responses: {total - refusals - null_choices:,} ({(total - refusals - null_choices)/total:.1%})")
        
        return refusal_stats
    
    def analyze_response_structure(self) -> Dict:
        """Analyze response structure and formatting"""
        print("\n📋 RESPONSE STRUCTURE ANALYSIS")
        print("-" * 30)
        
        structure_stats = {}
        
        for model in self.df['model'].unique():
            model_data = self.df[self.df['model'] == model]
            
            # Analyze response patterns
            has_number = 0
            has_rating = 0
            has_reasoning = 0
            well_formatted = 0
            
            for _, row in model_data.iterrows():
                response = str(row['response'])
                
                # Check for numerical ratings (1-10)
                if re.search(r'\b([1-9]|10)\b', response):
                    has_number += 1
                
                # Check for explicit acceptable/unacceptable
                if re.search(r'(acceptable|unacceptable)', response.lower()):
                    has_rating += 1
                
                # Check for reasoning (longer responses)
                if len(response) > 50:
                    has_reasoning += 1
                
                # Check for well-formatted responses (has all components)
                if (re.search(r'\b([1-9]|10)\b', response) and 
                    re.search(r'(acceptable|unacceptable)', response.lower()) and 
                    len(response) > 30):
                    well_formatted += 1
            
            total = len(model_data)
            structure_stats[model] = {
                'has_number_rate': has_number / total,
                'has_rating_rate': has_rating / total,
                'has_reasoning_rate': has_reasoning / total,
                'well_formatted_rate': well_formatted / total
            }
            
            print(f"\n{model}:")
            print(f"  Has numerical rating: {has_number/total:.1%}")
            print(f"  Has explicit judgment: {has_rating/total:.1%}")
            print(f"  Has reasoning: {has_reasoning/total:.1%}")
            print(f"  Well formatted: {well_formatted/total:.1%}")
        
        return structure_stats
    
    def analyze_choice_extraction(self) -> Dict:
        """Analyze moral choice extraction accuracy"""
        print("\n📋 MORAL CHOICE EXTRACTION ANALYSIS")
        print("-" * 30)
        
        choice_stats = {}
        
        for model in self.df['model'].unique():
            model_data = self.df[self.df['model'] == model]
            
            choice_counts = model_data['choice'].value_counts()
            total = len(model_data)
            
            choice_stats[model] = {
                'choice_distribution': choice_counts.to_dict(),
                'extraction_success_rate': (total - choice_counts.get(None, 0) - choice_counts.get('unknown', 0)) / total,
                'total_responses': total
            }
            
            print(f"\n{model}:")
            for choice, count in choice_counts.items():
                print(f"  {choice}: {count:,} ({count/total:.1%})")
        
        return choice_stats
    
    def compare_models(self) -> Dict:
        """Compare model performance"""
        print("\n📋 MODEL COMPARISON")
        print("-" * 30)
        
        comparison = {}
        
        for model in self.df['model'].unique():
            model_data = self.df[self.df['model'] == model]
            
            # Calculate key metrics
            total = len(model_data)
            successful = len(model_data[model_data['choice'].notna() & (model_data['choice'] != 'unknown')])
            avg_response_length = model_data['response'].str.len().mean()
            avg_inference_time = model_data['inference_time'].mean()
            
            comparison[model] = {
                'total_samples': total,
                'successful_extractions': successful,
                'success_rate': successful / total,
                'avg_response_length': avg_response_length,
                'avg_inference_time': avg_inference_time,
                'cached_rate': model_data['cached'].mean() if 'cached' in model_data.columns else 0
            }
        
        # Sort by success rate
        sorted_models = sorted(comparison.items(), key=lambda x: x[1]['success_rate'], reverse=True)
        
        print("Ranking by Success Rate:")
        for i, (model, stats) in enumerate(sorted_models, 1):
            print(f"{i}. {model}:")
            print(f"   Success rate: {stats['success_rate']:.1%}")
            print(f"   Avg response length: {stats['avg_response_length']:.0f} chars")
            print(f"   Avg inference time: {stats['avg_inference_time']:.2f}s")
            print(f"   Cache hit rate: {stats['cached_rate']:.1%}")
        
        return comparison
    
    def identify_problematic_responses(self) -> List[Dict]:
        """Identify problematic responses for manual review"""
        print("\n📋 PROBLEMATIC RESPONSES SAMPLE")
        print("-" * 30)
        
        problematic = []
        
        # Find responses that are refusals or have extraction issues
        problem_responses = self.df[
            (self.df['choice'].isna()) | 
            (self.df['choice'] == 'unknown') |
            (self.df['response'].str.len() < 20)
        ]
        
        # Sample from each model
        sample_size = 3
        for model in problem_responses['model'].unique():
            model_problems = problem_responses[problem_responses['model'] == model].head(sample_size)
            
            print(f"\n{model} - Sample problematic responses:")
            for i, (_, row) in enumerate(model_problems.iterrows(), 1):
                print(f"  {i}. Sample: {row['sample_id']}")
                print(f"     Response: {row['response'][:100]}...")
                print(f"     Choice: {row['choice']}")
                
                problematic.append({
                    'model': row['model'],
                    'sample_id': row['sample_id'], 
                    'response': row['response'],
                    'choice': row['choice'],
                    'issue': 'refusal' if 'cannot' in str(row['response']).lower() else 'extraction_failed'
                })
        
        return problematic
    
    def generate_quality_summary(self) -> Dict:
        """Generate overall quality summary"""
        print("\n🎯 OVERALL QUALITY SUMMARY")
        print("=" * 30)
        
        total_responses = len(self.df)
        successful_extractions = len(self.df[self.df['choice'].notna() & (self.df['choice'] != 'unknown')])
        
        # Model rankings
        model_success_rates = {}
        for model in self.df['model'].unique():
            model_data = self.df[self.df['model'] == model]
            successful = len(model_data[model_data['choice'].notna() & (model_data['choice'] != 'unknown')])
            model_success_rates[model] = successful / len(model_data)
        
        best_model = max(model_success_rates.items(), key=lambda x: x[1])
        worst_model = min(model_success_rates.items(), key=lambda x: x[1])
        
        summary = {
            'total_responses': total_responses,
            'overall_success_rate': successful_extractions / total_responses,
            'best_performing_model': best_model,
            'worst_performing_model': worst_model,
            'model_success_rates': model_success_rates
        }
        
        print(f"📊 Total responses analyzed: {total_responses:,}")
        print(f"📊 Overall success rate: {successful_extractions/total_responses:.1%}")
        print(f"🏆 Best model: {best_model[0]} ({best_model[1]:.1%})")
        print(f"⚠️  Worst model: {worst_model[0]} ({worst_model[1]:.1%})")
        
        return summary

def main():
    """Main execution"""
    results_file = "outputs/server_sync_evaluation/run_20250902_165021/local/local_results.json"
    
    if not Path(results_file).exists():
        print(f"❌ Results file not found: {results_file}")
        return
    
    analyzer = LocalResultsAnalyzer(results_file)
    analysis = analyzer.analyze_response_patterns()
    summary = analyzer.generate_quality_summary()
    
    # Save analysis
    output_file = Path(results_file).parent / "quality_analysis.json"
    with open(output_file, 'w') as f:
        json.dump({
            'analysis': analysis,
            'summary': summary
        }, f, indent=2, default=str)
    
    print(f"\n💾 Detailed analysis saved to: {output_file}")

if __name__ == "__main__":
    main()