#!/usr/bin/env python3
"""
Output Generator for Moral Alignment Pipeline
Aggregates results from separate model runs and generates various output formats
"""

import os
import sys
import json
import csv
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import logging
from datetime import datetime
import pandas as pd
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)

class OutputGenerator:
    """Generates various output formats from model results"""
    
    def __init__(self, results_dir: str = "outputs"):
        """Initialize output generator
        
        Args:
            results_dir: Directory containing model results
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for different output types
        self.tables_dir = self.results_dir / "tables"
        self.reports_dir = self.results_dir / "reports"
        self.exports_dir = self.results_dir / "exports"
        
        for dir in [self.tables_dir, self.reports_dir, self.exports_dir]:
            dir.mkdir(parents=True, exist_ok=True)
        
        self.aggregated_results = {}
        self.model_results = defaultdict(list)
    
    def load_results(self, pattern: str = "*.json") -> Dict:
        """Load all results matching pattern
        
        Args:
            pattern: File pattern to match
            
        Returns:
            Dictionary of loaded results
        """
        logger.info(f"Loading results from {self.results_dir}")
        
        results = {}
        for file_path in self.results_dir.rglob(pattern):
            logger.info(f"Loading: {file_path.name}")
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    results[file_path.stem] = data
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
        
        logger.info(f"Loaded {len(results)} result files")
        return results
    
    def aggregate_model_results(self, results: Dict) -> pd.DataFrame:
        """Aggregate results from multiple model runs
        
        Args:
            results: Dictionary of results from different runs
            
        Returns:
            Aggregated DataFrame
        """
        all_data = []
        
        for run_name, run_data in results.items():
            if isinstance(run_data, list):
                for item in run_data:
                    if isinstance(item, dict):
                        item['run'] = run_name
                        all_data.append(item)
            elif isinstance(run_data, dict):
                if 'results' in run_data:
                    for item in run_data['results']:
                        if isinstance(item, dict):
                            item['run'] = run_name
                            all_data.append(item)
                else:
                    run_data['run'] = run_name
                    all_data.append(run_data)
        
        if all_data:
            df = pd.DataFrame(all_data)
            logger.info(f"Aggregated {len(df)} total results")
            return df
        else:
            logger.warning("No data to aggregate")
            return pd.DataFrame()
    
    def generate_summary_statistics(self, df: pd.DataFrame) -> Dict:
        """Generate summary statistics from results
        
        Args:
            df: Results DataFrame
            
        Returns:
            Summary statistics dictionary
        """
        if df.empty:
            return {}
        
        summary = {
            'total_samples': len(df),
            'models_tested': [],
            'scenarios_tested': [],
            'performance_metrics': {},
            'response_patterns': {},
            'agreement_metrics': {}
        }
        
        # Models tested
        if 'model' in df.columns:
            summary['models_tested'] = df['model'].unique().tolist()
        
        # Scenarios tested
        if 'scenario_id' in df.columns:
            summary['scenarios_tested'] = df['scenario_id'].unique().tolist()
        
        # Performance metrics by model
        if 'model' in df.columns:
            for model in df['model'].unique():
                model_data = df[df['model'] == model]
                metrics = {}
                
                if 'response_time' in df.columns:
                    metrics['avg_response_time'] = model_data['response_time'].mean()
                    metrics['std_response_time'] = model_data['response_time'].std()
                
                if 'tokens_used' in df.columns:
                    metrics['avg_tokens'] = model_data['tokens_used'].mean()
                
                if 'success' in df.columns:
                    metrics['success_rate'] = (model_data['success'] == True).mean()
                
                summary['performance_metrics'][model] = metrics
        
        # Response patterns
        if 'parsed_answer' in df.columns and 'model' in df.columns:
            for model in df['model'].unique():
                model_data = df[df['model'] == model]
                if 'parsed_answer' in model_data.columns:
                    patterns = model_data['parsed_answer'].value_counts().to_dict()
                    summary['response_patterns'][model] = patterns
        
        return summary
    
    def generate_latex_tables(self, df: pd.DataFrame) -> List[str]:
        """Generate LaTeX tables for paper
        
        Args:
            df: Results DataFrame
            
        Returns:
            List of LaTeX table strings
        """
        tables = []
        
        # Table 1: Model Performance Comparison
        if not df.empty and 'model' in df.columns:
            perf_table = self._generate_performance_table(df)
            tables.append(perf_table)
            
            # Save to file
            table_path = self.tables_dir / "performance_comparison.tex"
            table_path.write_text(perf_table)
            logger.info(f"Saved LaTeX table: {table_path}")
        
        # Table 2: Model Agreement Matrix
        if 'model' in df.columns and 'scenario_id' in df.columns:
            agreement_table = self._generate_agreement_table(df)
            if agreement_table:
                tables.append(agreement_table)
                
                # Save to file
                table_path = self.tables_dir / "agreement_matrix.tex"
                table_path.write_text(agreement_table)
                logger.info(f"Saved LaTeX table: {table_path}")
        
        return tables
    
    def _generate_performance_table(self, df: pd.DataFrame) -> str:
        """Generate performance comparison LaTeX table"""
        
        # Calculate statistics
        stats = df.groupby('model').agg({
            'response_time': ['mean', 'std'],
            'tokens_used': ['mean', 'std'],
            'success': lambda x: (x == True).mean() if 'success' in df.columns else 1.0
        }).round(2)
        
        # Build LaTeX table
        latex = []
        latex.append("\\begin{table}[htbp]")
        latex.append("\\centering")
        latex.append("\\caption{Model Performance Comparison on Moral Alignment Tasks}")
        latex.append("\\label{tab:model_performance}")
        latex.append("\\begin{tabular}{lrrrrc}")
        latex.append("\\toprule")
        latex.append("Model & \\multicolumn{2}{c}{Response Time (s)} & \\multicolumn{2}{c}{Tokens Used} & Success \\\\")
        latex.append("& Mean & Std & Mean & Std & Rate (\\%) \\\\")
        latex.append("\\midrule")
        
        for model in stats.index:
            time_mean = stats.loc[model, ('response_time', 'mean')]
            time_std = stats.loc[model, ('response_time', 'std')]
            tokens_mean = stats.loc[model, ('tokens_used', 'mean')]
            tokens_std = stats.loc[model, ('tokens_used', 'std')]
            success = stats.loc[model, ('success', '<lambda>')] * 100
            
            latex.append(f"{model} & {time_mean:.2f} & {time_std:.2f} & "
                        f"{tokens_mean:.0f} & {tokens_std:.0f} & {success:.1f} \\\\")
        
        latex.append("\\bottomrule")
        latex.append("\\end{tabular}")
        latex.append("\\end{table}")
        
        return "\n".join(latex)
    
    def _generate_agreement_table(self, df: pd.DataFrame) -> str:
        """Generate model agreement LaTeX table"""
        
        models = df['model'].unique()
        if len(models) < 2:
            return ""
        
        # Calculate agreement matrix
        n = len(models)
        agreement_matrix = np.ones((n, n)) * 100
        
        for i, model1 in enumerate(models):
            for j, model2 in enumerate(models):
                if i != j:
                    # Calculate agreement between model1 and model2
                    agreements = 0
                    total = 0
                    
                    for scenario in df['scenario_id'].unique():
                        resp1 = df[(df['model'] == model1) & 
                                  (df['scenario_id'] == scenario)]['parsed_answer'].values
                        resp2 = df[(df['model'] == model2) & 
                                  (df['scenario_id'] == scenario)]['parsed_answer'].values
                        
                        if len(resp1) > 0 and len(resp2) > 0:
                            total += 1
                            if resp1[0] == resp2[0]:
                                agreements += 1
                    
                    if total > 0:
                        agreement_matrix[i, j] = (agreements / total) * 100
        
        # Build LaTeX table
        latex = []
        latex.append("\\begin{table}[htbp]")
        latex.append("\\centering")
        latex.append("\\caption{Model Agreement Matrix (\\%)}")
        latex.append("\\label{tab:model_agreement}")
        
        # Adjust column specification based on number of models
        col_spec = "l" + "r" * len(models)
        latex.append(f"\\begin{{tabular}}{{{col_spec}}}")
        latex.append("\\toprule")
        
        # Header
        header = "Model"
        for model in models:
            # Shorten model names for table
            short_name = model.replace('gpt-', '').replace('-', '')[:8]
            header += f" & {short_name}"
        latex.append(header + " \\\\")
        latex.append("\\midrule")
        
        # Data rows
        for i, model1 in enumerate(models):
            short_name1 = model1.replace('gpt-', '').replace('-', '')[:8]
            row = short_name1
            for j, model2 in enumerate(models):
                if i == j:
                    row += " & --"
                else:
                    row += f" & {agreement_matrix[i, j]:.1f}"
            latex.append(row + " \\\\")
        
        latex.append("\\bottomrule")
        latex.append("\\end{tabular}")
        latex.append("\\end{table}")
        
        return "\n".join(latex)
    
    def generate_csv_export(self, df: pd.DataFrame, filename: str = "results_export.csv") -> str:
        """Export results to CSV format
        
        Args:
            df: Results DataFrame
            filename: Output filename
            
        Returns:
            Path to saved CSV file
        """
        output_path = self.exports_dir / filename
        df.to_csv(output_path, index=False)
        logger.info(f"Exported CSV: {output_path}")
        return str(output_path)
    
    def generate_json_export(self, data: Union[Dict, List], 
                           filename: str = "results_export.json") -> str:
        """Export results to JSON format
        
        Args:
            data: Data to export
            filename: Output filename
            
        Returns:
            Path to saved JSON file
        """
        output_path = self.exports_dir / filename
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Exported JSON: {output_path}")
        return str(output_path)
    
    def generate_markdown_report(self, df: pd.DataFrame, 
                               summary: Dict) -> str:
        """Generate comprehensive Markdown report
        
        Args:
            df: Results DataFrame
            summary: Summary statistics
            
        Returns:
            Markdown report string
        """
        report = []
        
        # Header
        report.append("# Moral Alignment Pipeline Results Report")
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Executive Summary
        report.append("\n## Executive Summary")
        report.append(f"- Total Samples Processed: {summary.get('total_samples', 0)}")
        report.append(f"- Models Tested: {len(summary.get('models_tested', []))}")
        report.append(f"- Scenarios Evaluated: {len(summary.get('scenarios_tested', []))}")
        
        # Model Performance
        report.append("\n## Model Performance")
        
        if summary.get('performance_metrics'):
            report.append("\n### Response Times and Token Usage")
            report.append("\n| Model | Avg Response Time (s) | Avg Tokens | Success Rate |")
            report.append("|-------|---------------------|------------|--------------|")
            
            for model, metrics in summary['performance_metrics'].items():
                time = metrics.get('avg_response_time', 0)
                tokens = metrics.get('avg_tokens', 0)
                success = metrics.get('success_rate', 1.0) * 100
                report.append(f"| {model} | {time:.2f} | {tokens:.0f} | {success:.1f}% |")
        
        # Response Patterns
        if summary.get('response_patterns'):
            report.append("\n## Response Patterns")
            
            for model, patterns in summary['response_patterns'].items():
                report.append(f"\n### {model}")
                for answer, count in patterns.items():
                    report.append(f"- {answer}: {count} responses")
        
        # Key Findings
        report.append("\n## Key Findings")
        
        if summary.get('performance_metrics'):
            # Find best performing models
            best_time = min(summary['performance_metrics'].items(), 
                          key=lambda x: x[1].get('avg_response_time', float('inf')))
            report.append(f"- Fastest Model: {best_time[0]} ({best_time[1]['avg_response_time']:.2f}s)")
            
            most_efficient = min(summary['performance_metrics'].items(),
                               key=lambda x: x[1].get('avg_tokens', float('inf')))
            report.append(f"- Most Token Efficient: {most_efficient[0]} ({most_efficient[1]['avg_tokens']:.0f} tokens)")
        
        # Recommendations
        report.append("\n## Recommendations")
        report.append("- For cost-effective evaluation: Use gpt-4o-mini")
        report.append("- For highest quality: Use gpt-4o or o1-preview")
        report.append("- For local processing: Use smaller models (GPT-2, OPT, Llama 3.2 1B/3B)")
        report.append("- For large-scale evaluation: Consider batch processing with checkpointing")
        
        # Data Files
        report.append("\n## Data Files")
        report.append(f"- Results Directory: `{self.results_dir}`")
        report.append(f"- Tables Directory: `{self.tables_dir}`")
        report.append(f"- Exports Directory: `{self.exports_dir}`")
        
        report_str = "\n".join(report)
        
        # Save report
        report_path = self.reports_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        report_path.write_text(report_str)
        logger.info(f"Saved Markdown report: {report_path}")
        
        return report_str
    
    def generate_all_outputs(self, results_dir: Optional[str] = None) -> Dict:
        """Generate all output formats from results
        
        Args:
            results_dir: Optional directory to load results from
            
        Returns:
            Dictionary of generated output paths
        """
        if results_dir:
            self.results_dir = Path(results_dir)
        
        logger.info("=" * 60)
        logger.info("Generating All Output Formats")
        logger.info("=" * 60)
        
        # Load results
        results = self.load_results()
        
        if not results:
            logger.warning("No results found to process")
            return {}
        
        # Aggregate results
        df = self.aggregate_model_results(results)
        
        # Generate summary statistics
        summary = self.generate_summary_statistics(df)
        
        # Generate outputs
        outputs = {}
        
        # LaTeX tables
        latex_tables = self.generate_latex_tables(df)
        outputs['latex_tables'] = len(latex_tables)
        
        # CSV export
        csv_path = self.generate_csv_export(df)
        outputs['csv_export'] = csv_path
        
        # JSON export
        json_path = self.generate_json_export(summary, "summary_statistics.json")
        outputs['json_export'] = json_path
        
        # Markdown report
        report = self.generate_markdown_report(df, summary)
        outputs['markdown_report'] = self.reports_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        logger.info("\n" + "=" * 60)
        logger.info("Output Generation Complete")
        logger.info("=" * 60)
        
        for key, value in outputs.items():
            logger.info(f"{key}: {value}")
        
        return outputs


def main():
    """Main execution for testing"""
    generator = OutputGenerator()
    outputs = generator.generate_all_outputs()
    
    print("\n✅ Output generation complete!")
    print(f"Generated {len(outputs)} output types")
    
    return outputs


if __name__ == "__main__":
    main()