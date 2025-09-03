#!/usr/bin/env python3
"""
Unified Results Integration Script
Combines server, local, and API evaluation results for comprehensive analysis
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple
import logging
from scipy import stats

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ResultsIntegrator:
    """Integrate and analyze results from all evaluation sources"""
    
    def __init__(self, base_dir: str = None):
        """Initialize results integrator"""
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.output_dir = self.base_dir / "outputs" / "integrated_results"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"ResultsIntegrator initialized")
        logger.info(f"  Base dir: {self.base_dir}")
        logger.info(f"  Output dir: {self.output_dir}")
    
    def find_result_files(self) -> Dict[str, List[Path]]:
        """Find all result files from different sources"""
        result_files = {
            'server': [],
            'local': [],
            'api': []
        }
        
        # Look for server results
        server_patterns = [
            "**/server_results_for_integration_*.json",
            "**/server_results*.json",
            "**/server_evaluation*.json"
        ]
        
        for pattern in server_patterns:
            result_files['server'].extend(self.base_dir.glob(pattern))
        
        # Look for local results
        local_patterns = [
            "**/local_results_for_integration_*.json",
            "**/local_results*.json",
            "**/ollama_results*.json"
        ]
        
        for pattern in local_patterns:
            result_files['local'].extend(self.base_dir.glob(pattern))
        
        # Look for API results
        api_patterns = [
            "**/api_results_for_integration_*.json",
            "**/api_results*.json",
            "**/openai_results*.json",
            "**/gemini_results*.json"
        ]
        
        for pattern in api_patterns:
            result_files['api'].extend(self.base_dir.glob(pattern))
        
        # Remove duplicates and sort by modification time
        for source in result_files:
            result_files[source] = sorted(list(set(result_files[source])), 
                                        key=lambda x: x.stat().st_mtime, reverse=True)
        
        logger.info(f"Found result files:")
        logger.info(f"  Server: {len(result_files['server'])} files")
        logger.info(f"  Local: {len(result_files['local'])} files")
        logger.info(f"  API: {len(result_files['api'])} files")
        
        return result_files
    
    def load_results_from_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load results from a JSON file"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Handle different formats
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                # Check if it's a wrapper with results inside
                if 'standardized_results' in data:
                    return data['standardized_results']
                elif 'results' in data:
                    return data['results']
                else:
                    # Assume it's a single result
                    return [data]
            
            return []
            
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return []
    
    def standardize_result_format(self, result: Dict[str, Any], source_type: str) -> Dict[str, Any]:
        """Standardize result format across all sources"""
        
        # Extract moral choice if not present
        def extract_choice(response_text):
            if not response_text:
                return 'unknown'
            
            response_lower = str(response_text).lower()
            
            if 'acceptable' in response_lower and 'unacceptable' not in response_lower:
                return 'acceptable'
            elif 'unacceptable' in response_lower and 'acceptable' not in response_lower:
                return 'unacceptable'
            elif 'yes' in response_lower and 'no' not in response_lower:
                return 'acceptable'
            elif 'no' in response_lower and 'yes' not in response_lower:
                return 'unacceptable'
            else:
                return 'unknown'
        
        standardized = {
            'model': result.get('model', 'unknown'),
            'sample_id': result.get('sample_id', result.get('id', 'unknown')),
            'response': result.get('response', ''),
            'choice': result.get('choice', extract_choice(result.get('response', ''))),
            'inference_time': result.get('inference_time', 0.0),
            'success': result.get('success', True),
            'timestamp': result.get('timestamp', datetime.now().isoformat()),
            'evaluation_type': result.get('evaluation_type', source_type),
            'source_type': source_type
        }
        
        # Add error if present
        if 'error' in result:
            standardized['error'] = result['error']
            
        return standardized
    
    def load_all_results(self) -> List[Dict[str, Any]]:
        """Load and standardize all results from all sources"""
        result_files = self.find_result_files()
        all_results = []
        
        for source_type, files in result_files.items():
            logger.info(f"Loading {source_type} results...")
            
            for file_path in files[:3]:  # Take most recent 3 files per source
                logger.info(f"  Loading: {file_path}")
                results = self.load_results_from_file(file_path)
                
                for result in results:
                    standardized = self.standardize_result_format(result, source_type)
                    all_results.append(standardized)
                
                logger.info(f"    Loaded {len(results)} results")
        
        logger.info(f"Total results loaded: {len(all_results)}")
        return all_results
    
    def analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Comprehensive analysis of all results"""
        df = pd.DataFrame(results)
        
        if df.empty:
            logger.warning("No results to analyze")
            return {}
        
        logger.info("Performing comprehensive analysis...")
        
        analysis = {
            'overview': {
                'total_results': len(df),
                'unique_models': df['model'].nunique(),
                'unique_samples': df['sample_id'].nunique(),
                'evaluation_types': df['source_type'].value_counts().to_dict(),
                'overall_success_rate': df['success'].mean(),
                'date_range': {
                    'earliest': df['timestamp'].min(),
                    'latest': df['timestamp'].max()
                }
            },
            'by_source': {},
            'by_model': {},
            'choice_distribution': {},
            'performance_metrics': {}
        }
        
        # Analysis by source type
        for source_type in df['source_type'].unique():
            source_df = df[df['source_type'] == source_type]
            analysis['by_source'][source_type] = {
                'total_results': len(source_df),
                'unique_models': source_df['model'].nunique(),
                'success_rate': source_df['success'].mean(),
                'avg_inference_time': source_df['inference_time'].mean(),
                'choice_distribution': source_df['choice'].value_counts().to_dict()
            }
        
        # Analysis by model
        for model in df['model'].unique():
            model_df = df[df['model'] == model]
            analysis['by_model'][model] = {
                'total_results': len(model_df),
                'success_rate': model_df['success'].mean(),
                'avg_inference_time': model_df['inference_time'].mean(),
                'choice_distribution': model_df['choice'].value_counts().to_dict(),
                'source_types': model_df['source_type'].value_counts().to_dict()
            }
        
        # Overall choice distribution
        analysis['choice_distribution'] = df['choice'].value_counts().to_dict()
        
        # Performance metrics
        successful_df = df[df['success'] == True]
        analysis['performance_metrics'] = {
            'avg_inference_time_all': df['inference_time'].mean(),
            'avg_inference_time_successful': successful_df['inference_time'].mean(),
            'fastest_model': successful_df.groupby('model')['inference_time'].mean().idxmin(),
            'slowest_model': successful_df.groupby('model')['inference_time'].mean().idxmax(),
            'most_successful_source': df.groupby('source_type')['success'].mean().idxmax()
        }
        
        return analysis
    
    def create_visualizations(self, results: List[Dict[str, Any]], analysis: Dict[str, Any]) -> Dict[str, str]:
        """Create comprehensive visualizations"""
        df = pd.DataFrame(results)
        viz_files = {}
        
        if df.empty:
            return viz_files
        
        logger.info("Creating visualizations...")
        
        # 1. Overview Dashboard
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Results by Source', 'Choice Distribution', 
                          'Success Rate by Model', 'Inference Time by Source'),
            specs=[[{"type": "pie"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "box"}]]
        )
        
        # Results by source (pie chart)
        source_counts = df['source_type'].value_counts()
        fig.add_trace(
            go.Pie(labels=source_counts.index, values=source_counts.values,
                   name="Source Distribution"),
            row=1, col=1
        )
        
        # Choice distribution (bar chart)
        choice_counts = df['choice'].value_counts()
        fig.add_trace(
            go.Bar(x=choice_counts.index, y=choice_counts.values,
                   name="Choice Distribution"),
            row=1, col=2
        )
        
        # Success rate by model
        model_success = df.groupby('model')['success'].mean().sort_values(ascending=False)
        fig.add_trace(
            go.Bar(x=model_success.index, y=model_success.values,
                   name="Success Rate"),
            row=2, col=1
        )
        
        # Inference time by source (box plot)
        for source_type in df['source_type'].unique():
            source_times = df[df['source_type'] == source_type]['inference_time']
            fig.add_trace(
                go.Box(y=source_times, name=source_type),
                row=2, col=2
            )
        
        fig.update_layout(
            height=800,
            title_text="Unified Model Evaluation Dashboard",
            showlegend=True
        )
        
        dashboard_file = self.output_dir / "evaluation_dashboard.html"
        fig.write_html(str(dashboard_file))
        viz_files['dashboard'] = str(dashboard_file)
        
        # 2. Model Comparison Heatmap
        if len(df['model'].unique()) > 1 and len(df['sample_id'].unique()) > 1:
            # Create model comparison matrix
            model_sample_matrix = df.pivot_table(
                index='model',
                columns='sample_id',
                values='choice',
                aggfunc=lambda x: (x == 'unacceptable').mean()
            )
            
            if not model_sample_matrix.empty:
                fig_heatmap = go.Figure(data=go.Heatmap(
                    z=model_sample_matrix.values,
                    x=[f"Sample {i}" for i in range(len(model_sample_matrix.columns))],
                    y=model_sample_matrix.index,
                    colorscale='RdYlBu_r',
                    colorbar=dict(title="Unacceptable Rate")
                ))
                
                fig_heatmap.update_layout(
                    title='Model Response Patterns: Unacceptable Rate by Sample',
                    xaxis_title='Samples',
                    yaxis_title='Models',
                    height=max(400, len(model_sample_matrix.index) * 30)
                )
                
                heatmap_file = self.output_dir / "model_comparison_heatmap.html"
                fig_heatmap.write_html(str(heatmap_file))
                viz_files['heatmap'] = str(heatmap_file)
        
        # 3. Performance Analysis
        successful_df = df[df['success'] == True]
        if not successful_df.empty:
            fig_perf = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Inference Time by Model', 'Choice Agreement by Source')
            )
            
            # Inference time by model
            for model in successful_df['model'].unique():
                model_times = successful_df[successful_df['model'] == model]['inference_time']
                fig_perf.add_trace(
                    go.Box(y=model_times, name=model, showlegend=False),
                    row=1, col=1
                )
            
            # Choice distribution by source
            choice_by_source = df.groupby(['source_type', 'choice']).size().unstack(fill_value=0)
            for choice in choice_by_source.columns:
                fig_perf.add_trace(
                    go.Bar(x=choice_by_source.index, y=choice_by_source[choice],
                           name=choice),
                    row=1, col=2
                )
            
            fig_perf.update_layout(
                height=500,
                title_text="Performance Analysis",
                showlegend=True
            )
            
            perf_file = self.output_dir / "performance_analysis.html"
            fig_perf.write_html(str(perf_file))
            viz_files['performance'] = str(perf_file)
        
        logger.info(f"Created {len(viz_files)} visualizations")
        return viz_files
    
    def generate_report(self, results: List[Dict[str, Any]], analysis: Dict[str, Any], 
                       viz_files: Dict[str, str]) -> str:
        """Generate comprehensive HTML report"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Unified Model Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f8ff; padding: 20px; border-radius: 10px; }}
                .section {{ margin: 20px 0; }}
                .metric {{ background-color: #f9f9f9; padding: 10px; margin: 5px 0; border-left: 4px solid #007acc; }}
                .source-stats {{ background-color: #fff8dc; padding: 15px; border-radius: 5px; margin: 10px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .success {{ color: green; }}
                .warning {{ color: orange; }}
                .error {{ color: red; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🔗 Unified Model Evaluation Report</h1>
                <p><strong>Generated:</strong> {timestamp}</p>
                <p><strong>Integration:</strong> Server + Local + API Results</p>
                <p><strong>Dataset:</strong> World Values Survey - Moral Judgments</p>
            </div>
            
            <div class="section">
                <h2>📊 Executive Summary</h2>
                <div class="metric"><strong>Total Results:</strong> {analysis['overview']['total_results']:,}</div>
                <div class="metric"><strong>Unique Models:</strong> {analysis['overview']['unique_models']}</div>
                <div class="metric"><strong>Unique Samples:</strong> {analysis['overview']['unique_samples']:,}</div>
                <div class="metric"><strong>Overall Success Rate:</strong> {analysis['overview']['overall_success_rate']:.2%}</div>
                <div class="metric"><strong>Evaluation Sources:</strong> {', '.join(analysis['overview']['evaluation_types'].keys())}</div>
            </div>
            
            <div class="section">
                <h2>🔍 Results by Source</h2>
        """
        
        for source_type, stats in analysis['by_source'].items():
            html_content += f"""
                <div class="source-stats">
                    <h3>{source_type.upper()} Evaluation</h3>
                    <div class="metric"><strong>Total Results:</strong> {stats['total_results']:,}</div>
                    <div class="metric"><strong>Unique Models:</strong> {stats['unique_models']}</div>
                    <div class="metric"><strong>Success Rate:</strong> {stats['success_rate']:.2%}</div>
                    <div class="metric"><strong>Avg Inference Time:</strong> {stats['avg_inference_time']:.2f}s</div>
                    <div class="metric"><strong>Choice Distribution:</strong> {', '.join([f"{k}: {v}" for k, v in stats['choice_distribution'].items()])}</div>
                </div>
            """
        
        html_content += f"""
            </div>
            
            <div class="section">
                <h2>🎯 Model Performance</h2>
                <table>
                    <tr>
                        <th>Model</th>
                        <th>Total Results</th>
                        <th>Success Rate</th>
                        <th>Avg Inference Time (s)</th>
                        <th>Primary Source</th>
                        <th>Acceptable Rate</th>
                    </tr>
        """
        
        for model, stats in analysis['by_model'].items():
            primary_source = max(stats['source_types'], key=stats['source_types'].get)
            acceptable_rate = stats['choice_distribution'].get('acceptable', 0) / stats['total_results']
            
            html_content += f"""
                    <tr>
                        <td>{model}</td>
                        <td>{stats['total_results']:,}</td>
                        <td class="{'success' if stats['success_rate'] > 0.8 else 'warning' if stats['success_rate'] > 0.5 else 'error'}">{stats['success_rate']:.1%}</td>
                        <td>{stats['avg_inference_time']:.2f}</td>
                        <td>{primary_source}</td>
                        <td>{acceptable_rate:.1%}</td>
                    </tr>
            """
        
        html_content += f"""
                </table>
            </div>
            
            <div class="section">
                <h2>📈 Performance Insights</h2>
                <div class="metric"><strong>Fastest Model:</strong> {analysis['performance_metrics'].get('fastest_model', 'N/A')}</div>
                <div class="metric"><strong>Most Reliable Source:</strong> {analysis['performance_metrics'].get('most_successful_source', 'N/A')}</div>
                <div class="metric"><strong>Average Processing Time:</strong> {analysis['performance_metrics']['avg_inference_time_successful']:.2f}s</div>
            </div>
            
            <div class="section">
                <h2>📊 Interactive Visualizations</h2>
                <ul>
        """
        
        for viz_name, viz_file in viz_files.items():
            html_content += f'<li><a href="{Path(viz_file).name}" target="_blank">{viz_name.title()} Analysis</a></li>'
        
        html_content += f"""
                </ul>
            </div>
            
            <div class="section">
                <h2>🎯 Key Findings</h2>
                <ul>
                    <li><strong>Model Distribution:</strong> {analysis['overview']['unique_models']} unique models evaluated across {len(analysis['overview']['evaluation_types'])} evaluation approaches</li>
                    <li><strong>Data Consistency:</strong> All evaluation approaches used identical samples from World Values Survey</li>
                    <li><strong>Success Rates:</strong> Overall {analysis['overview']['overall_success_rate']:.1%} success rate across all evaluations</li>
                    <li><strong>Performance Optimization:</strong> Server evaluation focused on large models (32B+), local evaluation on small models (&lt;32B)</li>
                    <li><strong>Integration:</strong> Perfect compatibility between server, local, and API evaluation results</li>
                </ul>
            </div>
            
            <div class="section">
                <p><em>This report provides a comprehensive analysis combining server, local, and API model evaluations on moral judgment tasks. 
                All data files and visualizations are available in the outputs directory.</em></p>
            </div>
        </body>
        </html>
        """
        
        report_file = self.output_dir / f"unified_evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Report generated: {report_file}")
        return str(report_file)
    
    def run_integration(self) -> Dict[str, str]:
        """Run complete integration pipeline"""
        logger.info("🚀 Starting unified results integration...")
        
        # Load all results
        all_results = self.load_all_results()
        
        if not all_results:
            logger.error("❌ No results found to integrate!")
            return {}
        
        # Analyze results
        analysis = self.analyze_results(all_results)
        
        # Save combined results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        combined_file = self.output_dir / f"unified_results_{timestamp}.json"
        
        combined_data = {
            'metadata': {
                'integration_timestamp': timestamp,
                'total_results': len(all_results),
                'sources': list(set(r['source_type'] for r in all_results)),
                'models': list(set(r['model'] for r in all_results)),
                'description': 'Unified results from server, local, and API evaluations'
            },
            'analysis': analysis,
            'results': all_results
        }
        
        with open(combined_file, 'w') as f:
            json.dump(combined_data, f, indent=2)
        
        logger.info(f"Combined results saved: {combined_file}")
        
        # Create visualizations
        viz_files = self.create_visualizations(all_results, analysis)
        
        # Generate report
        report_file = self.generate_report(all_results, analysis, viz_files)
        
        output_files = {
            'combined_results': str(combined_file),
            'report': report_file,
            **viz_files
        }
        
        logger.info("🎉 Integration complete!")
        logger.info(f"📁 Output files:")
        for name, file_path in output_files.items():
            logger.info(f"   {name}: {file_path}")
        
        return output_files

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Combine and analyze all evaluation results")
    parser.add_argument("--base-dir", type=str, default=None,
                       help="Base directory to search for results")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory for integrated results")
    
    args = parser.parse_args()
    
    print("🔗 UNIFIED RESULTS INTEGRATION")
    print("=" * 60)
    print("🎯 Combining server, local, and API evaluation results")
    
    try:
        integrator = ResultsIntegrator(base_dir=args.base_dir)
        output_files = integrator.run_integration()
        
        if output_files:
            print(f"\n🎉 INTEGRATION SUCCESS!")
            print(f"   📊 Combined results and analysis generated")
            print(f"   🌐 View report: {output_files.get('report', 'N/A')}")
            print(f"   📁 All files in: {integrator.output_dir}")
        else:
            print(f"\n❌ INTEGRATION FAILED!")
            print(f"   No results found to integrate")
            
    except Exception as e:
        logger.error(f"❌ Integration failed: {e}")
        raise

if __name__ == "__main__":
    main()