#!/usr/bin/env python3
"""
Visualization Engine for Moral Alignment Pipeline
Creates comprehensive plots and visualizations for model comparisons
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class VisualizationEngine:
    """Creates visualizations for moral alignment analysis"""
    
    def __init__(self, output_dir: str = "outputs/plots"):
        """Initialize visualization engine
        
        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Color schemes for different model types
        self.color_schemes = {
            'openai': '#10A37F',  # OpenAI green
            'anthropic': '#AA7BC3',  # Claude purple
            'google': '#4285F4',  # Google blue
            'meta': '#0084FF',  # Meta blue
            'local': '#FF6B6B',  # Red for local
            'mistral': '#FF9500',  # Orange
            'cohere': '#39C3E6'  # Cyan
        }
        
        # Model categories
        self.model_categories = {
            'gpt-4o': 'openai',
            'gpt-4o-mini': 'openai',
            'gpt-4-turbo': 'openai',
            'o1-preview': 'openai',
            'o1-mini': 'openai',
            'claude-3.5-sonnet': 'anthropic',
            'claude-3.5-haiku': 'anthropic',
            'gemini-1.5-pro': 'google',
            'gemini-1.5-flash': 'google',
            'llama': 'meta',
            'gpt2': 'local',
            'opt': 'local',
            'gemma': 'google',
            'mistral': 'mistral',
            'command': 'cohere'
        }
    
    def get_model_color(self, model_name: str) -> str:
        """Get color for a model based on its provider
        
        Args:
            model_name: Name of the model
            
        Returns:
            Hex color code
        """
        for key, category in self.model_categories.items():
            if key in model_name.lower():
                return self.color_schemes.get(category, '#808080')
        return '#808080'  # Default gray
    
    def plot_model_performance_comparison(self, results_df: pd.DataFrame, 
                                         save_name: str = "model_performance") -> str:
        """Create performance comparison chart
        
        Args:
            results_df: DataFrame with model results
            save_name: Name for saved plot
            
        Returns:
            Path to saved plot
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Response Time', 'Token Usage', 
                          'Success Rate', 'Cost Efficiency'),
            specs=[[{'type': 'bar'}, {'type': 'bar'}],
                   [{'type': 'bar'}, {'type': 'scatter'}]]
        )
        
        if 'model' in results_df.columns:
            # Group by model
            model_stats = results_df.groupby('model').agg({
                'response_time': 'mean',
                'tokens_used': 'mean',
                'success': 'mean'
            }).reset_index()
            
            # Add cost efficiency (tokens per second)
            model_stats['efficiency'] = model_stats['tokens_used'] / model_stats['response_time']
            
            # Response Time
            fig.add_trace(
                go.Bar(
                    x=model_stats['model'],
                    y=model_stats['response_time'],
                    marker_color=[self.get_model_color(m) for m in model_stats['model']],
                    showlegend=False
                ),
                row=1, col=1
            )
            
            # Token Usage
            fig.add_trace(
                go.Bar(
                    x=model_stats['model'],
                    y=model_stats['tokens_used'],
                    marker_color=[self.get_model_color(m) for m in model_stats['model']],
                    showlegend=False
                ),
                row=1, col=2
            )
            
            # Success Rate
            fig.add_trace(
                go.Bar(
                    x=model_stats['model'],
                    y=model_stats['success'] * 100,
                    marker_color=[self.get_model_color(m) for m in model_stats['model']],
                    showlegend=False
                ),
                row=2, col=1
            )
            
            # Cost Efficiency
            fig.add_trace(
                go.Scatter(
                    x=model_stats['response_time'],
                    y=model_stats['tokens_used'],
                    mode='markers+text',
                    text=model_stats['model'],
                    textposition='top center',
                    marker=dict(
                        size=15,
                        color=[self.get_model_color(m) for m in model_stats['model']]
                    ),
                    showlegend=False
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_xaxes(title_text="Model", row=1, col=1)
        fig.update_xaxes(title_text="Model", row=1, col=2)
        fig.update_xaxes(title_text="Model", row=2, col=1)
        fig.update_xaxes(title_text="Response Time (s)", row=2, col=2)
        
        fig.update_yaxes(title_text="Time (s)", row=1, col=1)
        fig.update_yaxes(title_text="Tokens", row=1, col=2)
        fig.update_yaxes(title_text="Success %", row=2, col=1)
        fig.update_yaxes(title_text="Tokens Used", row=2, col=2)
        
        fig.update_layout(
            title="Model Performance Comparison",
            height=800,
            showlegend=False
        )
        
        # Save plot
        output_path = self.output_dir / f"{save_name}.html"
        fig.write_html(str(output_path))
        
        # Also save as static image if kaleido is installed
        try:
            fig.write_image(str(self.output_dir / f"{save_name}.png"))
        except:
            pass
        
        logger.info(f"Performance comparison saved to: {output_path}")
        return str(output_path)
    
    def plot_model_agreement_heatmap(self, agreement_matrix: Dict,
                                    save_name: str = "model_agreement") -> str:
        """Create heatmap of model agreement
        
        Args:
            agreement_matrix: Dictionary with agreement rates
            save_name: Name for saved plot
            
        Returns:
            Path to saved plot
        """
        # Parse agreement matrix into DataFrame
        models = set()
        for key in agreement_matrix.keys():
            m1, m2 = key.split(' vs ')
            models.add(m1)
            models.add(m2)
        
        models = sorted(list(models))
        n = len(models)
        
        # Create matrix
        matrix = np.ones((n, n))
        for i, m1 in enumerate(models):
            for j, m2 in enumerate(models):
                if i != j:
                    key1 = f"{m1} vs {m2}"
                    key2 = f"{m2} vs {m1}"
                    if key1 in agreement_matrix:
                        matrix[i, j] = agreement_matrix[key1]['agreement_rate']
                    elif key2 in agreement_matrix:
                        matrix[i, j] = agreement_matrix[key2]['agreement_rate']
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=models,
            y=models,
            colorscale='RdYlGn',
            text=np.round(matrix * 100, 1),
            texttemplate='%{text}%',
            textfont={"size": 10},
            colorbar=dict(title="Agreement %")
        ))
        
        fig.update_layout(
            title="Model Agreement Matrix",
            xaxis_title="Model",
            yaxis_title="Model",
            height=600,
            width=800
        )
        
        # Save plot
        output_path = self.output_dir / f"{save_name}.html"
        fig.write_html(str(output_path))
        
        logger.info(f"Agreement heatmap saved to: {output_path}")
        return str(output_path)
    
    def plot_moral_alignment_scores(self, scores_df: pd.DataFrame,
                                   save_name: str = "moral_alignment") -> str:
        """Plot moral alignment scores across models
        
        Args:
            scores_df: DataFrame with alignment scores
            save_name: Name for saved plot
            
        Returns:
            Path to saved plot
        """
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Alignment Scores by Model', 'Score Distribution'),
            specs=[[{'type': 'box'}, {'type': 'violin'}]]
        )
        
        if 'model' in scores_df.columns and 'score' in scores_df.columns:
            models = scores_df['model'].unique()
            
            # Box plot
            for model in models:
                model_data = scores_df[scores_df['model'] == model]['score']
                fig.add_trace(
                    go.Box(
                        y=model_data,
                        name=model,
                        marker_color=self.get_model_color(model)
                    ),
                    row=1, col=1
                )
            
            # Violin plot
            for model in models:
                model_data = scores_df[scores_df['model'] == model]['score']
                fig.add_trace(
                    go.Violin(
                        y=model_data,
                        name=model,
                        marker_color=self.get_model_color(model),
                        showlegend=False
                    ),
                    row=1, col=2
                )
        
        fig.update_layout(
            title="Moral Alignment Score Analysis",
            height=500,
            showlegend=True
        )
        
        fig.update_yaxes(title_text="Alignment Score", row=1, col=1)
        fig.update_yaxes(title_text="Alignment Score", row=1, col=2)
        
        # Save plot
        output_path = self.output_dir / f"{save_name}.html"
        fig.write_html(str(output_path))
        
        logger.info(f"Alignment scores plot saved to: {output_path}")
        return str(output_path)
    
    def plot_response_patterns(self, responses_df: pd.DataFrame,
                              save_name: str = "response_patterns") -> str:
        """Analyze and plot response patterns
        
        Args:
            responses_df: DataFrame with model responses
            save_name: Name for saved plot
            
        Returns:
            Path to saved plot
        """
        # Create sunburst chart for response patterns
        if 'model' in responses_df.columns and 'parsed_answer' in responses_df.columns:
            pattern_data = responses_df.groupby(['model', 'parsed_answer']).size().reset_index(name='count')
            
            fig = px.sunburst(
                pattern_data,
                path=['model', 'parsed_answer'],
                values='count',
                title='Response Patterns by Model',
                color_discrete_map={
                    'yes': '#2ECC40',
                    'no': '#FF4136',
                    'it depends': '#FFDC00'
                }
            )
            
            fig.update_layout(height=600)
            
            # Save plot
            output_path = self.output_dir / f"{save_name}.html"
            fig.write_html(str(output_path))
            
            logger.info(f"Response patterns saved to: {output_path}")
            return str(output_path)
        
        return ""
    
    def plot_cost_analysis(self, models: List[str], samples: int = 1000,
                          save_name: str = "cost_analysis") -> str:
        """Create cost analysis visualization
        
        Args:
            models: List of model names
            samples: Number of samples for cost estimation
            save_name: Name for saved plot
            
        Returns:
            Path to saved plot
        """
        from env_loader import get_env_loader
        loader = get_env_loader()
        
        cost_data = []
        for model in models:
            cost_info = loader.estimate_costs(model, samples)
            if cost_info['is_api_model']:
                cost_data.append({
                    'model': model,
                    'cost': cost_info['estimated_cost_usd'],
                    'cost_per_1000': cost_info['cost_per_1000']
                })
        
        if cost_data:
            df = pd.DataFrame(cost_data)
            
            fig = go.Figure()
            
            # Bar chart for total cost
            fig.add_trace(go.Bar(
                x=df['model'],
                y=df['cost'],
                text=[f"${c:.2f}" for c in df['cost']],
                textposition='outside',
                marker_color=[self.get_model_color(m) for m in df['model']],
                name=f'Cost for {samples} samples'
            ))
            
            fig.update_layout(
                title=f"API Cost Analysis ({samples} samples)",
                xaxis_title="Model",
                yaxis_title="Estimated Cost (USD)",
                height=500,
                showlegend=True
            )
            
            # Save plot
            output_path = self.output_dir / f"{save_name}.html"
            fig.write_html(str(output_path))
            
            logger.info(f"Cost analysis saved to: {output_path}")
            return str(output_path)
        
        return ""
    
    def create_dashboard(self, results_df: pd.DataFrame,
                        save_name: str = "dashboard") -> str:
        """Create comprehensive dashboard with all visualizations
        
        Args:
            results_df: DataFrame with all results
            save_name: Name for saved dashboard
            
        Returns:
            Path to saved dashboard
        """
        from plotly.subplots import make_subplots
        
        # Create multi-panel dashboard
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Model Performance', 'Response Time Distribution',
                          'Success Rates', 'Token Efficiency',
                          'Model Categories', 'Performance Overview'),
            specs=[[{'type': 'bar'}, {'type': 'box'}],
                   [{'type': 'bar'}, {'type': 'scatter'}],
                   [{'type': 'pie'}, {'type': 'indicator'}]],
            vertical_spacing=0.1,
            horizontal_spacing=0.15
        )
        
        if not results_df.empty and 'model' in results_df.columns:
            # Performance metrics
            model_stats = results_df.groupby('model').agg({
                'response_time': 'mean',
                'tokens_used': 'mean',
                'success': lambda x: (x == True).mean() if 'success' in results_df else 1.0
            }).reset_index()
            
            # 1. Model Performance Bar
            fig.add_trace(
                go.Bar(
                    x=model_stats['model'],
                    y=model_stats['response_time'],
                    marker_color=[self.get_model_color(m) for m in model_stats['model']],
                    showlegend=False
                ),
                row=1, col=1
            )
            
            # 2. Response Time Distribution
            for model in model_stats['model']:
                model_times = results_df[results_df['model'] == model]['response_time']
                fig.add_trace(
                    go.Box(
                        y=model_times,
                        name=model,
                        marker_color=self.get_model_color(model),
                        showlegend=False
                    ),
                    row=1, col=2
                )
            
            # 3. Success Rates
            fig.add_trace(
                go.Bar(
                    x=model_stats['model'],
                    y=model_stats['success'] * 100,
                    marker_color=[self.get_model_color(m) for m in model_stats['model']],
                    text=[f"{s:.1f}%" for s in model_stats['success'] * 100],
                    textposition='outside',
                    showlegend=False
                ),
                row=2, col=1
            )
            
            # 4. Token Efficiency
            fig.add_trace(
                go.Scatter(
                    x=model_stats['response_time'],
                    y=model_stats['tokens_used'],
                    mode='markers+text',
                    text=model_stats['model'],
                    textposition='top center',
                    marker=dict(
                        size=20,
                        color=[self.get_model_color(m) for m in model_stats['model']]
                    ),
                    showlegend=False
                ),
                row=2, col=2
            )
            
            # 5. Model Categories Pie
            categories = [self.model_categories.get(m.split('-')[0].lower(), 'other') 
                         for m in model_stats['model']]
            category_counts = pd.Series(categories).value_counts()
            
            fig.add_trace(
                go.Pie(
                    labels=category_counts.index,
                    values=category_counts.values,
                    marker=dict(colors=[self.color_schemes.get(c, '#808080') 
                               for c in category_counts.index]),
                    showlegend=True
                ),
                row=3, col=1
            )
            
            # 6. Performance Indicator
            avg_success = model_stats['success'].mean() * 100
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=avg_success,
                    title={'text': "Overall Success Rate"},
                    gauge={'axis': {'range': [0, 100]},
                          'bar': {'color': "darkgreen"},
                          'steps': [
                              {'range': [0, 50], 'color': "lightgray"},
                              {'range': [50, 80], 'color': "yellow"},
                              {'range': [80, 100], 'color': "lightgreen"}],
                          'threshold': {'line': {'color': "red", 'width': 4},
                                      'thickness': 0.75, 'value': 90}}
                ),
                row=3, col=2
            )
        
        # Update layout
        fig.update_layout(
            title="Moral Alignment Pipeline Dashboard",
            height=1200,
            showlegend=False
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Model", row=1, col=1)
        fig.update_xaxes(title_text="Model", row=2, col=1)
        fig.update_xaxes(title_text="Response Time (s)", row=2, col=2)
        
        fig.update_yaxes(title_text="Time (s)", row=1, col=1)
        fig.update_yaxes(title_text="Time (s)", row=1, col=2)
        fig.update_yaxes(title_text="Success %", row=2, col=1)
        fig.update_yaxes(title_text="Tokens", row=2, col=2)
        
        # Save dashboard
        output_path = self.output_dir / f"{save_name}.html"
        fig.write_html(str(output_path))
        
        logger.info(f"Dashboard saved to: {output_path}")
        return str(output_path)
    
    def generate_latex_table(self, results_df: pd.DataFrame,
                           save_name: str = "results_table") -> str:
        """Generate LaTeX table for paper
        
        Args:
            results_df: DataFrame with results
            save_name: Name for saved table
            
        Returns:
            LaTeX table string
        """
        if results_df.empty:
            return ""
        
        # Aggregate statistics
        stats = results_df.groupby('model').agg({
            'response_time': ['mean', 'std'],
            'tokens_used': ['mean', 'std'],
            'success': lambda x: (x == True).mean() if 'success' in results_df else 1.0
        }).round(2)
        
        # Create LaTeX table
        latex = []
        latex.append("\\begin{table}[h]")
        latex.append("\\centering")
        latex.append("\\caption{Model Performance Comparison}")
        latex.append("\\label{tab:model_performance}")
        latex.append("\\begin{tabular}{lcccc}")
        latex.append("\\hline")
        latex.append("Model & Avg Time (s) & Std Time & Avg Tokens & Success Rate \\\\")
        latex.append("\\hline")
        
        for model in stats.index:
            time_mean = stats.loc[model, ('response_time', 'mean')]
            time_std = stats.loc[model, ('response_time', 'std')]
            tokens_mean = stats.loc[model, ('tokens_used', 'mean')]
            success = stats.loc[model, ('success', '<lambda>')] * 100
            
            latex.append(f"{model} & {time_mean:.2f} & {time_std:.2f} & {tokens_mean:.0f} & {success:.1f}\\% \\\\")
        
        latex.append("\\hline")
        latex.append("\\end{tabular}")
        latex.append("\\end{table}")
        
        latex_str = "\n".join(latex)
        
        # Save to file
        output_path = self.output_dir.parent / "tables" / f"{save_name}.tex"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(latex_str)
        
        logger.info(f"LaTeX table saved to: {output_path}")
        return latex_str


def create_sample_visualizations():
    """Create sample visualizations with dummy data"""
    logger.info("Creating sample visualizations...")
    
    # Create visualization engine
    viz = VisualizationEngine()
    
    # Create sample data
    models = ['gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo', 'o1-preview', 'o1-mini']
    scenarios = ['trolley_problem', 'white_lie', 'stealing_medicine']
    
    results = []
    for model in models:
        for scenario in scenarios:
            results.append({
                'model': model,
                'scenario_id': scenario,
                'response_time': np.random.uniform(0.5, 3.0),
                'tokens_used': np.random.randint(100, 500),
                'success': True,
                'parsed_answer': np.random.choice(['yes', 'no', 'it depends']),
                'score': np.random.uniform(0.6, 1.0)
            })
    
    df = pd.DataFrame(results)
    
    # Create visualizations
    plots = []
    plots.append(viz.plot_model_performance_comparison(df))
    plots.append(viz.plot_response_patterns(df))
    plots.append(viz.plot_cost_analysis(models, 5000))
    plots.append(viz.create_dashboard(df))
    
    # Generate LaTeX table
    latex_table = viz.generate_latex_table(df)
    
    logger.info(f"✅ Created {len(plots)} visualizations")
    logger.info(f"Plots saved in: {viz.output_dir}")
    
    return plots


if __name__ == "__main__":
    create_sample_visualizations()