#!/usr/bin/env python3
"""
Moral Alignment Visualization Engine
Creates publication-ready figures for moral alignment evaluation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional
import json
import warnings
warnings.filterwarnings('ignore')

class MoralVisualizationEngine:
    """Create visualizations for moral alignment evaluation"""
    
    def __init__(self, output_dir: str = "outputs/figures"):
        """Initialize visualization engine
        
        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style for publication-quality figures
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
    def plot_model_correlations(self, results: Dict, save_name: str = "model_correlations") -> str:
        """Plot model correlations comparison
        
        Args:
            results: Dictionary with model results
            save_name: Name for saved plot
            
        Returns:
            Path to saved plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        models = []
        lp_corrs = []
        dir_corrs = []
        
        for model, data in results.items():
            if 'metrics' in data:
                models.append(model)
                lp_corrs.append(data['metrics'].get('correlation_logprob', 0))
                dir_corrs.append(data['metrics'].get('correlation_direct', 0))
        
        if models:
            x = np.arange(len(models))
            width = 0.35
            
            # Correlation comparison
            ax1.bar(x - width/2, lp_corrs, width, label='Log-Probability', color='#2E86AB')
            ax1.bar(x + width/2, dir_corrs, width, label='Direct', color='#A23B72')
            ax1.set_xlabel('Model')
            ax1.set_ylabel('Pearson Correlation (ρ)')
            ax1.set_title('Model Correlations with Human Judgments')
            ax1.set_xticks(x)
            ax1.set_xticklabels(models, rotation=45, ha='right')
            ax1.legend()
            ax1.grid(axis='y', alpha=0.3)
            ax1.set_ylim([0, 1])
            
            # MAE comparison
            lp_mae = [results[m]['metrics'].get('mae_logprob', 0) for m in models]
            dir_mae = [results[m]['metrics'].get('mae_direct', 0) for m in models]
            
            ax2.bar(x - width/2, lp_mae, width, label='Log-Probability', color='#2E86AB')
            ax2.bar(x + width/2, dir_mae, width, label='Direct', color='#A23B72')
            ax2.set_xlabel('Model')
            ax2.set_ylabel('Mean Absolute Error')
            ax2.set_title('Prediction Error by Method')
            ax2.set_xticks(x)
            ax2.set_xticklabels(models, rotation=45, ha='right')
            ax2.legend()
            ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.output_dir / f"{save_name}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
    
    def plot_country_heatmap(self, df: pd.DataFrame, save_name: str = "country_heatmap") -> str:
        """Create country-model alignment heatmap
        
        Args:
            df: DataFrame with results
            save_name: Name for saved plot
            
        Returns:
            Path to saved plot
        """
        if 'country' not in df.columns or 'model' not in df.columns:
            return ""
        
        # Calculate mean scores by country and model
        pivot = df.groupby(['country', 'model'])['model_score'].mean().reset_index()
        pivot_table = pivot.pivot(index='country', columns='model', values='model_score')
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_table, cmap='RdBu_r', center=0, 
                    cbar_kws={'label': 'Mean Moral Score'},
                    vmin=-1, vmax=1, annot=True, fmt='.2f')
        plt.title('Country-Model Moral Alignment Patterns')
        plt.xlabel('Model')
        plt.ylabel('Country')
        plt.tight_layout()
        
        plot_path = self.output_dir / f"{save_name}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
    
    def plot_topic_comparison(self, df: pd.DataFrame, save_name: str = "topic_comparison") -> str:
        """Create topic-wise comparison
        
        Args:
            df: DataFrame with results
            save_name: Name for saved plot
            
        Returns:
            Path to saved plot
        """
        if 'topic' not in df.columns:
            return ""
        
        # Calculate mean scores and errors by topic
        topic_stats = df.groupby('topic').agg({
            'model_score': 'mean',
            'ground_truth': 'mean'
        }).reset_index()
        
        topic_stats['error'] = abs(topic_stats['model_score'] - topic_stats['ground_truth'])
        topic_stats = topic_stats.sort_values('error', ascending=False).head(10)
        
        # Create bar plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(topic_stats))
        width = 0.35
        
        ax.bar(x - width/2, topic_stats['ground_truth'], width, 
               label='Human Judgment', color='#2E86AB', alpha=0.8)
        ax.bar(x + width/2, topic_stats['model_score'], width,
               label='Model Prediction', color='#A23B72', alpha=0.8)
        
        ax.set_xlabel('Moral Topic')
        ax.set_ylabel('Mean Score')
        ax.set_title('Topic-wise Model Alignment with Human Judgments')
        ax.set_xticks(x)
        ax.set_xticklabels(topic_stats['topic'], rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        plot_path = self.output_dir / f"{save_name}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
    
    def plot_error_distribution(self, df: pd.DataFrame, save_name: str = "error_distribution") -> str:
        """Plot error distribution by method
        
        Args:
            df: DataFrame with results
            save_name: Name for saved plot
            
        Returns:
            Path to saved plot
        """
        if 'method' not in df.columns:
            return ""
        
        df['error'] = df['model_score'] - df['ground_truth']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Error distribution by method
        for method in df['method'].unique():
            method_df = df[df['method'] == method]
            ax1.hist(method_df['error'], bins=30, alpha=0.6, label=method, density=True)
        
        ax1.set_xlabel('Prediction Error')
        ax1.set_ylabel('Density')
        ax1.set_title('Error Distribution by Scoring Method')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        ax1.axvline(x=0, color='black', linestyle='--', linewidth=1)
        
        # Absolute error by method
        methods = df['method'].unique()
        abs_errors = [df[df['method'] == m]['error'].abs().mean() for m in methods]
        
        ax2.bar(methods, abs_errors, color=['#2E86AB', '#A23B72'])
        ax2.set_xlabel('Scoring Method')
        ax2.set_ylabel('Mean Absolute Error')
        ax2.set_title('Average Prediction Error by Method')
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.output_dir / f"{save_name}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
    
    def plot_scatter_alignment(self, df: pd.DataFrame, save_name: str = "scatter_alignment") -> str:
        """Create scatter plot of model vs human scores
        
        Args:
            df: DataFrame with results
            save_name: Name for saved plot
            
        Returns:
            Path to saved plot
        """
        if 'model' not in df.columns:
            # If no model column, create single plot
            fig, ax = plt.subplots(figsize=(8, 8))
            
            ax.scatter(df['ground_truth'], df['model_score'], alpha=0.5, s=20)
            
            # Add diagonal line
            lims = [-1, 1]
            ax.plot(lims, lims, 'k-', alpha=0.3, zorder=0)
            
            # Calculate correlation
            corr = df[['ground_truth', 'model_score']].corr().iloc[0, 1]
            
            ax.set_xlabel('Human Judgment')
            ax.set_ylabel('Model Prediction')
            ax.set_title(f'Model-Human Alignment (ρ={corr:.3f})')
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
        else:
            # Create subplot for each model
            models = df['model'].unique()
            n_models = len(models)
            fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 6))
            
            if n_models == 1:
                axes = [axes]
            
            for ax, model in zip(axes, models):
                model_df = df[df['model'] == model]
                
                # Color by method if available
                if 'method' in model_df.columns:
                    for method in model_df['method'].unique():
                        method_df = model_df[model_df['method'] == method]
                        ax.scatter(method_df['ground_truth'], method_df['model_score'], 
                                 alpha=0.5, s=20, label=method)
                else:
                    ax.scatter(model_df['ground_truth'], model_df['model_score'], 
                             alpha=0.5, s=20)
                
                # Add diagonal line
                lims = [-1, 1]
                ax.plot(lims, lims, 'k-', alpha=0.3, zorder=0)
                
                # Calculate correlation
                corr = model_df[['ground_truth', 'model_score']].corr().iloc[0, 1]
                
                ax.set_xlabel('Human Judgment')
                ax.set_ylabel('Model Prediction')
                ax.set_title(f'{model} (ρ={corr:.3f})')
                ax.set_xlim(lims)
                ax.set_ylim(lims)
                ax.grid(True, alpha=0.3)
                ax.set_aspect('equal')
                
                if 'method' in model_df.columns:
                    ax.legend()
        
        plt.tight_layout()
        plot_path = self.output_dir / f"{save_name}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
    
    def create_all_plots(self, results: Dict = None, df: pd.DataFrame = None) -> List[str]:
        """Create all visualizations
        
        Args:
            results: Dictionary with model results
            df: DataFrame with combined results
            
        Returns:
            List of paths to saved plots
        """
        plots = []
        
        # If we have results dictionary, create correlation plot
        if results:
            plot = self.plot_model_correlations(results)
            if plot:
                plots.append(plot)
                print(f"✅ Created model correlations plot: {plot}")
        
        # If we have DataFrame, create other plots
        if df is not None and not df.empty:
            # Country heatmap
            plot = self.plot_country_heatmap(df)
            if plot:
                plots.append(plot)
                print(f"✅ Created country heatmap: {plot}")
            
            # Topic comparison
            plot = self.plot_topic_comparison(df)
            if plot:
                plots.append(plot)
                print(f"✅ Created topic comparison: {plot}")
            
            # Error distribution
            plot = self.plot_error_distribution(df)
            if plot:
                plots.append(plot)
                print(f"✅ Created error distribution: {plot}")
            
            # Scatter alignment
            plot = self.plot_scatter_alignment(df)
            if plot:
                plots.append(plot)
                print(f"✅ Created scatter alignment: {plot}")
        
        return plots


def main():
    """Test visualization with mock data"""
    
    print("Testing Moral Alignment Visualizations...")
    
    # Load mock results
    mock_file = Path("outputs/paper_demo/mock_results.json")
    if not mock_file.exists():
        print("❌ Mock results not found. Run generate_paper_demo.py first")
        return
    
    with open(mock_file, 'r') as f:
        results = json.load(f)
    
    # Convert to DataFrame
    all_scores = []
    for model, data in results.items():
        for score in data['scores']:
            score['model'] = model
            all_scores.append(score)
    
    df = pd.DataFrame(all_scores)
    
    # Create visualizations
    viz = MoralVisualizationEngine(output_dir="outputs/paper_demo/figures")
    plots = viz.create_all_plots(results=results, df=df)
    
    print(f"\n✅ Created {len(plots)} visualizations")
    print(f"📁 Saved to: outputs/paper_demo/figures/")


if __name__ == "__main__":
    main()