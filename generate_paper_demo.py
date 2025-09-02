#!/usr/bin/env python3
"""
Paper Output Demonstration
Shows how to represent moral alignment results in academic paper format
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from paper_outputs import PaperOutputGenerator
from moral_visualization import MoralVisualizationEngine

def create_realistic_mock_results():
    """Create realistic mock results based on expected model behavior"""
    
    # Load human baseline for reference
    with open('outputs/wvs_processed/human_baseline.json', 'r') as f:
        baseline = json.load(f)
    
    # Create realistic model results with varying performance
    models = {
        'gpt-4o': {
            'correlation': 0.72,  # High performance
            'noise': 0.15,
            'bias': 0.05
        },
        'gpt-4o-mini': {
            'correlation': 0.65,  # Good performance
            'noise': 0.20,
            'bias': 0.08
        },
        'gpt-3.5-turbo': {
            'correlation': 0.58,  # Moderate performance
            'noise': 0.25,
            'bias': 0.10
        }
    }
    
    # Generate scores for each model
    all_results = {}
    np.random.seed(42)  # For reproducibility
    
    # Sample countries and topics
    countries = list(baseline['by_country'].keys())[:10]  # Top 10 countries
    topics = list(baseline['by_topic'].keys())[:10]  # Top 10 topics
    
    for model_name, params in models.items():
        scores = []
        
        for country in countries:
            country_data = baseline['by_country'].get(country, {})
            country_mean = country_data.get('mean', -0.5)
            
            for topic in topics:
                topic_data = baseline['by_topic'].get(topic, {})
                topic_mean = topic_data.get('mean', -0.5)
                
                # Combine country and topic effects
                true_score = (country_mean + topic_mean) / 2
                
                # Add model-specific noise and bias
                model_score_lp = true_score * params['correlation'] + \
                               np.random.normal(0, params['noise']) + \
                               params['bias']
                
                model_score_dir = true_score * (params['correlation'] - 0.05) + \
                                 np.random.normal(0, params['noise'] * 1.2) + \
                                 params['bias'] * 1.1
                
                # Clip to valid range
                model_score_lp = np.clip(model_score_lp, -1, 1)
                model_score_dir = np.clip(model_score_dir, -1, 1)
                
                # Create score entries
                scores.append({
                    'country': country,
                    'topic': topic,
                    'ground_truth': true_score,
                    'model_score': model_score_lp,
                    'method': 'logprob',
                    'confidence': 0.8 + np.random.random() * 0.2,
                    'raw_logprobs': {
                        'acceptable': np.random.random(),
                        'unacceptable': np.random.random()
                    }
                })
                
                scores.append({
                    'country': country,
                    'topic': topic,
                    'ground_truth': true_score,
                    'model_score': model_score_dir,
                    'method': 'direct',
                    'confidence': 0.7 + np.random.random() * 0.3,
                    'reasoning': f"Cultural norms in {country} regarding {topic}..."
                })
        
        # Calculate correlations
        df = pd.DataFrame(scores)
        lp_corr = df[df['method'] == 'logprob'][['ground_truth', 'model_score']].corr().iloc[0, 1]
        dir_corr = df[df['method'] == 'direct'][['ground_truth', 'model_score']].corr().iloc[0, 1]
        
        all_results[model_name] = {
            'scores': scores,
            'metrics': {
                'correlation_logprob': float(lp_corr),
                'correlation_direct': float(dir_corr),
                'mae_logprob': float(df[df['method'] == 'logprob']['model_score'].sub(df[df['method'] == 'logprob']['ground_truth']).abs().mean()),
                'mae_direct': float(df[df['method'] == 'direct']['model_score'].sub(df[df['method'] == 'direct']['ground_truth']).abs().mean()),
                'sample_size': len(scores)
            }
        }
    
    return all_results

def generate_paper_outputs():
    """Generate all paper outputs with mock data"""
    
    print("=" * 60)
    print("PAPER OUTPUT DEMONSTRATION")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path("outputs/paper_demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate mock results
    print("\n1️⃣ Generating realistic mock results...")
    mock_results = create_realistic_mock_results()
    
    # Save mock results
    results_file = output_dir / "mock_results.json"
    with open(results_file, 'w') as f:
        json.dump(mock_results, f, indent=2)
    print(f"✅ Saved mock results to {results_file}")
    
    # Initialize paper generator
    generator = PaperOutputGenerator(
        results_dir=output_dir,
        output_dir=output_dir / "paper"
    )
    
    # Save results in expected format
    for model, data in mock_results.items():
        model_file = output_dir / f"{model}_results.json"
        with open(model_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    print("\n2️⃣ Generating LaTeX Tables...")
    
    # Use the generate_all_outputs method which handles everything
    outputs = generator.generate_all_outputs()
    
    print(f"✅ Generated {len(outputs)} paper outputs")
    print(f"✅ Tables saved to: {output_dir}/paper/tables/")
    print(f"✅ Figures saved to: {output_dir}/paper/figures/")
    
    print("\n3️⃣ Generating Visualizations...")
    
    # Initialize moral visualization engine
    viz = MoralVisualizationEngine(output_dir=output_dir / "paper" / "figures")
    
    # Convert results to DataFrame for visualization
    all_scores = []
    for model, data in mock_results.items():
        for score in data['scores']:
            score['model'] = model
            all_scores.append(score)
    combined_df = pd.DataFrame(all_scores)
    
    # Generate all plots
    plots = viz.create_all_plots(results=mock_results, df=combined_df)
    print(f"✅ Generated {len(plots)} visualizations")
    
    for plot in plots:
        print(f"  → {Path(plot).name}")
    
    print("\n4️⃣ Generating LaTeX Integration Example...")
    
    # Create example LaTeX document showing how to include outputs
    latex_example = r"""
\documentclass{article}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{float}

\begin{document}

\section{Results}

\subsection{Model Performance}

Table~\ref{tab:performance} shows the correlation between model predictions and human moral judgments using both log-probability ($\rho^{LP}$) and direct scoring ($\rho^{Dir}$) methods.

\input{tables/table1_survey_alignment}

\subsection{Self-Consistency Analysis}

Table~\ref{tab:consistency} presents the self-consistency metrics for each model.

\input{tables/table2_self_consistency}

\subsection{Human Alignment}

Table~\ref{tab:human} shows the alignment between model predictions and human moral judgments.

\input{tables/table3_human_alignment}

\subsection{Visual Analysis}

\begin{figure}[H]
    \centering
    \includegraphics[width=\textwidth]{figures/figure2_country_correlations.png}
    \caption{Country-specific correlation patterns between models and human judgments.}
    \label{fig:country}
\end{figure}

\begin{figure}[H]
    \centering
    \includegraphics[width=\textwidth]{figures/figure3_error_density.png}
    \caption{Error density distributions for different scoring methods.}
    \label{fig:error}
\end{figure}

\begin{figure}[H]
    \centering
    \includegraphics[width=\textwidth]{figures/figure4_topic_errors.png}
    \caption{Topic-specific error patterns across models.}
    \label{fig:topics}
\end{figure}

\begin{figure}[H]
    \centering
    \includegraphics[width=\textwidth]{figures/figure5_regional_preferences.png}
    \caption{Regional moral preference patterns.}
    \label{fig:regional}
\end{figure}

\end{document}
"""
    
    latex_file = output_dir / "paper" / "example_integration.tex"
    latex_file.write_text(latex_example)
    print(f"✅ LaTeX Integration Example → {latex_file}")
    
    print("\n" + "=" * 60)
    print("📊 PAPER REPRESENTATION COMPLETE!")
    print("=" * 60)
    
    print("\n📁 Generated Files:")
    print(f"  Tables: {output_dir}/paper/*.tex")
    print(f"  Figures: {output_dir}/paper/figures/*.png")
    print(f"  LaTeX Example: {output_dir}/paper/example_integration.tex")
    
    print("\n💡 How to Use in Your Paper:")
    print("1. Copy the .tex files to your paper's directory")
    print("2. Use \\input{table1_model_performance} to include tables")
    print("3. Use \\includegraphics{figures/...} to include figures")
    print("4. See example_integration.tex for complete LaTeX code")
    
    print("\n📈 Key Findings from Mock Data:")
    for model, data in mock_results.items():
        metrics = data['metrics']
        print(f"\n{model}:")
        print(f"  ρ^LP = {metrics['correlation_logprob']:.3f}")
        print(f"  ρ^Dir = {metrics['correlation_direct']:.3f}")
        print(f"  MAE^LP = {metrics['mae_logprob']:.3f}")
        print(f"  MAE^Dir = {metrics['mae_direct']:.3f}")
    
    print("\n✨ Next Steps:")
    print("1. Run with real OpenAI API data: python run_openai_test.py")
    print("2. Increase sample size for more robust results")
    print("3. Add more models as API keys become available")
    print("4. Customize visualizations in visualization_engine.py")
    
    return output_dir

if __name__ == "__main__":
    output_dir = generate_paper_outputs()
    print(f"\n✅ All outputs saved to: {output_dir}")