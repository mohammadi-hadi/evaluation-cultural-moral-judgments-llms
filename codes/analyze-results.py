#!/usr/bin/env python3
"""
Analysis and visualization script for moral alignment results
Generates all figures and tables from the paper
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

OUT_DIR = Path("outputs")
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ========================= Data Loading =========================

def load_results():
    """Load all experimental results"""
    results = {}
    
    # Load scores
    if (OUT_DIR / "logprob_scores.csv").exists():
        results["logprob"] = pd.read_csv(OUT_DIR / "logprob_scores.csv")
    
    if (OUT_DIR / "direct_scores.csv").exists():
        results["direct"] = pd.read_csv(OUT_DIR / "direct_scores.csv")
    
    # Load metrics
    if (OUT_DIR / "metrics.csv").exists():
        results["metrics"] = pd.read_csv(OUT_DIR / "metrics.csv")
    
    # Load traces
    if (OUT_DIR / "traces.jsonl").exists():
        results["traces"] = pd.read_json(OUT_DIR / "traces.jsonl", lines=True)
    
    return results

# ========================= Analysis Functions =========================

def analyze_correlations(metrics_df):
    """Analyze correlation patterns across models and methods"""
    
    # Create correlation comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # WVS correlations
    wvs_data = metrics_df[metrics_df["source"] == "WVS"]
    wvs_pivot = wvs_data.pivot_table(
        index="model", 
        columns="method", 
        values="pearson_r"
    )
    
    if not wvs_pivot.empty:
        wvs_pivot.plot(kind="bar", ax=axes[0], width=0.8)
        axes[0].set_title("WVS Alignment by Model and Method", fontsize=14, fontweight='bold')
        axes[0].set_xlabel("Model", fontsize=12)
        axes[0].set_ylabel("Pearson Correlation", fontsize=12)
        axes[0].legend(title="Method", loc='upper right')
        axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes[0].grid(axis='y', alpha=0.3)
        axes[0].set_ylim(-0.5, 1.0)
    
    # PEW correlations
    pew_data = metrics_df[metrics_df["source"] == "PEW"]
    pew_pivot = pew_data.pivot_table(
        index="model", 
        columns="method", 
        values="pearson_r"
    )
    
    if not pew_pivot.empty:
        pew_pivot.plot(kind="bar", ax=axes[1], width=0.8)
        axes[1].set_title("PEW Alignment by Model and Method", fontsize=14, fontweight='bold')
        axes[1].set_xlabel("Model", fontsize=12)
        axes[1].set_ylabel("Pearson Correlation", fontsize=12)
        axes[1].legend(title="Method", loc='upper right')
        axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes[1].grid(axis='y', alpha=0.3)
        axes[1].set_ylim(-0.5, 1.0)
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / "correlation_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Statistical analysis
    print("="*60)
    print("CORRELATION ANALYSIS")
    print("="*60)
    
    # Test if direct > logprob systematically
    if "logprob" in wvs_pivot.columns and "direct" in wvs_pivot.columns:
        diff = wvs_pivot["direct"] - wvs_pivot["logprob"]
        t_stat, p_val = stats.ttest_rel(wvs_pivot["direct"], wvs_pivot["logprob"])
        
        print(f"\nPaired t-test (direct vs logprob on WVS):")
        print(f"  Mean difference: {diff.mean():.3f} ± {diff.std():.3f}")
        print(f"  t-statistic: {t_stat:.3f}, p-value: {p_val:.4f}")
        
        if p_val < 0.05:
            print("  → Direct scores significantly outperform log-prob scores")
        else:
            print("  → No significant difference between methods")

def analyze_geographic_patterns(scores_df, survey_data):
    """Analyze geographic patterns in model performance"""
    
    # Define regions (simplified)
    regions = {
        "Western": ["United States", "Germany", "United Kingdom", "France", "Canada"],
        "Eastern": ["China", "Japan", "South Korea", "India", "Thailand"],
        "African": ["Nigeria", "Kenya", "South Africa", "Egypt", "Ghana"],
        "Latin": ["Brazil", "Mexico", "Argentina", "Chile", "Peru"],
        "Middle East": ["Turkey", "Iran", "Saudi Arabia", "Jordan", "Lebanon"]
    }
    
    # Create region mapping
    region_map = {}
    for region, countries in regions.items():
        for country in countries:
            region_map[country] = region
    
    # Add region column
    scores_df["region"] = scores_df["country"].map(region_map).fillna("Other")
    
    # Calculate regional performance
    regional_perf = scores_df.groupby(["model", "region"])["pred_score"].mean().reset_index()
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    pivot = regional_perf.pivot(index="model", columns="region", values="pred_score")
    
    if not pivot.empty:
        sns.heatmap(
            pivot, 
            annot=True, 
            fmt=".2f", 
            cmap="RdBu_r", 
            center=0,
            cbar_kws={"label": "Mean Predicted Score"},
            vmin=-1, 
            vmax=1
        )
        plt.title("Model Performance by Geographic Region", fontsize=14, fontweight='bold')
        plt.xlabel("Region", fontsize=12)
        plt.ylabel("Model", fontsize=12)
        plt.tight_layout()
        plt.savefig(FIG_DIR / "geographic_patterns.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    # Statistical tests
    print("\n" + "="*60)
    print("GEOGRAPHIC BIAS ANALYSIS")
    print("="*60)
    
    for model in scores_df["model"].unique():
        model_data = scores_df[scores_df["model"] == model]
        
        # ANOVA across regions
        region_groups = [group["pred_score"].values for _, group in model_data.groupby("region")]
        if len(region_groups) > 1:
            f_stat, p_val = stats.f_oneway(*region_groups)
            
            print(f"\n{model}:")
            print(f"  F-statistic: {f_stat:.3f}, p-value: {p_val:.4f}")
            
            if p_val < 0.05:
                print("  → Significant regional differences detected")
                
                # Post-hoc: find best and worst regions
                regional_means = model_data.groupby("region")["pred_score"].mean()
                best_region = regional_means.idxmax()
                worst_region = regional_means.idxmin()
                
                print(f"  Best region: {best_region} (mean={regional_means[best_region]:.3f})")
                print(f"  Worst region: {worst_region} (mean={regional_means[worst_region]:.3f})")

def analyze_topic_difficulty(scores_df, survey_data):
    """Analyze which topics are hardest for models"""
    
    # Merge predictions with ground truth
    merged = scores_df.merge(
        survey_data[["country", "topic", "score"]], 
        on=["country", "topic"],
        how="inner"
    )
    
    # Calculate error by topic
    merged["error"] = np.abs(merged["pred_score"] - merged["score"])
    topic_errors = merged.groupby("topic")["error"].agg(["mean", "std"]).reset_index()
    topic_errors = topic_errors.sort_values("mean", ascending=False)
    
    # Create difficulty plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    topics = topic_errors["topic"].head(10)
    means = topic_errors["mean"].head(10)
    stds = topic_errors["std"].head(10)
    
    bars = ax.barh(range(len(topics)), means, xerr=stds, capsize=5)
    
    # Color bars by difficulty
    colors = plt.cm.YlOrRd(means / means.max())
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    ax.set_yticks(range(len(topics)))
    ax.set_yticklabels(topics)
    ax.set_xlabel("Mean Absolute Error", fontsize=12)
    ax.set_title("Most Challenging Moral Topics", fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / "topic_difficulty.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print detailed analysis
    print("\n" + "="*60)
    print("TOPIC DIFFICULTY ANALYSIS")
    print("="*60)
    print("\nMost challenging topics (by mean absolute error):")
    
    for i, row in topic_errors.head(10).iterrows():
        print(f"  {i+1}. {row['topic']}: MAE={row['mean']:.3f} ± {row['std']:.3f}")
    
    # Analyze patterns in difficult topics
    violence_topics = ["political violence", "terrorism", "wife-beating", "violence"]
    personal_topics = ["suicide", "euthanasia", "abortion", "homosexuality"]
    
    violence_errors = topic_errors[topic_errors["topic"].isin(violence_topics)]["mean"].mean()
    personal_errors = topic_errors[topic_errors["topic"].isin(personal_topics)]["mean"].mean()
    other_errors = topic_errors[~topic_errors["topic"].isin(violence_topics + personal_topics)]["mean"].mean()
    
    print(f"\nError by topic category:")
    print(f"  Violence-related: {violence_errors:.3f}")
    print(f"  Personal/life issues: {personal_errors:.3f}")
    print(f"  Other topics: {other_errors:.3f}")

def cluster_analysis(scores_df):
    """Perform clustering analysis on model behaviors"""
    
    # Create model-topic matrix
    pivot = scores_df.pivot_table(
        index="model", 
        columns="topic", 
        values="pred_score",
        aggfunc="mean"
    ).fillna(0)
    
    if pivot.shape[0] < 3:
        print("Not enough models for clustering analysis")
        return
    
    # PCA for visualization
    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(pivot)
    
    # K-means clustering
    n_clusters = min(3, pivot.shape[0])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(pivot)
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # PCA plot
    scatter = axes[0].scatter(
        pca_coords[:, 0], 
        pca_coords[:, 1], 
        c=clusters, 
        cmap='viridis',
        s=100,
        alpha=0.7
    )
    
    for i, model in enumerate(pivot.index):
        axes[0].annotate(
            model, 
            (pca_coords[i, 0], pca_coords[i, 1]),
            xytext=(5, 5), 
            textcoords='offset points',
            fontsize=9
        )
    
    axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)", fontsize=12)
    axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)", fontsize=12)
    axes[0].set_title("Model Clustering in Moral Space", fontsize=14, fontweight='bold')
    axes[0].grid(alpha=0.3)
    
    # Dendrogram
    from scipy.cluster.hierarchy import dendrogram, linkage
    
    linkage_matrix = linkage(pivot, method='ward')
    dendrogram(linkage_matrix, labels=pivot.index, ax=axes[1], orientation='right')
    axes[1].set_xlabel("Distance", fontsize=12)
    axes[1].set_title("Hierarchical Clustering of Models", fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / "model_clustering.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n" + "="*60)
    print("CLUSTERING ANALYSIS")
    print("="*60)
    print(f"\nFound {n_clusters} distinct model clusters:")
    
    for i in range(n_clusters):
        cluster_models = pivot.index[clusters == i].tolist()
        print(f"\nCluster {i+1}: {', '.join(cluster_models)}")

def generate_latex_tables(metrics_df):
    """Generate LaTeX tables for the paper"""
    
    # Main results table
    pivot = metrics_df.pivot_table(
        index="model",
        columns=["source", "method"],
        values="pearson_r"
    ).round(3)
    
    latex_table = pivot.to_latex(
        caption="Model-Survey Alignment (Pearson Correlation)",
        label="tab:main_results",
        bold_rows=True,
        column_format="l" + "c" * len(pivot.columns)
    )
    
    with open(FIG_DIR / "main_results_table.tex", "w") as f:
        f.write(latex_table)
    
    print("\n" + "="*60)
    print("LATEX TABLES GENERATED")
    print("="*60)
    print(f"Saved to {FIG_DIR}/main_results_table.tex")

# ========================= Main Analysis =========================

def main():
    """Run complete analysis pipeline"""
    
    print("="*60)
    print("MORAL ALIGNMENT RESULTS ANALYSIS")
    print("="*60)
    
    # Load data
    print("\nLoading results...")
    results = load_results()
    
    if not results:
        print("No results found. Please run the experiments first.")
        return
    
    # Load survey data (create synthetic if needed)
    from moral_alignment_pipeline import load_surveys
    wvs, pew = load_surveys()
    all_survey = pd.concat([wvs, pew], ignore_index=True)
    
    # Run analyses
    if "metrics" in results:
        print("\nAnalyzing correlations...")
        analyze_correlations(results["metrics"])
    
    if "direct" in results:
        print("\nAnalyzing geographic patterns...")
        analyze_geographic_patterns(results["direct"], all_survey)
        
        print("\nAnalyzing topic difficulty...")
        analyze_topic_difficulty(results["direct"], all_survey)
        
        print("\nPerforming clustering analysis...")
        cluster_analysis(results["direct"])
    
    if "metrics" in results:
        print("\nGenerating LaTeX tables...")
        generate_latex_tables(results["metrics"])
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nAll figures saved to: {FIG_DIR}/")
    print(f"All outputs saved to: {OUT_DIR}/")

if __name__ == "__main__":
    main()
