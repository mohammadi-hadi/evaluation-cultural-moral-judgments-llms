"""
Visualization module for plotting model evaluation results.
"""

import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def setup_plotting_style():
    """Set up consistent plotting style."""
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except:
        plt.style.use("seaborn-whitegrid")
    sns.set_palette("husl")
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["font.size"] = 12


def plot_correlation_scatter(
    df: pd.DataFrame, dataset_name: str, model_name: str, save_path: str = None
) -> None:
    """
    Create scatter plot of survey scores vs model predictions.

    Args:
        df: DataFrame with evaluation results
        dataset_name: 'WVS' or 'PEW'
        model_name: Name of the model
        save_path: Directory to save plot (optional)
    """
    if df.empty or "correlation" not in df.columns:
        print(f"No data to plot for {model_name} on {dataset_name}")
        return

    setup_plotting_style()

    # Get correlation values
    correlation = df["correlation"].iloc[0]
    p_value = df["p_value"].iloc[0]

    # Create scatter plot
    plt.figure(figsize=(10, 8))

    # Plot by country with different colors
    countries = df["country"].unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(countries)))

    for i, country in enumerate(countries):
        country_data = df[df["country"] == country]
        plt.scatter(
            country_data["survey_score"],
            country_data["normalized_log_prob_diff"],
            label=country,
            alpha=0.6,
            s=50,
            color=colors[i],
        )

    # Add correlation line
    x = df["survey_score"]
    y = df["normalized_log_prob_diff"]
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    plt.plot(x, p(x), "r--", alpha=0.8, linewidth=2)

    # Labels and title
    plt.xlabel(f"{dataset_name} Survey Score", fontsize=14)
    plt.ylabel("Normalized Log Probability Difference", fontsize=14)
    plt.title(
        f"{model_name} - {dataset_name}\nPearson r = {correlation:.3f}, p = {p_value:.2e}",
        fontsize=16,
    )

    # Add reference lines
    plt.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    plt.axvline(x=0, color="gray", linestyle="--", alpha=0.5)

    # Legend
    if len(countries) <= 10:
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)

    plt.tight_layout()

    # Save if path provided
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        filename = f"{model_name.replace('/', '_')}_{dataset_name}_scatter.png"
        plt.savefig(os.path.join(save_path, filename), dpi=300, bbox_inches="tight")
        print(f"Saved scatter plot to {os.path.join(save_path, filename)}")

    plt.show()
    plt.close()


def plot_model_comparison(
    results_dict: Dict[str, pd.DataFrame], dataset_name: str, save_path: str = None
) -> None:
    """
    Create bar plot comparing correlations across models.

    Args:
        results_dict: Dictionary mapping model names to result DataFrames
        dataset_name: 'WVS' or 'PEW'
        save_path: Directory to save plot (optional)
    """
    setup_plotting_style()

    # Extract correlations
    model_data = []
    for model_name, df in results_dict.items():
        if not df.empty and "correlation" in df.columns:
            correlation = df["correlation"].iloc[0]
            p_value = df["p_value"].iloc[0]
            model_data.append(
                {
                    "model": model_name.split("/")[-1],  # Simplify model name
                    "correlation": correlation,
                    "p_value": p_value,
                    "significant": p_value < 0.05,
                }
            )

    if not model_data:
        print("No correlation data to compare")
        return

    # Create DataFrame and sort
    comp_df = pd.DataFrame(model_data)
    comp_df = comp_df.sort_values("correlation", ascending=False)

    # Create bar plot
    plt.figure(figsize=(12, 8))

    # Color bars based on significance
    colors = ["darkgreen" if sig else "lightcoral" for sig in comp_df["significant"]]

    bars = plt.bar(comp_df["model"], comp_df["correlation"], color=colors, alpha=0.8)

    # Add value labels on bars
    for bar, corr, pval in zip(bars, comp_df["correlation"], comp_df["p_value"]):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{corr:.3f}\n(p={pval:.2e})",
            ha="center",
            va="bottom" if height > 0 else "top",
            fontsize=10,
        )

    # Labels and title
    plt.xlabel("Model", fontsize=14)
    plt.ylabel("Pearson Correlation (r)", fontsize=14)
    plt.title(f"Model Performance Comparison on {dataset_name}", fontsize=16)

    # Add reference line at 0
    plt.axhline(y=0, color="black", linestyle="-", alpha=0.3)

    # Rotate x labels
    plt.xticks(rotation=45, ha="right")

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="darkgreen", alpha=0.8, label="Significant (p < 0.05)"),
        Patch(facecolor="lightcoral", alpha=0.8, label="Not Significant"),
    ]
    plt.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()

    # Save if path provided
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        filename = f"model_comparison_{dataset_name}_bar.png"
        plt.savefig(os.path.join(save_path, filename), dpi=300, bbox_inches="tight")
        print(f"Saved comparison plot to {os.path.join(save_path, filename)}")

    plt.show()
    plt.close()


def plot_topic_heatmap(
    df: pd.DataFrame, model_name: str, dataset_name: str, save_path: str = None
) -> None:
    """
    Create heatmap showing model performance across topics and countries.

    Args:
        df: DataFrame with evaluation results
        model_name: Name of the model
        dataset_name: 'WVS' or 'PEW'
        save_path: Directory to save plot (optional)
    """
    if df.empty:
        print(f"No data to create heatmap for {model_name}")
        return

    setup_plotting_style()

    # Pivot data for heatmap
    pivot_df = df.pivot_table(
        values="normalized_log_prob_diff",
        index="topic",
        columns="country",
        aggfunc="mean",
    )

    # Create figure
    plt.figure(figsize=(20, 10))

    # Create heatmap
    sns.heatmap(
        pivot_df,
        cmap="RdBu_r",
        center=0,
        cbar_kws={"label": "Normalized Log Prob Difference"},
        fmt=".2f",
        linewidths=0.5,
    )

    plt.title(
        f"{model_name} - {dataset_name}\nMoral Judgments by Topic and Country",
        fontsize=16,
    )
    plt.xlabel("Country", fontsize=14)
    plt.ylabel("Topic", fontsize=14)

    plt.tight_layout()

    # Save if path provided
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        filename = f"{model_name.replace('/', '_')}_{dataset_name}_heatmap.png"
        plt.savefig(os.path.join(save_path, filename), dpi=300, bbox_inches="tight")
        print(f"Saved heatmap to {os.path.join(save_path, filename)}")

    plt.show()
    plt.close()


def plot_country_performance(
    df: pd.DataFrame,
    model_name: str,
    dataset_name: str,
    top_n: int = 10,
    save_path: str = None,
) -> None:
    """
    Plot model performance by country.

    Args:
        df: DataFrame with evaluation results
        model_name: Name of the model
        dataset_name: 'WVS' or 'PEW'
        top_n: Number of top/bottom countries to show
        save_path: Directory to save plot (optional)
    """
    if df.empty:
        print(f"No data to plot country performance for {model_name}")
        return

    setup_plotting_style()

    # Calculate correlation by country
    country_corrs = []
    for country in df["country"].unique():
        country_data = df[df["country"] == country]
        if len(country_data) > 3:  # Need at least 4 points for correlation
            corr = country_data["survey_score"].corr(
                country_data["normalized_log_prob_diff"]
            )
            country_corrs.append({"country": country, "correlation": corr})

    if not country_corrs:
        print("Not enough data points per country for correlation")
        return

    # Create DataFrame and sort
    country_df = pd.DataFrame(country_corrs)
    country_df = country_df.sort_values("correlation", ascending=False)

    # Select top and bottom countries
    top_countries = country_df.head(top_n)
    bottom_countries = country_df.tail(top_n)
    plot_df = pd.concat([top_countries, bottom_countries])

    # Create plot
    plt.figure(figsize=(12, 8))

    # Color based on positive/negative correlation
    colors = ["green" if c > 0 else "red" for c in plot_df["correlation"]]

    plt.barh(plot_df["country"], plot_df["correlation"], color=colors, alpha=0.7)

    # Labels and title
    plt.xlabel("Correlation with Survey Data", fontsize=14)
    plt.ylabel("Country", fontsize=14)
    plt.title(
        f"{model_name} - {dataset_name}\nTop and Bottom {top_n} Countries by Correlation",
        fontsize=16,
    )

    # Add vertical line at 0
    plt.axvline(x=0, color="black", linestyle="-", alpha=0.3)

    plt.tight_layout()

    # Save if path provided
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        filename = (
            f"{model_name.replace('/', '_')}_{dataset_name}_country_performance.png"
        )
        plt.savefig(os.path.join(save_path, filename), dpi=300, bbox_inches="tight")
        print(f"Saved country performance plot to {os.path.join(save_path, filename)}")

    plt.show()
    plt.close()


def create_summary_report(
    results_dict: Dict[str, Dict[str, pd.DataFrame]], save_path: str = None
) -> pd.DataFrame:
    """
    Create a summary report of all model evaluations.

    Args:
        results_dict: Nested dict {model_name: {dataset: DataFrame}}
        save_path: Directory to save report (optional)

    Returns:
        Summary DataFrame
    """
    summary_data = []

    for model_name, datasets in results_dict.items():
        for dataset_name, df in datasets.items():
            if not df.empty and "correlation" in df.columns:
                summary_data.append(
                    {
                        "Model": model_name,
                        "Dataset": dataset_name.upper(),
                        "Correlation": df["correlation"].iloc[0],
                        "P-value": df["p_value"].iloc[0],
                        "N_samples": len(df),
                        "N_countries": df["country"].nunique(),
                        "N_topics": df["topic"].nunique(),
                    }
                )

    summary_df = pd.DataFrame(summary_data)

    if not summary_df.empty:
        # Sort by correlation
        summary_df = summary_df.sort_values(
            ["Dataset", "Correlation"], ascending=[True, False]
        )

        # Format for display
        summary_df["Correlation"] = summary_df["Correlation"].round(3)
        summary_df["P-value"] = summary_df["P-value"].apply(lambda x: f"{x:.2e}")

        # Save if path provided
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            summary_df.to_csv(
                os.path.join(save_path, "model_evaluation_summary.csv"), index=False
            )
            print(
                f"Saved summary report to {os.path.join(save_path, 'model_evaluation_summary.csv')}"
            )

    return summary_df
