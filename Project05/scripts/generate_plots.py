#!/usr/bin/env python3
"""
Script to generate plots from existing model evaluation results.
"""

import argparse
import glob
import os
import sys

import pandas as pd

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import (
    create_summary_report,
    plot_correlation_scatter,
    plot_country_performance,
    plot_model_comparison,
    plot_topic_heatmap,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate plots from model evaluation results"
    )

    parser.add_argument(
        "--results-dir",
        default="results/model_outputs",
        help="Directory containing model output CSV files",
    )
    parser.add_argument(
        "--output-dir", default="results/figures", help="Directory to save plots"
    )
    parser.add_argument(
        "--plot-types",
        nargs="+",
        choices=["scatter", "comparison", "heatmap", "country", "all"],
        default=["all"],
        help="Types of plots to generate",
    )

    return parser.parse_args()


def load_results(results_dir: str) -> dict:
    """Load all result CSV files from directory."""
    results = {}

    csv_files = glob.glob(os.path.join(results_dir, "*.csv"))

    for csv_file in csv_files:
        # Extract model name and dataset from filename
        filename = os.path.basename(csv_file)
        parts = filename.replace(".csv", "").split("_")

        # Assume format: model_name_dataset.csv
        if len(parts) >= 2:
            dataset = parts[-1].lower()
            model_name = "_".join(parts[:-1])

            # Load DataFrame
            df = pd.read_csv(csv_file)

            if model_name not in results:
                results[model_name] = {}

            results[model_name][dataset] = df
            print(f"Loaded {model_name} - {dataset}: {len(df)} records")

    return results


def main():
    """Main function to generate plots."""
    args = parse_args()

    # Load results
    print("Loading results...")
    results = load_results(args.results_dir)

    if not results:
        print("No results found to plot!")
        return 1

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    plot_all = "all" in args.plot_types

    # Generate plots for each model
    for model_name, model_results in results.items():
        print(f"\nGenerating plots for {model_name}...")

        for dataset, df in model_results.items():
            if plot_all or "scatter" in args.plot_types:
                print(f"  - Creating scatter plot for {dataset}")
                plot_correlation_scatter(
                    df, dataset.upper(), model_name, args.output_dir
                )

            if plot_all or "heatmap" in args.plot_types:
                print(f"  - Creating heatmap for {dataset}")
                plot_topic_heatmap(df, model_name, dataset.upper(), args.output_dir)

            if plot_all or "country" in args.plot_types:
                print(f"  - Creating country performance plot for {dataset}")
                plot_country_performance(
                    df, model_name, dataset.upper(), top_n=10, save_path=args.output_dir
                )

    # Generate comparison plots
    if plot_all or "comparison" in args.plot_types:
        print("\nGenerating comparison plots...")

        # Group by dataset
        for dataset in ["wvs", "pew"]:
            dataset_results = {
                model: res[dataset] for model, res in results.items() if dataset in res
            }

            if dataset_results:
                print(f"  - Creating comparison plot for {dataset.upper()}")
                plot_model_comparison(dataset_results, dataset.upper(), args.output_dir)

    # Create summary report
    print("\nCreating summary report...")
    summary = create_summary_report(results, args.output_dir)
    print("\nSummary:")
    print(summary.to_string(index=False))

    print(f"\nAll plots saved to {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
