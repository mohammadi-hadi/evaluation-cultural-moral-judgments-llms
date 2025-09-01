#!/usr/bin/env python3
"""
Main script to evaluate all models on WVS and PEW datasets.
"""

import argparse
import os
import sys
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import (
    create_summary_report,
    ensure_directories,
    evaluate_model,
    format_time,
    get_available_models,
    load_pew_data,
    load_wvs_data,
    plot_correlation_scatter,
    plot_model_comparison,
    save_results,
    setup_logging,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate LLMs on cultural moral judgment tasks"
    )

    parser.add_argument(
        "--models",
        nargs="+",
        help="List of models to evaluate (default: all recommended models)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["wvs", "pew", "both"],
        default=["both"],
        help="Datasets to evaluate on",
    )
    parser.add_argument(
        "--output-dir", default="results", help="Directory to save results"
    )
    parser.add_argument(
        "--data-dir", default="data", help="Directory containing data files"
    )
    parser.add_argument(
        "--no-cuda", action="store_true", help="Disable CUDA even if available"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Run in debug mode with fewer samples"
    )
    parser.add_argument("--log-file", help="Path to log file")

    return parser.parse_args()


def main():
    """Main evaluation pipeline."""
    args = parse_args()

    # Setup
    ensure_directories()
    setup_logging(args.log_file)

    # Determine models to evaluate
    if args.models:
        models = args.models
    else:
        models = get_available_models()
        # Filter out API-based models for now
        models = [m for m in models if not m.startswith("gpt-4")]

    # Determine datasets
    datasets = []
    if "both" in args.datasets:
        datasets = ["wvs", "pew"]
    else:
        datasets = args.datasets

    # Load data
    print("Loading survey data...")
    data = {}

    if "wvs" in datasets:
        wvs_path = os.path.join(args.data_dir, "raw", "WVS_Moral.csv")
        country_codes_path = os.path.join(
            args.data_dir, "raw", "Country_Codes_Names.csv"
        )

        if os.path.exists(wvs_path):
            data["wvs"] = load_wvs_data(wvs_path, country_codes_path)
            print(f"Loaded WVS data: {len(data['wvs'])} records")
        else:
            print(f"Warning: WVS data not found at {wvs_path}")
            datasets.remove("wvs")

    if "pew" in datasets:
        pew_path = os.path.join(
            args.data_dir,
            "raw",
            "Pew Research Global Attitudes Project Spring 2013 Dataset for web.sav",
        )

        if os.path.exists(pew_path):
            data["pew"] = load_pew_data(pew_path)
            print(f"Loaded PEW data: {len(data['pew'])} records")
        else:
            print(f"Warning: PEW data not found at {pew_path}")
            datasets.remove("pew")

    if not datasets:
        print("Error: No datasets available for evaluation")
        return 1

    # Evaluate models
    results = {}
    use_cuda = not args.no_cuda

    for model_name in models:
        print(f"\n{'=' * 60}")
        print(f"Evaluating model: {model_name}")
        print(f"{'=' * 60}")

        model_results = {}
        start_time = time.time()

        try:
            for dataset in datasets:
                print(f"\nEvaluating on {dataset.upper()} dataset...")

                # Run evaluation
                df = evaluate_model(
                    model_name=model_name,
                    dataset=dataset,
                    use_cuda=use_cuda,
                    data_df=data[dataset],
                    cultures=None
                    if not args.debug
                    else ["United States", "China", "Germany"],
                )

                # Save results
                output_path = os.path.join(
                    args.output_dir,
                    "model_outputs",
                    f"{model_name.replace('/', '_')}_{dataset}.csv",
                )
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                df.to_csv(output_path, index=False)

                # Store for plotting
                model_results[dataset] = df

                # Print summary
                if not df.empty:
                    corr = df["correlation"].iloc[0]
                    pval = df["p_value"].iloc[0]
                    print(f"Results: correlation = {corr:.3f}, p-value = {pval:.2e}")

                    # Create scatter plot
                    plot_correlation_scatter(
                        df,
                        dataset.upper(),
                        model_name,
                        save_path=os.path.join(args.output_dir, "figures"),
                    )
                else:
                    print("No valid results obtained")

            results[model_name] = model_results

        except Exception as e:
            print(f"Error evaluating {model_name}: {str(e)}")
            continue

        elapsed = time.time() - start_time
        print(f"\nCompleted {model_name} in {format_time(elapsed)}")

    # Create comparison plots and summary
    if results:
        print(f"\n{'=' * 60}")
        print("Creating comparison plots and summary...")
        print(f"{'=' * 60}")

        # Comparison plots for each dataset
        for dataset in datasets:
            dataset_results = {
                model: res[dataset] for model, res in results.items() if dataset in res
            }

            if dataset_results:
                plot_model_comparison(
                    dataset_results,
                    dataset.upper(),
                    save_path=os.path.join(args.output_dir, "figures"),
                )

        # Create summary report
        summary = create_summary_report(results, save_path=args.output_dir)

        print("\nSummary Report:")
        print(summary.to_string(index=False))

        # Save full results
        save_results(
            {"models": list(results.keys()), "summary": summary.to_dict()},
            os.path.join(args.output_dir, "evaluation_results.json"),
        )

    print("\nEvaluation complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
