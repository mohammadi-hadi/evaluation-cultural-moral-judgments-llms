"""
Utility functions for the cultural moral judgments project.
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List

import yaml


def setup_logging(log_file: str = None, level: str = "INFO"):
    """
    Set up logging configuration.

    Args:
        log_file: Path to log file (optional)
        level: Logging level
    """
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        logging.basicConfig(
            level=getattr(logging, level),
            format=log_format,
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )
    else:
        logging.basicConfig(level=getattr(logging, level), format=log_format)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary
    """
    with open(config_path, "r") as f:
        if config_path.endswith(".yaml") or config_path.endswith(".yml"):
            return yaml.safe_load(f)
        elif config_path.endswith(".json"):
            return json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path}")


def save_results(results: Dict[str, Any], output_path: str):
    """
    Save results to JSON file.

    Args:
        results: Results dictionary
        output_path: Path to save results
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Add metadata
    results["metadata"] = {"timestamp": datetime.now().isoformat(), "version": "1.0.0"}

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)


def get_model_short_name(model_name: str) -> str:
    """
    Get shortened version of model name for display.

    Args:
        model_name: Full model name

    Returns:
        Shortened name
    """
    # Common patterns to simplify
    replacements = {
        "meta-llama/": "",
        "google/": "",
        "openai/": "",
        "Meta-Llama-3": "Llama-3",
        "-Instruct": "-I",
        "gemma-": "Gemma-",
    }

    short_name = model_name
    for old, new in replacements.items():
        short_name = short_name.replace(old, new)

    return short_name


def ensure_directories(base_path: str = "."):
    """
    Ensure all required directories exist.

    Args:
        base_path: Base directory path
    """
    directories = [
        "data/raw",
        "data/processed",
        "results/model_outputs",
        "results/figures",
        "logs",
    ]

    for directory in directories:
        full_path = os.path.join(base_path, directory)
        os.makedirs(full_path, exist_ok=True)

        # Create .gitkeep files for empty directories
        gitkeep_path = os.path.join(full_path, ".gitkeep")
        if not os.path.exists(gitkeep_path):
            open(gitkeep_path, "a").close()


def get_available_models() -> List[str]:
    """
    Get list of recommended models for evaluation.

    Returns:
        List of model names
    """
    return [
        # Smaller/Earlier models
        "gpt2",
        "gpt2-medium",
        "gpt2-large",
        "facebook/opt-1.3b",
        "facebook/opt-2.7b",
        "bigscience/bloomz-560m",
        "bigscience/bloomz-1b7",
        "Qwen/Qwen-1_8B",
        # Instruction-tuned models
        "google/gemma-2-9b-it",
        "meta-llama/Llama-3.3-70B-Instruct",
        "meta-llama/Meta-Llama-3-8B-Instruct",
        # Note: GPT-4 models require OpenAI API
        # "gpt-4o",
        # "gpt-4o-mini"
    ]


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"
