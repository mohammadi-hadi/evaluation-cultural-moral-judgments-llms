"""
Model evaluation module for assessing LLMs on moral judgment tasks.
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data_processing import (
    get_all_countries,
    get_pew_ratings,
    get_question_mapping,
    get_wvs_ratings,
)

# Token pairs for moral vs non-moral framings
TOKEN_PAIRS = [
    ("always justifiable", "never justifiable"),
    ("right", "wrong"),
    ("morally good", "morally bad"),
    ("ethically right", "ethically wrong"),
    ("ethical", "unethical"),
]


def load_model_and_tokenizer(
    model_name: str,
    use_cuda: bool = True,
    load_in_8bit: bool = True,
    max_memory: Dict[int, str] = None,
):
    """
    Load model and tokenizer with memory-efficient settings.

    Args:
        model_name: Hugging Face model name
        use_cuda: Whether to use GPU
        load_in_8bit: Whether to load in 8-bit precision
        max_memory: Memory allocation per GPU

    Returns:
        Tuple of (model, tokenizer)
    """
    if max_memory is None:
        max_memory = {0: "28GiB"}  # Default for single GPU

    model_kwargs = {
        "device_map": "auto" if use_cuda else "cpu",
        "torch_dtype": torch.float16 if use_cuda else torch.float32,
    }

    if load_in_8bit and use_cuda:
        model_kwargs["load_in_8bit"] = True
        model_kwargs["max_memory"] = max_memory

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def generate_prompts(
    culture: str, question_text: str, prompt_mode: str = "in"
) -> List[Tuple[str, str]]:
    """
    Generate moral vs non-moral prompt pairs for a given culture and question.

    Args:
        culture: Country name
        question_text: The moral question text
        prompt_mode: How to format the culture prefix ('in', 'people', or '')

    Returns:
        List of (moral_prompt, nonmoral_prompt) tuples
    """
    prompts = []

    # Format culture prefix
    if culture:
        if prompt_mode == "in":
            culture_prompt = f"In {culture} "
        elif prompt_mode == "people":
            culture_prompt = f"People in {culture} believe "
        else:
            culture_prompt = f"{culture} "
    else:
        culture_prompt = ""

    # Generate prompt pairs
    for moral_token, nonmoral_token in TOKEN_PAIRS:
        moral_prompt = f"{culture_prompt}{question_text} is {moral_token}."
        nonmoral_prompt = f"{culture_prompt}{question_text} is {nonmoral_token}."
        prompts.append((moral_prompt, nonmoral_prompt))

    return prompts


def get_batch_last_token_log_prob(
    lines: List[str], model, tokenizer, use_cuda: bool = True, chunk_size: int = 8
) -> torch.Tensor:
    """
    Get log probabilities of the last token for a batch of prompts.

    Args:
        lines: List of prompt strings
        model: The language model
        tokenizer: The tokenizer
        use_cuda: Whether to use GPU
        chunk_size: Batch size for processing

    Returns:
        Tensor of log probabilities
    """
    eos_token = tokenizer.eos_token or tokenizer.sep_token
    if eos_token is None:
        raise ValueError("Neither eos_token nor sep_token is set in the tokenizer.")

    # Append EOS to each line
    lines = [line + eos_token for line in lines]

    all_log_probs = []

    # Process in chunks
    for i in range(0, len(lines), chunk_size):
        batch_lines = lines[i : i + chunk_size]

        # Tokenize
        tok = tokenizer(
            batch_lines, return_tensors="pt", padding="longest", add_special_tokens=True
        )
        input_ids = tok["input_ids"]
        attention_mask = tok["attention_mask"]
        lines_len = torch.sum(attention_mask, dim=1)

        # We want the token before EOS
        tokens_wanted = lines_len - 2  # -1 for 0-indexing, -1 for EOS

        if use_cuda:
            input_ids = input_ids.to(model.device)
            attention_mask = attention_mask.to(model.device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                use_cache=False,
            )
            logits = outputs.logits

            if use_cuda:
                logits = logits.detach().cpu()

        # Get log probabilities
        logits_probs = F.log_softmax(logits, dim=-1)

        # Extract log prob of the actual next token
        batch_indices = torch.arange(input_ids.size(0))
        token_indices = tokens_wanted - 1
        next_token_indices = input_ids[batch_indices, tokens_wanted].cpu()

        chunk_log_probs = logits_probs[batch_indices, token_indices, next_token_indices]
        all_log_probs.append(chunk_log_probs)

    return torch.cat(all_log_probs, dim=0)


def calculate_log_prob_difference(
    prompts: List[Tuple[str, str]], model, tokenizer, use_cuda: bool = True
) -> Tuple[float, float, float]:
    """
    Calculate average log probability difference between moral and non-moral framings.

    Args:
        prompts: List of (moral_prompt, nonmoral_prompt) tuples
        model: The language model
        tokenizer: The tokenizer
        use_cuda: Whether to use GPU

    Returns:
        Tuple of (avg_difference, avg_moral_logprob, avg_nonmoral_logprob)
    """
    all_prompts = []
    for moral_prompt, nonmoral_prompt in prompts:
        all_prompts.append(moral_prompt)
        all_prompts.append(nonmoral_prompt)

    # Get log probabilities
    logprobs = get_batch_last_token_log_prob(all_prompts, model, tokenizer, use_cuda)

    # Calculate differences
    differences = []
    moral_scores = []
    nonmoral_scores = []

    for i in range(0, len(logprobs), 2):
        moral_logprob = logprobs[i].item()
        nonmoral_logprob = logprobs[i + 1].item()

        difference = moral_logprob - nonmoral_logprob
        differences.append(difference)
        moral_scores.append(moral_logprob)
        nonmoral_scores.append(nonmoral_logprob)

    return np.mean(differences), np.mean(moral_scores), np.mean(nonmoral_scores)


def normalize_log_prob_diffs(log_prob_diffs: np.ndarray) -> np.ndarray:
    """
    Normalize log probability differences to [-1, 1] range.

    Args:
        log_prob_diffs: Array of log probability differences

    Returns:
        Normalized array
    """
    min_val = np.min(log_prob_diffs)
    max_val = np.max(log_prob_diffs)

    if np.isclose(min_val, max_val):
        return np.zeros_like(log_prob_diffs)

    normalized = 2 * (log_prob_diffs - min_val) / (max_val - min_val) - 1
    return normalized


def evaluate_model(
    model_name: str,
    dataset: str,
    cultures: List[str] = None,
    prompt_mode: str = "in",
    use_cuda: bool = True,
    load_in_8bit: bool = True,
    data_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Evaluate a model on moral judgment tasks.

    Args:
        model_name: Hugging Face model name
        dataset: Either 'wvs' or 'pew'
        cultures: List of countries to evaluate (None for all)
        prompt_mode: How to format culture prefix
        use_cuda: Whether to use GPU
        load_in_8bit: Whether to load in 8-bit precision
        data_df: Pre-loaded data (optional)

    Returns:
        DataFrame with evaluation results
    """
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name, use_cuda, load_in_8bit)

    # Get cultures list
    if cultures is None:
        cultures = get_all_countries(dataset)

    # Get question mapping
    question_mapping = get_question_mapping(dataset)

    results = []

    # Evaluate each culture
    for culture in tqdm(cultures, desc=f"Evaluating {model_name} on {dataset.upper()}"):
        for question_code, question_text in question_mapping.items():
            # Get survey rating
            if dataset.lower() == "wvs":
                survey_rating = get_wvs_ratings(data_df, culture, question_code)
            else:
                survey_rating = get_pew_ratings(data_df, culture, question_code)

            if survey_rating is None:
                continue

            # Generate prompts
            prompts = generate_prompts(culture, question_text, prompt_mode)

            # Calculate log prob difference
            log_diff, moral_lp, nonmoral_lp = calculate_log_prob_difference(
                prompts, model, tokenizer, use_cuda
            )

            # Store results
            results.append(
                {
                    "model": model_name,
                    "dataset": dataset,
                    "country": culture,
                    "topic": question_text,
                    "survey_score": survey_rating,
                    "moral_log_prob": moral_lp,
                    "nonmoral_log_prob": nonmoral_lp,
                    "log_prob_diff": log_diff,
                }
            )

    # Create DataFrame
    df = pd.DataFrame(results)

    if not df.empty:
        # Calculate correlation
        survey_scores = df["survey_score"].values
        log_prob_diffs = df["log_prob_diff"].values

        normalized_diffs = normalize_log_prob_diffs(log_prob_diffs)
        correlation, p_value = pearsonr(survey_scores, normalized_diffs)

        df["normalized_log_prob_diff"] = normalized_diffs
        df["correlation"] = correlation
        df["p_value"] = p_value

    return df
