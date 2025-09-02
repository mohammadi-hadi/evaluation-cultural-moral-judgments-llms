#!/usr/bin/env python3
"""
Create Test Dataset for Moral Alignment Evaluation
Generates a smaller random dataset for efficient testing
"""

import pandas as pd
import numpy as np
from pathlib import Path
import random
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_dataset(
    input_file: str = "sample_data/WVS_Moral.csv",
    output_file: str = "sample_data/test_dataset_5k.csv",
    n_samples: int = 5000,
    random_seed: int = 42,
    stratify_by_country: bool = True,
    min_samples_per_country: int = 10
):
    """
    Create a smaller test dataset from WVS data
    
    Args:
        input_file: Path to full WVS dataset
        output_file: Path to save test dataset
        n_samples: Number of samples to include
        random_seed: Random seed for reproducibility
        stratify_by_country: Whether to stratify by country
        min_samples_per_country: Minimum samples per country
    """
    logger.info(f"Loading WVS data from {input_file}")
    
    # Set random seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Load data
    df = pd.read_csv(input_file)
    logger.info(f"Loaded {len(df)} samples from {df['B_COUNTRY'].nunique()} countries")
    
    # Key moral questions to ensure are included
    moral_questions = ['Q176', 'Q177', 'Q178', 'Q179', 'Q180', 'Q181', 'Q182', 
                      'Q183', 'Q184', 'Q185', 'Q186', 'Q187', 'Q188', 'Q189',
                      'Q190', 'Q191', 'Q192', 'Q193', 'Q194', 'Q195', 'Q196', 
                      'Q197', 'Q198']
    
    if stratify_by_country:
        # Stratified sampling by country
        countries = df['B_COUNTRY'].unique()
        samples_per_country = max(min_samples_per_country, n_samples // len(countries))
        
        sampled_dfs = []
        for country in countries:
            country_df = df[df['B_COUNTRY'] == country]
            n_country_samples = min(len(country_df), samples_per_country)
            
            if len(country_df) > 0:
                country_sample = country_df.sample(n=n_country_samples, random_state=random_seed)
                sampled_dfs.append(country_sample)
        
        # Combine all samples
        test_df = pd.concat(sampled_dfs, ignore_index=True)
        
        # If we have more than needed, randomly sample down
        if len(test_df) > n_samples:
            test_df = test_df.sample(n=n_samples, random_state=random_seed)
        # If we have less, add more random samples
        elif len(test_df) < n_samples:
            remaining = n_samples - len(test_df)
            # Get samples not already included
            excluded_indices = set(df.index) - set(test_df.index)
            additional_samples = df.loc[list(excluded_indices)].sample(
                n=min(remaining, len(excluded_indices)), 
                random_state=random_seed
            )
            test_df = pd.concat([test_df, additional_samples], ignore_index=True)
    else:
        # Simple random sampling
        test_df = df.sample(n=min(n_samples, len(df)), random_state=random_seed)
    
    # Ensure we have responses for key moral questions
    logger.info(f"Test dataset contains {len(test_df)} samples")
    
    # Check coverage of moral questions
    for q in moral_questions:
        if q in test_df.columns:
            valid_responses = test_df[q].notna().sum()
            logger.info(f"  {q}: {valid_responses} valid responses ({valid_responses/len(test_df)*100:.1f}%)")
    
    # Save test dataset
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    test_df.to_csv(output_path, index=False)
    logger.info(f"Saved test dataset to {output_path}")
    
    # Print summary statistics
    logger.info("\n=== Test Dataset Summary ===")
    logger.info(f"Total samples: {len(test_df)}")
    logger.info(f"Countries: {test_df['B_COUNTRY'].nunique()}")
    
    # Check if year column exists
    if 'year' in test_df.columns:
        logger.info(f"Years: {test_df['year'].unique()}")
    
    # Sample distribution by country
    country_dist = test_df['B_COUNTRY'].value_counts().head(10)
    logger.info("\nTop 10 countries by sample count:")
    for country, count in country_dist.items():
        logger.info(f"  {country}: {count} samples")
    
    return test_df

def create_multiple_test_sets():
    """Create multiple test datasets of different sizes"""
    sizes = [1000, 2500, 5000]
    
    for size in sizes:
        logger.info(f"\n{'='*50}")
        logger.info(f"Creating test dataset with {size} samples")
        logger.info(f"{'='*50}")
        
        create_test_dataset(
            output_file=f"sample_data/test_dataset_{size}.csv",
            n_samples=size,
            stratify_by_country=True
        )

if __name__ == "__main__":
    # Create main 5K test dataset
    logger.info("Creating main 5000-sample test dataset")
    create_test_dataset(
        n_samples=5000,
        stratify_by_country=True
    )
    
    # Also create smaller datasets for quick testing
    logger.info("\nCreating additional test datasets")
    create_multiple_test_sets()