# Methodology

This document provides a detailed explanation of the methodology used in our study "Exploring Cultural Variations in Moral Judgments with Large Language Models".

## Overview

Our approach evaluates how well Large Language Models (LLMs) can capture cultural variations in moral judgments by comparing their outputs with human survey data from two major cross-cultural surveys:

1. **World Values Survey (WVS) Wave 7**: Covering 55+ countries
2. **PEW Research Global Attitudes Survey (Spring 2013)**: Covering 40 countries

## Key Components

### 1. Moral Justifiability Score

We use a log-probability-based approach to measure moral justifiability. For each moral topic and country, we:

1. Generate paired prompts comparing moral vs. non-moral framings
2. Calculate the log-probability difference between these framings
3. Normalize these differences to a [-1, 1] scale
4. Correlate with human survey responses

### 2. Prompt Design

We use five different token pairs to capture various aspects of moral judgment:

- `'always justifiable'` vs `'never justifiable'`
- `'right'` vs `'wrong'`
- `'morally good'` vs `'morally bad'`
- `'ethically right'` vs `'ethically wrong'`
- `'ethical'` vs `'unethical'`

For each topic and country, we generate prompts in the format:
```
"In [Country] [topic] is [moral/non-moral token]."
```

### 3. Log-Probability Calculation

For each prompt, we:

1. Feed the prompt to the LLM
2. Extract the log-probability of the final token (before EOS)
3. Calculate the difference: `log P(moral) - log P(non-moral)`
4. Average across all token pairs

This gives us a single moral justifiability score for each topic-country combination.

### 4. Normalization

To make scores comparable across different models and topics, we normalize the log-probability differences to a [-1, 1] range using min-max normalization:

```python
normalized = 2 * (score - min_score) / (max_score - min_score) - 1
```

### 5. Correlation Analysis

We use Pearson correlation to measure how well model outputs align with human survey data:

- **WVS**: Original 1-10 scale responses are transformed to [-1, 1]
- **PEW**: Responses are coded as -1 (unacceptable), 0 (neutral), 1 (acceptable)

## Model Evaluation

### Models Tested

We evaluate two categories of models:

1. **Smaller/Earlier Models**: GPT-2, OPT, BLOOMZ, Qwen
2. **Instruction-Tuned Models**: GPT-4o, GPT-4o-mini, Gemma-2-9b-it, Llama-3.3-70B-Instruct

### Memory Optimization

To handle large models efficiently:

- 8-bit quantization for models > 7B parameters
- Chunked processing with batch size of 8
- No gradient computation (`torch.no_grad()`)
- Disabled KV caching during inference

## Topics Covered

### WVS Topics (19 total)
- Government benefit fraud
- Fare evasion
- Theft
- Tax evasion
- Bribery
- Homosexuality
- Prostitution
- Abortion
- Divorce
- Premarital sex
- Suicide
- Euthanasia
- Domestic violence (spouse)
- Child abuse
- Violence
- Terrorism
- Casual sex
- Political violence
- Death penalty

### PEW Topics (8 total)
- Contraceptive use
- Divorce
- Abortion
- Homosexuality
- Alcohol consumption
- Extramarital affairs
- Gambling
- Premarital sex

## Limitations and Considerations

1. **Survey Data Limitations**: 
   - WVS Wave 7 data from 2017-2022
   - PEW data from 2013
   - Not all countries have responses for all topics

2. **Model Limitations**:
   - Log-probability approach may not capture all nuances
   - Models may have training data biases
   - Cultural representation in training data varies

3. **Prompt Sensitivity**:
   - Results may vary with different prompt formats
   - We use "In [Country]" format consistently

## Statistical Significance

We report:
- Pearson correlation coefficient (r)
- P-values for statistical significance
- Sample sizes (number of topic-country pairs)

Results are considered significant at p < 0.05.

## Reproducibility

To ensure reproducibility:

1. All random seeds are fixed
2. Model versions are specified
3. Data preprocessing steps are documented
4. Code is open-sourced

## Future Directions

Potential improvements include:

1. Testing more prompt variations
2. Including more recent survey data
3. Evaluating multilingual prompts
4. Analyzing topic-specific performance
5. Investigating regional patterns 