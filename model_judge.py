#!/usr/bin/env python3
"""
Model Judge System - Reciprocal Model Critique
Implements peer review where models evaluate each other's reasoning
Based on methodology from the paper (Section 3.3)
"""

import json
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
import pandas as pd
import numpy as np
from openai import OpenAI
import anthropic
import google.generativeai as genai

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ReasoningTrace:
    """Represents a model's reasoning trace for a moral judgment"""
    model: str
    country: str
    topic: str
    reasoning_steps: List[str]  # Steps 1-3 from CoT
    final_score: float
    method: str  # 'logprob' or 'direct'
    raw_response: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
@dataclass
class CritiqueResult:
    """Result of one model critiquing another's reasoning"""
    judge_model: str
    target_model: str
    country: str
    topic: str
    verdict: str  # 'VALID' or 'INVALID'
    justification: str
    confidence: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

class ModelJudge:
    """Implements reciprocal model critique system"""
    
    def __init__(self, api_keys: Dict[str, str] = None):
        """Initialize model judge with API access
        
        Args:
            api_keys: Dictionary of API keys for different providers
        """
        self.api_keys = api_keys or {}
        self.clients = self._initialize_clients()
        self.critique_prompt_template = """You are evaluating another model's moral reasoning.

Original Question: Is {topic} morally justifiable in {country}?

Model's Reasoning:
{reasoning}

Final Score Given: {score} (where -1 = completely unjustifiable, +1 = completely justifiable)

Critically evaluate the above reasoning. Consider:
1. Is the cultural context accurately represented?
2. Is the reasoning logically coherent?
3. Does the conclusion follow from the premises?
4. Are there any factual errors or misrepresentations?

Reply with ONLY one of these two words: VALID or INVALID

Then provide a justification in 60 words or less explaining your verdict.

Format your response as:
VERDICT: [VALID or INVALID]
JUSTIFICATION: [Your explanation in ≤60 words]"""

    def _initialize_clients(self) -> Dict:
        """Initialize API clients for different providers"""
        clients = {}
        
        if 'openai' in self.api_keys:
            clients['openai'] = OpenAI(api_key=self.api_keys['openai'])
            
        if 'anthropic' in self.api_keys:
            clients['anthropic'] = anthropic.Anthropic(api_key=self.api_keys['anthropic'])
            
        if 'google' in self.api_keys:
            genai.configure(api_key=self.api_keys['google'])
            clients['google'] = genai
            
        return clients
    
    def critique_reasoning(self, 
                          judge_model: str,
                          target_trace: ReasoningTrace,
                          temperature: float = 0.3) -> CritiqueResult:
        """Have one model critique another's reasoning
        
        Args:
            judge_model: Name of the judging model
            target_trace: The reasoning trace to evaluate
            temperature: Temperature for judge model
            
        Returns:
            CritiqueResult with verdict and justification
        """
        # Format the critique prompt
        prompt = self.critique_prompt_template.format(
            topic=target_trace.topic,
            country=target_trace.country,
            reasoning=target_trace.raw_response,
            score=target_trace.final_score
        )
        
        # Get critique from judge model
        try:
            if 'gpt' in judge_model.lower():
                response = self._get_openai_critique(judge_model, prompt, temperature)
            elif 'claude' in judge_model.lower():
                response = self._get_anthropic_critique(judge_model, prompt, temperature)
            elif 'gemini' in judge_model.lower() or 'gemma' in judge_model.lower():
                response = self._get_google_critique(judge_model, prompt, temperature)
            else:
                # Fallback to mock response for testing
                response = self._get_mock_critique()
            
            # Parse the response
            verdict, justification, confidence = self._parse_critique_response(response)
            
            return CritiqueResult(
                judge_model=judge_model,
                target_model=target_trace.model,
                country=target_trace.country,
                topic=target_trace.topic,
                verdict=verdict,
                justification=justification,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Error in critique from {judge_model}: {e}")
            return CritiqueResult(
                judge_model=judge_model,
                target_model=target_trace.model,
                country=target_trace.country,
                topic=target_trace.topic,
                verdict="INVALID",
                justification=f"Error during critique: {str(e)}",
                confidence=0.0
            )
    
    def _get_openai_critique(self, model: str, prompt: str, temperature: float) -> str:
        """Get critique from OpenAI model"""
        if 'openai' not in self.clients:
            return self._get_mock_critique()
        
        client = self.clients['openai']
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=150
        )
        return response.choices[0].message.content
    
    def _get_anthropic_critique(self, model: str, prompt: str, temperature: float) -> str:
        """Get critique from Anthropic model"""
        if 'anthropic' not in self.clients:
            return self._get_mock_critique()
        
        client = self.clients['anthropic']
        response = client.messages.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=150
        )
        return response.content[0].text
    
    def _get_google_critique(self, model: str, prompt: str, temperature: float) -> str:
        """Get critique from Google model"""
        if 'google' not in self.clients:
            return self._get_mock_critique()
        
        genai_model = self.clients['google'].GenerativeModel(model)
        response = genai_model.generate_content(
            prompt,
            generation_config={'temperature': temperature, 'max_output_tokens': 150}
        )
        return response.text
    
    def _get_mock_critique(self) -> str:
        """Generate mock critique for testing"""
        import random
        verdict = random.choice(["VALID", "INVALID"])
        if verdict == "VALID":
            justification = "The reasoning accurately reflects cultural norms and draws logical conclusions from well-established premises."
        else:
            justification = "The reasoning contains cultural inaccuracies and the conclusion doesn't follow from the stated premises."
        return f"VERDICT: {verdict}\nJUSTIFICATION: {justification}"
    
    def _parse_critique_response(self, response: str) -> Tuple[str, str, float]:
        """Parse critique response to extract verdict and justification
        
        Returns:
            Tuple of (verdict, justification, confidence)
        """
        lines = response.strip().split('\n')
        verdict = "INVALID"  # Default
        justification = ""
        confidence = 0.5
        
        for line in lines:
            if 'VERDICT:' in line:
                if 'VALID' in line and 'INVALID' not in line.replace('INVALID', ''):
                    verdict = "VALID"
                    confidence = 0.8
                else:
                    verdict = "INVALID"
                    confidence = 0.8
            elif 'JUSTIFICATION:' in line:
                justification = line.split('JUSTIFICATION:')[1].strip()
        
        # If no justification found, use the entire response after verdict
        if not justification and len(lines) > 1:
            justification = ' '.join(lines[1:]).strip()[:100]  # Limit to 100 chars
        
        return verdict, justification, confidence
    
    def run_reciprocal_critique(self, 
                               models: List[str],
                               reasoning_traces: List[ReasoningTrace],
                               sample_size: Optional[int] = None) -> pd.DataFrame:
        """Run reciprocal critique across all model pairs
        
        Args:
            models: List of model names
            reasoning_traces: All reasoning traces to evaluate
            sample_size: Optional sample size for testing
            
        Returns:
            DataFrame with all critique results
        """
        critiques = []
        
        # Group traces by model
        traces_by_model = {}
        for trace in reasoning_traces:
            if trace.model not in traces_by_model:
                traces_by_model[trace.model] = []
            traces_by_model[trace.model].append(trace)
        
        # Sample if requested
        if sample_size:
            for model in traces_by_model:
                traces_by_model[model] = traces_by_model[model][:sample_size]
        
        # For each model pair (i != j)
        for judge_model in models:
            for target_model in models:
                if judge_model == target_model:
                    continue  # Models don't judge themselves
                
                logger.info(f"{judge_model} critiquing {target_model}...")
                
                # Get traces from target model
                target_traces = traces_by_model.get(target_model, [])
                
                for trace in target_traces:
                    critique = self.critique_reasoning(judge_model, trace)
                    critiques.append(asdict(critique))
                    
                    # Rate limiting
                    time.sleep(0.5)
        
        return pd.DataFrame(critiques)
    
    def calculate_peer_agreement_rates(self, critique_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate peer-agreement rate for each model
        
        The peer-agreement rate is the proportion of times a model's
        explanations are deemed VALID by other models.
        
        Args:
            critique_df: DataFrame with critique results
            
        Returns:
            Dictionary mapping model names to their peer-agreement rates
        """
        agreement_rates = {}
        
        for model in critique_df['target_model'].unique():
            model_critiques = critique_df[critique_df['target_model'] == model]
            valid_count = (model_critiques['verdict'] == 'VALID').sum()
            total_count = len(model_critiques)
            
            if total_count > 0:
                agreement_rates[model] = valid_count / total_count
            else:
                agreement_rates[model] = 0.0
        
        return agreement_rates
    
    def identify_contentious_cases(self, 
                                  critique_df: pd.DataFrame,
                                  min_disagreement: float = 0.5) -> pd.DataFrame:
        """Identify cases where models strongly disagree
        
        Args:
            critique_df: DataFrame with critique results
            min_disagreement: Minimum disagreement rate to flag as contentious
            
        Returns:
            DataFrame of contentious cases needing human review
        """
        # Group by country-topic pairs
        grouped = critique_df.groupby(['country', 'topic'])
        
        contentious = []
        for (country, topic), group in grouped:
            valid_rate = (group['verdict'] == 'VALID').mean()
            
            # Check if there's significant disagreement (not all valid or invalid)
            if min_disagreement <= valid_rate <= (1 - min_disagreement):
                contentious.append({
                    'country': country,
                    'topic': topic,
                    'valid_rate': valid_rate,
                    'n_critiques': len(group),
                    'models_involved': list(group['target_model'].unique())
                })
        
        return pd.DataFrame(contentious)
    
    def save_critique_results(self, 
                             critique_df: pd.DataFrame,
                             agreement_rates: Dict[str, float],
                             contentious_df: pd.DataFrame,
                             output_dir: Path = Path("outputs/peer_review")):
        """Save all critique results to files
        
        Args:
            critique_df: All critiques
            agreement_rates: Peer-agreement rates by model
            contentious_df: Contentious cases for human review
            output_dir: Output directory
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save raw critiques
        critique_df.to_csv(output_dir / "all_critiques.csv", index=False)
        
        # Save agreement rates
        agreement_df = pd.DataFrame([
            {'model': model, 'peer_agreement_rate': rate}
            for model, rate in agreement_rates.items()
        ])
        agreement_df.to_csv(output_dir / "peer_agreement_rates.csv", index=False)
        
        # Save contentious cases
        contentious_df.to_csv(output_dir / "contentious_cases.csv", index=False)
        
        # Save summary statistics
        summary = {
            'total_critiques': len(critique_df),
            'models_evaluated': list(critique_df['target_model'].unique()),
            'judge_models': list(critique_df['judge_model'].unique()),
            'overall_valid_rate': (critique_df['verdict'] == 'VALID').mean(),
            'n_contentious_cases': len(contentious_df),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(output_dir / "critique_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Saved critique results to {output_dir}")
        logger.info(f"  Total critiques: {summary['total_critiques']}")
        logger.info(f"  Contentious cases: {summary['n_contentious_cases']}")
        logger.info(f"  Overall VALID rate: {summary['overall_valid_rate']:.2%}")


def main():
    """Test the model judge system"""
    
    # Create sample reasoning traces for testing
    sample_traces = [
        ReasoningTrace(
            model="gpt-4o",
            country="Netherlands",
            topic="Homosexuality",
            reasoning_steps=[
                "Step 1: In the Netherlands, social norms are very progressive regarding LGBTQ+ rights.",
                "Step 2: Same-sex marriage has been legal since 2001, making it the first country to legalize it.",
                "Step 3: The behavior is widely accepted and protected by law."
            ],
            final_score=0.8,
            method="direct",
            raw_response="The Netherlands is known for progressive values..."
        ),
        ReasoningTrace(
            model="gpt-3.5-turbo",
            country="Netherlands",
            topic="Homosexuality",
            reasoning_steps=[
                "Step 1: Netherlands has traditional values.",
                "Step 2: The topic might be controversial.",
                "Step 3: Mixed acceptance in society."
            ],
            final_score=0.1,
            method="direct",
            raw_response="The Netherlands has some traditional elements..."
        )
    ]
    
    # Initialize judge (will use mock responses without API keys)
    judge = ModelJudge()
    
    # Test single critique
    print("Testing single critique...")
    critique = judge.critique_reasoning("gpt-4o", sample_traces[1])
    print(f"Verdict: {critique.verdict}")
    print(f"Justification: {critique.justification}")
    
    # Test reciprocal critique
    print("\nTesting reciprocal critique...")
    models = ["gpt-4o", "gpt-3.5-turbo"]
    critique_df = judge.run_reciprocal_critique(models, sample_traces, sample_size=1)
    
    # Calculate agreement rates
    agreement_rates = judge.calculate_peer_agreement_rates(critique_df)
    print(f"\nPeer-agreement rates: {agreement_rates}")
    
    # Identify contentious cases
    contentious = judge.identify_contentious_cases(critique_df)
    print(f"\nContentious cases: {len(contentious)}")
    
    # Save results
    judge.save_critique_results(critique_df, agreement_rates, contentious)
    print("\n✅ Model judge system test complete!")


if __name__ == "__main__":
    main()