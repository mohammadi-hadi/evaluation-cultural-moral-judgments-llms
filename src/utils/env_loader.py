#!/usr/bin/env python3
"""
Environment and API Key Management for Moral Alignment Pipeline
Secure loading and validation of API keys with fallback handling
"""

import os
import sys
from pathlib import Path
from typing import Dict, Optional, List
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class EnvLoader:
    """Manages API keys and environment configuration"""
    
    def __init__(self, env_path: Optional[str] = None):
        """Initialize environment loader
        
        Args:
            env_path: Path to .env file (defaults to project root)
        """
        if env_path:
            self.env_path = Path(env_path)
        else:
            self.env_path = Path(__file__).parent / '.env'
        
        # Load environment variables
        if self.env_path.exists():
            load_dotenv(self.env_path)
            logger.info(f"Loaded environment from {self.env_path}")
        else:
            logger.warning(f"No .env file found at {self.env_path}")
        
        # Track available APIs
        self.available_apis = self._check_available_apis()
        
    def _check_available_apis(self) -> Dict[str, bool]:
        """Check which API keys are available"""
        apis = {
            'openai': bool(os.getenv('OPENAI_API_KEY') and 
                          os.getenv('OPENAI_API_KEY') != 'your-key-here'),
            'anthropic': bool(os.getenv('ANTHROPIC_API_KEY') and 
                             os.getenv('ANTHROPIC_API_KEY') != 'your-anthropic-key-here'),
            'google': bool(os.getenv('GEMINI_API_KEY') and 
                          os.getenv('GEMINI_API_KEY') != 'your-gemini-key-here'),
            'mistral': bool(os.getenv('MISTRAL_API_KEY') and 
                           os.getenv('MISTRAL_API_KEY') != 'your-mistral-key-here'),
            'cohere': bool(os.getenv('COHERE_API_KEY') and 
                          os.getenv('COHERE_API_KEY') != 'your-cohere-key-here'),
        }
        
        # Log available APIs
        available = [api for api, is_available in apis.items() if is_available]
        unavailable = [api for api, is_available in apis.items() if not is_available]
        
        if available:
            logger.info(f"Available APIs: {', '.join(available)}")
        if unavailable:
            logger.info(f"Unavailable APIs (placeholders): {', '.join(unavailable)}")
        
        return apis
    
    def get_api_key(self, provider: str) -> Optional[str]:
        """Get API key for a specific provider
        
        Args:
            provider: API provider name (openai, anthropic, google, mistral, cohere)
            
        Returns:
            API key if available, None otherwise
        """
        key_mapping = {
            'openai': 'OPENAI_API_KEY',
            'anthropic': 'ANTHROPIC_API_KEY',
            'google': 'GEMINI_API_KEY',
            'gemini': 'GEMINI_API_KEY',
            'mistral': 'MISTRAL_API_KEY',
            'cohere': 'COHERE_API_KEY',
        }
        
        env_var = key_mapping.get(provider.lower())
        if not env_var:
            logger.warning(f"Unknown API provider: {provider}")
            return None
        
        api_key = os.getenv(env_var)
        
        # Check if it's a placeholder
        if api_key and 'your-' in api_key and '-key-here' in api_key:
            logger.debug(f"API key for {provider} is a placeholder")
            return None
        
        return api_key
    
    def get_available_models(self) -> Dict[str, List[str]]:
        """Get list of available models based on API keys
        
        Returns:
            Dictionary mapping model categories to available models
        """
        available_models = {
            'api': [],
            'local': [],
            'server': []
        }
        
        # OpenAI models
        if self.available_apis.get('openai'):
            available_models['api'].extend([
                'gpt-4o',
                'gpt-4o-mini',
                'gpt-4-turbo',
                'o1-preview',
                'o1-mini'
            ])
        
        # Anthropic models (future)
        if self.available_apis.get('anthropic'):
            available_models['api'].extend([
                'claude-3.5-sonnet',
                'claude-3.5-haiku',
                'claude-3-opus'
            ])
        
        # Google models (future)
        if self.available_apis.get('google'):
            available_models['api'].extend([
                'gemini-1.5-pro',
                'gemini-1.5-flash',
                'gemini-1.5-flash-8b',
                'gemini-2.0-flash-exp'
            ])
        
        # Mistral models (future)
        if self.available_apis.get('mistral'):
            available_models['api'].extend([
                'mistral-large-latest',
                'mistral-small-latest'
            ])
        
        # Cohere models (future)
        if self.available_apis.get('cohere'):
            available_models['api'].extend([
                'command-r-plus-api',
                'command-r-api'
            ])
        
        # Local models (always available on M4 Max)
        available_models['local'] = [
            'gpt2', 'gpt2-medium', 'gpt2-large',
            'opt-125m', 'opt-350m', 'opt-1.3b', 'opt-2.7b',
            'llama-3.2-1b-instruct', 'llama-3.2-3b-instruct',
            'gemma-2-2b-it',
            'qwen-0.5b', 'qwen-1.8b',
            'bloomz-560m', 'bloomz-1b7',
            'falcon-7b'
        ]
        
        # Server models (for SURF)
        available_models['server'] = [
            'gpt2-xl',
            'llama-3.3-70b-instruct', 'llama-3.2-90b-instruct', 'llama-3.2-11b-instruct',
            'gemma-2-27b-it', 'gemma-2-9b-it',
            'mistral-large-2', 'mixtral-8x22b-instruct',
            'command-r-plus',
            'bloom-176b', 'qwen-72b',
            'gpt-neox-20b', 'dolly-v2-12b', 'falcon-40b-instruct'
        ]
        
        return available_models
    
    def validate_environment(self) -> bool:
        """Validate that at least one API is available
        
        Returns:
            True if at least one API is configured, False otherwise
        """
        if not any(self.available_apis.values()):
            logger.warning("No API keys configured. Only local models will be available.")
            return False
        return True
    
    def get_environment_info(self) -> Dict:
        """Get comprehensive environment information
        
        Returns:
            Dictionary with environment details
        """
        models = self.get_available_models()
        
        return {
            'available_apis': self.available_apis,
            'api_model_count': len(models['api']),
            'local_model_count': len(models['local']),
            'server_model_count': len(models['server']),
            'total_available': len(models['api']) + len(models['local']),
            'environment': os.getenv('MORAL_ALIGNMENT_ENV', 'local'),
            'has_openai': self.available_apis.get('openai', False),
            'api_models': models['api'],
            'local_models': models['local'][:5],  # Show first 5 for brevity
            'server_models': models['server'][:5]  # Show first 5 for brevity
        }
    
    def estimate_costs(self, model: str, num_samples: int = 1000) -> Dict:
        """Estimate API costs for a model
        
        Args:
            model: Model name
            num_samples: Number of samples to process
            
        Returns:
            Cost estimation dictionary
        """
        # Rough cost estimates per 1000 samples (in USD)
        cost_map = {
            'gpt-4o': 10.0,
            'gpt-4o-mini': 1.0,
            'gpt-4-turbo': 15.0,
            'o1-preview': 30.0,
            'o1-mini': 20.0,
            'claude-3.5-sonnet': 12.0,
            'claude-3.5-haiku': 3.0,
            'claude-3-opus': 20.0,
            'gemini-1.5-pro': 8.0,
            'gemini-1.5-flash': 2.0,
            'mistral-large-latest': 10.0,
            'mistral-small-latest': 3.0,
        }
        
        cost_per_1000 = cost_map.get(model, 0)
        total_cost = (num_samples / 1000) * cost_per_1000
        
        return {
            'model': model,
            'samples': num_samples,
            'estimated_cost_usd': round(total_cost, 2),
            'cost_per_1000': cost_per_1000,
            'is_api_model': cost_per_1000 > 0
        }


def get_env_loader() -> EnvLoader:
    """Get or create singleton environment loader"""
    if not hasattr(get_env_loader, '_instance'):
        get_env_loader._instance = EnvLoader()
    return get_env_loader._instance


if __name__ == "__main__":
    # Test environment loading
    loader = get_env_loader()
    
    print("=" * 60)
    print("Environment Configuration")
    print("=" * 60)
    
    info = loader.get_environment_info()
    for key, value in info.items():
        if isinstance(value, list) and len(value) > 5:
            print(f"{key}: [{len(value)} items]")
        else:
            print(f"{key}: {value}")
    
    print("\n" + "=" * 60)
    print("Cost Estimates for OpenAI Models")
    print("=" * 60)
    
    if info['has_openai']:
        for model in ['gpt-4o', 'gpt-4o-mini', 'o1-preview']:
            cost = loader.estimate_costs(model, 5000)
            print(f"{model}: ${cost['estimated_cost_usd']} for {cost['samples']} samples")
    else:
        print("OpenAI API key not configured")
    
    print("\n✅ Environment loader ready!")