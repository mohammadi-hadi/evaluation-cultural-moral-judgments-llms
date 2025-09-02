#!/usr/bin/env python3
"""
Run o3 model first to upgrade usage tier
"""

import os
import json
import asyncio
import aiohttp
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_o3_model():
    """Test o3 model to upgrade usage tier"""
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "o3",
        "messages": [{"role": "user", "content": "Hello, this is a test message."}],
        "max_tokens": 50,
        "temperature": 0.5
    }
    
    logger.info("🚀 Testing o3 model to upgrade usage tier...")
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                "https://api.openai.com/v1/chat/completions",
                json=payload,
                headers=headers
            ) as response:
                
                if response.status == 200:
                    data = await response.json()
                    logger.info("✅ o3 model successful!")
                    logger.info(f"Response: {data['choices'][0]['message']['content']}")
                    return True
                else:
                    error_text = await response.text()
                    logger.error(f"o3 model failed: {response.status} - {error_text}")
                    return False
                    
        except Exception as e:
            logger.error(f"Exception testing o3: {e}")
            return False

async def run_available_models():
    """Run only the available models"""
    
    api_key = os.getenv("OPENAI_API_KEY")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Test which models are actually available
    available_models = ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"]
    
    logger.info("🔍 Testing model availability...")
    
    test_payload = {
        "messages": [{"role": "user", "content": "Test"}],
        "max_tokens": 5,
        "temperature": 0.1
    }
    
    working_models = []
    
    async with aiohttp.ClientSession() as session:
        for model in available_models:
            try:
                test_payload["model"] = model
                async with session.post(
                    "https://api.openai.com/v1/chat/completions",
                    json=test_payload,
                    headers=headers
                ) as response:
                    
                    if response.status == 200:
                        logger.info(f"✅ {model} - Available")
                        working_models.append(model)
                    else:
                        logger.warning(f"❌ {model} - Not available ({response.status})")
                        
                await asyncio.sleep(1)  # Rate limit protection
                        
            except Exception as e:
                logger.error(f"Error testing {model}: {e}")
    
    logger.info(f"📊 Available models: {working_models}")
    return working_models

def main():
    """Main execution"""
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        return
    
    async def run():
        # First test o3 to upgrade tier
        o3_success = await test_o3_model()
        
        if o3_success:
            logger.info("🎉 o3 test successful - usage tier should be upgraded!")
        else:
            logger.warning("⚠️ o3 test failed - continuing with available models")
        
        # Test available models
        working_models = await run_available_models()
        
        if working_models:
            logger.info(f"✅ Ready to run evaluation with: {working_models}")
        else:
            logger.error("❌ No models available for evaluation")
    
    asyncio.run(run())

if __name__ == "__main__":
    main()