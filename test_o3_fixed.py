#!/usr/bin/env python3
"""
Test o3 model with correct parameters to upgrade usage tier
"""

import os
import json
import asyncio
import aiohttp
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_o3_correct():
    """Test o3 with correct parameters"""
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Correct payload for o3 model
    payload = {
        "model": "o3",
        "messages": [{"role": "user", "content": "Hello, please respond with a simple greeting."}],
        "max_completion_tokens": 50,  # Use max_completion_tokens for o3
        "temperature": 0.5
    }
    
    logger.info("🚀 Testing o3 model with correct parameters...")
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                "https://api.openai.com/v1/chat/completions",
                json=payload,
                headers=headers
            ) as response:
                
                response_text = await response.text()
                
                if response.status == 200:
                    data = json.loads(response_text)
                    logger.info("✅ o3 model successful!")
                    logger.info(f"Response: {data['choices'][0]['message']['content']}")
                    logger.info("🎉 Usage tier should now be upgraded!")
                    return True
                else:
                    logger.error(f"o3 model failed: {response.status}")
                    logger.error(f"Response: {response_text}")
                    return False
                    
        except Exception as e:
            logger.error(f"Exception testing o3: {e}")
            return False

def main():
    """Main execution"""
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        return
    
    success = asyncio.run(test_o3_correct())
    
    if success:
        print("✅ o3 model test successful - usage tier upgraded!")
        print("🔄 Now you can retry the full API evaluation with higher limits")
    else:
        print("❌ o3 model test failed")

if __name__ == "__main__":
    main()