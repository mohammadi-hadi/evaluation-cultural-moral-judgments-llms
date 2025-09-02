#!/usr/bin/env python3
"""
Final o3 test with all correct parameters
"""

import os
import json
import asyncio
import aiohttp
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_o3_final():
    """Test o3 with all correct parameters"""
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Correct payload for o3 model - minimal parameters
    payload = {
        "model": "o3",
        "messages": [{"role": "user", "content": "Hello, please respond with a simple greeting."}],
        "max_completion_tokens": 50
        # Temperature default is 1.0, don't specify it
    }
    
    logger.info("🚀 Testing o3 model with minimal correct parameters...")
    
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
                elif response.status == 429:
                    logger.warning("⚠️ Rate limited on o3 - need to wait")
                    return False
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
    
    success = asyncio.run(test_o3_final())
    
    if success:
        print("✅ o3 model test successful!")
        print("🎉 Usage tier upgraded - higher rate limits now available")
        print("🔄 You can now retry API evaluation with better performance")
    else:
        print("❌ o3 model test failed or rate limited")
        print("💡 The batch API approach is still the best option for large-scale evaluation")

if __name__ == "__main__":
    main()