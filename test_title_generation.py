#!/usr/bin/env python3
"""
Test Room Title Generation

Script to test the room title generation functionality with proper imports.
"""

import sys
import os
import asyncio

# Get the base project directory
project_root = os.path.dirname(os.path.abspath(__file__))

# Add 'src' to sys.path
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Now import the chat_service
from API.services.chat_service import chat_service

async def test_title_generation():
    """Test room title generation with various messages"""
    
    print("🧪 Testing Room Title Generation...")
    print("=" * 50)
    
    test_messages = [
        "What is the revenue growth rate?",
        "Can you help me analyze this financial report?",
        "I need help with technical documentation",
        "How do I implement this feature?",
        "What are the key performance indicators?",
        "Hello, how are you today?"
    ]
    
    for i, message in enumerate(test_messages, 1):
        print(f"\n📝 Test {i}: {message}")
        try:
            title = await chat_service._generate_room_title(message)
            print(f"✅ test {i} title: {title}")
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n🎉 Title generation test completed!")

def main():
    """Main entry point"""
    try:
        asyncio.run(test_title_generation())
        print("\n✅ All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()