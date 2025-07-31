#!/usr/bin/env python3
"""
Test script to verify the new API payload structure
"""

import requests
import json
import uuid
from datetime import datetime

# API Configuration
API_BASE = "http://localhost:5000"
ROOM_ID = str(uuid.uuid4())

def test_chat_message():
    """Test the new chat message payload structure"""
    
    print("🧪 Testing new API payload structure...")
    print(f"📝 Room ID: {ROOM_ID}")
    print(f"🌐 API Base: {API_BASE}")
    print()
    
    # Test 1: Send first message
    print("📤 Sending first message...")
    response1 = requests.post(f"{API_BASE}/api/chat/message", json={
        "room_id": ROOM_ID,
        "message": "Hello! This is a test message.",
        "user_id": "test-user-1"
    })
    
    if response1.status_code == 200:
        data1 = response1.json()
        print("✅ First message sent successfully")
        print(f"📊 Response status: {data1.get('status')}")
        
        # Check data structure
        if 'data' in data1:
            data = data1['data']
            print(f"📋 Room ID: {data.get('room_id')}")
            print(f"📋 Room Title: {data.get('room_title')}")
            print(f"📋 Created At: {data.get('createdAt')}")
            print(f"📋 Updated At: {data.get('updatedAt')}")
            print(f"📋 Messages Count: {len(data.get('messages', []))}")
            
            # Check message structure
            messages = data.get('messages', [])
            if messages:
                print("\n📝 Message structure:")
                for i, msg in enumerate(messages):
                    print(f"  Message {i+1}:")
                    print(f"    msg_id: {msg.get('msg_id')}")
                    print(f"    role: {msg.get('role')}")
                    print(f"    content: {msg.get('content')[:50]}...")
                    print(f"    timestamp: {msg.get('timestamp')}")
                    print(f"    files: {msg.get('files')}")
        else:
            print("❌ No 'data' field in response")
    else:
        print(f"❌ Failed to send first message: {response1.status_code}")
        print(response1.text)
        return
    
    print("\n" + "="*50 + "\n")
    
    # Test 2: Send second message
    print("📤 Sending second message...")
    response2 = requests.post(f"{API_BASE}/api/chat/message", json={
        "room_id": ROOM_ID,
        "message": "This is a follow-up message to test conversation flow.",
        "user_id": "test-user-1"
    })
    
    if response2.status_code == 200:
        data2 = response2.json()
        print("✅ Second message sent successfully")
        
        # Check data structure
        if 'data' in data2:
            data = data2['data']
            messages = data.get('messages', [])
            print(f"📋 Total Messages: {len(messages)}")
            
            # Verify messages are sorted by timestamp
            timestamps = [msg.get('timestamp') for msg in messages]
            is_sorted = timestamps == sorted(timestamps)
            print(f"📋 Messages sorted by timestamp: {is_sorted}")
            
            # Show conversation flow
            print("\n💬 Conversation flow:")
            for i, msg in enumerate(messages):
                role_emoji = "👤" if msg.get('role') == 'user' else "🤖"
                print(f"  {role_emoji} {msg.get('role').upper()}: {msg.get('content')[:60]}...")
        else:
            print("❌ No 'data' field in response")
    else:
        print(f"❌ Failed to send second message: {response2.status_code}")
        print(response2.text)
        return
    
    print("\n" + "="*50 + "\n")
    
    # Test 3: Get chat history
    print("📚 Getting chat history...")
    response3 = requests.get(f"{API_BASE}/api/chat/history/{ROOM_ID}")
    
    if response3.status_code == 200:
        data3 = response3.json()
        print("✅ Chat history retrieved successfully")
        
        if 'data' in data3:
            data = data3['data']
            messages = data.get('messages', [])
            print(f"📋 History Messages: {len(messages)}")
            
            # Verify same structure
            if messages:
                first_msg = messages[0]
                print(f"📋 First message structure: {list(first_msg.keys())}")
        else:
            print("❌ No 'data' field in history response")
    else:
        print(f"❌ Failed to get chat history: {response3.status_code}")
        print(response3.text)

if __name__ == "__main__":
    try:
        test_chat_message()
        print("\n🎉 API payload structure test completed!")
    except Exception as e:
        print(f"❌ Test failed: {e}") 