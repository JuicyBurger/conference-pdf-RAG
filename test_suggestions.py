#!/usr/bin/env python3
"""
Test script for the suggestions feature.
This script tests the suggestion generation and retrieval functionality.
"""

import sys
import os

# Add the src directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.rag.suggestions import (
    init_suggestion_collection,
    generate_suggestions_for_doc,
    retrieve_suggestions,
    get_all_doc_ids_with_suggestions
)
from src.rag.retriever import get_all_doc_ids

def test_suggestion_flow():
    """Test the complete suggestion flow"""
    print("🧪 Testing suggestion system...")
    
    # Step 1: Initialize the suggestion collection
    print("\n1. Initializing suggestion collection...")
    try:
        init_suggestion_collection()
        print("✅ Suggestion collection initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize collection: {e}")
        return False
    
    # Step 2: Get available documents
    print("\n2. Getting available documents...")
    doc_ids = get_all_doc_ids()
    
    if not doc_ids:
        print("❌ No documents found in the index. Please index some documents first.")
        print("   Use: python -m src.cli index 'path/to/your/*.pdf'")
        return False
    
    print(f"✅ Found {len(doc_ids)} documents: {doc_ids[:3]}{'...' if len(doc_ids) > 3 else ''}")
    
    # Step 3: Test suggestion generation for the first document
    test_doc_id = doc_ids[0]
    print(f"\n3. Generating suggestions for document: {test_doc_id}")
    
    try:
        success = generate_suggestions_for_doc(
            doc_id=test_doc_id,
            num_questions=3,  # Generate just 3 for testing
            auto_init_collection=False  # Already initialized
        )
        
        if success:
            print("✅ Suggestions generated successfully")
        else:
            print("❌ Suggestion generation failed")
            return False
            
    except Exception as e:
        print(f"❌ Exception during suggestion generation: {e}")
        return False
    
    # Step 4: Test suggestion retrieval
    print(f"\n4. Retrieving suggestions for document: {test_doc_id}")
    
    try:
        suggestions = retrieve_suggestions(
            doc_id=test_doc_id,
            k=5
        )
        
        if suggestions:
            print(f"✅ Retrieved {len(suggestions)} suggestions:")
            for i, suggestion in enumerate(suggestions, 1):
                print(f"   {i}. {suggestion['question_text']}")
        else:
            print("❌ No suggestions retrieved")
            return False
            
    except Exception as e:
        print(f"❌ Exception during suggestion retrieval: {e}")
        return False
    
    # Step 5: Test topic-based retrieval
    print(f"\n5. Testing topic-based suggestion retrieval...")
    
    try:
        topic_suggestions = retrieve_suggestions(
            doc_id=test_doc_id,
            topic="財務",  # Financial topic
            k=3
        )
        
        print(f"✅ Retrieved {len(topic_suggestions)} topic-based suggestions:")
        for i, suggestion in enumerate(topic_suggestions, 1):
            print(f"   {i}. {suggestion['question_text']}")
            if suggestion.get('relevance_score'):
                print(f"      (relevance: {suggestion['relevance_score']:.3f})")
                
    except Exception as e:
        print(f"❌ Exception during topic-based retrieval: {e}")
        return False
    
    # Step 6: Test listing documents with suggestions
    print(f"\n6. Listing all documents with suggestions...")
    
    try:
        docs_with_suggestions = get_all_doc_ids_with_suggestions()
        print(f"✅ Found {len(docs_with_suggestions)} documents with suggestions: {docs_with_suggestions}")
        
    except Exception as e:
        print(f"❌ Exception during doc listing: {e}")
        return False
    
    print("\n🎉 All tests passed! Suggestion system is working correctly.")
    return True

def test_api_simulation():
    """Simulate API calls to test the endpoint logic"""
    print("\n🌐 Testing API simulation...")
    
    # This would normally be done through HTTP requests
    # but we'll test the core logic directly
    
    from API.routes.suggestions import suggestions_bp
    from flask import Flask
    
    app = Flask(__name__)
    app.register_blueprint(suggestions_bp)
    
    with app.test_client() as client:
        # Test health endpoint
        print("Testing /suggestions/health...")
        response = client.get('/suggestions/health')
        print(f"Health check response: {response.status_code}")
        
        # Test docs listing
        print("Testing /suggestions/docs...")
        response = client.get('/suggestions/docs')
        print(f"Docs listing response: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ API simulation successful")
        else:
            print("❌ API simulation failed")

if __name__ == "__main__":
    print("🚀 Starting suggestion system tests...\n")
    
    # Run the main test
    success = test_suggestion_flow()
    
    if success:
        # Run API simulation if main test passes
        try:
            test_api_simulation()
        except Exception as e:
            print(f"⚠️ API simulation failed: {e}")
    
    print("\n✨ Test completed!")