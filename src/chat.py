#!/usr/bin/env python3
"""
Simple chat interface to test document retrieval quality
"""

import sys
import os
from .rag.retriever import retrieve
from .models.reranker import rerank
from .rag.generator import generate_answer

def chat_interface():
    """Interactive chat loop for testing document retrieval"""
    
    print("🤖 Document Chat Interface")
    print("=" * 50)
    print("💡 Ask questions about your indexed documents")
    print("❓ Commands: 'quit', 'exit', 'q' to exit")
    print("-" * 50)
    
    while True:
        try:
            # Get user input
            question = input("\n❓ Your question: ").strip()
            
            # Handle commands
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
                
            elif not question:
                print("⚠️ Please enter a question")
                continue
            
            # Process the question
            print("🔍 Searching documents...")
            
            # Retrieve relevant chunks with optimized parameters
            hits = retrieve(query=question, top_k=5, score_threshold=0.3)
            if not hits:
                print("❌ No relevant documents found (try lowering search criteria)")
                continue
                
            # Rerank for better relevance (only if we have multiple hits)
            if len(hits) > 1:
                print("📊 Reranking results...")
                hits = rerank(question, hits)
            
            # Generate answer
            print("🤖 Generating answer...")
            answer = generate_answer(question, hits)
            
            # Display results
            print("\n" + "=" * 50)
            print("📝 ANSWER:")
            print(answer)
            print("=" * 50)
            
            # Show source information
            print("\n📚 SOURCES:")
            for i, hit in enumerate(hits[:3], 1):  # Show top 3 sources
                page = hit.payload.get('page', 'Unknown')
                doc_id = hit.payload.get('doc_id', 'Unknown')
                score = hit.score if hasattr(hit, 'score') else 'N/A'
                print(f"  {i}. Document: {doc_id}, Page: {page}, Score: {score:.3f}")
            
            # Show text preview from top source
            if hits:
                top_hit = hits[0]
                text_preview = top_hit.payload.get('text', '')[:200]
                if len(text_preview) > 200:
                    text_preview += "..."
                print(f"\n📖 Top source preview: {text_preview}")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            print("💡 Try asking a different question")

def main():
    """Main function"""
    if len(sys.argv) > 1:
        print("Usage: python -m src.chat")
        print("No arguments needed - just run the module")
        sys.exit(1)
    
    # Check if we have indexed documents
    try:
        # Test connection to Qdrant
        from qdrant_client import QdrantClient
        from .config import QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION
        
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        
        # Check if collection exists and has data
        collections = client.get_collections()
        collection_names = [col.name for col in collections.collections]
        
        if QDRANT_COLLECTION not in collection_names:
            print("❌ No indexed documents found!")
            print("💡 Please run: python -m src.cli index 'data/raw/*.pdf' first")
            sys.exit(1)
        
        # Check if collection has points
        collection_info = client.get_collection(QDRANT_COLLECTION)
        if collection_info.points_count == 0:
            print("❌ Collection is empty!")
            print("💡 Please run: python -m src.cli index 'data/raw/*.pdf' first")
            sys.exit(1)
            
        print(f"✅ Found {collection_info.points_count} indexed documents")
        
    except Exception as e:
        print(f"❌ Error connecting to Qdrant: {e}")
        print("💡 Please check your .env configuration")
        sys.exit(1)
    
    # Start chat interface
    chat_interface()

if __name__ == "__main__":
    main()