#!/usr/bin/env python3
"""
Debug script to check embedding dimensions and identify the source of the 768 dimension.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def debug_embedding_dimension():
    """Debug the embedding dimension issue."""
    print("🔍 Debugging embedding dimension issue...")
    
    try:
        from src.models.embedder import embed, model
        
        print(f"🔧 Model: {model.model_name}")
        
        # Test the dynamic dimension detection
        print("\n📊 Testing dynamic dimension detection...")
        dimension = model.get_sentence_embedding_dimension()
        print(f"   Detected dimension: {dimension}")
        
        # Test with a simple text
        print("\n📝 Testing actual embedding...")
        test_text = "Hello world"
        embedding = embed(test_text)
        actual_dimension = len(embedding)
        print(f"   Actual embedding dimension: {actual_dimension}")
        
        # Test with a longer text
        print("\n📝 Testing with longer text...")
        long_text = "This is a longer text to test if the embedding dimension changes with different input lengths."
        long_embedding = embed(long_text)
        long_dimension = len(long_embedding)
        print(f"   Long text embedding dimension: {long_dimension}")
        
        # Check if dimensions are consistent
        print(f"\n🔍 Dimension consistency check:")
        print(f"   Dynamic detection: {dimension}")
        print(f"   Short text: {actual_dimension}")
        print(f"   Long text: {long_dimension}")
        
        if dimension == actual_dimension == long_dimension:
            print(f"   ✅ All dimensions are consistent!")
        else:
            print(f"   ❌ Dimensions are inconsistent!")
            return False
        
        # Test batch embedding
        print(f"\n📝 Testing batch embedding...")
        batch_texts = ["Text 1", "Text 2", "Text 3"]
        batch_embeddings = embed(batch_texts)
        batch_dimensions = [len(emb) for emb in batch_embeddings]
        print(f"   Batch embedding dimensions: {batch_dimensions}")
        
        if all(dim == dimension for dim in batch_dimensions):
            print(f"   ✅ Batch dimensions are consistent!")
        else:
            print(f"   ❌ Batch dimensions are inconsistent!")
            return False
        
        print(f"\n🎉 All embedding tests passed!")
        print(f"📊 Final dimension: {dimension}")
        return True
        
    except Exception as e:
        print(f"❌ Error during debugging: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔍 Debug Embedding Dimension Issue")
    print("=" * 50)
    
    success = debug_embedding_dimension()
    
    print("=" * 50)
    if success:
        print("✅ Debug completed successfully!")
        print("💡 If you're still getting 768 dimensions, the issue might be:")
        print("   1. Cached embeddings from before the model change")
        print("   2. A different code path using an old embedding model")
        print("   3. Environment variable pointing to wrong model")
    else:
        print("❌ Debug failed!")
