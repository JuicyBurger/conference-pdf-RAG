from .rag.indexing.indexer import index_pdf

def ingest(path):
    """Legacy ingest function - now just calls the modern index_pdf"""
    print(f"📄 Processing {path}...")
    total_chunks = index_pdf(path)
    print(f"✅ Indexed {total_chunks} chunks from {path}")

if __name__ == "__main__":
    import sys
    ingest(sys.argv[1])
