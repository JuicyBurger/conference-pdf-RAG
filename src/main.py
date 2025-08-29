from .indexing import index_nodes_vector
from .data.pdf_ingestor import build_page_nodes
import os

def ingest(path):
    """Ingest a PDF using the new indexing system"""
    print(f"📄 Processing {path}...")
    
    # Extract nodes from PDF
    nodes = build_page_nodes(path)
    doc_id = os.path.splitext(os.path.basename(path))[0]
    
    # Index using new system
    result = index_nodes_vector(nodes, doc_id)
    total_chunks = result.indexed_count if result.success else 0
    
    print(f"✅ Indexed {total_chunks} chunks from {path}")

if __name__ == "__main__":
    import sys
    ingest(sys.argv[1])
