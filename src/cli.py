import argparse
import glob
import os
import json
import uuid
import sys

from .indexing import VectorIndexer, index_nodes_vector
from .data.pdf_ingestor import build_page_nodes
from .rag.retrieval.retrieval_service import retrieval_service
from .models.reranker  import rerank
from .rag.generation import generate_answer
from .rag.generation import generate_qa_pairs_for_doc, generate_suggestions_for_doc, batch_generate_suggestions
from .data.pdf_ingestor import extract_text_pages, extract_tables_per_page, build_page_nodes

# New imports for reset command
from qdrant_client import QdrantClient
from .config import QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION, QDRANT_QA_DB
from .rag.graph.graph_store import get_driver, ensure_graph_indexes
from .config import NEO4J_DATABASE



def cmd_index(args):
    # Initialize vector indexer
    indexer = VectorIndexer()
    if args.recreate:
        indexer.initialize()
    
    paths = glob.glob(args.pattern)
    total = 0
    for p in paths:
        # Extract nodes from PDF
        nodes = build_page_nodes(p)
        doc_id = os.path.splitext(os.path.basename(p))[0]
        
        # Index using new system
        result = indexer.index_nodes(nodes, doc_id)
        total += result.indexed_count if result.success else 0
        
    print(f"[INDEX] Done: {total} chunks")


def cmd_query(args):
    hits = retrieval_service.retrieve(args.question, top_k=args.topk)
    hits = rerank(args.question, hits)
    print(generate_answer(args.question, hits))


def cmd_list_docs(args):
    """List all available document IDs"""
    try:
        doc_ids = retrieval_service.get_all_doc_ids()
        if doc_ids:
            print("📚 Available documents:")
            for doc_id in sorted(doc_ids):
                print(f"  - {doc_id}")
        else:
            print("📚 No documents found in the index")
    except Exception as e:
        print(f"❌ Failed to list documents: {e}")


def cmd_qa_generate(args):
    doc_id = args.doc_id
    print(f"🗂  Processing document: '{doc_id}'…")

    try:
        # Directly get a list of QA dicts
        pairs = generate_qa_pairs_for_doc(
            doc_id,
            num_pairs=args.num
        )
    except Exception as e:
        print(f"❌ Failed to generate QA pairs for {doc_id}: {e}")
        return

    # Ensure every item has a unique id
    for item in pairs:
        if "id" not in item or not item["id"]:
            item["id"] = str(uuid.uuid4())

    # Write out the results
    with open(args.output, "w", encoding="utf-8") as fout:
        json.dump(pairs, fout, ensure_ascii=False, indent=2)
    print(f"✨ Wrote {len(pairs)} QA pairs to {args.output}")


# --- Reset helpers -----------------------------------------------------------

def _reset_qdrant() -> None:
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60.0)
    try:
        collections = {c.name for c in client.get_collections().collections}
    except Exception:
        collections = set()

    # Candidate collections to drop
    candidates = {QDRANT_COLLECTION, QDRANT_QA_DB, f"{QDRANT_COLLECTION}_questions"}
    for col in sorted(candidates):
        try:
            if col in collections:
                client.delete_collection(col)
                print(f"🗑️  Deleted Qdrant collection '{col}'")
            else:
                # Try anyway; Qdrant returns error if not exists
                client.delete_collection(col)
                print(f"🗑️  Requested delete for Qdrant collection '{col}' (may not have existed)")
        except Exception as e:
            print(f"⚠️  Could not delete Qdrant collection '{col}': {e}")


def _reset_neo4j(hard: bool = True) -> None:
    driver = get_driver()
    if hard:
        # Drop and recreate database via system DB (requires Neo4j 4+ and privileges)
        try:
            with driver.session(database="system") as sess:
                sess.run(f"DROP DATABASE {NEO4J_DATABASE} IF EXISTS")
                sess.run(f"CREATE DATABASE {NEO4J_DATABASE}")
            print(f"🗑️  Dropped and recreated Neo4j database '{NEO4J_DATABASE}'")
        except Exception as e:
            print(f"⚠️  Could not drop/recreate Neo4j DB (falling back to soft reset): {e}")
            hard = False

    if not hard:
        # Soft reset: delete all nodes/relationships
        try:
            with driver.session(database=NEO4J_DATABASE) as sess:
                sess.run("MATCH (n) DETACH DELETE n")
            print(f"🧹 Cleared all nodes/relationships in Neo4j database '{NEO4J_DATABASE}'")
        except Exception as e:
            print(f"❌ Failed to clear Neo4j database '{NEO4J_DATABASE}': {e}")

    # Recreate constraints/indexes
    try:
        ensure_graph_indexes()
        print("✅ Ensured Neo4j constraints and indexes")
    except Exception as e:
        print(f"⚠️  Failed to ensure Neo4j constraints/indexes: {e}")


def cmd_reset(args):
    do_qdrant = not args.neo4j_only
    do_neo4j = not args.qdrant_only

    if do_qdrant:
        print("🚨 Resetting Qdrant…")
        _reset_qdrant()

    if do_neo4j:
        print("🚨 Resetting Neo4j…")
        _reset_neo4j(hard=not args.soft_neo4j)

    print("🎉 Reset complete.")


def cmd_chat(args):
    """Interactive chat interface for testing document retrieval"""
    print("🤖 Document Chat Interface (Hybrid RAG)")
    print("=" * 50)
    print("💡 Ask questions about your indexed documents")
    print("🔍 Using hybrid retrieval: Qdrant + Neo4j")
    print("❓ Commands: 'quit', 'exit', 'q' to exit")
    print("-" * 50)
    
    # Check if we have indexed documents
    try:
        # Test connection to Qdrant
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        
        # Check if collection exists and has data
        collections = client.get_collections()
        collection_names = [col.name for col in collections.collections]
        
        if QDRANT_COLLECTION not in collection_names:
            print("❌ No indexed documents found!")
            print("💡 Please run: python -m src.cli index 'data/raw/*.pdf' first")
            return
        
        # Check if collection has points
        collection_info = client.get_collection(QDRANT_COLLECTION)
        if collection_info.points_count == 0:
            print("❌ Collection is empty!")
            print("💡 Please run: python -m src.cli index 'data/raw/*.pdf' first")
            return
            
        print(f"✅ Found {collection_info.points_count} indexed documents")
        
        # Test Neo4j connection
        try:
            driver = get_driver()
            with driver.session(database=NEO4J_DATABASE) as session:
                result = session.run("MATCH (n) RETURN count(n) as count").single()
                node_count = result.get("count", 0)
                print(f"✅ Found {node_count} nodes in Neo4j")
        except Exception as e:
            print(f"⚠️ Neo4j connection issue: {e}")
            print("💡 Graph features may be limited")
        
    except Exception as e:
        print(f"❌ Error connecting to databases: {e}")
        print("💡 Please check your .env configuration")
        return
    
    # Import the router for hybrid retrieval
    try:
        from .rag.router import router
        print("✅ Hybrid RAG router loaded")
    except Exception as e:
        print(f"❌ Failed to load hybrid RAG router: {e}")
        print("💡 Falling back to vector-only retrieval")
        router = None
    
    # Determine retrieval method based on command-line arguments
    use_hybrid = not args.vector_only  # Default to hybrid unless --vector-only is specified
    if use_hybrid and router:
        print("🔍 Using hybrid retrieval (Qdrant + Neo4j)")
    else:
        print("🔍 Using vector-only retrieval (Qdrant only)")
        router = None  # Force vector-only mode
    
    # Start chat loop
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
            
            if router:
                # Use hybrid retrieval (Qdrant + Neo4j)
                print("🔄 Using hybrid retrieval (Qdrant + Neo4j)...")
                try:
                    # Use the same method as the API but get the full response
                    from .rag.engine.models import AnswerRequest
                    request = AnswerRequest(query=question, room_id=None)
                    response = router.get_engine().answer(request)
                    answer = response.answer
                    evidence = response.evidence
                    
                    print("✅ Hybrid retrieval completed")
                    
                    # Display results
                    print("\n" + "=" * 50)
                    print("📝 ANSWER:")
                    print(answer)
                    print("=" * 50)
                    
                    # Display source information from evidence
                    print("\n📚 SOURCES:")
                    if evidence:
                        print(f"  Hybrid retrieval found {len(evidence)} sources:")
                        
                        # Group sources by type
                        vector_sources = []
                        graph_sources = []
                        
                        for i, ev in enumerate(evidence[:5], 1):  # Show top 5 sources
                            try:
                                # Extract source information from evidence using the correct structure
                                if hasattr(ev, 'payload') and ev.payload:
                                    # Use the payload directly from Evidence object
                                    payload = ev.payload
                                    page = payload.get('page', 'Unknown')
                                    doc_id = payload.get('doc_id', 'Unknown')
                                    
                                    # Try to get source type from different possible fields
                                    source_type = payload.get('type', 'Unknown')
                                    if source_type == 'Unknown':
                                        # For table embeddings, use 'level' field
                                        level = payload.get('level', 'Unknown')
                                        if level == 'row':
                                            source_type = 'table_row'
                                        elif level == 'table':
                                            source_type = 'table_summary'
                                        else:
                                            source_type = level
                                    
                                    score = getattr(ev, 'score', 'N/A')
                                    
                                    # Get text preview for better context
                                    text_preview = payload.get('text', '')[:100]
                                    if len(text_preview) > 100:
                                        text_preview += "..."
                                    
                                    source_info = f"  {i}. {source_type}: {doc_id}, Page: {page}, Score: {score:.3f}"
                                    if text_preview:
                                        source_info += f"\n      Preview: {text_preview}"
                                    
                                    # Categorize by source type
                                    if source_type in ['paragraph', 'table_summary', 'table_record', 'table_column', 'table_row']:
                                        vector_sources.append(source_info)
                                    else:
                                        graph_sources.append(source_info)
                                        
                                elif hasattr(ev, 'raw') and hasattr(ev.raw, 'payload'):
                                    # Fallback to raw.payload structure
                                    payload = ev.raw.payload
                                    page = payload.get('page', 'Unknown')
                                    doc_id = payload.get('doc_id', 'Unknown')
                                    source_type = payload.get('type', 'Unknown')
                                    score = getattr(ev, 'score', 'N/A')
                                    
                                    # Get text preview for better context
                                    text_preview = payload.get('text', '')[:100]
                                    if len(text_preview) > 100:
                                        text_preview += "..."
                                    
                                    source_info = f"  {i}. {source_type}: {doc_id}, Page: {page}, Score: {score:.3f}"
                                    if text_preview:
                                        source_info += f"\n      Preview: {text_preview}"
                                    
                                    # Categorize by source type
                                    if source_type in ['paragraph', 'table_summary', 'table_record', 'table_column']:
                                        vector_sources.append(source_info)
                                    else:
                                        graph_sources.append(source_info)
                                        
                                elif hasattr(ev, 'id'):
                                    # Try to get info from evidence ID
                                    ev_id = ev.id
                                    score = getattr(ev, 'score', 'N/A')
                                    source_info = f"  {i}. Evidence ID: {ev_id}, Score: {score:.3f}"
                                    vector_sources.append(source_info)
                                    
                                else:
                                    # Fallback for graph evidence
                                    source_info = f"  {i}. Graph source: {getattr(ev, 'id', 'Unknown')}"
                                    graph_sources.append(source_info)
                                    
                            except Exception as e:
                                print(f"  {i}. Error extracting source info: {e}")
                        
                        # Display categorized sources
                        if vector_sources:
                            print("  📄 Vector sources (Qdrant):")
                            for source in vector_sources:
                                print(source)
                        
                        if graph_sources:
                            print("  🕸️ Graph sources (Neo4j):")
                            for source in graph_sources:
                                print(source)
                        
                    else:
                        print("  No sources found in hybrid retrieval")
                    
                except Exception as e:
                    print(f"❌ Hybrid retrieval failed: {e}")
                    print("🔄 Falling back to vector-only retrieval...")
                    router = None  # Fall back to old method
            
            if not router:
                # Fallback to old vector-only method
                print("🔄 Using vector-only retrieval (Qdrant)...")
                
                # Use original query since this is a simple interface
                rewritten_query = question  # Placeholder for future enhancement
                
                # Retrieve relevant chunks with optimized parameters
                hits = retrieval_service.retrieve(query=rewritten_query, top_k=10, score_threshold=0.3)
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
    parser = argparse.ArgumentParser(prog="rag")
    sub = parser.add_subparsers()

    ix = sub.add_parser("index")
    ix.add_argument("pattern", help="glob pattern, e.g. data/raw/*.pdf")
    ix.add_argument("--no-recreate", dest="recreate", action="store_false")
    ix.set_defaults(func=cmd_index, recreate=True)

    q = sub.add_parser("query")
    q.add_argument("question")
    q.add_argument("--topk", type=int, default=5)
    q.set_defaults(func=cmd_query)

    l = sub.add_parser("list-docs", help="List all available document IDs")
    l.set_defaults(func=cmd_list_docs)
    
    # Add chat subcommand
    c = sub.add_parser("chat", help="Start an interactive chat session with your documents")
    c.add_argument("--vector-only", action="store_true", 
                   help="Use vector-only retrieval (Qdrant only, no Neo4j)")
    c.add_argument("--hybrid", action="store_true", 
                   help="Use hybrid retrieval (Qdrant + Neo4j) - default")
    c.set_defaults(func=cmd_chat)

    g = sub.add_parser("qa-generate", help="Generate Q&A pairs from indexed document")
    g.add_argument("doc_id", help="Document ID to generate QA pairs for")
    g.add_argument("-n", "--num", type=int, default=5,
                help="How many Q&A pairs to generate")
    g.add_argument("-o", "--output", default="qa_pairs.json",
                help="JSON file to write the results")
    g.set_defaults(func=cmd_qa_generate)
    
    # Debug extract command: dump extraction output to test_results JSON
    def cmd_debug_extract(args):
        pdf_path = args.pdf
        out = {
            "meta": {
                "pdf": os.path.basename(pdf_path),
            }
        }
        pages = extract_text_pages(pdf_path)
        tables = extract_tables_per_page(pdf_path)
        nodes = build_page_nodes(pdf_path)

        out["pages"] = pages
        out["tables"] = {}
        for k, dfs in tables.items():
            # Convert first few rows to plain lists for JSON compactness
            out["tables"][str(k)] = [list(map(str, row)) for row in dfs[0].values.tolist()] if dfs else []
        out["nodes"] = nodes

        os.makedirs("test_results", exist_ok=True)
        fname = os.path.join("test_results", f"extract_{os.path.basename(pdf_path)}.json")
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"✅ Saved extraction debug to {fname}")
        print(f"Pages: {len(pages)} | Tables(pages): {len(tables)} | Nodes: {len(nodes)}")

    dbg = sub.add_parser("debug-extract", help="Extract text/tables/nodes and save JSON to test_results/")
    dbg.add_argument("pdf", help="Path to PDF")
    dbg.set_defaults(func=cmd_debug_extract)
    
    # Reset subcommand
    r = sub.add_parser("reset", help="Reset Qdrant collections and Neo4j database")
    r.add_argument("--qdrant-only", action="store_true", help="Only reset Qdrant")
    r.add_argument("--neo4j-only", action="store_true", help="Only reset Neo4j")
    r.add_argument("--soft-neo4j", action="store_true", help="Soft reset Neo4j (delete nodes) instead of drop/create DB")
    r.set_defaults(func=cmd_reset)
    

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()