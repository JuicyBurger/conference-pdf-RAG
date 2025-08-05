import argparse
import glob
import os
import json
import uuid


from .data.pdf_ingestor import ingest_pdf
from .rag.indexing.indexer import init_collection, index_pdf
from .rag.retriever    import retrieve, get_all_doc_ids
from .models.reranker  import rerank
from .rag.generator    import generate_answer, generate_qa_pairs_for_doc
from .rag.suggestions  import generate_suggestions_for_doc, batch_generate_suggestions


def cmd_ingest(args):
    doc = ingest_pdf(args.pdf)
    print(f"[INGEST] Extracted {len(doc['pages'])} pages, {len(doc['images'])} images")

def cmd_index(args):
    if args.recreate:
        init_collection()
    paths = glob.glob(args.pattern)
    total = 0
    for p in paths:
        total += index_pdf(p)
    print(f"[INDEX] Done: {total} chunks")

def cmd_query(args):
    hits = retrieve(args.question, top_k=args.topk)
    hits = rerank(args.question, hits)
    print(generate_answer(args.question, hits))

def cmd_list_docs(args):
    """List all available document IDs"""
    try:
        doc_ids = get_all_doc_ids()
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

def cmd_suggestions_generate(args):
    """Generate question suggestions for a document or all documents"""
    if args.doc_id:
        # Generate for specific document
        success = generate_suggestions_for_doc(
            doc_id=args.doc_id,
            num_questions=args.num,
            auto_init_collection=True,
            use_lightweight=not args.full_qa  # Use lightweight unless --full-qa is specified
        )
        if success:
            print(f"✨ Generated suggestions for document: {args.doc_id}")
        else:
            print(f"❌ Failed to generate suggestions for document: {args.doc_id}")
    else:
        # Generate for all documents
        doc_ids = get_all_doc_ids()
        if not doc_ids:
            print("❌ No documents found in the index")
            return
        
        print(f"🚀 Generating suggestions for {len(doc_ids)} documents...")
        results = batch_generate_suggestions(
            doc_ids, 
            args.num,
            use_lightweight=not args.full_qa  # Use lightweight unless --full-qa is specified
        )
        
        print(f"✨ Batch generation complete:")
        print(f"   Successful: {results['successful']}")
        print(f"   Failed: {results['failed']}")
        if results['failed_docs']:
            print(f"   Failed docs: {', '.join(results['failed_docs'])}")

def main():
    parser = argparse.ArgumentParser(prog="rag")
    sub = parser.add_subparsers()

    i = sub.add_parser("ingest")
    i.add_argument("pdf")
    i.set_defaults(func=cmd_ingest)

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

    g = sub.add_parser("qa-generate", help="Generate Q&A pairs from indexed document")
    g.add_argument("doc_id", help="Document ID to generate QA pairs for")
    g.add_argument("-n", "--num", type=int, default=5,
                help="How many Q&A pairs to generate")
    g.add_argument("-o", "--output", default="qa_pairs.json",
                help="JSON file to write the results")
    g.set_defaults(func=cmd_qa_generate)
    
    s = sub.add_parser("suggestions-generate", help="Generate question suggestions for documents")
    s.add_argument("--doc-id", help="Specific document ID (if not provided, generates for all documents)")
    s.add_argument("-n", "--num", type=int, default=8,
                help="How many question suggestions to generate per document")
    s.add_argument("--full-qa", action="store_true", 
                help="Use full QA generation (slower but more detailed) instead of lightweight question-only generation")
    s.set_defaults(func=cmd_suggestions_generate)
    
    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()