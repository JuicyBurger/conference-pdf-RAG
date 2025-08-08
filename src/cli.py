import argparse
import glob
import os
import json
import uuid



from .rag.indexing.indexer import init_collection, index_pdf
from .rag.retriever    import retrieve, get_all_doc_ids
from .models.reranker  import rerank
from .rag.generator    import generate_answer, generate_qa_pairs_for_doc
from .rag.suggestions  import generate_suggestions_for_doc, batch_generate_suggestions
from .data.pdf_ingestor import extract_text_pages, extract_tables_per_page, build_page_nodes




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
    

    
    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()