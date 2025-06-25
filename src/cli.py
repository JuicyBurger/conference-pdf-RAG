import argparse
import glob
import os
import json
import uuid

from .pdf_ingestor import ingest_pdf
from .chunker      import chunk_text
from .embedder     import embed
from .indexer      import init_collection, index_pdf
from .retriever    import retrieve
from .reranker     import rerank
from .generator    import generate_answer, generate_qa_pairs_for_doc

def safe_json_loads(raw: str):
    """
    1) Extract the first […] substring
    2) Try json.loads
    3) If that fails, fall back to ast.literal_eval
    """
    # 1) find the JSON array
    start = raw.find('[')
    end   = raw.rfind(']')
    if start != -1 and end != -1:
        fragment = raw[start:end+1]
    else:
        fragment = raw

    # 2) try strict JSON
    try:
        return json.loads(fragment)
    except json.JSONDecodeError:
        pass

    # 3) fallback to Python literal eval (handles single quotes, trailing commas)
    try:
        return ast.literal_eval(fragment)
    except Exception as e:
        raise ValueError(f"Could not parse JSON: {e}\nFragment was:\n{fragment}")


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
    print(generate_answer(args.question, hits, lang=args.lang))

def cmd_qa_generate(args):
    pdf_paths = glob.glob(args.pattern)[:2]
    if not pdf_paths:
        print(f"❌ No PDFs matched pattern: {args.pattern}")
        return

    all_pairs = []
    for path in pdf_paths:
        doc_id = os.path.splitext(os.path.basename(path))[0]
        print(f"🗂  Processing '{doc_id}'…")
        try:
            result = generate_qa_pairs_for_doc(
                doc_id,
                num_pairs=args.num,
                lang=args.lang
            )

            raw = result.get("response", "")
            print("parsing data...")
            qa_list = safe_json_loads(raw)

            for item in qa_list:
                if "id" not in item:
                    item["id"] = str(uuid.uuid4())
                all_pairs.append(item)

            print(f"[+] Parsed {len(qa_list)} pairs for {doc_id}")

        except Exception as e:
            print(f"❌ Failed on {doc_id}: {e}")
            continue

    # write them out
    with open(args.output, "w", encoding="utf-8") as fout:
        json.dump(all_pairs, fout, ensure_ascii=False, indent=2)
    print(f"✨ Wrote {len(all_pairs)} QA pairs to {args.output}")

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
    q.add_argument(
        "--lang", choices=["zh","en"], default="zh",
        help="Output language: 'zh' for Traditional Chinese, 'en' for English"
    )
    q.set_defaults(func=cmd_query)

    g = sub.add_parser("qa-generate", help="Generate Q&A pairs from PDFs")
    g.add_argument("pattern", help="glob pattern, e.g. data/raw/*.pdf")
    g.add_argument("-n", "--num", type=int, default=5,
                help="How many Q&A pairs per document")
    g.add_argument("-l", "--lang", choices=["zh","en"], default="zh",
                help="Language for output")
    g.add_argument("-o", "--output", default="qa_pairs.json",
                help="JSON file to write the results")
    g.set_defaults(func=cmd_qa_generate)
    
    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()