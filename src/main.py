from pdf_parser import parse_pdf
from chunker import chunk_text
from embedder import embed
from indexer import index_chunks

def ingest(path):
    text = parse_pdf(path)
    chunks = chunk_text(text, lang="zh")
    vectors = [embed(c) for c in chunks]
    index_chunks(chunks, vectors)

if __name__ == "__main__":
    import sys
    ingest(sys.argv[1])
