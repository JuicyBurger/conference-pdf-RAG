#!/usr/bin/env python3
"""
Generate QA pairs for a single document (legacy monolithic path) and export to JSON.
"""

import sys
import os
import time
import json
from datetime import datetime
import logging

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.rag.generator import generate_qa_pairs_for_doc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

DOC_ID = "112年報 20240531"
NUM_PAIRS = 5


def export_results(pairs, duration):
    """Save QA pairs and metadata to test_results/*.json"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = "test_results"
    os.makedirs(out_dir, exist_ok=True)

    meta = {
        "timestamp": ts,
        "doc_id": DOC_ID,
        "num_pairs": NUM_PAIRS,
        "generation_time_sec": duration,
        "total_sources": sum(len(p.get("source", [])) for p in pairs),
        "avg_sources_per_pair": sum(len(p.get("source", [])) for p in pairs) / len(pairs) if pairs else 0,
    }

    with open(os.path.join(out_dir, f"qa_pairs_{ts}.json"), "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "qa_pairs": pairs}, f, ensure_ascii=False, indent=2)

    print(f"\n📄 Results exported to {out_dir}/qa_pairs_{ts}.json")


def main():
    print("🚀 Generating QA pairs (legacy monolithic approach)...")
    start = time.time()
    pairs = generate_qa_pairs_for_doc(DOC_ID, NUM_PAIRS, timeout=120.0, context_top_k=30)
    duration = time.time() - start

    print(f"✅ Generated {len(pairs)} pairs in {duration:.2f}s\n")

    # Brief console summary
    for i, p in enumerate(pairs, 1):
        print(f"Q{i}: {p.get('question')}")
        print(f"   sources: {len(p.get('source', []))}\n")

    export_results(pairs, duration)


if __name__ == "__main__":
    main()
