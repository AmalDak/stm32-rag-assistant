# ingestion/run_ingestion.py
# Orchestrates the full ingestion pipeline:
# parse → chunk → embed → store
# Run this once before starting the API.

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.pdf_parser import parse_all
from ingestion.chunker   import chunk_pages, print_stats
from ingestion.embedder  import init_collection, embed_and_store

DATASHEETS_DIR = "data/datasheets"


def run():
    print("=== STM32 RAG — Ingestion Pipeline ===")

    # Step 1 — Parse PDFs
    print("\n[1/3] Parsing PDFs...")
    pages = parse_all(DATASHEETS_DIR)
    print(f"  → {len(pages)} pages extracted")

    # Step 2 — Chunk pages
    print("\n[2/3] Chunking pages...")
    chunks = chunk_pages(pages)
    print_stats(chunks)

    # Step 3 — Embed and store in Qdrant
    print("\n[3/3] Embedding and storing in Qdrant...")
    init_collection(recreate=True)
    embed_and_store(chunks)

    print("\n=== Ingestion complete. Ready to query. ===")


if __name__ == "__main__":
    run()