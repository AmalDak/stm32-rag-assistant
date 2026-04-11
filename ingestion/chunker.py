# ingestion/chunker.py
# Splits extracted pages into chunks ready for embedding

from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict


# --- Tuning constants -------------------------------------------
# CHUNK_SIZE: how many characters per chunk
#   Too small → chunks lose context ("1.7 V" with no label)
#   Too large → chunks become noisy, retrieval gets imprecise
#   500–800 is the sweet spot for datasheets
CHUNK_SIZE = 600

# CHUNK_OVERLAP: characters shared between adjacent chunks
#   Prevents losing info that falls exactly on a boundary
#   10–15% of CHUNK_SIZE is a good rule of thumb
CHUNK_OVERLAP = 80

# Separators tried IN ORDER — stops at first match per split
# "\n\n" catches section breaks
# "\n"   catches line breaks (table rows, spec lines)
# ". "   catches sentence endings inside paragraphs
# " "    last resort word boundary
SEPARATORS = ["\n\n", "\n", ". ", " "]
# ----------------------------------------------------------------


_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=SEPARATORS,
    length_function=len,
)


def chunk_pages(pages: List[Dict]) -> List[Dict]:
    """
    Takes the output of pdf_parser.parse_all() and returns
    a flat list of chunks, each with full metadata:
      {
        text        : the chunk content,
        source_file : which PDF it came from,
        page_num    : which page,
        chunk_index : position within that page
      }
    """
    all_chunks = []

    for page in pages:
        raw_chunks = _splitter.split_text(page["text"])

        for i, chunk_text in enumerate(raw_chunks):

            # Skip chunks that are too short to be meaningful
            # (leftover headers, page numbers, single values)
            if len(chunk_text.strip()) < 60:
                continue

            all_chunks.append({
                "text": chunk_text.strip(),
                "source_file": page["source_file"],
                "page_num": page["page_num"],
                "chunk_index": i,
            })

    return all_chunks


def print_stats(chunks: List[Dict]):
    """Utility — call this after chunking to sanity check results."""
    lengths = [len(c["text"]) for c in chunks]
    sources = {}
    for c in chunks:
        sources[c["source_file"]] = sources.get(c["source_file"], 0) + 1

    print(f"Total chunks : {len(chunks)}")
    print(f"Avg length   : {sum(lengths) // len(lengths)} chars")
    print(f"Min length   : {min(lengths)} chars")
    print(f"Max length   : {max(lengths)} chars")
    print(f"\nChunks per document:")
    for src, count in sources.items():
        print(f"  {src}: {count} chunks")


# Quick test — run directly to verify chunking looks sane
if __name__ == "__main__":
    from pdf_parser import parse_all

    pages = parse_all("../data/datasheets")
    chunks = chunk_pages(pages)
    print_stats(chunks)

    print(f"\n--- Sample chunk ---")
    print(chunks[10]["text"])
    print(f"\nFrom: {chunks[10]['source_file']} p.{chunks[10]['page_num']}")