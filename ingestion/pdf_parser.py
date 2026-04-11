# ingestion/pdf_parser.py
# Extracts clean text from STM32 datasheets (PDF format)

import fitz  # PyMuPDF
import os
from pathlib import Path
from typing import List, Dict


def extract_pages(pdf_path: str) -> List[Dict]:
    """
    Extract text from each page of a PDF.
    Returns a list of dicts, one per page:
      { page_num, text, source_file }
    """
    doc = fitz.open(pdf_path)
    source_file = Path(pdf_path).name
    pages = []

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        text = _clean(text)

        # Skip pages with almost no text (cover pages, figures)
        if len(text) < 100:
            continue

        pages.append({
            "page_num": page_num,
            "text": text,
            "source_file": source_file,
        })

    doc.close()
    return pages


def _clean(text: str) -> str:
    """
    Basic cleaning:
    - Collapse excessive whitespace
    - Remove form-feed characters
    - Strip leading/trailing space per line
    """
    lines = text.split("\n")
    cleaned = []

    for line in lines:
        line = line.strip()
        if line:
            cleaned.append(line)

    return "\n".join(cleaned)


def parse_all(datasheets_dir: str) -> List[Dict]:
    """
    Parse every PDF in the datasheets directory.
    Returns a flat list of all pages across all documents.
    """
    all_pages = []
    pdf_files = list(Path(datasheets_dir).glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDFs found in {datasheets_dir}"
        )

    for pdf_path in pdf_files:
        print(f"Parsing: {pdf_path.name}")
        pages = extract_pages(str(pdf_path))
        all_pages.extend(pages)
        print(f"  → {len(pages)} pages extracted")

    return all_pages


# Quick test — run this file directly to verify it works
if __name__ == "__main__":
    pages = parse_all("../data/datasheets")
    print(f"\nTotal pages extracted: {len(pages)}")
    print(f"Sample (page 1 of first doc):\n")
    print(pages[0]["text"][:500])