# retrieval/vector_retriever.py
# Searches Qdrant for relevant chunks, filtered to only
# the documents belonging to candidate chips.

from qdrant_client.models import Filter, FieldCondition, MatchAny
from ingestion.embedder import get_client, get_model
from typing import List, Dict

COLLECTION_NAME = "stm32_docs"
TOP_K           = 10   # chunks to retrieve before reranking


def retrieve(query: str, candidates: List[Dict]) -> List[Dict]:
    """
    Embeds the query and searches Qdrant.
    Restricts search to documents of the candidate chips
    returned by the catalog filter — this is the key
    design decision that makes retrieval precise.

    Returns top-k chunks as a list of dicts:
      { text, source_file, page_num, score }
    """
    client = get_client()
    model  = get_model()

    # Collect the datasheet filenames for candidate chips
    target_files = list({
        c["datasheet_file"] for c in candidates
        if c.get("datasheet_file")
    })

    # Embed the query
    query_vector = model.encode(query, convert_to_numpy=True).tolist()

    # Build Qdrant metadata filter — only search relevant docs
    search_filter = Filter(
        must=[
            FieldCondition(
                key="source_file",
                match=MatchAny(any=target_files)
            )
        ]
    ) if target_files else None

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=search_filter,
        limit=TOP_K,
        with_payload=True
    ).points

    chunks = []
    for hit in results:
        chunks.append({
            "text"        : hit.payload["text"],
            "source_file" : hit.payload["source_file"],
            "page_num"    : hit.payload["page_num"],
            "score"       : hit.score,
        })

    return chunks