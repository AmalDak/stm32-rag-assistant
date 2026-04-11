# retrieval/reranker.py
# Cross-encoder reranking of retrieved chunks.
# Improves precision by scoring query-chunk pairs directly.

from sentence_transformers import CrossEncoder
from typing import List, Dict

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOP_N          = 5   # chunks to keep after reranking

_reranker = CrossEncoder(RERANKER_MODEL)


def rerank(query: str, chunks: List[Dict]) -> List[Dict]:
    """
    Takes the top-k chunks from vector retrieval and
    reranks them using a cross-encoder model.

    Cross-encoders look at query + chunk together (not
    separately like bi-encoders), which gives much better
    relevance scores at the cost of speed.
    Returns top-N chunks sorted by rerank score.
    """
    if not chunks:
        return []

    pairs  = [(query, c["text"]) for c in chunks]
    scores = _reranker.predict(pairs)

    # Attach rerank score to each chunk
    for i, chunk in enumerate(chunks):
        chunk["rerank_score"] = float(scores[i])

    # Sort descending by rerank score, keep top N
    reranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:TOP_N]