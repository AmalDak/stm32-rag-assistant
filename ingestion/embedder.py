# ingestion/embedder.py
# Embeds chunks and stores them in Qdrant with metadata

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct
)
from typing import List, Dict
import uuid
import os


# --- Constants --------------------------------------------------
COLLECTION_NAME = "stm32_docs"
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"
VECTOR_DIM       = 384   # output dim of all-MiniLM-L6-v2
BATCH_SIZE       = 64    # chunks per upsert call
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT      = 6333
# ----------------------------------------------------------------

_model  = None
_client = None

def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model

def _get_client():
    global _client
    if _client is None:
        _client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    return _client


def init_collection(recreate: bool = False):
    """
    Create the Qdrant collection if it doesn't exist.
    Set recreate=True to wipe and rebuild from scratch.
    """
    existing = [c.name for c in _get_client().get_collections().collections]

    if COLLECTION_NAME in existing and not recreate:
        print(f"Collection '{COLLECTION_NAME}' already exists — skipping creation.")
        return

    if COLLECTION_NAME in existing and recreate:
        _get_client().delete_collection(COLLECTION_NAME)
        print(f"Deleted existing collection '{COLLECTION_NAME}'.")

    _get_client().create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_DIM,
            distance=Distance.COSINE
        )
    )
    print(f"Created collection '{COLLECTION_NAME}'.")


def embed_and_store(chunks: List[Dict]):
    """
    Embeds all chunks in batches and upserts into Qdrant.
    Each point stores the full metadata as payload so we
    can retrieve source_file and page_num at query time.
    """
    total   = len(chunks)
    stored  = 0

    for i in range(0, total, BATCH_SIZE):
        batch  = chunks[i : i + BATCH_SIZE]
        texts  = [c["text"] for c in batch]

        # Generate embeddings for the whole batch at once
        vectors = _get_model().encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True
        )

        points = [
            PointStruct(
                id      = str(uuid.uuid4()),
                vector  = vectors[j].tolist(),
                payload = {
                    "text"        : batch[j]["text"],
                    "source_file" : batch[j]["source_file"],
                    "page_num"    : batch[j]["page_num"],
                    "chunk_index" : batch[j]["chunk_index"],
                }
            )
            for j in range(len(batch))
        ]

        _get_client().upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )

        stored += len(batch)
        print(f"  Stored {stored}/{total} chunks...")

    print(f"Done. {stored} chunks indexed in Qdrant.")


def get_client() -> QdrantClient:
    """Returns the shared Qdrant client for use in retrieval."""
    return _get_client()


def get_model() -> SentenceTransformer:
    """Returns the shared embedding model for use in retrieval."""
    return _get_model()