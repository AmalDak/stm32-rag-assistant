# api/main.py
# FastAPI backend — exposes the full pipeline as a REST endpoint.

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict

from retrieval.requirement_extractor import extract_requirements
from retrieval.catalog_filter        import filter_candidates
from retrieval.vector_retriever      import retrieve
from retrieval.reranker              import rerank
from generation.prompt_builder       import build_prompt
from generation.llm_client           import generate

app = FastAPI(
    title="STM32 RAG Assistant",
    description="Pre-sales consultant for STM32 products",
    version="1.0.0"
)


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer          : str
    candidates      : List[Dict]
    sources         : List[Dict]
    requirements    : Dict


@app.get("/")
def health():
    return {"status": "ok", "service": "STM32 RAG Assistant"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    # Step 1 — Extract structured requirements
    requirements = extract_requirements(request.query)

    # Step 2 — Filter catalog deterministically
    candidates = filter_candidates(requirements)

    # Step 3 — Retrieve relevant chunks from Qdrant
    chunks = retrieve(request.query, candidates)

    # Step 4 — Rerank for precision
    chunks = rerank(request.query, chunks)

    # Step 5 — Build prompt and generate answer
    prompt = build_prompt(request.query, candidates, chunks)
    answer = generate(prompt)

    # Sources returned for UI citation display
    sources = [
        {
            "file"  : c["source_file"],
            "page"  : c["page_num"],
            "score" : round(c.get("rerank_score", 0), 3)
        }
        for c in chunks
    ]

    return QueryResponse(
        answer       = answer,
        candidates   = candidates,
        sources      = sources,
        requirements = requirements,
    )