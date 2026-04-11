# STM32 RAG Pre-Sales Consultant

A domain-specific AI assistant that recommends STM32 microcontrollers
based on natural language requirements, grounded in official ST documentation.

## Architecture

- **Retrieval**: sentence-transformers embeddings + Qdrant vector database
- **Filtering**: deterministic catalog filtering before vector search
- **Reranking**: cross-encoder reranking for precision
- **Generation**: Gemini 2.0 Flash with cited sources
- **Backend**: FastAPI
- **Frontend**: Streamlit
- **Infrastructure**: Docker Compose

## How to run

1. Clone the repo
2. Add your datasheets to `data/datasheets/`
3. Create a .env file based on .env.example and add your API key
4. Run:

\```bash
docker compose up -d
docker compose exec backend python -m ingestion.run_ingestion
\```

5. Open http://localhost:8501

## Stack

Python · FastAPI · Streamlit · Qdrant · sentence-transformers · Docker