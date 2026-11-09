# main.py
# The backend of the chatbot: a small web API with ONE endpoint.
#   GET /rag/search?q=...   the chunks of the insurance conditions closest to a question
#
# Start it from the backend/ folder (port 8001, so it doesn't clash with other projects):
#   .venv/bin/uvicorn main:app --reload --port 8001
# Automatic documentation: http://localhost:8001/docs

from collections.abc import Iterator

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database import engine
from rag.search import search_chunks

# The application object. Every endpoint is attached to it.
app = FastAPI(title="Betingelser RAG chatbot")

# CORS: a browser refuses to let a page from one address (the React app on
# localhost:5174) read answers from another (this API on localhost:8001) unless
# the API says it is allowed. This block gives that permission.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5174"],
    allow_methods=["GET"],
)


# A database session for ONE request. FastAPI calls this (see "Depends" below),
# hands the session to the endpoint and closes it afterwards.
def get_session() -> Iterator[Session]:
    with Session(engine) as session:
        yield session


# The shape of one search result: a chunk of a PDF, where it came from, and how
# close it is to the question.
class ChunkHit(BaseModel):
    file: str  # "betingelser-cyberforsikring.pdf"
    doc_type: str  # "Betingelser" or "Produktinformation"
    product: str  # "Cyberforsikring"
    page: int  # the page where the chunk starts
    chunk_number: int
    score: float  # 0..1, bigger = closer in meaning to the question
    text: str


# GET /rag/search?q=Hvad er karenstid?&limit=5
# Turns the question into a vector and returns the `limit` closest chunks.
@app.get("/rag/search", response_model=list[ChunkHit])
def rag_search(
    q: str = Query(min_length=2, max_length=500),  # the question; 2-500 characters
    limit: int = Query(5, ge=1, le=20),  # how many chunks (default 5, max 20)
    session: Session = Depends(get_session),
):
    try:
        return search_chunks(session, q, limit)
    except RuntimeError as error:
        # search_chunks raises RuntimeError when the embedding service (OpenRouter)
        # fails. 502 = "Bad Gateway": a service WE depend on failed, not our code.
        raise HTTPException(status_code=502, detail=f"Embedding service failed: {error}")
