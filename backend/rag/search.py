# search.py
# STEP 4e: SEARCH. Given a question, return the chunks whose meaning is closest.
#
#   question --(AI: embed_query)--> vector --(maths in Postgres)--> 5 nearest chunks
#
# Only the first arrow is AI. The second is a plain SQL query using pgvector's
# "cosine distance" operator (<=>), which is the same calculation as
# cosine_similarity() in embeddings.py, but done by the database.
#
# We do an EXACT search: Postgres compares the question with every one of the 4,257
# rows. That takes a few milliseconds at our size, so no index is needed yet.
#
# Try it in the terminal (from backend/):
#   .venv/bin/python -m rag.search "Hvad er karenstid i cyberforsikringen?"

import sys

from sqlalchemy import select
from sqlalchemy.orm import Session

from database import engine
from rag.embeddings import embed_query
from rag.rag_models import DocumentChunk


def search_chunks(session: Session, question: str, limit: int = 5) -> list[dict]:
    """Return the `limit` chunks closest in meaning to `question`, best first."""

    # 1) THE AI STEP: turn the question into a vector (with Qwen's query instruction).
    question_vector = embed_query(question)

    # 2) cosine_distance(...) is pgvector's "<=>" operator: 0 = same direction,
    #    larger = less similar. .label("distance") names the column in the result.
    distance = DocumentChunk.embedding.cosine_distance(question_vector).label("distance")

    # 3) THE SQL QUERY. In plain SQL this is roughly:
    #      SELECT *, embedding <=> :question_vector AS distance
    #      FROM document_chunks
    #      ORDER BY distance      -- smallest distance first = most similar first
    #      LIMIT 5;
    rows = session.execute(
        select(DocumentChunk, distance).order_by(distance).limit(limit)
    ).all()

    # 4) Turn each (chunk, distance) pair into a simple dictionary.
    #    score = 1 - distance, so BIGGER = MORE similar (easier to read).
    return [
        {
            "file": chunk.file,
            "doc_type": chunk.doc_type,
            "product": chunk.product,
            "page": chunk.page,
            "chunk_number": chunk.chunk_number,
            "score": round(1 - dist, 4),
            "text": chunk.text,
        }
        for chunk, dist in rows
    ]


# Lets us test search from the terminal without the website.
if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "Hvad er karenstid i cyberforsikringen?"
    print(f"QUESTION: {question}")
    with Session(engine) as session:
        for rank, hit in enumerate(search_chunks(session, question), start=1):
            print(f"\n#{rank}  score {hit['score']:.3f}  |  {hit['product']} ({hit['doc_type']})"
                  f"  |  {hit['file']}, page {hit['page']}")
            print(hit["text"][:400].replace("\n", " "))
