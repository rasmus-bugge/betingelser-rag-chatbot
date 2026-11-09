# pgvector_store.py
# VECTOR STORE no. 1: PostgreSQL with the pgvector extension (runs on your own machine).
#
# A "vector store" is the database that holds the chunk vectors and can answer:
# "which stored vectors are closest to THIS vector?". This project has two of them
# (this file and qdrant_store.py). Both offer the same function shape:
#       search(question_vector, limit) -> list of chunks, best first
# so search.py can use either one without caring which.

from sqlalchemy import select
from sqlalchemy.orm import Session

from database import engine
from rag.rag_models import DocumentChunk


def search_pgvector(
    question_vector: list[float], limit: int, filters: dict[str, str]
) -> list[dict]:
    """Return the `limit` chunks whose vectors are closest to `question_vector`.
    `filters` keeps only matching chunks, e.g. {"product": "Cyberforsikring"}."""

    # cosine_distance(...) is pgvector's "<=>" operator: 0 = same direction,
    # larger = less similar. .label("distance") names the column in the result.
    distance = DocumentChunk.embedding.cosine_distance(question_vector).label("distance")

    # THE SQL QUERY. In plain SQL this is roughly:
    #   SELECT *, embedding <=> :question_vector AS distance
    #   FROM document_chunks
    #   WHERE language = 'da'  -- one condition per filter
    #   ORDER BY distance      -- smallest distance first = most similar first
    #   LIMIT 5;
    # It is an EXACT search: Postgres compares the question with every matching row.
    query = select(DocumentChunk, distance)
    for column, value in filters.items():
        query = query.where(getattr(DocumentChunk, column) == value)

    with Session(engine) as session:
        rows = session.execute(query.order_by(distance).limit(limit)).all()

        # score = 1 - distance, so BIGGER = MORE similar (easier to read).
        return [
            {
                "file": chunk.file,
                "doc_type": chunk.doc_type,
                "product": chunk.product,
                "variant": chunk.variant,
                "language": chunk.language,
                "page": chunk.page,
                "chunk_number": chunk.chunk_number,
                "section": chunk.section,
                "score": round(1 - dist, 4),
                "text": chunk.text,
            }
            for chunk, dist in rows
        ]
