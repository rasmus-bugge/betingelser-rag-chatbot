# search.py
# SEARCH. Given a question, return the chunks whose meaning is closest.
#
#   question --(AI: embed_query)--> vector --(maths in the vector store)--> nearest chunks
#
# Only the first arrow is AI. The second is a "nearest vectors" lookup, done by one
# of our two vector stores:
#   pgvector  PostgreSQL on your own machine   (rag/pgvector_store.py)   <- default
#   qdrant    Qdrant Cloud, hosted             (rag/qdrant_store.py)
# The setting VECTOR_STORE (in .env, or typed before a command) picks which one.
#
# Try it in the terminal (from backend/):
#   .venv/bin/python -m rag.search "Hvad er karenstid i cyberforsikringen?"
#   VECTOR_STORE=qdrant .venv/bin/python -m rag.search "Hvad er karenstid i cyberforsikringen?"

import os
import sys

from rag.embeddings import embed_query   # (importing this also loads the .env file)


def current_store() -> str:
    """Which vector store are we using? Read from the VECTOR_STORE setting."""
    return os.environ.get("VECTOR_STORE", "pgvector").lower()


def search_chunks(
    question: str,
    limit: int = 5,
    question_vector: list[float] | None = None,
    filters: dict[str, str] | None = None,
) -> list[dict]:
    """Return the `limit` chunks closest in meaning to `question`, best first.

    `question_vector` is optional: normally we embed the question here. The
    evaluation passes in a vector it saved earlier, so that two runs use exactly
    the same numbers (see evaluate.py for why that matters).

    `filters` limits the search to chunks with these labels, for example
    {"product": "Cyberforsikring"}. Without it we search the Danish chunks only,
    because some English documents are translations that would fill the top results
    with the same clause twice."""

    if filters is None:
        filters = {"language": "da"}

    # 1) THE AI STEP: turn the question into a vector (with Qwen's query instruction).
    if question_vector is None:
        question_vector = embed_query(question)

    # 2) THE LOOKUP: ask the chosen store for the nearest chunk vectors.
    #    The imports sit INSIDE the if-blocks on purpose: when we run on Qdrant only
    #    (for example on a web host), Python never loads the Postgres code, so no
    #    Postgres database is needed there.
    store = current_store()
    if store == "qdrant":
        from rag.qdrant_store import search_qdrant
        return search_qdrant(question_vector, limit, filters)
    if store == "pgvector":
        from rag.pgvector_store import search_pgvector
        return search_pgvector(question_vector, limit, filters)
    raise ValueError(f"Unknown VECTOR_STORE '{store}'. Use 'pgvector' or 'qdrant'.")


# Lets us test search from the terminal without the website.
if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "Hvad er karenstid i cyberforsikringen?"
    print(f"QUESTION: {question}   (vector store: {current_store()})")
    for rank, hit in enumerate(search_chunks(question), start=1):
        print(f"\n#{rank}  score {hit['score']:.3f}  |  {hit['product']} ({hit['doc_type']})"
              f"  |  {hit['file']}, page {hit['page']}")
        print(hit["text"][:400].replace("\n", " "))
