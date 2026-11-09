# qdrant_store.py
# VECTOR STORE no. 2: Qdrant Cloud (a hosted database built only for vectors).
#
# Why a second store? Postgres runs on your own machine, so a website on the
# internet cannot reach it. Qdrant Cloud is hosted, so a deployed app can.
#
# Qdrant's words for things:
#   collection = a table                      (ours is called "document_chunks")
#   point      = a row: an id + a vector + a "payload"
#   payload    = the extra data stored next to the vector (file, page, text ...)

import os
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

from rag.embeddings import DIMENSIONS   # 1024: the size of our vectors

# Load the .env file in the project root (three folders up: rag/ -> backend/ -> root).
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

COLLECTION = "document_chunks"


def get_client() -> QdrantClient:
    """Connect to our Qdrant cluster. The address and key come from .env."""
    return QdrantClient(
        url=os.environ["QDRANT_ENDPOINT"],
        api_key=os.environ["QDRANT_API_KEY"],
        timeout=60,   # seconds to wait for an answer before giving up
    )


def recreate_collection(client: QdrantClient) -> None:
    """Delete the collection if it exists and create an empty one. Starting clean
    means we can never end up with old and new chunks mixed together."""
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    client.create_collection(
        collection_name=COLLECTION,
        # Every vector has 1024 numbers, and "closeness" means cosine similarity,
        # the same measure we use in Postgres.
        vectors_config=models.VectorParams(size=DIMENSIONS, distance=models.Distance.COSINE),
    )
    # A payload index lets Qdrant apply a filter quickly and accurately, like an
    # index on a column in Postgres. We index the fields the search filters on.
    for field in ("product", "variant", "language"):
        client.create_payload_index(
            collection_name=COLLECTION,
            field_name=field,
            field_schema=models.PayloadSchemaType.KEYWORD,   # "exact text match"
        )


def upload_points(client: QdrantClient, rows: list[dict]) -> None:
    """Store a batch of chunks. Each row is a dictionary with an "id", a "vector"
    and the chunk's label and text."""
    client.upsert(   # "upsert" = insert, or update if the id already exists
        collection_name=COLLECTION,
        points=[
            models.PointStruct(
                id=row["id"],
                vector=row["vector"],
                # Everything except id and vector goes into the payload.
                payload={key: value for key, value in row.items() if key not in ("id", "vector")},
            )
            for row in rows
        ],
        wait=True,   # don't return until Qdrant has really saved them
    )


def search_qdrant(
    question_vector: list[float], limit: int, filters: dict[str, str]
) -> list[dict]:
    """Return the `limit` chunks whose vectors are closest to `question_vector`.
    `filters` keeps only matching chunks, e.g. {"product": "Cyberforsikring"}."""
    client = get_client()

    # "must" = every condition has to be true (the same as WHERE ... AND ... in SQL).
    query_filter = models.Filter(
        must=[
            models.FieldCondition(key=field, match=models.MatchValue(value=value))
            for field, value in filters.items()
        ]
    )

    # Unlike our Postgres search, Qdrant uses an INDEX (called HNSW): it does not
    # compare with every point, but walks a pre-built network of neighbours. That
    # is much faster on millions of points and ALMOST always finds the true best.
    result = client.query_points(
        collection_name=COLLECTION,
        query=question_vector,
        query_filter=query_filter,
        limit=limit,
        with_payload=True,   # also send back file, page, text ...
    )

    # With cosine distance, Qdrant's score already IS the similarity (bigger = closer).
    return [
        {
            "file": point.payload["file"],
            "doc_type": point.payload["doc_type"],
            "product": point.payload["product"],
            "variant": point.payload["variant"],
            "language": point.payload["language"],
            "page": point.payload["page"],
            "chunk_number": point.payload["chunk_number"],
            "section": point.payload["section"],
            "score": round(point.score, 4),
            "text": point.payload["text"],
        }
        for point in result.points
    ]
