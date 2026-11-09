# copy_to_qdrant.py
# Copy every chunk (label, text AND vector) from Postgres to Qdrant Cloud.
#
# The vectors already exist in Postgres, so nothing is embedded again: this costs
# no API calls. Run it after `python -m rag.load_chunks`, whenever the chunks change:
#   .venv/bin/python -m rag.copy_to_qdrant
#
# Safe to run again: it empties the Qdrant collection first, then refills it.

from sqlalchemy import select
from sqlalchemy.orm import Session

from database import engine
from rag.qdrant_store import COLLECTION, get_client, recreate_collection, upload_points
from rag.rag_models import DocumentChunk

# How many chunks we send to Qdrant per request.
BATCH_SIZE = 100


def main() -> None:
    client = get_client()
    recreate_collection(client)
    print(f"Collection '{COLLECTION}' is empty and ready.")

    # Read all chunks from Postgres, in a stable order.
    with Session(engine) as session:
        chunks = session.scalars(select(DocumentChunk).order_by(DocumentChunk.id)).all()

        rows = [
            {
                "id": chunk.id,
                # Postgres hands the vector over as a numeric array; Qdrant wants a
                # plain Python list of numbers. float(x) converts each one.
                "vector": [float(x) for x in chunk.embedding],
                "file": chunk.file,
                "doc_type": chunk.doc_type,
                "product": chunk.product,
                "variant": chunk.variant,
                "language": chunk.language,
                "page": chunk.page,
                "chunk_number": chunk.chunk_number,
                "section": chunk.section,
                "text": chunk.text,
            }
            for chunk in chunks
        ]
    print(f"Read {len(rows)} chunks from Postgres.")

    # Upload in batches of 100.
    for start in range(0, len(rows), BATCH_SIZE):
        upload_points(client, rows[start:start + BATCH_SIZE])
        print(f"  uploaded {min(start + BATCH_SIZE, len(rows))} / {len(rows)}")

    # Check: does Qdrant now hold as many points as we sent?
    stored = client.count(collection_name=COLLECTION, exact=True).count
    print(f"\nDone. Qdrant holds {stored} points (expected {len(rows)}).")


if __name__ == "__main__":
    main()
