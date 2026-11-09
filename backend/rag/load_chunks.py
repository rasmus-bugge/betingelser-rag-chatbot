# load_chunks.py
# STEP 4d: embed every chunk and store it in Postgres.
#
# The pipeline so far:   PDFs -> clean -> chunks (data/chunks_v1.jsonl)
# This script does:      chunks -> embeddings (one API call per 64 chunks) -> database
#
# This is the "indexing" half of RAG. It runs ONCE (or whenever the chunks change),
# not on every question. Cost: well under 1 US cent on OpenRouter.
#
# Run from the backend/ folder, after `python -m rag.chunker` and
# `python -m rag.create_chunk_table`:
#   .venv/bin/python -m rag.load_chunks
#
# Safe to run again: it first EMPTIES the table, then reloads everything, so you
# never end up with duplicates (useful when we change chunking in v2).

import json
import time
from pathlib import Path

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from database import engine
from rag.embeddings import BATCH_SIZE, embed_documents
from rag.rag_models import DocumentChunk

CHUNKS_FILE = Path(__file__).resolve().parent.parent.parent / "data" / "chunks_v1.jsonl"


def text_to_embed(chunk: dict) -> str:
    """The text we turn into a vector. We put the product name IN FRONT of the chunk,
    so a chunk like '5.4.5 Karenstid ...' becomes 'Cyberforsikring (Betingelser): 5.4.5
    Karenstid ...'. Now the vector knows WHICH insurance it is about; otherwise the
    'Karenstid' of cyber and of loss-of-profit insurance would look identical.
    (We still store the ORIGINAL text in the database, without the prefix.)"""
    return f"{chunk['product']} ({chunk['doc_type']}): {chunk['text']}"


def main() -> None:
    # Read the JSON Lines file: one chunk per line.
    with open(CHUNKS_FILE, encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]
    print(f"Read {len(chunks)} chunks from {CHUNKS_FILE.name}")

    started = time.time()

    # A Session = a conversation with the database. Changes are only saved when we
    # call session.commit().
    with Session(engine) as session:
        # Start clean: delete all existing rows in the table.
        session.execute(delete(DocumentChunk))
        session.commit()

        # Work through the chunks 64 at a time.
        for start in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[start:start + BATCH_SIZE]

            # THE AI STEP: 64 texts in, 64 vectors out (one API request).
            vectors = embed_documents([text_to_embed(c) for c in batch])

            # zip pairs each chunk with ITS vector. We build one table row per pair.
            session.add_all(
                DocumentChunk(
                    file=chunk["file"],
                    doc_type=chunk["doc_type"],
                    product=chunk["product"],
                    page=chunk["page"],
                    chunk_number=chunk["chunk_number"],
                    text=chunk["text"],
                    embedding=vector,
                )
                for chunk, vector in zip(batch, vectors)
            )
            # Save after EVERY batch, so a crash half-way doesn't throw away the work done.
            session.commit()

            done = min(start + BATCH_SIZE, len(chunks))
            print(f"  embedded and saved {done} / {len(chunks)}  ({time.time() - started:.0f}s)")

        # Check: does the database now hold as many rows as we sent?
        stored = session.scalar(select(func.count()).select_from(DocumentChunk))
        print(f"\nDone. document_chunks now holds {stored} rows (expected {len(chunks)}).")


if __name__ == "__main__":
    main()
