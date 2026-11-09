# create_chunk_table.py
# Switches on the pgvector extension and creates the document_chunks table.
#
# Run from the backend/ folder (pgvector must be installed on the Postgres server,
# e.g. `brew install pgvector`):
#   .venv/bin/python -m rag.create_chunk_table
#
# Safe to run again: it only creates what is missing.

from sqlalchemy import text

from database import engine
from rag.rag_models import DocumentChunk

# 1) Switch on the pgvector EXTENSION in OUR database.
#    An extension is a plug-in. Installing the files on the machine (brew install)
#    is one thing; "CREATE EXTENSION" turns it on inside one specific database.
#    engine.begin() opens a connection and commits (saves) when the block ends.
with engine.begin() as connection:
    connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    version = connection.execute(
        text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
    ).scalar()
    print(f"pgvector is on (version {version}).")

# 2) Create the document_chunks table. checkfirst=True = skip if it already exists.
DocumentChunk.__table__.create(engine, checkfirst=True)
print("Table document_chunks is ready.")
