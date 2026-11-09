# rag_models.py
# The database table that holds our chunks AND their embeddings.
#
# One row = one chunk:
#
#   id | file                       | product         | page | text ...          | embedding
#   1  | betingelser-cyber....pdf   | Cyberforsikring | 6    | 5.4.5 Karenstid.. | [0.012, -0.03, ...1024 numbers]
#
# The "embedding" column has the special type VECTOR(1024), provided by the
# pgvector extension. It stores the 1024 numbers AND lets Postgres compare them
# (find the nearest vectors) with ordinary SQL.

from pgvector.sqlalchemy import Vector            # the column type for vectors
from sqlalchemy import String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from rag.embeddings import DIMENSIONS             # 1024: ONE place decides the size


# All table classes are built on Base, so SQLAlchemy can find them.
class Base(DeclarativeBase):
    pass


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id: Mapped[int] = mapped_column(primary_key=True)

    # --- The label: where does this chunk come from? (needed for source citations) ---
    file: Mapped[str] = mapped_column(String(255))       # "betingelser-cyberforsikring.pdf"
    doc_type: Mapped[str] = mapped_column(String(50))    # "Betingelser" or "Produktinformation"
    product: Mapped[str] = mapped_column(String(255))    # "Cyberforsikring"
    variant: Mapped[str] = mapped_column(String(100))    # "2303", or "" if only one version
    language: Mapped[str] = mapped_column(String(5))     # "da" or "en"
    page: Mapped[int]                                    # page where the chunk STARTS
    chunk_number: Mapped[int]                            # 0, 1, 2 ... within its file
    # The headings above the chunk, e.g. "6000 Svamp > 6013 Forsikringen dækker ikke".
    section: Mapped[str] = mapped_column(Text)

    # --- The content ---
    # Text (not String) = unlimited length. Our chunks are ~1000 characters.
    text: Mapped[str] = mapped_column(Text)

    # --- The embedding ---
    # Vector(1024) = a column holding exactly 1024 numbers.
    embedding: Mapped[list[float]] = mapped_column(Vector(DIMENSIONS))

    # A file can't have two chunks with the same number. This also protects us
    # from loading the same chunk twice by accident.
    __table_args__ = (UniqueConstraint("file", "chunk_number"),)
