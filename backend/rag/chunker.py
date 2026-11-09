# chunker.py
# STEP 3b of the RAG project (v1): cut every PDF into "chunks".
#
# WHAT IS A CHUNK? A small piece of a document (here ~1000 characters, about a
# paragraph or a clause). Later we search over chunks, not whole documents, and give
# only the best few to Claude. Think of it as cutting a 60-page binder into index
# cards: each card must make sense on its own.
#
# The pipeline for each PDF:
#   1. read every page with pypdf
#   2. clean each page (clean_text.py)
#   3. glue the pages into ONE long text (so a clause that runs over a page break
#      isn't cut in half), remembering where each page starts
#   4. cut the long text with LangChain's RecursiveCharacterTextSplitter
#   5. attach a label (file, product, page) to every chunk
#
# Run from the backend/ folder. "-m rag.chunker" means "run the file chunker.py
# inside the rag package" (the same style as `python -m scripts.import_cvr`).
# Run all 90 files and print statistics (also saves data/chunks_v1.jsonl):
#   .venv/bin/python -m rag.chunker
# Print the chunks of ONE file (any part of the file name works):
#   .venv/bin/python -m rag.chunker cyberforsikring

import json                    # to save chunks as text lines (JSON Lines format)
import statistics              # median etc. for the report
import sys                     # command-line arguments
from bisect import bisect_right  # fast "which bucket does this number fall into?" lookup
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from rag.clean_text import clean_page   # our cleaning function from step 3a

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PDF_ROOT = PROJECT_ROOT / "data" / "tryg"
OUTPUT_FILE = PROJECT_ROOT / "data" / "chunks_v1.jsonl"   # data/ is git-ignored

# --- Chunk settings (the two most important "knobs" in a RAG system) -----------------
# Maximum characters per chunk. Too small = a chunk lacks context ("60 %" of what?).
# Too big = a chunk contains many topics, so it matches too vaguely and wastes space.
CHUNK_SIZE = 1000
# Characters repeated at the end of one chunk and the start of the next, so a sentence
# cut at the boundary still appears whole in at least one chunk.
CHUNK_OVERLAP = 150

# --- Where the splitter is ALLOWED to cut, in priority order -------------------------
# The splitter tries the first separator. Pieces that are still bigger than CHUNK_SIZE
# are cut again with the NEXT separator, and so on. Small neighbouring pieces are
# merged back together up to CHUNK_SIZE.
# "(?=...)" is a "look ahead": cut at a line break that is FOLLOWED by this pattern,
# so the clause number stays at the START of its chunk (not the end of the previous one).
SEPARATORS = [
    r"\n(?=\d{3,4}\.\d{2}\b)",              # 1) before a clause number like "2010.10" or "7601.13"
    r"\n(?=\d{1,2}(?:\.\d{1,2})+\s+[A-ZÆØÅ])",  # 2) before a dotted heading like "5.4.5 Karenstid"
    r"\n(?=\d{1,2}\.\s+[A-ZÆØÅ])",          # 3) before a heading like "1. Hvad skal sikrede"
    r"\n(?=Stk\. \d)",                      # 4) before "Stk. 2" (paragraph style in old documents)
    "\n\n",                                 # 5) a blank line (paragraph)
    "\n",                                   # 6) any line break
    " ",                                    # 7) between words
    "",                                     # 8) last resort: between any two characters
]

# add_start_index=True makes the splitter record, for each chunk, at which character of
# the long text it starts. We use that to find which PAGE the chunk came from.
# is_separator_regex=True tells it our separators are patterns, not plain text.
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=SEPARATORS,
    is_separator_regex=True,
    add_start_index=True,
)


def product_from_header(header: str, fallback: str) -> str:
    """Pull the product name out of a header like
    '15/67 | Tryg | Bygningsforsikring 657-03 | Almindelige betingelser'.
    Splitting on '|' gives ['15/67', 'Tryg', 'Bygningsforsikring 657-03', ...];
    the product is the third part (index 2). If there's no header, use `fallback`."""
    parts = [p.strip() for p in header.split("|")]
    return parts[2] if len(parts) >= 3 and parts[2] else fallback


def chunk_pdf(pdf_path: Path) -> list[dict]:
    """Turn one PDF into a list of chunks. Each chunk is a dictionary (a small
    record) with the text plus its label."""

    reader = PdfReader(pdf_path)

    long_text = ""         # all pages glued together
    page_starts = []       # page_starts[i] = at which character page i+1 begins
    product = pdf_path.stem  # fallback label = the file name without ".pdf"

    for index, page in enumerate(reader.pages):
        header, body = clean_page(page.extract_text() or "")
        if header and index == 0:
            product = product_from_header(header, product)   # take the product name from page 1

        page_starts.append(len(long_text))   # this page begins where the text is right now
        long_text += body + "\n"             # add the page, plus a line break between pages

    # THE ACTUAL CUTTING. Returns "Document" objects with .page_content (the text)
    # and .metadata (a dictionary, here containing "start_index").
    documents = splitter.create_documents([long_text])

    chunks = []
    for number, doc in enumerate(documents):
        # bisect_right(page_starts, start) = how many pages start at or before this
        # chunk's start. That count IS the page number (pages count from 1).
        page = bisect_right(page_starts, doc.metadata["start_index"])
        chunks.append({
            "file": pdf_path.name,
            "doc_type": pdf_path.parent.name,   # "Betingelser" or "Produktinformation"
            "product": product,
            "page": page,
            "chunk_number": number,
            "text": doc.page_content,
        })
    return chunks


def main() -> None:
    pdf_files = sorted(PDF_ROOT.rglob("*.pdf"))

    # --- Mode 1: show the chunks of one file ---------------------------------------
    if len(sys.argv) > 1:
        wanted = sys.argv[1]
        matches = [p for p in pdf_files if wanted in p.name]
        if not matches:
            print(f"No file name contains '{wanted}'")
            return
        for chunk in chunk_pdf(matches[0]):
            print(f"\n######## chunk {chunk['chunk_number']} | page {chunk['page']} | "
                  f"{len(chunk['text'])} chars | {chunk['product']} ########")
            print(chunk["text"])
        return

    # --- Mode 2: chunk everything and print statistics ------------------------------
    all_chunks = []
    per_file = []   # (file name, number of chunks)
    for pdf_path in pdf_files:
        chunks = chunk_pdf(pdf_path)
        all_chunks.extend(chunks)
        per_file.append((pdf_path.name, len(chunks)))

    # Save one JSON object per line. ensure_ascii=False keeps æ, ø, å readable.
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    sizes = [len(c["text"]) for c in all_chunks]
    print(f"Files chunked:   {len(pdf_files)}")
    print(f"Total chunks:    {len(all_chunks)}")
    print(f"Chunk size (characters): min {min(sizes)}, median {int(statistics.median(sizes))}, "
          f"average {int(statistics.mean(sizes))}, max {max(sizes)}")
    print(f"Tiny chunks (< 150 chars):    {sum(1 for s in sizes if s < 150)}")
    print(f"Small chunks (< 300 chars):   {sum(1 for s in sizes if s < 300)}")
    print(f"Oversized (> {CHUNK_SIZE} chars):      {sum(1 for s in sizes if s > CHUNK_SIZE)}")
    print(f"\nSaved to {OUTPUT_FILE}")

    print("\nChunks per file (most -> fewest, top 8 and bottom 5):")
    per_file.sort(key=lambda item: item[1], reverse=True)
    for name, count in per_file[:8] + per_file[-5:]:
        print(f"  {count:>5}  {name}")


if __name__ == "__main__":
    main()
