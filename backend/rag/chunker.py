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
#   6. note which SECTION each chunk sits in (the headings above it)
#
# Run from the backend/ folder. "-m rag.chunker" means "run the file chunker.py
# inside the rag package" (the same style as `python -m scripts.import_cvr`).
# Run all 56 conditions files and print statistics (also saves data/chunks_v1.jsonl):
#   .venv/bin/python -m rag.chunker
# Print the chunks of ONE file (any part of the file name works):
#   .venv/bin/python -m rag.chunker cyberforsikring

import json                    # to save chunks as text lines (JSON Lines format)
import re                      # patterns for recognising headings
import statistics              # median etc. for the report
import sys                     # command-line arguments
from bisect import bisect_right  # fast "which bucket does this number fall into?" lookup
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from rag.clean_text import clean_page   # our cleaning function from step 3a
from rag.products import load_product_map   # our hand-written list of what each PDF is

# Read once at start-up: {file name: its product, variant and language}.
PRODUCT_MAP = load_product_map()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
# Only the conditions are indexed. The short product sheets (data/tryg/Produktinformation)
# stay on disk but are left out on purpose: the conditions are the binding text.
PDF_ROOT = PROJECT_ROOT / "data" / "tryg" / "Betingelser"
OUTPUT_FILE = PROJECT_ROOT / "data" / "chunks_v1.jsonl"   # data/ is git-ignored

# --- Chunk settings (the two most important "knobs" in a RAG system) -----------------
# Maximum characters per chunk. Too small = a chunk lacks context ("60 %" of what?).
# Too big = a chunk contains many topics, so it matches too vaguely and wastes space.
CHUNK_SIZE = 1000
# Characters repeated at the end of one chunk and the start of the next, so a sentence
# cut at the boundary still appears whole in at least one chunk.
CHUNK_OVERLAP = 150
# A piece shorter than this is too small to stand alone. It is usually a heading
# that got cut off from its text ("2. Hvem er omfattet"). Such a chunk matches
# questions very well but contains no answer, so we glue it onto the next chunk.
MIN_CHUNK_CHARS = 200

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


def join_without_repeat(first: str, second: str) -> str:
    """Glue two pieces of text together. Because of the overlap setting, `second`
    may already START with the END of `first`; then we must not repeat that part.
    We look for the longest piece that is both the end of `first` and the start of
    `second`, and include it only once."""
    for size in range(min(len(first), len(second)), 9, -1):   # try long overlaps first
        if first.endswith(second[:size]):
            return first + second[size:]
    return first + "\n" + second   # no shared part: just put a line break between them


def merge_small_pieces(pieces: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """Glue every too-small piece onto the piece AFTER it (a heading belongs with
    the text below it). Each piece is a pair: (start position in the document, text).
    The merged piece keeps the start position of the small one, so its page number
    is the page where the heading is."""

    merged: list[tuple[int, str]] = []
    waiting: tuple[int, str] | None = None   # a small piece waiting for the next one

    for start, text in pieces:
        if waiting is not None:
            start, text = waiting[0], join_without_repeat(waiting[1], text)
            waiting = None

        if len(text) < MIN_CHUNK_CHARS:
            waiting = (start, text)   # still too small: hold it for the next piece
        else:
            merged.append((start, text))

    # A small piece at the very END of the document has no "next" piece:
    # glue it onto the previous one instead (or keep it if it is the only piece).
    if waiting is not None:
        if merged:
            last_start, last_text = merged[-1]
            merged[-1] = (last_start, join_without_repeat(last_text, waiting[1]))
        else:
            merged.append(waiting)

    return merged


# --- Which section is a chunk in? ---------------------------------------------------
# A chunk is cut out of its surroundings. A line like
#     "5421.50 Følgeskader i form af svamp, skimmelsvamp og råd."
# means the OPPOSITE depending on whether the heading above it says "Forsikringen
# dækker" or "Forsikringen dækker ikke". So every chunk gets the headings above it:
#     "5400 Udvidet vandskade > 5420 Hvilke skader dækkes? > 5421.30 Forsikringen dækker ikke"

# A NUMBERED HEADING: a clause number and a short title on one line, e.g.
# "5421.30 Forsikringen dækker ikke" or "5.4.5 Karenstid". (A clause with body text
# has its number alone on the line, or runs on as a sentence.)
HEADING_PATTERN = re.compile(r"^(\d{1,4}(?:\.\d{1,2}){0,3})\.?[ \t]+([A-ZÆØÅ][^\n]{2,70})$")

# An UNNUMBERED LIST HEADING, common in the newer documents: a short line such as
# "Forsikringen dækker ikke", "Kaskoforsikringen dækker ikke" or "Forsikringen dækker:".
MARKER_PATTERN = re.compile(
    r"^(?:[A-ZÆØÅ][\w\-]*(?: [\w\-]+){0,3} (?:dækker|omfatter)(?: dog)?(?: ikke)?"
    r"(?:,? når| ansvar for| udgifter til| følgende)?"
    r"|The insurance (?:shall|does) not cover[\w ]{0,25}"
    r"|Undtagelser|Exclusions):?$"
)
# The GLOSSARY at the end of a document usually has a heading without a number. It
# must still end the section before it, or its definitions get that section's label.
GLOSSARY_PATTERN = re.compile(r"^(Ordforklaring(er)?|Ordliste|Definitioner|Definitions|Glossary)$")
# A line that continues a list: a bullet, "a)", a number, or a lower-case continuation.
LIST_LINE_PATTERN = re.compile(r"^(?:[•\-–]|[a-zæøå]\)|\d|[a-zæøå(])")


def heading_depth(number: str) -> int:
    """How deep in the document's outline is this clause number? 1 = a main section.
    Two numbering styles exist:
      four-digit style: 5400 -> 1, 5420 -> 2, 5421 -> 3, 5421.30 -> 4, 5421.31 -> 5
      dotted style:     5 -> 1,    5.4 -> 2,  5.4.5 -> 3"""
    parts = number.split(".")
    if len(parts[0]) < 3:
        return len(parts)
    if len(parts) > 1:
        return 4 if parts[1].endswith("0") else 5
    return 1 if parts[0].endswith("00") else 2 if parts[0].endswith("0") else 3


def is_above(upper: str, lower: str) -> bool:
    """Does heading number `upper` sit above `lower` in the outline?"""
    upper_parts, lower_parts = upper.split("."), lower.split(".")
    if len(upper_parts[0]) < 3 and len(lower_parts[0]) < 3:
        # Dotted style: "5.4" is above "5.4.5" because it is the start of it.
        return len(upper_parts) < len(lower_parts) and lower_parts[:len(upper_parts)] == upper_parts
    if heading_depth(upper) >= heading_depth(lower):
        return False
    # Four-digit style: 5400 is above 5420 and 5421.30, because they start with "54".
    # (Without this check, "9999 Ordforklaring" would land under the section before it.)
    start = upper_parts[0] if len(upper_parts) > 1 else upper_parts[0].rstrip("0")
    return lower_parts[0].startswith(start)


def find_headings(long_text: str) -> tuple[list[tuple[int, str, str]], list[tuple[int, int, str]]]:
    """Find all headings in a document. Returns two lists:
      numbered: (position, number, title)
      markers:  (position, end of the list it introduces, title)"""
    lines = long_text.split("\n")
    positions = []          # positions[i] = at which character line i starts
    position = 0
    for line in lines:
        positions.append(position)
        position += len(line) + 1

    numbered, markers = [], []
    for i, raw_line in enumerate(lines):
        line = raw_line.strip()
        next_line = next((x.strip() for x in lines[i + 1:i + 3] if x.strip()), "")

        if GLOSSARY_PATTERN.match(line):
            numbered.append((positions[i], "", line))   # "" = a heading without a number
            continue

        match = HEADING_PATTERN.match(line)
        if match:
            title = match.group(2).strip()
            # A title that continues in lower case on the next line is normally a
            # sentence, not a heading. The exception is a main number such as "160":
            # those never carry body text, so there it is just a long, wrapped title.
            number = match.group(1)
            is_main_number = "." not in number and len(number) >= 3
            runs_on = next_line[:1].islower() and not is_main_number
            is_sentence = title.endswith((".", ",", ";")) or runs_on
            is_contents_line = re.search(r"\d{1,3}$", title)   # "Karenstid 6" = a leftover contents line
            if not is_sentence and not is_contents_line:
                numbered.append((positions[i], number, title))
            continue

        if len(line) <= 60 and MARKER_PATTERN.match(line):
            # The marker covers the list right below it and nothing more: we stop at
            # the first line that does not look like part of a list. Stopping too
            # early only loses a label; stopping too late would put "dækker ikke"
            # on text that is not an exclusion, which is the worse mistake.
            end = i + 1
            while end < len(lines) and (not lines[end].strip() or LIST_LINE_PATTERN.match(lines[end].strip())):
                end += 1
            end_position = positions[end] if end < len(lines) else len(long_text)
            markers.append((positions[i], end_position, line.rstrip(":")))

    return numbered, markers


def section_at(position: int, numbered: list, markers: list) -> str:
    """The headings above `position`, outermost first, joined with " > ".
    A heading that sits exactly AT the position counts too: when a chunk starts with
    a new heading, the previous section is over."""
    stack: list[tuple[int, str, str]] = []   # the headings we are currently "inside"
    for heading in numbered:
        if heading[0] > position:
            break
        # A new heading closes every heading that is not above it in the outline.
        # (A glossary heading, with number "", closes everything and is closed by anything.)
        while stack and ("" in (stack[-1][1], heading[1]) or not is_above(stack[-1][1], heading[1])):
            stack.pop()
        stack.append(heading)

    parts = [f"{number} {title}".strip() for _, number, title in stack[-3:]]   # the 3 innermost
    last_numbered = stack[-1][0] if stack else -1
    for start, end, title in markers:
        # Use a marker only if we are inside its list and no numbered heading came after it.
        if last_numbered < start < position < end:
            parts.append(title)
    return " > ".join(parts)


def chunk_pdf(pdf_path: Path) -> list[dict]:
    """Turn one PDF into a list of chunks. Each chunk is a dictionary (a small
    record) with the text plus its label."""

    reader = PdfReader(pdf_path)

    long_text = ""         # all pages glued together
    page_starts = []       # page_starts[i] = at which character page i+1 begins

    # The label comes from our hand-written list. A PDF that is missing from the list
    # stops the run on purpose: an unlabelled chunk could never be found by a filter.
    if pdf_path.name not in PRODUCT_MAP:
        raise KeyError(f"{pdf_path.name} is missing from rag/product_map.csv")
    label = PRODUCT_MAP[pdf_path.name]

    for page in reader.pages:
        _header, body = clean_page(page.extract_text() or "")

        page_starts.append(len(long_text))   # this page begins where the text is right now
        long_text += body + "\n"             # add the page, plus a line break between pages

    # THE ACTUAL CUTTING. Returns "Document" objects with .page_content (the text)
    # and .metadata (a dictionary, here containing "start_index").
    documents = splitter.create_documents([long_text])

    # Glue too-small pieces (mostly orphan headings) onto the piece after them.
    pieces = merge_small_pieces(
        [(doc.metadata["start_index"], doc.page_content) for doc in documents]
    )

    numbered_headings, list_markers = find_headings(long_text)

    chunks = []
    for number, (start, text) in enumerate(pieces):
        # bisect_right(page_starts, start) = how many pages start at or before this
        # chunk's start. That count IS the page number (pages count from 1).
        page = bisect_right(page_starts, start)
        chunks.append({
            "file": pdf_path.name,
            "doc_type": pdf_path.parent.name,   # "Betingelser" or "Produktinformation"
            "product": label["product"],     # e.g. "Varebilforsikring"
            "variant": label["variant"],     # e.g. "2303" ("" if only one version)
            "language": label["language"],   # "da" or "en"
            "page": page,
            "chunk_number": number,
            # The headings above the chunk's first line ("" if none were found).
            "section": section_at(start, numbered_headings, list_markers),
            "text": text,
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
