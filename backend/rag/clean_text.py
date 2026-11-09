# clean_text.py
# STEP 3a of the RAG project (v1): clean the raw text that pypdf extracts from one page.
# We fix three problems we saw in the health report:
#   1. A repeated header at the top of every page      -> remove it, but KEEP it as a label
#   2. Words split across lines, e.g. "føl -\ngende"   -> glue back to "følgende"
#   3. "T ryg" with a stray space                       -> "Tryg"
#
# Run it from the backend/ folder to see a before/after demo:
#   .venv/bin/python rag/clean_text.py

import re                      # regular expressions: a mini-language for matching text patterns
from pathlib import Path       # clean file/folder paths

from pypdf import PdfReader    # opens PDFs and extracts text

# --- Pattern 1: the page header --------------------------------------------------
# Example: "15/67 | Tryg | Bygningsforsikring 657-03 | Almindelige betingelser"
# Reading the pattern piece by piece:
#   ^           start of the text
#   \s*         optional spaces
#   \d+\s*/\s*\d+   page number "15/67" (digits, slash, digits; spaces allowed)
#   \s*\|       a "|" separator
#   \s*T\s?ryg  the word Tryg, possibly written "T ryg"
#   .*          the rest of the line (product name etc.)
# re.IGNORECASE = don't care about upper/lower case.
HEADER_PATTERN = re.compile(r"^\s*\d+\s*/\s*\d+\s*\|\s*T\s?ryg.*", re.IGNORECASE)

# --- Pattern 2: a word split across two lines ------------------------------------
# Example text:  "...følgende andel. Erstatningen kan dog føl -\ngende..."
# Pattern: letters, a space, a hyphen, a line break, then letters.
# Why a space BEFORE the hyphen? Real hyphens like "gas- og vand" have no space
# before them, so this pattern only catches the broken-word case.
# The parentheses "capture" the two word halves so we can glue them back together.
SPLIT_WORD_PATTERN = re.compile(r"(\w+) -\n(\w+)")

# --- Pattern 2b: the OTHER kind of split word ------------------------------------
# Older PDFs (e.g. the 2002 crime insurance) split words like this:
#   "formue- \ntab"   -> should be "formuetab"
# Here the hyphen touches the letters, and a space may follow it before the line break.
# Reading the pattern:
#   (\w)        one letter right before the hyphen (captured, so we can keep it)
#   -[ \t]*\n   a hyphen, optional spaces/tabs, then a line break
#   (?=[a-zæøå]) "look ahead": the next line must START with a lowercase letter, but
#               don't consume it. This protects real dashes before a capital/number.
# Known weakness: a REAL hyphenated word at a line end ("el-\ninstallation") would be
# glued too. That's rare, so we accept it in v1.
SPLIT_WORD_PATTERN_2 = re.compile(r"(\w)-[ \t]*\n(?=[a-zæøå])")

# --- Pattern 3: "T ryg" -> "Tryg" --------------------------------------------------
# \b = a word boundary, so we don't touch letters inside other words.
SPLIT_TRYG_PATTERN = re.compile(r"\bT ryg\b")

# --- Pattern 4: table-of-contents lines ----------------------------------------------
# A contents line is a heading followed by a page number: "5.4.5 Karenstid 6".
# Such lines contain every heading in the document but answer nothing, so they
# match almost any question and push real answers down the result list.
# Reading the pattern:
#   [A-Za-zæøåÆØÅ)\-]   a letter (or ")" or "-") ...
#   \s+                 ... then space(s) ...
#   \d{1,3}\s*$         ... then a 1-3 digit number at the very END of the line.
TOC_LINE_PATTERN = re.compile(r"[A-Za-zæøåÆØÅ)\-]\s+\d{1,3}\s*$")

# The labels that sit around a table of contents: "Indholdsfortegnelse", "Side  Side  Side".
TOC_LABEL_PATTERN = re.compile(
    r"^\s*(Indholdsfortegnelse|Indhold|Contents|Table of contents|(Side\s*)+|(Page\s*)+)\s*$",
    re.IGNORECASE,
)

# A page counts as a contents page when it has at least 9 contents-lines AND they
# make up at least a quarter of the page. (Measured on all 982 pages: 57 pages pass,
# all of them real tables of contents. A normal page has only a stray line or two
# that happens to end in a number, so it is left alone.)
TOC_MIN_LINES = 9
TOC_MIN_SHARE = 0.25


def remove_contents_lines(lines: list[str]) -> list[str]:
    """If the page holds a table of contents, remove the contents-lines and keep
    everything else. We remove LINES, not the whole page, because some contents
    pages also hold real text (an introduction, or even the first clauses).

    Known weakness: a contents entry that wraps over two lines leaves its first
    half behind ("1.4 Ændring af forsikringsbetingelser og"), because only the
    second half ends in a page number."""

    real_lines = [line for line in lines if line.strip()]          # ignore empty lines
    toc_lines = [line for line in real_lines if TOC_LINE_PATTERN.search(line)]

    is_contents_page = (
        len(toc_lines) >= TOC_MIN_LINES
        and len(toc_lines) / len(real_lines) >= TOC_MIN_SHARE
    )
    if not is_contents_page:
        return lines   # a normal page: change nothing

    return [
        line for line in lines
        if not TOC_LINE_PATTERN.search(line) and not TOC_LABEL_PATTERN.match(line)
    ]


def clean_page(raw_text: str) -> tuple[str, str]:
    """Clean one page of raw text.

    Returns TWO things: (header, body)
      header = the removed header line (e.g. "15/67 | Tryg | Bygningsforsikring ..."),
               or "" if the page had none. We keep it as a label for later.
      body   = the cleaned page text without the header.
    """

    # Split the page into lines so we can look at the first one.
    lines = raw_text.splitlines()

    header = ""
    # lines[0] is the first line; we only check there, because the header is always on top.
    if lines and HEADER_PATTERN.match(lines[0]):
        header = lines[0].strip()   # save it
        lines = lines[1:]           # drop it from the text (everything except the first line)

    # If this is a table-of-contents page, drop the contents-lines.
    lines = remove_contents_lines(lines)

    # Glue the lines back together into one string, with line breaks between them.
    body = "\n".join(lines)

    # Fix words split across lines: r"\1\2" means "first half + second half, no gap".
    body = SPLIT_WORD_PATTERN.sub(r"\1\2", body)

    # Fix the second kind: keep the letter (\1), drop the hyphen and the line break.
    body = SPLIT_WORD_PATTERN_2.sub(r"\1", body)

    # Fix "T ryg".
    body = SPLIT_TRYG_PATTERN.sub("Tryg", body)

    return header, body


# --- Demo: show before/after on the page we studied -----------------------------------
if __name__ == "__main__":
    pdf = Path(__file__).resolve().parent.parent.parent / "data/tryg/Betingelser/betingelser-bygningsforsikring.pdf"
    reader = PdfReader(pdf)

    raw = reader.pages[14].extract_text()      # page 15 (counting starts at 0)
    header, body = clean_page(raw)

    print("HEADER KEPT AS LABEL:", header)
    print("\n--- BEFORE (first 8 lines) ---")
    print("\n".join(raw.splitlines()[:8]))
    print("\n--- AFTER (first 8 lines) ---")
    print("\n".join(body.splitlines()[:8]))

    # Count how many split words we repaired across the WHOLE document.
    total_fixes = 0
    pages_without_header = 0
    for page in reader.pages:
        text = page.extract_text() or ""
        # findall returns a list of every match; len() counts them.
        total_fixes += len(SPLIT_WORD_PATTERN.findall(text))
        if not HEADER_PATTERN.match(text.splitlines()[0] if text else ""):
            pages_without_header += 1
    print(f"\nWhole file: {total_fixes} split words repaired, "
          f"{pages_without_header} of {len(reader.pages)} pages had no recognised header")
