# health_report.py
# STEP 2 of the RAG project: extract text from ALL the PDFs and print a "health report".
# Goal: find problems NOW (empty files, scanned pages, table-heavy pages)
# instead of after we have built chunking and embeddings on top of bad text.
#
# Run it from the backend/ folder:
#   .venv/bin/python rag/health_report.py

import re                      # "regular expressions": a mini-language for matching text patterns
from pathlib import Path       # a clean way to work with file/folder paths

from pypdf import PdfReader    # opens PDFs and extracts text

# Project root = three levels up from this file (backend/rag/health_report.py).
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# The folder that holds both Betingelser/ and Produktinformation/.
PDF_ROOT = PROJECT_ROOT / "data" / "tryg"

# A page with fewer characters than this is suspicious. A normal text page has
# 1500-3000. Near-zero usually means the page is an image (a scan) with no real text.
MIN_CHARS_PER_PAGE = 200

# A "money line" is a line that contains an amount or percentage, e.g.
# "10.000 kr.", "5 %", "DKK 500.000", "2 mio.". The pattern is:
#   a number, optional space, then kr / % / DKK / mio  (case-insensitive).
# Tables of sums, deductibles (selvrisiko) and limits (dækningssum) are full of these.
MONEY_PATTERN = re.compile(r"\d[\d.,]*\s*(kr|%|dkk|mio)", re.IGNORECASE)

# If a page has at least this many money lines, we call it "table-like".
# This is a rough guess, not a precise table detector.
TABLE_LINES_THRESHOLD = 5


def analyse_pdf(pdf_path: Path) -> dict:
    """Read one PDF and return a small dictionary of facts about its text quality."""

    reader = PdfReader(pdf_path)
    pages = len(reader.pages)

    total_chars = 0        # characters across the whole document
    thin_pages = 0         # pages with almost no text (possible scans)
    table_like_pages = 0   # pages with many amount/percentage lines

    for page in reader.pages:
        # `or ""` protects us if extract_text() ever returns None for an odd page.
        text = page.extract_text() or ""
        total_chars += len(text)

        if len(text) < MIN_CHARS_PER_PAGE:
            thin_pages += 1

        # Count lines on this page that match the money pattern.
        money_lines = sum(1 for line in text.splitlines() if MONEY_PATTERN.search(line))
        if money_lines >= TABLE_LINES_THRESHOLD:
            table_like_pages += 1

    return {
        "name": pdf_path.name,
        "folder": pdf_path.parent.name,
        "pages": pages,
        "chars": total_chars,
        "chars_per_page": total_chars // pages if pages else 0,  # // = whole-number division
        "thin_pages": thin_pages,
        "table_like_pages": table_like_pages,
    }


def main() -> None:
    # rglob("*.pdf") = find every .pdf in PDF_ROOT and all sub-folders. sorted() = stable order.
    pdf_files = sorted(PDF_ROOT.rglob("*.pdf"))
    print(f"Found {len(pdf_files)} PDFs under {PDF_ROOT}\n")

    results = []     # successfully read files
    failures = []    # files that crashed, with the reason

    for pdf_path in pdf_files:
        # try/except: if ONE file is broken we still want the report for the other 89.
        try:
            results.append(analyse_pdf(pdf_path))
        except Exception as error:
            failures.append((pdf_path.name, repr(error)))

    # --- Table: one line per file ---
    # f"{x:<58}" = left-align in 58 characters; {x:>5} = right-align in 5. Just formatting.
    print(f"{'file':<58}{'pages':>6}{'chars/pg':>9}{'thin':>6}{'tables?':>8}")
    print("-" * 87)
    for r in results:
        print(f"{r['folder'][:1]}/{r['name']:<56}{r['pages']:>6}{r['chars_per_page']:>9}"
              f"{r['thin_pages']:>6}{r['table_like_pages']:>8}")

    # --- Summary ---
    total_pages = sum(r["pages"] for r in results)
    total_chars = sum(r["chars"] for r in results)
    print("\n" + "=" * 87)
    print(f"Files read OK:      {len(results)} / {len(pdf_files)}")
    print(f"Total pages:        {total_pages}")
    print(f"Total characters:   {total_chars:,}")
    # Rule of thumb: ~4 characters per token. Useful later for cost estimates.
    print(f"Rough token count:  {total_chars // 4:,}")
    print(f"Files with thin pages (possible scans): "
          f"{sum(1 for r in results if r['thin_pages'] > 0)}")
    print(f"Files with table-like pages:            "
          f"{sum(1 for r in results if r['table_like_pages'] > 0)}")

    if failures:
        print("\nFAILED TO READ:")
        for name, reason in failures:
            print(f"  {name}: {reason}")


if __name__ == "__main__":
    main()
