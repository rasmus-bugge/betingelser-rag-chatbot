# peek_pdf.py
# STEP 1 of the RAG project: look at what the computer "sees" when it opens a PDF.
# We do NOT chunk, embed or store anything yet. We only read a PDF and print
# its text, so we can judge how clean the raw material is.
#
# Run it from the backend/ folder:
#   .venv/bin/python rag/peek_pdf.py
# Or pick another PDF / number of pages:
#   .venv/bin/python rag/peek_pdf.py ../data/tryg/Betingelser/betingelser-tandforsikring.pdf 3

import sys                     # gives us access to command-line arguments
from pathlib import Path       # a clean way to work with file/folder paths

from pypdf import PdfReader    # the library that opens PDFs and extracts text

# The project root is two folders above this file (backend/rag/peek_pdf.py).
# __file__ = this file; .parent = rag/; .parent.parent = backend/; one more = root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# The PDF we look at if you don't name another one on the command line.
DEFAULT_PDF = PROJECT_ROOT / "data" / "tryg" / "Betingelser" / "betingelser-cyberforsikring.pdf"


def peek(pdf_path: Path, pages_to_show: int) -> None:
    """Open one PDF and print the text of its first `pages_to_show` pages."""

    # PdfReader opens the file. Nothing is read in detail yet; it just
    # figures out how the file is organised (like opening a binder).
    reader = PdfReader(pdf_path)

    # len(reader.pages) = how many pages the PDF has in total.
    total_pages = len(reader.pages)
    print(f"File:  {pdf_path.name}")
    print(f"Pages: {total_pages}")

    # Only show as many pages as exist (a 2-page PDF can't show 3 pages).
    for page_number in range(min(pages_to_show, total_pages)):
        page = reader.pages[page_number]

        # extract_text() asks pypdf to pull the letters out of the page and
        # glue them into one string. Page numbers in code start at 0, so we
        # print page_number + 1 to match what a human sees.
        text = page.extract_text()

        print(f"\n{'=' * 20} PAGE {page_number + 1} ({len(text)} characters) {'=' * 20}")
        print(text)


# This block only runs when you start the file directly (not when imported).
if __name__ == "__main__":
    # sys.argv is the list of words typed after "python".
    # sys.argv[0] is the script name, [1] is the first extra word, and so on.
    pdf = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PDF
    pages = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    peek(pdf, pages)
