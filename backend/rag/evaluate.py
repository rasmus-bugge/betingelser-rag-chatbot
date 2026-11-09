# evaluate.py
# EVALUATION: measure how good the SEARCH is, with a number instead of a feeling.
#
# The idea: we write a list of test questions (eval_questions.json). For each one we
# also write down which chunk is the RIGHT one to find:
#     "files":        the PDF(s) the answer is in
#     "must_contain": a short piece of text that only the right chunk contains
# Then we run every question through the real search and note at which RANK the
# right chunk shows up (1 = top result).
#
# The numbers we report:
#   hit@1   how often the right chunk is the very first result
#   hit@5   how often it is among the 5 we show (and later give to Claude)
#           -> THE number we want to push up
#   hit@30  how often it is among the first 30
#           -> shows what a "reranker" could rescue: it can only re-sort what the
#              first search found
#
# Run from the backend/ folder (costs one embedding call per question):
#   .venv/bin/python -m rag.evaluate
#
# RULE: when we change something (chunking, cleaning, model ...), we run this
# before and after, and keep the change only if the numbers get better.

import json
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from database import engine
from rag.rag_models import DocumentChunk
from rag.search import search_chunks

QUESTIONS_FILE = Path(__file__).resolve().parent / "eval_questions.json"

# How far down the result list we look for the right chunk.
SEARCH_DEPTH = 30
# How many chunks the chatbot actually shows.
TOP_K = 5


def normalise(text: str) -> str:
    """Collapse all line breaks and repeated spaces into single spaces.
    PDF text has line breaks in random places, so 'mere end\n3 timer' and
    'mere end 3 timer' must count as the same text when we compare."""
    return " ".join(text.split())


def is_right_chunk(file: str, text: str, item: dict) -> bool:
    """Is this chunk the one the test question is looking for?"""
    return file in item["files"] and normalise(item["must_contain"]) in normalise(text)


def main() -> None:
    with open(QUESTIONS_FILE, encoding="utf-8") as f:
        items = json.load(f)

    ranks: list[int | None] = []   # for each question: the rank found, or None = not in top 30

    with Session(engine) as session:
        print(f"{'rank':>5}  {'type':<11} question")
        print("-" * 100)

        for item in items:
            # SANITY CHECK: does the right chunk exist in the database at all?
            # If not, the test question itself is broken (e.g. a typo in must_contain),
            # and counting it as a "miss" would blame the search unfairly.
            stored = session.execute(
                select(DocumentChunk.file, DocumentChunk.text)
                .where(DocumentChunk.file.in_(item["files"]))
            ).all()
            candidates = sum(1 for file, text in stored if is_right_chunk(file, text, item))
            if candidates == 0:
                print(f"{'??':>5}  {item['type']:<11} {item['question']}")
                print("       ^ BROKEN TEST: no stored chunk matches must_contain. Skipped.")
                continue

            # Run the real search and look for the right chunk in the results.
            hits = search_chunks(session, item["question"], limit=SEARCH_DEPTH)
            rank = None
            for position, hit in enumerate(hits, start=1):
                if is_right_chunk(hit["file"], hit["text"], item):
                    rank = position
                    break
            ranks.append(rank)

            shown = str(rank) if rank else f">{SEARCH_DEPTH}"
            marker = "" if rank and rank <= TOP_K else "   <-- MISS"
            print(f"{shown:>5}  {item['type']:<11} {item['question']}{marker}")

    # --- Summary ---
    total = len(ranks)
    hit_1 = sum(1 for r in ranks if r == 1)
    hit_k = sum(1 for r in ranks if r and r <= TOP_K)
    hit_deep = sum(1 for r in ranks if r)
    print("\n" + "=" * 100)
    print(f"Questions evaluated: {total}")
    print(f"hit@1:  {hit_1}/{total}  ({hit_1 / total:.0%})   right chunk is the top result")
    print(f"hit@{TOP_K}:  {hit_k}/{total}  ({hit_k / total:.0%})   right chunk is among the {TOP_K} shown")
    print(f"hit@{SEARCH_DEPTH}: {hit_deep}/{total}  ({hit_deep / total:.0%})   right chunk is in the top {SEARCH_DEPTH} "
          f"(what a reranker could work with)")


if __name__ == "__main__":
    main()
