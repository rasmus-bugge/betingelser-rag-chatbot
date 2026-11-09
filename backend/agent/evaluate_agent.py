# evaluate_agent.py
# Measures the two decisions the agent adds on top of the search:
#   1. ROUTING: does each question take the right route?
#   2. FILTERING: does searching within the chosen product find the right chunk more
#      often than rag/evaluate.py, which searches everything?
#
#   3. ANSWERS: does the written answer say "covered" / "not covered" correctly?
#
# Uses the same questions as rag/evaluate.py. The numbers move a little from run to
# run, because the model does not pick exactly the same search terms every time.
# Run from backend/:
#   .venv/bin/python -m agent.evaluate_agent

import json

from agent.graph import ask, classify_question, filters_for, find_product, needs_follow_up, search_text
from agent.llm import chat_json
from rag.evaluate import QUESTIONS_FILE, SEARCH_DEPTH, TOP_K, is_right_chunk
from rag.products import load_product_map
from rag.search import search_chunks

# Extra routing tests: (question, the route it should take). The insurance ones are
# deliberately close to the internal ones, e.g. both are about cancelling.
ROUTING_TESTS = [
    ("Hvordan opsiger jeg en police i Rumba?", "internal"),
    ("Hvor finder jeg kundens police i Rumba?", "internal"),
    ("Hvordan registrerer jeg en ny skade i systemet?", "internal"),
    ("Hvem skal godkende en kulancesag internt?", "internal"),
    ("How do I change the address on a policy in Rumba?", "internal"),
    ("Hvornår kan jeg opsige min varebilforsikring?", "insurance"),
    ("Does the dental insurance cover teeth cleaning?", "insurance"),
    ("Hvad er selvrisikoen på bygningsforsikringen?", "insurance"),
]

# Follow-up tests: (question, should the agent ask which product is meant?).
FOLLOW_UP_TESTS = [
    ("Dækker min bilforsikring, hvis føreren ikke har kørekort?", True),
    ("Hvad er selvrisikoen på min ansvarsforsikring?", True),
    ("Dækker forsikringen frostskader på min ejendom?", True),
    ("Hvad er karenstid i cyberforsikringen?", False),
    ("Hvad koster en pizza?", False),
    ("Hvad betyder ordet selvrisiko?", False),
]


# Answer tests: (question, the verdicts that count as correct). The verdict is what
# the answer SAYS about cover: "yes", "no", "depends" (only with an optional cover or
# on a condition) or "unknown" (the documents don't say). Checked against the PDFs by hand.
ANSWER_TESTS = [
    ("Er råd dækket på bygningsforsikringen?", {"no"}),
    ("Dækker bygningsforsikringen svamp og råd, der opstår efter at regnvand er trængt ind?", {"no", "depends"}),
    ("Kan jeg få dækket råd i træværket på landboforsikringen?", {"depends", "yes"}),
    ("Dækker varebilforsikringen, hvis chaufføren ikke har kørekort?", {"no", "depends"}),
    ("Dækker tandforsikringen tandrensning?", {"yes", "depends"}),
    ("Hvad er forsikringssummen for månelandinger i patientforsikringen?", {"unknown"}),
]

JUDGE_PROMPT = """You grade an answer from an insurance chatbot. Decide what the answer says about whether the thing asked about is covered.
- "yes": it is covered
- "no": it is not covered
- "depends": it is covered only if an optional cover was chosen, or only under a stated condition
- "unknown": the answer says the documents do not contain the information
Reply with JSON only: {"verdict": "yes" or "no" or "depends" or "unknown"}"""


def would_ask(found: dict) -> bool:
    """The same rule as needs_follow_up in graph.py, before any reply exists."""
    return needs_follow_up(found) == "ask_user"


def main() -> None:
    with open(QUESTIONS_FILE, encoding="utf-8") as f:
        items = json.load(f)
    product_map = load_product_map()

    print("ROUTING")
    route_right = 0
    for question, expected in ROUTING_TESTS:
        route = classify_question({"question": question})["route"]
        route_right += route == expected
        print(f"  {'ok ' if route == expected else 'BAD'} {route:<10} {question}")

    print("\nFOLLOW-UP")
    follow_ups_right = 0
    for question, expected in FOLLOW_UP_TESTS:
        found = find_product({"question": question})
        asked = would_ask(found)
        follow_ups_right += asked == expected
        print(f"  {'ok ' if asked == expected else 'BAD'} asks={str(asked):<5} {question}  {found['candidates']}")

    print(f"\n{'rank':>5}  {'product chosen':<34} question")
    print("-" * 100)
    ranks = []
    products_right = 0
    asked_count = 0
    for item in items:
        question = item["question"]
        decision = classify_question({"question": question})
        route_right += decision["route"] == "insurance"
        found = find_product({"question": question})
        products_right += found["product"] in {product_map[f]["product"] for f in item["files"]}
        asked_count += would_ask(found)

        filters = filters_for(found["product"], found["variant"], decision["language"])
        hits = search_chunks(
            search_text(question, found["search_terms"]), limit=SEARCH_DEPTH, filters=filters
        )
        rank = next(
            (n for n, hit in enumerate(hits, start=1) if is_right_chunk(hit["file"], hit["text"], item)),
            None,
        )
        ranks.append(rank)
        shown = str(rank) if rank else f">{SEARCH_DEPTH}"
        marker = "" if rank and rank <= TOP_K else "   <-- MISS"
        print(f"{shown:>5}  {str(found['product']):<34} {question[:60]}{marker}")

    print("\nANSWERS")
    answers_right = 0
    for question, accepted in ANSWER_TESTS:
        result = ask(question)
        answer = result.get("answer") or "(asked a follow-up question instead of answering)"
        verdict = chat_json(JUDGE_PROMPT, f"Question: {question}\n\nAnswer: {answer}").get("verdict")
        answers_right += verdict in accepted
        print(f"  {'ok ' if verdict in accepted else 'BAD'} says={str(verdict):<8} {question}")

    total = len(ranks)
    print("\n" + "=" * 100)
    print(f"Route correct:   {route_right}/{len(ROUTING_TESTS) + total}")
    print(f"Product correct: {products_right}/{total}")
    print(f"Follow-up tests: {follow_ups_right}/{len(FOLLOW_UP_TESTS)}")
    print(f"Answer tests:    {answers_right}/{len(ANSWER_TESTS)}")
    print(f"Would ask a follow-up on the {total} search questions: {asked_count}/{total}")
    print(f"hit@1:  {sum(r == 1 for r in ranks)}/{total}")
    print(f"hit@{TOP_K}:  {sum(1 for r in ranks if r and r <= TOP_K)}/{total}")
    print(f"hit@{SEARCH_DEPTH}: {sum(1 for r in ranks if r)}/{total}")


if __name__ == "__main__":
    main()
