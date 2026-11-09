# graph.py
# THE AGENT, built with LangGraph.
#
# LangGraph describes a process as boxes (NODES) and arrows (EDGES). Every node reads
# from and writes to one shared notepad (the STATE). Here is the whole flow:
#
#   question
#      |
#      v
#   classify_question        What kind of question is this?
#      |
#      +-- "internal" --> internal_reply --> END       (fixed reply, no search)
#      |
#      +-- "insurance" --> find_product --> search --> answer --> add_references --> END
#                            |      ^
#              in doubt:     v      |  the person's reply
#                            ask_user
#
# ask_user is HUMAN-IN-THE-LOOP: the graph stops, shows the person a question, and
# continues from the same place when the reply arrives. To make that possible,
# LangGraph saves the notepad after every step (a CHECKPOINT) under a thread id.
#
# The fork after classify_question is QUESTION ROUTING: one flow per kind of question.
# A new ability later (for example a tool that looks something up) is one more node
# and one more branch. The existing flows do not change.
#
# Try it in the terminal (from backend/):
#   .venv/bin/python -m agent.graph "Hvad er karenstid i cyberforsikringen?"
#   .venv/bin/python -m agent.graph "Hvordan opsiger jeg en police i Rumba?"

import json
import re
import sys
import uuid
from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from agent.llm import chat, chat_json
from rag.products import product_catalog
from rag.search import search_chunks

CATALOG = product_catalog()   # {product: {"variants": [...], "languages": [...]}}

# When a product has several versions and the user names none, we search this one.
# Varebil 2303 is the newest of the three.
DEFAULT_VARIANT = {"Varebilforsikring": "2303"}

CHUNKS_TO_READ = 5   # how many excerpts the answer step gets


# THE NOTEPAD: everything the nodes can read and write.
# total=False = a field may still be missing (it is filled in along the way).
class AgentState(TypedDict, total=False):
    question: str
    route: str              # "insurance" or "internal"
    language: str           # the language of the question: "da" or "en"
    product: str | None     # None = the question doesn't name a product we know
    variant: str | None     # None = no specific version
    candidates: list[str]   # products the question might mean, when we are in doubt
    search_terms: list[str] # the question in the conditions' own words, for the search
    reply: str              # the person's answer to our follow-up question
    chunks: list[dict]      # the excerpts found by the search
    answer: str             # the text shown to the user


# --- Node 1: what kind of question is this? ------------------------------------------

CLASSIFY_PROMPT = """You sort questions sent to an assistant for employees at the Danish insurer Tryg.

Choose one route:
- "internal": how to do something in Tryg's own systems or daily work, for example in the system Rumba: where to click, how to register or process something, internal procedures, who to contact.
- "insurance": what an insurance covers or what the insurance conditions say, including excess, waiting periods, compensation and notice periods. Use this when in doubt.

Example: "Hvordan opsiger jeg en police i Rumba?" is internal, because it asks how to do it in the system.
Example: "Med hvilket varsel kan cyberforsikringen opsiges?" is insurance, because it asks what the conditions say.

Also give the language of the question: "da" or "en".
Reply with JSON only: {"route": "insurance" or "internal", "language": "da" or "en"}"""


def classify_question(state: AgentState) -> AgentState:
    decision = chat_json(CLASSIFY_PROMPT, state["question"])

    # Never trust the model's output blindly. Anything unexpected falls back to the
    # insurance route, where the worst outcome is "the documents don't cover this".
    route = "internal" if decision.get("route") == "internal" else "insurance"
    language = "en" if decision.get("language") == "en" else "da"
    return {"route": route, "language": language}


def pick_route(state: AgentState) -> str:
    """Not a node but the FORK: LangGraph calls it to decide which arrow to follow."""
    return state["route"]


# --- The internal route --------------------------------------------------------------

INTERNAL_REPLY = {
    "da": (
        "Jeg kan se, at det her er et spørgsmål om arbejdsgange. Den viden har jeg "
        "desværre ikke fået endnu, så spørg min lillesøster Elvira om det."
    ),
    "en": (
        "I can see this is a question about workflows. Unfortunately I haven't added "
        "that to my knowledge yet, so please ask my little sister Elvira about it."
    ),
}


def internal_reply(state: AgentState) -> AgentState:
    # A fixed text: no language model and no search, so it costs nothing.
    return {"answer": INTERNAL_REPLY[state["language"]], "chunks": []}


# --- The insurance route, step 1: which product is it about? -------------------------
# This step is what replaces a product dropdown on the page.

def _catalog_as_text() -> str:
    """The product list as the model will read it, one product per line."""
    lines = []
    for product, info in CATALOG.items():
        versions = f"  (versions: {', '.join(info['variants'])})" if info["variants"] else ""
        lines.append(f"- {product}{versions}")
    return "\n".join(lines)


FIND_PRODUCT_PROMPT = f"""You work out which Tryg insurance product a question is about.

Products:
{_catalog_as_text()}

Rules:
- Pick a product only if the question names it or clearly describes it. Otherwise use null.
- Give a version only if the question explicitly mentions that version. Otherwise use null.
- If product is null and the answer depends on which insurance is meant, list the 2 to 4 most likely products in "candidates". Example: "min bilforsikring" could be several vehicle products.
- Use an empty "candidates" list if the question is general, or is not about insurance at all.
- Copy product names exactly as written in the list.
- In "search_terms", give 4 to 8 words or short phrases that the insurance conditions themselves would use for what is asked, in the language of the question. An everyday word often has a specific meaning in insurance: decide which meaning fits this product, and give the terms for that meaning, including compound and inflected forms. A question about cover usually asks about a kind of damage, an event or an insured object, so if a word could mean a kind of damage as well as something else, use the damage meaning. Example: for "vandskade fra et sprunget rør" give ["rørskade", "udstrømning af vand", "skjulte rør", "vandskade", "utæthed"].
Reply with JSON only: {{"product": "<name>" or null, "variant": "<version>" or null, "candidates": ["<name>", ...], "search_terms": ["<term>", ...]}}"""


def _question_with_reply(state: AgentState) -> str:
    """The question, plus the person's follow-up answer if there is one."""
    if state.get("reply"):
        return f"{state['question']} ({state['reply']})"
    return state["question"]


def find_product(state: AgentState) -> AgentState:
    decision = chat_json(FIND_PRODUCT_PROMPT, _question_with_reply(state))

    # The search terms are only a help for the search, so a missing or odd list is
    # simply left out.
    raw_terms = decision.get("search_terms")
    search_terms = [t for t in raw_terms if isinstance(t, str)][:8] if isinstance(raw_terms, list) else []

    # Check the model's choices against our own list, so a made-up name can't become
    # a filter that matches nothing, or a button that leads nowhere.
    product = decision.get("product")
    if not isinstance(product, str) or product not in CATALOG:
        raw = decision.get("candidates")
        candidates = [c for c in raw if isinstance(c, str) and c in CATALOG] if isinstance(raw, list) else []
        return {"product": None, "variant": None, "candidates": candidates[:4], "search_terms": search_terms}

    variant = decision.get("variant")
    if variant not in CATALOG[product]["variants"]:
        variant = None
    return {"product": product, "variant": variant, "candidates": [], "search_terms": search_terms}


def needs_follow_up(state: AgentState) -> str:
    """The FORK after find_product. We ask at most once: if the reply didn't settle
    it, we search as well as we can rather than keep questioning the person."""
    if len(state["candidates"]) >= 2 and not state.get("reply"):
        return "ask_user"
    return "search"


FOLLOW_UP_QUESTION = {
    "da": "Jeg er i tvivl om, hvilken forsikring du mener. Er det en af disse?",
    "en": "I'm not sure which insurance you mean. Is it one of these?",
}


def ask_user(state: AgentState) -> AgentState:
    # interrupt() STOPS the graph here and hands this dictionary to the caller. When
    # the graph is resumed, interrupt() returns the person's reply.
    reply = interrupt({
        "question": FOLLOW_UP_QUESTION[state["language"]],
        "options": state["candidates"],
    })
    return {"reply": reply}


# --- The insurance route, step 2: search --------------------------------------------

def filters_for(product: str | None, variant: str | None, language: str) -> dict[str, str]:
    """Turn the decisions so far into search filters."""
    filters: dict[str, str] = {}

    if product is None:
        # No product known. Danish questions search the Danish chunks. For English
        # questions we search everything, since only 6 documents are in English.
        if language == "da":
            filters["language"] = "da"
        return filters

    filters["product"] = product

    chosen_variant = variant or DEFAULT_VARIANT.get(product)
    if chosen_variant:
        filters["variant"] = chosen_variant

    # Filter on language only if this product exists in that language. Otherwise a
    # Danish question about an English-only product would find nothing.
    if language in CATALOG[product]["languages"]:
        filters["language"] = language
    return filters


def search_text(question: str, search_terms: list[str]) -> str:
    """The text we search with: the person's own words, plus the same question in
    the conditions' words. Example: "Er råd dækket? (råd, rådskade, svamp)".
    Without the terms, a question about rot ("råd") found chunks about advice
    ("rådgivning"), because the search could not tell which meaning was meant."""
    if not search_terms:
        return question
    return f"{question} ({', '.join(search_terms)})"


def search(state: AgentState) -> AgentState:
    filters = filters_for(state["product"], state["variant"], state["language"])
    text = search_text(_question_with_reply(state), state["search_terms"])
    return {"chunks": search_chunks(text, limit=CHUNKS_TO_READ, filters=filters)}


# --- The insurance route, step 3: write the answer ----------------------------------

ANSWER_PROMPT = """You answer questions about Tryg's insurance conditions, using ONLY the numbered excerpts you are given.

Rules:
- Answer in the same language as the question, briefly and directly, in plain text without Markdown formatting.
- After each fact, cite the excerpt it came from, like [1] or [2]. Do not add a list of sources at the end; that is done automatically.
- Each excerpt begins with a "Section" line: the headings it sits under in the document. The headings decide what the text means. A line under "Forsikringen dækker ikke" is an exclusion, even when the line itself does not say so.
- If something is only covered when an optional cover (tilvalg) has been chosen, or depends on what the policy says, say exactly that.
- Never use outside knowledge. If the excerpts do not answer the question, say that the conditions you were given do not cover it, and do not guess.
- If the product has several versions, say which version your answer comes from."""


def answer(state: AgentState) -> AgentState:
    excerpts = []
    for number, chunk in enumerate(state["chunks"], start=1):
        version = f", version {chunk['variant']}" if chunk["variant"] else ""
        excerpts.append(
            f"[{number}] {chunk['product']}{version} ({chunk['file']}, page {chunk['page']})\n"
            f"Section: {chunk['section'] or 'unknown'}\n"
            f"{chunk['text']}"
        )
    user_message = f"Question: {_question_with_reply(state)}\n\nExcerpts:\n\n" + "\n\n".join(excerpts)
    return {"answer": chat(ANSWER_PROMPT, user_message)}


# --- The insurance route, step 4: the references -------------------------------------
# Plain code, no language model: the references are built from the labels stored with
# each chunk, so the model cannot invent a document or a page number.

# Matches a citation in the answer: [2], or [1, 3].
CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def add_references(state: AgentState) -> AgentState:
    """Keep only the excerpts the answer really cited, and number them 1, 2, 3 ... in
    the order they appear. A citation of an excerpt that doesn't exist is removed."""
    found = state["chunks"]
    cited: list[int] = []   # the original numbers, in order of first appearance

    def renumber(match: re.Match) -> str:
        new_numbers = []
        for text in match.group(1).split(","):
            original = int(text)
            if not 1 <= original <= len(found):
                continue   # the model cited an excerpt it was never given
            if original not in cited:
                cited.append(original)
            new_numbers.append(cited.index(original) + 1)
        return "".join(f"[{number}]" for number in new_numbers)

    return {
        "answer": CITATION.sub(renumber, state["answer"]).replace("**", ""),   # the page shows plain text
        "chunks": [found[number - 1] for number in cited],
    }


# --- Drawing the graph: this is where the boxes and arrows are connected -------------

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("classify_question", classify_question)
    graph.add_node("internal_reply", internal_reply)
    graph.add_node("find_product", find_product)
    graph.add_node("ask_user", ask_user)
    graph.add_node("search", search)
    graph.add_node("answer", answer)
    graph.add_node("add_references", add_references)

    graph.add_edge(START, "classify_question")

    # The fork: pick_route returns a word, and this dictionary says where it leads.
    graph.add_conditional_edges(
        "classify_question",
        pick_route,
        {"insurance": "find_product", "internal": "internal_reply"},
    )

    graph.add_edge("internal_reply", END)
    graph.add_conditional_edges(
        "find_product", needs_follow_up, {"ask_user": "ask_user", "search": "search"}
    )
    graph.add_edge("ask_user", "find_product")   # with the reply, try again
    graph.add_edge("search", "answer")
    graph.add_edge("answer", "add_references")
    graph.add_edge("add_references", END)

    return graph.compile(checkpointer=CHECKPOINTS)


# Where paused conversations wait. In memory = lost when the server restarts, and
# only usable with one server process. A deployed version needs a database instead.
CHECKPOINTS = InMemorySaver()
AGENT = build_graph()


class ConversationNotFound(Exception):
    """A reply arrived for a conversation that isn't waiting for one."""


def _run(graph_input, thread_id: str) -> dict:
    """Run the graph until it finishes or stops to ask something."""
    config = {"configurable": {"thread_id": thread_id}}
    result = AGENT.invoke(graph_input, config)

    if "__interrupt__" in result:
        # Paused: the caller shows `follow_up` and later sends the reply with this id.
        return {**result, "follow_up": result["__interrupt__"][0].value, "thread_id": thread_id}

    CHECKPOINTS.delete_thread(thread_id)   # finished, so nothing needs to be kept
    return {**result, "follow_up": None, "thread_id": None}


def ask(question: str) -> dict:
    """Start a new conversation with one question."""
    return _run({"question": question}, str(uuid.uuid4()))


def continue_with_reply(thread_id: str, reply: str) -> dict:
    """Resume a paused conversation with the person's reply."""
    config = {"configurable": {"thread_id": thread_id}}
    if not AGENT.get_state(config).next:   # .next = the steps still waiting to run
        raise ConversationNotFound(thread_id)
    return _run(Command(resume=reply), thread_id)


if __name__ == "__main__":
    result = ask(" ".join(sys.argv[1:]) or "Hvad er karenstid i cyberforsikringen?")
    if result["follow_up"]:
        print(f"{result['follow_up']['question']}  {result['follow_up']['options']}")
        result = continue_with_reply(result["thread_id"], input("> "))
    decisions = {key: result.get(key) for key in ("route", "language", "product", "variant")}
    print(json.dumps(decisions, ensure_ascii=False))
    print("\n" + result["answer"])
    for number, chunk in enumerate(result["chunks"], start=1):
        version = f", version {chunk['variant']}" if chunk["variant"] else ""
        print(f"[{number}] {chunk['product']}{version}, page {chunk['page']}, {chunk['section']} ({chunk['file']})")
