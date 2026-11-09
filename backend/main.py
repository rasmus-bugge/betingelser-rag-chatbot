# main.py
# The backend of the chatbot: a small web API with two endpoints.
#   POST /agent/ask         the chatbot: routes the question and writes the answer
#   POST /agent/reply       the person's answer to a follow-up question from the chatbot
#   GET /rag/search?q=...   the search alone: the chunks closest to a question
#
# Start it from the backend/ folder (port 8001, so it doesn't clash with other projects):
#   .venv/bin/uvicorn main:app --reload --port 8001
# Automatic documentation: http://localhost:8001/docs

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from requests import RequestException

from agent.graph import ConversationNotFound, ask, continue_with_reply
from rag.search import search_chunks

# The application object. Every endpoint is attached to it.
app = FastAPI(title="Betingelser RAG chatbot")

# CORS: a browser refuses to let a page from one address (the React app on
# localhost:5174) read answers from another (this API on localhost:8001) unless
# the API says it is allowed. This block gives that permission.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5174"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],   # the POST request sends its question as JSON
)


# The shape of one search result: a chunk of a PDF, where it came from, and how
# close it is to the question.
class ChunkHit(BaseModel):
    file: str  # "betingelser-cyberforsikring.pdf"
    doc_type: str  # "Betingelser" or "Produktinformation"
    product: str  # "Cyberforsikring"
    variant: str  # "2303", or "" if the product has only one version
    language: str  # "da" or "en"
    page: int  # the page where the chunk starts
    chunk_number: int
    section: str  # the headings above the chunk, "" if none were found
    score: float  # 0..1, bigger = closer in meaning to the question
    text: str


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=500)


class ReplyRequest(BaseModel):
    thread_id: str
    reply: str = Field(min_length=1, max_length=500)


# A question from the chatbot back to the person.
class FollowUp(BaseModel):
    question: str
    options: list[str]  # the products to choose between


# What the chatbot sends back: EITHER an answer OR a follow-up question.
class AgentReply(BaseModel):
    route: str  # "insurance" or "internal"
    product: str | None  # None = no product recognised in the question
    answer: str | None  # None while the chatbot is waiting for a reply
    sources: list[ChunkHit]  # the excerpts the answer cites; sources[0] is its [1]
    follow_up: FollowUp | None
    thread_id: str | None  # send this back with the reply to a follow-up


def run_agent(step, *arguments) -> dict:
    """Run one agent call and shape its result for the page."""
    try:
        result = step(*arguments)
    except ConversationNotFound:
        # For example: the server restarted while the conversation was waiting.
        raise HTTPException(status_code=404, detail="Samtalen findes ikke længere.")
    except (RuntimeError, ValueError, RequestException) as error:
        # The language model or embedding service failed, or sent something unreadable.
        raise HTTPException(status_code=502, detail=f"AI service failed: {error}")
    return {
        "route": result["route"],
        "product": result.get("product"),
        "answer": result.get("answer"),
        "sources": result.get("chunks", []),
        "follow_up": result["follow_up"],
        "thread_id": result["thread_id"],
    }


# POST /agent/ask with {"question": "..."}
# Runs the question through the LangGraph flow in agent/graph.py.
@app.post("/agent/ask", response_model=AgentReply)
def agent_ask(request: AskRequest):
    return run_agent(ask, request.question)


# POST /agent/reply with {"thread_id": "...", "reply": "..."}
# Continues a conversation that stopped to ask the person something.
@app.post("/agent/reply", response_model=AgentReply)
def agent_reply(request: ReplyRequest):
    return run_agent(continue_with_reply, request.thread_id, request.reply)


# GET /rag/search?q=Hvad er karenstid?&limit=5
# Turns the question into a vector and returns the `limit` closest chunks.
# Which database is searched (Postgres or Qdrant) is decided in rag/search.py.
@app.get("/rag/search", response_model=list[ChunkHit])
def rag_search(
    q: str = Query(min_length=2, max_length=500),  # the question; 2-500 characters
    limit: int = Query(5, ge=1, le=20),  # how many chunks (default 5, max 20)
):
    try:
        return search_chunks(q, limit)
    except RuntimeError as error:
        # search_chunks raises RuntimeError when the embedding service (OpenRouter)
        # fails. 502 = "Bad Gateway": a service WE depend on failed, not our code.
        raise HTTPException(status_code=502, detail=f"Embedding service failed: {error}")
