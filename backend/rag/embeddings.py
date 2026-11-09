# embeddings.py
# STEP 4a of the RAG project: the EMBEDDING WRAPPER.
#
# WHAT IS AN EMBEDDING? A list of numbers (a "vector") that captures the MEANING of a
# text. Texts with similar meaning get vectors that point in similar directions.
#
# WHAT IS A WRAPPER? A small file that hides the details of one provider behind two
# simple functions. The rest of our code only ever calls:
#       embed_documents(["chunk 1", "chunk 2", ...])   -> list of vectors
#       embed_query("a user question")                 -> one vector
# If we later switch from Qwen to Voyage or OpenAI, we change THIS file only.
#
# We talk to OpenRouter, which forwards our request to a company that runs the model.

import os                      # read environment variables (where our API key lives)
import time                    # sleep() = pause, used when retrying a failed request
from pathlib import Path

import requests                # sends HTTP requests (a Python "browser" for APIs)
from dotenv import load_dotenv # reads the .env file into environment variables

# --- Settings you may want to change later ------------------------------------------
# The model name as OpenRouter spells it.
EMBEDDING_MODEL = "qwen/qwen3-embedding-8b"

# How many numbers each vector has. The model can produce up to 4096, but pgvector's
# search index only supports up to 2000, so we ask for a smaller vector (1024).
DIMENSIONS = 1024

# The web address (URL) of OpenRouter's embeddings "endpoint". An ENDPOINT is one
# specific address of an API that does one specific job.
API_URL = "https://openrouter.ai/api/v1/embeddings"

# How many texts we send in ONE request. Sending 4,000 chunks one by one would be slow;
# sending all at once could exceed size limits. 64 per request is a safe middle.
BATCH_SIZE = 64

# Qwen3 is trained so that QUESTIONS start with a short English instruction, while
# DOCUMENTS get no prefix (see the model card). Ignoring this makes search slightly worse.
QUERY_INSTRUCTION = (
    "Instruct: Given a question about Danish business insurance conditions, "
    "retrieve the passages that answer the question\nQuery: "
)

# Load the .env file in the project root (three folders up: rag/ -> backend/ -> root).
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")


def _embed(texts: list[str]) -> list[list[float]]:
    """Send a list of texts to OpenRouter, get one vector back per text.
    The leading underscore is a Python convention meaning "internal helper:
    other files shouldn't call this directly"."""

    # os.environ[...] raises an error if the key is missing: we want to fail loudly.
    api_key = os.environ["OPENROUTER_API_KEY"]

    all_vectors: list[list[float]] = []

    # range(0, len(texts), BATCH_SIZE) = 0, 64, 128, ...  Each step handles one batch.
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]   # a "slice": texts number start..start+63

        # RETRY: networks and busy servers hiccup. Codes 429 ("slow down") and 5xx
        # ("their problem") are usually temporary, so we wait a little and try again,
        # up to 4 times in total. Any other error (e.g. 401 = bad key) fails at once.
        for attempt in range(4):
            response = requests.post(
                API_URL,
                # The key travels in a header, never in the URL or in our code.
                headers={"Authorization": f"Bearer {api_key}"},
                # json=... turns this dictionary into JSON text and sends it as the body.
                json={
                    "model": EMBEDDING_MODEL,
                    "input": batch,
                    "dimensions": DIMENSIONS,
                    "encoding_format": "float",   # plain numbers (the alternative is compressed text)
                },
                timeout=120,   # give up after 120 seconds instead of hanging forever
            )
            if response.status_code in (429, 500, 502, 503, 504) and attempt < 3:
                time.sleep(2 ** attempt * 2)   # wait 2, 4, then 8 seconds
                continue                       # ...and try the same batch again
            break

        # If the server answered with an error code (401 = bad key, 429 = too many
        # requests, 500 = their problem...), stop and show what it said.
        if not response.ok:
            raise RuntimeError(f"OpenRouter error {response.status_code}: {response.text[:500]}")

        data = response.json()["data"]   # a list with one entry per input text

        # Each entry has an "index" (which input it belongs to). We sort by it so the
        # vectors come back in the same order as the texts we sent.
        data.sort(key=lambda item: item["index"])
        all_vectors.extend(item["embedding"] for item in data)

    return all_vectors


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed chunks of text that we want to be able to FIND later. No prefix."""
    return _embed(texts)


def embed_query(question: str) -> list[float]:
    """Embed a user's QUESTION. We add the instruction prefix first, then take the
    single vector out of the result list with [0]."""
    return _embed([QUERY_INSTRUCTION + question])[0]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """How similar are two vectors? Returns a number between -1 and 1:
       1 = same direction (same meaning), 0 = unrelated, -1 = opposite.
    The maths: dot product divided by the two vectors' lengths. Written out in plain
    Python so you can see there is no magic."""
    # zip pairs up the numbers: (a[0], b[0]), (a[1], b[1]), ...
    dot = sum(x * y for x, y in zip(a, b))
    length_a = sum(x * x for x in a) ** 0.5    # ** 0.5 = square root
    length_b = sum(y * y for y in b) ** 0.5
    return dot / (length_a * length_b)
