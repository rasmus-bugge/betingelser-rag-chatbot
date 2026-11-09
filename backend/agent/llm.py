# llm.py
# The LANGUAGE MODEL WRAPPER, built like embeddings.py: one small file hides the
# provider, and the rest of the code only ever calls
#       chat(system, user)        -> the model's answer as text
#       chat_json(system, user)   -> the model's answer as a dictionary
# To try another model (for example Claude), change LLM_MODEL below.

import json
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

LLM_MODEL = "z-ai/glm-5.3-flashx-legal"   # the model name as OpenRouter spells it
API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Load the .env file in the project root (three folders up: agent/ -> backend/ -> root).
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")


def chat(system: str, user: str, json_mode: bool = False) -> str:
    """Send one system instruction and one user message, return the answer as text."""
    body = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system},   # how the model should behave
            {"role": "user", "content": user},       # what it should work on
        ],
        "temperature": 0,   # as repeatable as possible: no creative variation
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}

    # RETRY, for the same reasons as in embeddings.py: 429 and 5xx are usually temporary.
    for attempt in range(4):
        response = requests.post(
            API_URL,
            headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
            json=body,
            timeout=120,
        )
        if response.status_code in (429, 500, 502, 503, 504) and attempt < 3:
            time.sleep(2 ** attempt * 2)   # wait 2, 4, then 8 seconds
            continue
        break

    if not response.ok:
        raise RuntimeError(f"OpenRouter error {response.status_code}: {response.text[:500]}")

    content = response.json()["choices"][0]["message"].get("content")
    if not content:
        raise RuntimeError("The language model returned an empty answer.")
    return content.strip()


def chat_json(system: str, user: str) -> dict:
    """Like chat(), but the answer is parsed into a dictionary. Used for the steps
    where the program must read the model's decision, not a person."""
    text = chat(system, user, json_mode=True)

    # Some models wrap JSON in ```json ... ``` despite being told not to, so we keep
    # only the part from the first "{" to the last "}".
    return json.loads(text[text.find("{"):text.rfind("}") + 1])
