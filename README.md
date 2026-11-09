# Betingelser RAG chatbot

A learning project: a chatbot that answers questions about Danish business
insurance conditions using **RAG** (Retrieval-Augmented Generation). The source
material is 56 sets of public policy conditions from Tryg, covering 40 products.

I built it step by step to understand each part of a RAG system, so the code is
heavily commented and every design choice is measured where possible.

> Not affiliated with Tryg. The PDFs are not included in this repository.

## Status

| Part | State |
|---|---|
| PDF text extraction and cleaning | Done |
| Chunking along clause numbers | Done (v1) |
| Embeddings stored in Postgres with pgvector | Done |
| Semantic search API and chat page | Done |
| Evaluation script with test questions | Done (16 questions) |
| LangGraph agent: question routing, product detection, cited answers | Done |
| Index clean-up (contents lines, orphan headings) | Done |
| Reranker, vision-based table extraction | Planned |

The search was evaluated on its own before an LLM was put on top of it.

## How it works

```
INDEXING (run once)
  PDFs -> extract text -> clean -> cut into chunks -> embed each chunk -> Postgres (pgvector)

QUESTION (every time)
  question -> classify_question
                |-- internal  -> fixed reply ("not in my knowledge yet")
                |-- insurance -> find_product -> filtered vector search -> answer with citations
                                  ^     |
                         reply    |     v   in doubt which product: ask the person (once)
```

- **Extraction** (`pypdf`): all files contain real text, so no OCR is needed.
- **Cleaning** (`rag/clean_text.py`): removes the repeated page header (kept as a
  label), and repairs words split across lines (`føl -` / `gende` -> `følgende`).
- **Chunking** (`rag/chunker.py`): LangChain's `RecursiveCharacterTextSplitter`
  with a priority list of cut points: clause numbers (`2010.10`, `5.4.5 Karenstid`)
  first, then paragraphs, lines and words. About 1,000 characters per chunk with
  150 characters of overlap. Pieces under 200 characters (mostly orphan headings)
  are glued onto the next chunk. Result: 3,795 chunks.
- **Section headings**: every chunk stores the headings above it, for example
  `5400 Udvidet vandskade > 5421.30 Forsikringen dækker ikke`. Without them, a line
  from a list of exclusions reads like cover once it is cut loose from its heading.
- **Labels** (`rag/product_map.csv`): a hand-written list gives every PDF its
  product, version and language, because the page headers are too unreliable to
  filter on.
- **Embeddings** (`rag/embeddings.py`): `qwen/qwen3-embedding-8b` through
  OpenRouter, 1,024 dimensions. The provider sits behind a small wrapper, so the
  model can be swapped in one file. Each chunk is embedded with its product name
  in front, so the same clause in two products does not look identical.
- **Search** (`rag/search.py`): embeds the question, then asks a vector store for
  the nearest chunks. Two stores sit behind the same function, chosen by the
  `VECTOR_STORE` setting: **pgvector** in local Postgres (exact search, one SQL
  query ordered by cosine distance) and **Qdrant Cloud** (hosted, HNSW index).
- **Agent** (`agent/graph.py`): a LangGraph flow. A first step sorts the question
  onto a route, so each kind of question has its own flow and new abilities can be
  added as new branches. On the insurance route a language model picks the product
  from the list of 40, which replaces a product dropdown. The same step rewrites the
  question in the conditions' own words, and the search uses both. (The Danish word
  "råd" means both advice and rot; without this step a question about rot found
  clauses about advice.) If the question could mean
  several products, the graph pauses and asks (human-in-the-loop, using LangGraph's
  `interrupt` and an in-memory checkpointer). The search then looks only
  in that product, and the model answers from the five excerpts, citing them. A last
  step without any AI turns the citations into references (document, version, page)
  from the stored labels, so a reference cannot be invented. The
  model (`z-ai/glm-5.3-flashx-legal` through OpenRouter) sits behind a small wrapper.

## Evaluation

`rag/evaluate.py` runs the test questions in `rag/eval_questions.json` through the
real search and reports at which rank the correct chunk appears.

| Metric | Baseline | After index clean-up | With agent's product filter | Meaning |
|---|---|---|---|---|
| hit@1 | 8 / 16 | 11 / 16 | 12 / 16 | correct chunk is the top result |
| hit@5 | 13 / 16 | 13 / 16 | 14 / 16 | correct chunk is among the five shown |
| hit@30 | 15 / 16 | 15 / 16 | 15 / 16 | correct chunk is in the top 30 |

`agent/evaluate_agent.py` measures the agent column, on the same questions and saved
vectors. Of the 16 questions, one improved: the netbank question went from rank 8 to
rank 1 once the search was limited to that product. That is one question, so treat it
as a good sign and not as proof. The agent also took the right route on 24 of 24 test
questions and named the right product on 15 of 16; the other question names no product.

The index clean-up removed table-of-contents lines and glued orphan headings onto
the text below them (4,257 -> 3,987 chunks). No model or search code changed. It
moved correct chunks up (two misses went from rank 11 and 21 to rank 7 and 9, and
three more questions reached rank 1), but not enough to change hit@5.

Two questions still miss the top five with the filter on. One is found at rank 8,
which is the case a reranker is meant to fix. The other is a table question where the
same table appears in several sections of the document; it is not in the top 30.

**The measurement is noisy.** The embedding service returns slightly different
vectors for the same text on each call (cosine similarity about 0.9999 between
two calls). That is enough to reorder chunks with nearly tied scores: the same
question landed at rank 4 in some runs and rank 7 in others. The evaluation
therefore saves the question vectors and reuses them, so two runs are comparable.

**pgvector and Qdrant give identical results.** With the same question vectors,
both stores return the correct chunk at the same rank for all 16 questions.

Limits of this number: 16 questions is a small sample, and I wrote them after
reading the documents, so they are probably easier than real user questions.

**Tried and not kept: keyword search.** I added BM25 keyword search and merged it
with the vector search (reciprocal rank fusion). On the test questions it did not
beat the rewritten question on its own and it varied more between runs, so it was
removed again.

## Known weaknesses

- **Tables** are extracted as loose lines. Side-by-side tables lose their column
  structure, and a header can end up attached to the wrong table.
- **Wrapped table-of-contents entries** leave a fragment behind, because only
  their last line ends in a page number.
- **Wrong product label**: at least one PDF holds two documents glued together,
  and the second half gets the first document's label.
- **Prøveskilte** has no conditions document, only a product sheet, and product
  sheets are not indexed, so the chatbot cannot answer questions about it.
- **Chunks mix clauses**, so a chunk can match on the wrong part.
- **Several versions of the same product** exist. For van insurance the agent uses
  the newest set of conditions unless the question names another one.

## Roadmap

1. ~~Clean the index (drop table-of-contents lines, attach headings to their text).~~ Done.
2. ~~Let an LLM write the answer from the retrieved chunks, with citations.~~ Done.
3. Handle PDFs that hold more than one document.
4. Add a reranker: fetch 30 candidates, re-sort, keep 5.
5. Give the internal route real knowledge and tools.
6. v2 extraction: use a vision model to rewrite tables as Markdown.
7. Re-run the evaluation after every step and keep only what improves it.

## Run it yourself

Requirements: Python 3.12, Node.js, PostgreSQL with the
[pgvector](https://github.com/pgvector/pgvector) extension, and an
[OpenRouter](https://openrouter.ai) API key.

```bash
# 1. Settings
cp .env.example .env          # then fill in DATABASE_URL and OPENROUTER_API_KEY

# 2. Documents: put the PDFs in data/tryg/Betingelser/

# 3. Backend
cd backend
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m rag.chunker              # PDFs -> data/chunks_v1.jsonl
.venv/bin/python -m rag.create_chunk_table   # pgvector extension + table
.venv/bin/python -m rag.load_chunks          # embed and store (a few minutes)
.venv/bin/python -m rag.copy_to_qdrant       # optional: copy the vectors to Qdrant Cloud
.venv/bin/python -m rag.evaluate             # optional: run the test questions
.venv/bin/python -m agent.evaluate_agent     # optional: test routing and product filter
.venv/bin/uvicorn main:app --reload --port 8001

# 4. Frontend (in a second terminal)
cd frontend
npm install
npm run dev                                  # http://localhost:5174
```

## Project layout

```
backend/
  main.py                  FastAPI app with POST /agent/ask and GET /rag/search
  database.py              connection to Postgres
  agent/
    graph.py               the LangGraph flow: routing, product, search, answer
    llm.py                 language model wrapper (OpenRouter)
    evaluate_agent.py      tests routing and the product filter
  rag/
    peek_pdf.py            print the raw text of one PDF
    health_report.py       text-quality report across all PDFs
    clean_text.py          clean one page of extracted text
    chunker.py             cut all PDFs into chunks
    products.py            reads product_map.csv
    product_map.csv        product, version and language of every PDF
    embeddings.py          embedding wrapper (OpenRouter)
    rag_models.py          the document_chunks table
    create_chunk_table.py  create extension and table
    load_chunks.py         embed all chunks and store them in Postgres
    copy_to_qdrant.py      copy chunks and vectors from Postgres to Qdrant
    pgvector_store.py      nearest-vector lookup in Postgres
    qdrant_store.py        nearest-vector lookup in Qdrant Cloud
    search.py              semantic search (embeds the question, picks the store)
    evaluate.py            evaluation script
    eval_questions.json    test questions and expected sources
frontend/
  src/Chatbot.tsx          the chat page
```

## Stack

Python, FastAPI, SQLAlchemy, PostgreSQL + pgvector, Qdrant, LangGraph, LangChain text
splitters, OpenRouter (Qwen3 embeddings, GLM language model), TypeScript, React, Vite.
