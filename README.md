# Betingelser RAG chatbot

A learning project: a chatbot that answers questions about Danish business
insurance conditions using **RAG** (Retrieval-Augmented Generation). The source
material is 90 public PDFs from Tryg (56 sets of policy conditions and 34 product
sheets, 982 pages in total).

I built it step by step to understand each part of a RAG system, so the code is
heavily commented and every design choice is measured where possible.

> Not affiliated with Tryg. The PDFs are not included in this repository.

## Status

| Part | State |
|---|---|
| PDF text extraction and cleaning | Done |
| Chunking along clause numbers | Done (v1) |
| Embeddings stored in Postgres with pgvector | Done |
| Semantic search API and chat page showing the top 5 chunks | Done |
| Evaluation script with test questions | Done (16 questions) |
| Answer written by an LLM, with source citations | Not built yet |
| Reranker, index clean-up, vision-based table extraction | Planned |

So today the "chatbot" shows the five most relevant excerpts for a question. It
does not write an answer yet. That is deliberate: the search is evaluated on its
own before an LLM is put on top of it.

## How it works

```
INDEXING (run once)
  PDFs -> extract text -> clean -> cut into chunks -> embed each chunk -> Postgres (pgvector)

QUESTION (every time)
  question -> embed -> find the nearest chunk vectors (SQL) -> top 5 chunks -> chat page
```

- **Extraction** (`pypdf`): all 90 files contain real text, so no OCR is needed.
- **Cleaning** (`rag/clean_text.py`): removes the repeated page header (kept as a
  label), and repairs words split across lines (`føl -` / `gende` -> `følgende`).
- **Chunking** (`rag/chunker.py`): LangChain's `RecursiveCharacterTextSplitter`
  with a priority list of cut points: clause numbers (`2010.10`, `5.4.5 Karenstid`)
  first, then paragraphs, lines and words. About 1,000 characters per chunk with
  150 characters of overlap. Result: 4,257 chunks.
- **Embeddings** (`rag/embeddings.py`): `qwen/qwen3-embedding-8b` through
  OpenRouter, 1,024 dimensions. The provider sits behind a small wrapper, so the
  model can be swapped in one file. Each chunk is embedded with its product name
  in front, so the same clause in two products does not look identical.
- **Search** (`rag/search.py`): one SQL query ordered by cosine distance. Exact
  search (no index), which takes milliseconds at this size.

## Evaluation

`rag/evaluate.py` runs the test questions in `rag/eval_questions.json` through the
real search and reports at which rank the correct chunk appears.

Baseline (v1):

| Metric | Result | Meaning |
|---|---|---|
| hit@1 | 8 / 16 (50 %) | correct chunk is the top result |
| hit@5 | 13 / 16 (81 %) | correct chunk is among the five shown |
| hit@30 | 15 / 16 (94 %) | correct chunk is in the top 30 |

The gap between hit@5 and hit@30 is the argument for adding a reranker: in two of
the three misses the correct chunk was found, but ranked 11th and 21st.

Limits of this number: 16 questions is a small sample, and I wrote them after
reading the documents, so they are probably easier than real user questions.

## Known weaknesses

- **Tables** are extracted as loose lines. Side-by-side tables lose their column
  structure, and a header can end up attached to the wrong table.
- **Table-of-contents chunks** match many questions without answering any.
- **Orphan headings**: some chunks are only a heading with no text.
- **Chunks mix clauses**, so a chunk can match on the wrong part.
- **Several versions of the same product** exist (for example three sets of van
  insurance conditions), and the search does not know which one a user means.

## Roadmap

1. Clean the index (drop table-of-contents chunks, attach headings to their text).
2. Add a reranker: fetch 30 candidates, re-sort, keep 5.
3. Let an LLM write the answer from the retrieved chunks, with file and page citations.
4. v2 extraction: use a vision model to rewrite tables as Markdown.
5. Re-run the evaluation after every step and keep only what improves it.

## Run it yourself

Requirements: Python 3.12, Node.js, PostgreSQL with the
[pgvector](https://github.com/pgvector/pgvector) extension, and an
[OpenRouter](https://openrouter.ai) API key.

```bash
# 1. Settings
cp .env.example .env          # then fill in DATABASE_URL and OPENROUTER_API_KEY

# 2. Documents: put the PDFs in data/tryg/Betingelser/ and data/tryg/Produktinformation/

# 3. Backend
cd backend
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m rag.chunker              # PDFs -> data/chunks_v1.jsonl
.venv/bin/python -m rag.create_chunk_table   # pgvector extension + table
.venv/bin/python -m rag.load_chunks          # embed and store (a few minutes)
.venv/bin/python -m rag.evaluate             # optional: run the test questions
.venv/bin/uvicorn main:app --reload --port 8001

# 4. Frontend (in a second terminal)
cd frontend
npm install
npm run dev                                  # http://localhost:5174
```

## Project layout

```
backend/
  main.py                  FastAPI app with GET /rag/search
  database.py              connection to Postgres
  rag/
    peek_pdf.py            print the raw text of one PDF
    health_report.py       text-quality report across all PDFs
    clean_text.py          clean one page of extracted text
    chunker.py             cut all PDFs into chunks
    embeddings.py          embedding wrapper (OpenRouter)
    rag_models.py          the document_chunks table
    create_chunk_table.py  create extension and table
    load_chunks.py         embed all chunks and store them
    search.py              semantic search
    evaluate.py            evaluation script
    eval_questions.json    test questions and expected sources
frontend/
  src/Chatbot.tsx          the chat page
```

## Stack

Python, FastAPI, SQLAlchemy, PostgreSQL + pgvector, LangChain text splitters,
OpenRouter (Qwen3 embeddings), TypeScript, React, Vite.
