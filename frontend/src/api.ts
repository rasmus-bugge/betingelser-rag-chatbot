// api.ts
// What the frontend needs to know about the backend: where it runs, and the
// SHAPE of the data it sends.

// Where our backend runs. (Start it with: uvicorn main:app --reload --port 8001)
export const API_URL = 'http://localhost:8001'

// One search result from GET /rag/search: a piece ("chunk") of a PDF.
// Mirrors ChunkHit in backend/main.py.
export type ChunkHit = {
  file: string // "betingelser-cyberforsikring.pdf"
  doc_type: string // "Betingelser" or "Produktinformation"
  product: string // "Cyberforsikring"
  page: number // the page where the chunk starts
  chunk_number: number
  score: number // 0..1, bigger = closer in meaning to the question
  text: string
}
