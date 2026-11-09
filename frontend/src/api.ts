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
  variant: string // "2303", or "" if the product has only one version
  language: string // "da" or "en"
  page: number // the page where the chunk starts
  chunk_number: number
  section: string // the headings above the chunk, "" if none were found
  score: number // 0..1, bigger = closer in meaning to the question
  text: string
}

// A question from the chatbot back to the person.
export type FollowUp = {
  question: string
  options: string[] // the products to choose between
}

// The chatbot's reply from POST /agent/ask or /agent/reply: EITHER an answer OR a
// follow-up question. Mirrors AgentReply in backend/main.py.
export type AgentReply = {
  route: 'insurance' | 'internal'
  product: string | null // null = no product recognised in the question
  answer: string | null // null while the chatbot is waiting for a reply
  sources: ChunkHit[] // the excerpts the answer cites; sources[0] is its [1]
  follow_up: FollowUp | null
  thread_id: string | null // sent back with the reply to a follow-up
}
