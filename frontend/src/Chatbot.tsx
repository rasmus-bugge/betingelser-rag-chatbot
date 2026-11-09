// Chatbot.tsx
// A chat page for asking questions about the Tryg insurance conditions.
//
// VERSION 1: there is no AI answer yet. For every question we only show the 5 pieces
// ("chunks") of the PDFs whose meaning is closest to the question. This lets us judge
// the SEARCH on its own. Later we hand these 5 chunks to Claude, which writes an answer
// above them; then the chunks become the "sources" under the answer.
//
// How a question travels:
//   you type -> this page calls GET /rag/search?q=... on the backend
//   -> the backend embeds the question and asks Postgres (pgvector) for the 5 nearest chunks
//   -> they come back as a list of ChunkHit -> we draw them as cards.

import { useEffect, useRef, useState } from 'react'
import type { FormEvent } from 'react'
import { API_URL } from './api'
import type { ChunkHit } from './api'
import './chatbot.css'

// One question and what happened to it. The page keeps a LIST of these (the history).
// status: 'loading' = waiting for the backend, 'done' = hits arrived, 'error' = failed.
type Turn = {
  id: number
  question: string
  status: 'loading' | 'done' | 'error'
  hits: ChunkHit[]
  error: string | null
}

// Clickable example questions, so you can try the search without typing.
const EXAMPLES = [
  'Hvad er karenstid i cyberforsikringen?',
  'Hvad får jeg i erstatning for en 5 år gammel genstand?',
  'Dækker erhvervsforsikringen skader fra frost?',
  'Hvornår kan Tryg opsige en forsikring?',
]

// Scores are numbers like 0.5503. Show them the Danish way: "0,55".
function formatScore(score: number): string {
  return score.toLocaleString('da-DK', { minimumFractionDigits: 2, maximumFractionDigits: 2 })
}

// One search result drawn as a card.
function HitCard({ hit, rank }: { hit: ChunkHit; rank: number }) {
  return (
    <article className="hit">
      <header className="hit-head">
        <span className="hit-rank">{rank}</span>
        <strong>{hit.product}</strong>
        <span className="badge grey">{hit.doc_type}</span>
        {/* The score tells us how close the chunk is to the question: bigger = closer */}
        <span className="hit-score" title="Lighed med spørgsmålet (cosinus-lighed)">
          {formatScore(hit.score)}
        </span>
      </header>
      {/* The source line: which PDF and page. Later this becomes the citation. */}
      <div className="hit-source muted small">
        {hit.file} · side {hit.page} · uddrag {hit.chunk_number}
      </div>
      {/* The chunk text itself. CSS "white-space: pre-wrap" keeps its line breaks. */}
      <pre className="hit-text">{hit.text}</pre>
    </article>
  )
}

export function Chatbot() {
  // STATE = values React remembers; when one changes, the page redraws.
  const [input, setInput] = useState('') // what is typed in the box right now
  const [turns, setTurns] = useState<Turn[]>([]) // every question asked so far

  // A counter for giving each question its own id. useRef = a value that survives
  // redraws but, unlike state, does NOT cause a redraw when it changes.
  const nextId = useRef(1)
  // A reference to an invisible element at the bottom, so we can scroll to it.
  const bottomRef = useRef<HTMLDivElement>(null)

  // Whenever the list of turns changes, scroll down to the newest one.
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [turns])

  // Change some fields of ONE turn (found by its id) and leave the others alone.
  function updateTurn(id: number, changes: Partial<Turn>) {
    setTurns((previous) => previous.map((t) => (t.id === id ? { ...t, ...changes } : t)))
  }

  // Ask the backend. "async/await" lets us wait for the answer without freezing the page.
  async function ask(rawQuestion: string) {
    const question = rawQuestion.trim()
    if (question.length < 2) return // the backend requires at least 2 characters

    const id = nextId.current++
    // Show the question at once, in the "loading" state, and clear the box.
    setTurns((previous) => [...previous, { id, question, status: 'loading', hits: [], error: null }])
    setInput('')

    try {
      // URLSearchParams safely builds "q=Hvad+er+...&limit=5" (handles spaces, æ, ø, å, ?).
      const params = new URLSearchParams({ q: question, limit: '5' })
      const response = await fetch(`${API_URL}/rag/search?${params}`)

      if (!response.ok) {
        // The backend explains errors as {"detail": "..."}. Use that text if it is there.
        const body = await response.json().catch(() => null)
        const detail = typeof body?.detail === 'string' ? body.detail : null
        throw new Error(detail ?? `Backend svarede ${response.status}`)
      }

      const hits: ChunkHit[] = await response.json()
      updateTurn(id, { status: 'done', hits })
    } catch (err) {
      updateTurn(id, {
        status: 'error',
        error: err instanceof Error ? err.message : 'Ukendt fejl',
      })
    }
  }

  // Runs when the form is submitted (Enter key or the button).
  function handleSubmit(event: FormEvent) {
    event.preventDefault() // stop the browser from reloading the page
    ask(input)
  }

  return (
    <div className="chat">
      <header className="page-header">
        <h1>Chatbot – Tryg betingelser</h1>
        <div className="meta">
          Stil et spørgsmål. Lige nu viser vi kun de 5 uddrag fra betingelserne, der ligner
          spørgsmålet mest – der er endnu ingen AI-svar.
        </div>
      </header>

      <div className="chat-log">
        {turns.length === 0 && (
          <div className="chat-empty">
            <p className="muted">Prøv et af disse, eller skriv dit eget:</p>
            <div className="chat-examples">
              {EXAMPLES.map((example) => (
                <button key={example} type="button" onClick={() => ask(example)}>
                  {example}
                </button>
              ))}
            </div>
          </div>
        )}

        {turns.map((turn) => (
          <div key={turn.id} className="turn">
            {/* The question, as a bubble on the right (like a chat app) */}
            <div className="bubble-question">{turn.question}</div>

            {turn.status === 'loading' && <p className="notice">Søger i betingelserne ...</p>}

            {turn.status === 'error' && (
              <p className="notice error">Søgningen fejlede: {turn.error}</p>
            )}

            {turn.status === 'done' && (
              <section aria-label="De fem bedste uddrag">
                <h2 className="hits-title">De {turn.hits.length} mest relevante uddrag</h2>
                {turn.hits.map((hit, index) => (
                  <HitCard key={`${hit.file}-${hit.chunk_number}`} hit={hit} rank={index + 1} />
                ))}
              </section>
            )}
          </div>
        ))}

        {/* An invisible marker; we scroll to it after each new question */}
        <div ref={bottomRef} />
      </div>

      {/* The input box, always visible at the bottom */}
      <form className="chat-input" onSubmit={handleSubmit}>
        <input
          value={input}
          onChange={(event) => setInput(event.target.value)}
          placeholder="Skriv dit spørgsmål om betingelserne ..."
          aria-label="Dit spørgsmål"
          maxLength={500}
        />
        <button type="submit" disabled={input.trim().length < 2}>
          Spørg
        </button>
      </form>
    </div>
  )
}
