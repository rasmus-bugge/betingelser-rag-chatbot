// Chatbot.tsx
// A chat page for asking questions about the Tryg insurance conditions.
//
// How a question travels:
//   you type -> this page calls POST /agent/ask on the backend
//   -> the agent (backend/agent/graph.py) decides what kind of question it is
//   -> insurance question: it finds the product, searches, and writes an answer
//      internal question: it sends a fixed reply, without searching
//   -> we draw the answer, with the excerpts it used as numbered sources below it.
// If the agent is unsure which product is meant, it asks first. The person's next
// message (or a click on an option) goes to POST /agent/reply and it continues.

import { useEffect, useRef, useState } from 'react'
import type { FormEvent } from 'react'
import { API_URL } from './api'
import type { AgentReply, ChunkHit } from './api'
import './chatbot.css'

// One question and what happened to it. The page keeps a LIST of these (the history).
// status: 'loading' = waiting for the backend, 'done' = reply arrived, 'error' = failed.
type Turn = {
  id: number
  question: string
  status: 'loading' | 'done' | 'error'
  reply: AgentReply | null
  error: string | null
  // Kept so the conversation stays visible after a follow-up has been answered.
  followUpAsked: string | null
  followUpReply: string | null
}

// Clickable example questions, so you can try the chatbot without typing.
const EXAMPLES = [
  'Hvad er karenstid i cyberforsikringen?',
  'Dækker varebilforsikringen, hvis chaufføren ikke har kørekort?',
  'Dækker erhvervsforsikringen skader fra frost?',
  'Hvordan opsiger jeg en police i Rumba?',
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

  // Call the backend and store its reply on turn `id`.
  // "async/await" lets us wait for the answer without freezing the page.
  async function send(id: number, path: string, body: object) {
    try {
      const response = await fetch(`${API_URL}${path}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })

      if (!response.ok) {
        // The backend explains errors as {"detail": "..."}. Use that text if it is there.
        const body = await response.json().catch(() => null)
        const detail = typeof body?.detail === 'string' ? body.detail : null
        throw new Error(detail ?? `Backend svarede ${response.status}`)
      }

      const reply: AgentReply = await response.json()
      updateTurn(id, { status: 'done', reply })
    } catch (err) {
      updateTurn(id, {
        status: 'error',
        error: err instanceof Error ? err.message : 'Ukendt fejl',
      })
    }
  }

  // Send what the person wrote: a reply if the chatbot is waiting for one, otherwise
  // a new question.
  function ask(rawText: string) {
    const text = rawText.trim()
    if (text.length < 2) return // the backend requires at least 2 characters
    setInput('')

    const waiting = turns.find((t) => t.status === 'done' && t.reply?.follow_up)
    if (waiting) {
      answerFollowUp(waiting, text)
      return
    }

    const id = nextId.current++
    // Show the question at once, in the "loading" state.
    setTurns((previous) => [
      ...previous,
      { id, question: text, status: 'loading', reply: null, error: null, followUpAsked: null, followUpReply: null },
    ])
    send(id, '/agent/ask', { question: text })
  }

  function answerFollowUp(turn: Turn, text: string) {
    const reply = turn.reply
    if (!reply?.follow_up || !reply.thread_id) return
    updateTurn(turn.id, {
      status: 'loading',
      reply: null,
      followUpAsked: reply.follow_up.question,
      followUpReply: text,
    })
    send(turn.id, '/agent/reply', { thread_id: reply.thread_id, reply: text })
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
          Stil et spørgsmål om forsikringsbetingelserne. Svaret henviser til de uddrag, det
          bygger på.
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

            {/* A follow-up that has already been answered */}
            {turn.followUpAsked && (
              <>
                <p className="bubble-answer">{turn.followUpAsked}</p>
                <div className="bubble-question">{turn.followUpReply}</div>
              </>
            )}

            {turn.status === 'loading' && <p className="notice">Tænker ...</p>}

            {turn.status === 'error' && (
              <p className="notice error">Det fejlede: {turn.error}</p>
            )}

            {/* The chatbot is waiting: show its question and the options to click */}
            {turn.status === 'done' && turn.reply?.follow_up && (
              <>
                <p className="bubble-answer">{turn.reply.follow_up.question}</p>
                <div className="chat-examples">
                  {turn.reply.follow_up.options.map((option) => (
                    <button key={option} type="button" onClick={() => ask(option)}>
                      {option}
                    </button>
                  ))}
                </div>
              </>
            )}

            {turn.status === 'done' && turn.reply?.answer && (
              <>
                <p className="bubble-answer">{turn.reply.answer}</p>
                {/* An answer that cites nothing has no references, so this part is skipped */}
                {turn.reply.sources.length > 0 && (
                  <section aria-label="Referencer">
                    <h2 className="hits-title">Referencer</h2>
                    <ul className="references">
                      {turn.reply.sources.map((hit: ChunkHit, index: number) => (
                        <li key={`${hit.file}-${hit.chunk_number}`}>
                          [{index + 1}] {hit.product}
                          {hit.variant && `, version ${hit.variant}`}, side {hit.page}{' '}
                          <span className="muted small">
                            {hit.section && `${hit.section} `}({hit.file})
                          </span>
                        </li>
                      ))}
                    </ul>
                    {/* <details> is the browser's built-in open/close box */}
                    <details>
                      <summary className="muted small">Vis de citerede uddrag</summary>
                      {turn.reply.sources.map((hit: ChunkHit, index: number) => (
                        <HitCard key={`${hit.file}-${hit.chunk_number}`} hit={hit} rank={index + 1} />
                      ))}
                    </details>
                  </section>
                )}
              </>
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
