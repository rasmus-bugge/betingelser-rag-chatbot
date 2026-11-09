// main.tsx
// The starting point of the frontend: it puts the chat page into the HTML page.

import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import { Chatbot } from './Chatbot'

// index.html contains an empty <div id="root">. React draws the app inside it.
createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <main className="page">
      <Chatbot />
    </main>
  </StrictMode>,
)
