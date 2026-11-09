import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // Port 5174, so this app can run next to other projects that use Vite's
  // default port 5173. strictPort = fail loudly instead of silently picking
  // another port (the backend only allows requests from localhost:5174).
  server: { port: 5174, strictPort: true },
})
