import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig(({ command }) => ({
  plugins: [react()],
  // GitHub Pages serves this as a project site under /2025-Canada-Election-Sentimen/,
  // but `npm run dev` still needs to run at root.
  base: command === 'build' ? '/2025-Canada-Election-Sentimen/' : '/',
}))
