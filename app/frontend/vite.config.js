import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      // Forward API calls to the Flask backend (python -m app.server)
      '/predict': 'http://127.0.0.1:5000',
      '/health': 'http://127.0.0.1:5000',
      '/upload-test': 'http://127.0.0.1:5000',
    },
  },
})
