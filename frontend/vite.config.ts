import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Development: `npm run dev` serves the app on http://127.0.0.1:5173 and forwards /api to the Python studio
// server (studio.py, default port 8770). Production: `npm run build` writes dist/, which studio.py serves itself.
const api = process.env.STUDIO_API ?? "http://127.0.0.1:8770";

export default defineConfig({
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    port: 5173,
    proxy: { "/api": { target: api, changeOrigin: false } },
  },
  build: {
    outDir: "dist",
    chunkSizeWarningLimit: 1500,
  },
});
