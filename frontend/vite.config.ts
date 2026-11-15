import { defineConfig } from "vite";

export default defineConfig({
  root: "frontend",
  build: {
    outDir: "../dist",
    emptyOutDir: true,
  },
  server: {
    port: 5173,
    allowedHosts: [".trycloudflare.com"],
    proxy: {
      "/api": "http://127.0.0.1:8000",
      "/map": "http://127.0.0.1:8000",
    },
  },
});
