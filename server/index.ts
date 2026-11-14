import { createServer, request as httpRequest } from "node:http";
import { readFile, stat } from "node:fs/promises";
import path from "node:path";
import { sql } from "../src/lib/db.ts";

const port = Number(process.env.PORT || 8000);
const pythonUrl = process.env.PYTHON_URL || "http://127.0.0.1:8010";
const dist = path.resolve("frontend/dist");

type SearchRow = {
  id: number;
  created_at: string;
  kind: string;
  source: string;
  compare_mode: string;
  surface: string;
  body_a: string | null;
  body_b: string | null;
  lat: number | null;
  lon: number | null;
  label: string | null;
  result_count: number;
  summary: string;
};

const types: Record<string, string> = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".webp": "image/webp",
};

function sendJson(res: import("node:http").ServerResponse, status: number, payload: unknown) {
  const body = Buffer.from(JSON.stringify(payload));
  res.writeHead(status, { "content-type": "application/json", "content-length": body.length, "cache-control": "no-store" });
  res.end(body);
}

async function record(url: URL, payload: { origin?: { body?: string; lat?: number; lon?: number }; matches?: unknown[] }) {
  const kind = url.pathname === "/api/query" ? "query" : "top";
  const matches = payload.matches ?? [];
  const first = matches[0] as { similarity?: number; b?: { body?: string } } | undefined;
  const summary = kind === "query"
    ? `${payload.origin?.body ?? url.searchParams.get("body")} ${Number(payload.origin?.lat ?? url.searchParams.get("lat")).toFixed(2)}, ${Number(payload.origin?.lon ?? url.searchParams.get("lon")).toFixed(2)} → ${first?.b?.body ?? "no match"} ${first?.similarity != null ? first.similarity.toFixed(1) + "%" : ""}`.trim()
    : `Top ${matches.length} ${url.searchParams.get("a") ?? ""} / ${url.searchParams.get("b") ?? ""}`;
  await sql(
    `insert into searches (kind, source, compare_mode, surface, body_a, body_b, lat, lon, label, result_count, summary)
     values ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)`,
    [
      kind,
      url.searchParams.get("source") ?? "",
      url.searchParams.get("mode") ?? "",
      url.searchParams.get("surface") ?? "any",
      kind === "query" ? url.searchParams.get("body") : url.searchParams.get("a"),
      kind === "query" ? url.searchParams.get("target") : url.searchParams.get("b"),
      url.searchParams.get("lat") ? Number(url.searchParams.get("lat")) : null,
      url.searchParams.get("lon") ? Number(url.searchParams.get("lon")) : null,
      null,
      matches.length,
      summary,
    ],
  );
}

function proxy(reqUrl: URL, res: import("node:http").ServerResponse) {
  const dest = new URL(reqUrl.pathname + reqUrl.search, pythonUrl);
  const preq = httpRequest(dest, { method: "GET" }, (pres) => {
    const chunks: Buffer[] = [];
    pres.on("data", (chunk: Buffer) => chunks.push(chunk));
    pres.on("end", () => {
      void (async () => {
      const body = Buffer.concat(chunks);
      const status = pres.statusCode ?? 502;
      if (status === 200 && (dest.pathname === "/api/query" || dest.pathname === "/api/top")) {
        try {
          await record(dest, JSON.parse(body.toString("utf8")));
        } catch {
          console.error("search save failed");
        }
      }
      res.writeHead(status, {
        "content-type": String(pres.headers["content-type"] || "application/octet-stream"),
        "content-length": body.length,
        "cache-control": String(pres.headers["cache-control"] || "no-store"),
      });
      res.end(body);
      })().catch(() => {
        if (!res.headersSent) sendJson(res, 500, { error: "request failed" });
      });
    });
  });
  preq.on("error", () => sendJson(res, 502, { error: "terrain service unavailable" }));
  preq.end();
}

async function serveStatic(pathname: string, res: import("node:http").ServerResponse) {
  const rel = pathname === "/" ? "/index.html" : pathname;
  const file = path.normalize(path.join(dist, rel));
  if (!file.startsWith(dist)) {
    sendJson(res, 403, { error: "forbidden" });
    return;
  }
  try {
    const info = await stat(file);
    if (!info.isFile()) throw new Error("not a file");
    const body = await readFile(file);
    res.writeHead(200, {
      "content-type": types[path.extname(file)] || "application/octet-stream",
      "content-length": body.length,
      "cache-control": file.endsWith("index.html") ? "no-store" : "public, max-age=3600",
    });
    res.end(body);
  } catch {
    sendJson(res, 404, { error: "not found" });
  }
}

const server = createServer(async (req, res) => {
  const url = new URL(req.url || "/", "http://127.0.0.1");
  try {
    if (url.pathname === "/api/searches") {
      const rows = await sql<SearchRow>(
        "select id, created_at, kind, source, compare_mode, surface, body_a, body_b, lat, lon, label, result_count, summary from searches order by id desc limit 20",
      );
      sendJson(res, 200, rows);
      return;
    }
    if (url.pathname.startsWith("/api/") || url.pathname.startsWith("/map/")) {
      proxy(url, res);
      return;
    }
    await serveStatic(url.pathname, res);
  } catch (error) {
    console.error(error instanceof Error ? error.message : "request failed");
    if (!res.headersSent) sendJson(res, 500, { error: "request failed" });
  }
});

server.listen(port, "127.0.0.1", () => {
  console.log(`Terrain bot web app at http://127.0.0.1:${port}`);
});
