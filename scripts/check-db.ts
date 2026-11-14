import { sql } from "../src/lib/db.ts";

await sql(
  `insert into searches (kind, source, compare_mode, surface, body_a, body_b, lat, lon, result_count, summary)
   values ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)`,
  ["query", "dem", "terrain", "any", "moon", "earth", 5.4, -158.6, 3, "smoke"],
);
const rows = await sql<{ summary: string }>("select summary from searches order by id desc limit 1");
if (rows[0]?.summary !== "smoke") {
  console.error(rows);
  process.exit(1);
}
console.log("db ok", rows[0].summary);
