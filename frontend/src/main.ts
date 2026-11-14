type BodyInfo = {
  name: string;
  source_name: string;
  lat_range: [number, number];
  lon_range: [number, number];
  width: number;
  height: number;
  pixel_km: number;
  color_min_m: number;
  color_max_m: number;
  landmarks: { kind: string; name: string; lat: number; lon: number; elevation_m: number }[];
  calibration: Record<string, { landmark: string; real_elevation_m: number; map_elevation_at_real_location_m: number; map_pixel_offset_km: number }>;
};

type Info = {
  default_mode: string;
  default_source: string;
  sources: Record<string, { label: string; bodies: Record<string, BodyInfo> }>;
  validation?: {
    body: string;
    source: string;
    spearman: number;
    median_abs_error_m: number;
    rmse_m: number;
    bias_m: number;
    underwater_fraction_image?: number;
    underwater_fraction_real?: number;
    land_water_agreement?: number;
  }[];
};

type Site = {
  body: string;
  lat: number;
  lon: number;
  surface: string;
  underwater: boolean;
  features: Record<string, number>;
};

type Breakdown = {
  label: string;
  unit: string;
  a: number;
  b: number;
  difference: number;
  z_difference: number;
  weight_share: number;
  contribution: number;
};

type Match = {
  rank: number;
  similarity: number;
  distance: number;
  elevation_difference_m: number;
  a: Site;
  b: Site;
  breakdown: Breakdown[];
};

type Step = { title: string; lines: string[] };

type QueryResult = { origin: Site; matches: Match[]; steps: Step[] };
type TopResult = { matches: Match[]; steps: Step[] };

type LayerInfo = {
  available: boolean;
  title?: string;
  source?: string;
  low?: string;
  high?: string;
  note: string;
  citation?: string;
  ramp?: "heat" | "rain" | "dust" | "photo";
};

type LayerCatalog = {
  layers: { id: string; label: string }[];
  bodies: Record<string, Record<string, LayerInfo>>;
};

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
  summary: string;
};

const $ = <T extends HTMLElement = HTMLElement>(sel: string) => {
  const el = document.querySelector(sel);
  if (!el) throw new Error(sel);
  return el as T;
};
const val = (sel: string) => (document.querySelector(sel) as HTMLInputElement | HTMLSelectElement).value;

let INFO: Info | null = null;
const GRADIENTS: Record<string, string> = {
  earth: "linear-gradient(90deg,#000000,#1f4a7a,#2f8a7a,#5aa85a,#a8a855,#c8a87a,#fdfdfd)",
  moon: "linear-gradient(90deg,#00224e,#575d6d,#a59c74,#fee838)",
  mars: "linear-gradient(90deg,#000000,#7f4f32,#ff9f65,#ffc77f)",
};
const RAMPS: Record<string, string> = {
  heat: "linear-gradient(90deg,#000004,#420a68,#932667,#dd513a,#fca50a,#fcffa4)",
  rain: "linear-gradient(90deg,#102018,#1a7a32,#c8e020,#f0a020,#e02020)",
  dust: "linear-gradient(90deg,#2a2118,#c47832,#f0c060,#fff4d0)",
  photo: "transparent",
};
const LANDMARK_LABELS: Record<string, string> = {
  "earth,highest": "Everest", "earth,lowest": "Mariana Trench",
  "moon,highest": "Selenean summit", "moon,lowest": "Antoniadi crater",
  "mars,highest": "Olympus Mons", "mars,lowest": "Hellas (Badwater)",
};

const fmtM = (v: number | null | undefined) => (v == null ? "n/a" : `${Math.round(v).toLocaleString()} m`);
const fmtNum = (v: number | null | undefined, d = 1) => (v == null ? "n/a" : v.toLocaleString(undefined, { maximumFractionDigits: d, minimumFractionDigits: d }));
const fmtLL = (lat: number, lon: number) => `${Math.abs(lat).toFixed(1)}°${lat >= 0 ? "N" : "S"} ${Math.abs(lon).toFixed(1)}°${lon >= 0 ? "E" : "W"}`;
const esc = (s: string) => String(s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" })[c] ?? c);
const source = () => val("#source");
const bodies = () => {
  if (!INFO) throw new Error("not ready");
  return INFO.sources[source()].bodies;
};
let searchMode = "on-earth";
let LAYERS: LayerCatalog | null = null;
let layerId = "elevation";
const earthMode = () => searchMode === "on-earth" && Boolean(bodies().earth) && Object.keys(bodies()).some((k) => k !== "earth");
const slots = () => {
  if (!earthMode()) return { A: val("#bodyA"), B: val("#bodyB") } as Record<string, string>;
  const extras = Object.keys(bodies()).filter((k) => k !== "earth");
  const letters = ["A", "B", "C"];
  const out: Record<string, string> = {};
  extras.forEach((k, i) => {
    const letter = letters[i];
    if (letter) out[letter] = k;
  });
  const earthSlot = letters[extras.length];
  if (earthSlot) out[earthSlot] = "earth";
  return out;
};
const slotOf = (body: string) => Object.entries(slots()).find(([, b]) => b === body)?.[0];
const otherOf = (body: string) => (slots().A === body ? slots().B : slots().A);
const params = () => `mode=${val("#mode")}&source=${source()}&surface=${val("#surface")}`;

async function api<T>(path: string): Promise<T> {
  const r = await fetch(path);
  const j = await r.json() as { error?: string };
  if (!r.ok) throw new Error(j.error || r.statusText);
  return j as T;
}

function toLatLon(body: string, ev: MouseEvent) {
  const b = bodies()[body];
  const rect = (ev.currentTarget as HTMLElement).getBoundingClientRect();
  const fx = (ev.clientX - rect.left) / rect.width, fy = (ev.clientY - rect.top) / rect.height;
  return {
    lat: b.lat_range[1] - fy * (b.lat_range[1] - b.lat_range[0]),
    lon: b.lon_range[0] + fx * (b.lon_range[1] - b.lon_range[0]),
  };
}

function place(body: string, lat: number, lon: number, cls: string, text: string, title?: string) {
  const slot = slotOf(body);
  if (!slot) return null;
  const b = bodies()[body];
  const el = document.createElement("div");
  el.className = `marker ${cls}`;
  el.style.left = `${((lon - b.lon_range[0]) / (b.lon_range[1] - b.lon_range[0])) * 100}%`;
  el.style.top = `${((b.lat_range[1] - lat) / (b.lat_range[1] - b.lat_range[0])) * 100}%`;
  el.textContent = text;
  if (title) el.title = title;
  $(`#map-${slot}`).appendChild(el);
  return el;
}

const clearMarkers = (all: boolean) => document.querySelectorAll(all ? ".marker" : ".marker:not(.lm)").forEach((m) => m.remove());
const bodyKey = (name: string) => name.toLowerCase();

function elevCell(p: Site) {
  const tag = p.body === "Earth" ? `<span class="tag ${p.underwater ? "sea" : "land"}">${p.underwater ? "underwater" : "land"}</span>` : "";
  return `${fmtM(p.features.elevation_m)}${tag}`;
}

function showSteps(steps: Step[]) {
  $("#steps").innerHTML = steps.map((s) =>
    `<li><b>${esc(s.title)}</b>${s.lines.map((l) => `<div>${esc(l)}</div>`).join("")}</li>`).join("");
}

const dp = (r: Breakdown) => (r.unit === "deg" ? 2 : 0);

function showBreakdown(m: Match) {
  $<HTMLElement>("#breakdownWrap").hidden = false;
  $("#bdRank").textContent = `#${m.rank}`;
  $("#bdSim").textContent = `${m.similarity.toFixed(1)}%`;
  $("#bdA").textContent = m.a.body;
  $("#bdB").textContent = m.b.body;
  $<HTMLTableSectionElement>("#breakdown tbody").innerHTML = m.breakdown.map((r) => `<tr>
      <td>${esc(r.label)}</td><td>${fmtNum(r.a, dp(r))} ${r.unit}</td><td>${fmtNum(r.b, dp(r))} ${r.unit}</td>
      <td>${r.difference >= 0 ? "+" : ""}${fmtNum(r.difference, dp(r))} ${r.unit}</td><td>${r.z_difference >= 0 ? "+" : ""}${r.z_difference.toFixed(3)}</td>
      <td>${(r.weight_share * 100).toFixed(1)}%</td><td>${r.contribution.toFixed(5)}</td></tr>`).join("");
  const total = m.breakdown.reduce((s, r) => s + r.contribution, 0);
  $("#bdTotal").textContent = `distance = √${total.toFixed(5)} = ${Math.sqrt(total).toFixed(4)}  →  similarity = 100 × exp(−${m.distance.toFixed(4)}) = ${m.similarity.toFixed(1)}%. ` +
    `${m.a.body}: ${m.a.surface}. ${m.b.body}: ${m.b.surface}.`;
}

function showResults(matches: Match[], origin: Site | null) {
  clearMarkers(false);
  const tbody = $<HTMLTableSectionElement>("#results tbody");
  tbody.innerHTML = "";
  $<HTMLElement>("#breakdownWrap").hidden = true;
  if (!matches.length) { $("#status").textContent = "No matches found."; $<HTMLElement>("#results").hidden = true; return; }
  $("#hA").textContent = matches[0].a.body + " location";
  $("#hB").textContent = matches[0].b.body + " location";
  if (origin) place(bodyKey(origin.body), origin.lat, origin.lon, "origin", "", `${origin.body} ${fmtLL(origin.lat, origin.lon)}`);
  const relief = (p: Site) => p.features.relief_75km_m;

  for (const m of matches) {
    const markers = [];
    if (!origin) markers.push(place(bodyKey(m.a.body), m.a.lat, m.a.lon, "", String(m.rank), `${m.a.body} #${m.rank}`));
    markers.push(place(bodyKey(m.b.body), m.b.lat, m.b.lon, "", String(m.rank), `${m.b.body} #${m.rank}`));
    const live = markers.filter((mk): mk is HTMLDivElement => Boolean(mk));
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${m.rank}</td><td><span class="sim">${m.similarity.toFixed(1)}%</span></td>
      <td>${fmtLL(m.a.lat, m.a.lon)}</td><td>${elevCell(m.a)}</td><td>${fmtM(relief(m.a))}</td>
      <td>${fmtLL(m.b.lat, m.b.lon)}</td><td>${elevCell(m.b)}</td><td>${fmtM(relief(m.b))}</td>
      <td>${m.elevation_difference_m >= 0 ? "+" : ""}${fmtM(m.elevation_difference_m)}</td>`;
    const activate = () => {
      document.querySelectorAll(".active").forEach((e) => e.classList.remove("active"));
      tr.classList.add("active");
      live.forEach((mk) => mk.classList.add("active"));
      showBreakdown(m);
    };
    tr.addEventListener("click", activate);
    live.forEach((mk) => mk.addEventListener("click", (e) => { e.stopPropagation(); activate(); tr.scrollIntoView({ block: "nearest" }); }));
    tbody.appendChild(tr);
  }
  $<HTMLElement>("#results").hidden = false;
  tbody.firstElementChild?.dispatchEvent(new Event("click"));
}

type LastQuery = { body: string; lat: number | string; lon: number | string; label?: string };
let lastQuery: LastQuery | null = null;

function setSearchMode(mode: string) {
  searchMode = mode;
  document.querySelectorAll<HTMLButtonElement>("#searchMode button").forEach((b) => {
    const on = b.dataset.mode === searchMode;
    b.classList.toggle("on", on);
    b.setAttribute("aria-pressed", on ? "true" : "false");
  });
}

async function runQuery(body: string, lat: number | string, lon: number | string, label?: string) {
  if (!slotOf(body)) return;
  if (earthMode() && body === "earth") return;
  lastQuery = { body, lat, lon, label };
  const target = earthMode() ? "earth" : otherOf(body);
  const from = $<HTMLSelectElement>("#earthFrom");
  if (earthMode() && from.querySelector(`option[value="${body}"]`)) from.value = body;
  const k = val("#k");
  $("#status").textContent = `Searching ${bodies()[target].name} for terrain like ${label || fmtLL(+lat, +lon)}…`;
  try {
    const res = await api<QueryResult>(`/api/query?body=${body}&target=${target}&lat=${lat}&lon=${lon}&k=${k}&${params()}`);
    const o = res.origin, f = o.features;
    $("#status").innerHTML = `Query: <b>${label ? esc(label) + ", " : ""}${o.body} ${fmtLL(o.lat, o.lon)}</b> is ${esc(o.surface)}; ` +
      `75 km roughness ${fmtM(f.relief_75km_m)}, slope ${f.slope_75km_deg.toFixed(2)}°. Most similar places on ${bodies()[target].name}:`;
    showResults(res.matches, o);
    showSteps(res.steps);
    void loadSearches();
  } catch (e) { $("#status").innerHTML = `<span class="err">${esc(e instanceof Error ? e.message : String(e))}</span>`; }
}

async function runTop() {
  lastQuery = null;
  const k = val("#k");
  const A = earthMode() ? val("#earthFrom") : slots().A;
  const B = earthMode() ? "earth" : slots().B;
  $("#status").textContent = earthMode()
    ? `Comparing every ${bodies()[A].name} pixel with every place on Earth…`
    : `Comparing every ${bodies()[A].name} pixel with every ${bodies()[B].name} pixel…`;
  try {
    const res = await api<TopResult>(`/api/top?n=${k}&a=${A}&b=${B}&${params()}`);
    $("#status").textContent = earthMode()
      ? `Top ${res.matches.length} closest ${bodies()[A].name} places on Earth (${val("#mode")} mode, ${INFO?.sources[source()].label}). Click a row to see why.`
      : `Top ${res.matches.length} most similar ${bodies()[A].name}/${bodies()[B].name} locations (${val("#mode")} mode, ${INFO?.sources[source()].label}). Click a row to see why.`;
    showResults(res.matches, null);
    showSteps(res.steps);
    void loadSearches();
  } catch (e) { $("#status").innerHTML = `<span class="err">${esc(e instanceof Error ? e.message : String(e))}</span>`; }
}

const rerun = () => (lastQuery && slotOf(lastQuery.body) ? runQuery(lastQuery.body, lastQuery.lat, lastQuery.lon, lastQuery.label) : runTop());

function wireMap(slot: string) {
  const wrap = $(`#map-${slot}`);
  let pending: ReturnType<typeof setTimeout> | undefined;
  wrap.addEventListener("mousemove", (ev) => {
    const body = slots()[slot];
    if (!body) return;
    const { lat, lon } = toLatLon(body, ev);
    clearTimeout(pending);
    pending = setTimeout(async () => {
      try {
        const h = await api<{ surface: string }>(`/api/height?body=${body}&lat=${lat}&lon=${lon}&source=${source()}`);
        $(`#hover-${slot}`).textContent = `${fmtLL(lat, lon)}: ${h.surface}`;
      } catch { /* hover is best-effort */ }
    }, 60);
  });
  wrap.addEventListener("click", (ev) => {
    const body = slots()[slot];
    if (!body || (earthMode() && body === "earth")) return;
    if (layerId !== "elevation" && LAYERS?.bodies[body]?.[layerId]?.available === false) return;
    const { lat, lon } = toLatLon(body, ev);
    void runQuery(body, lat.toFixed(3), lon.toFixed(3));
  });
}

function fillBodySelects() {
  const keys = Object.keys(bodies());
  for (const [id, fallback] of [["#bodyA", 0], ["#bodyB", 1]] as const) {
    const sel = $<HTMLSelectElement>(id), prev = sel.value;
    sel.innerHTML = "";
    keys.forEach((k) => sel.add(new Option(bodies()[k].name, k)));
    sel.value = keys.includes(prev) ? prev : keys[fallback];
  }
}

function applyChrome() {
  const onEarth = earthMode();
  $("#maps").classList.toggle("earth-mode", onEarth);
  $<HTMLElement>("#pairBodies").hidden = onEarth;
  $<HTMLElement>("#earthFromLabel").hidden = !onEarth;
  $("#topBtn").textContent = onEarth ? "Show closest places on Earth" : "Show global top matches";
  const from = $<HTMLSelectElement>("#earthFrom");
  const prev = from.value;
  from.innerHTML = "";
  Object.keys(bodies()).filter((k) => k !== "earth").forEach((k) => from.add(new Option(bodies()[k].name, k)));
  if (Array.from(from.options).some((o) => o.value === prev)) from.value = prev;
  for (const slot of ["A", "B", "C"]) $<HTMLElement>(`#slot${slot}`).hidden = !(slot in slots());
}

function renderPair() {
  const src = source();
  applyChrome();
  clearMarkers(true);
  for (const [slot, key] of Object.entries(slots())) {
    const b = bodies()[key];
    const panel = $<HTMLElement>(`#slot${slot}`);
    panel.hidden = false;
    panel.classList.toggle("earth-result", earthMode() && key === "earth");
    const thematic = layerId === "elevation" ? undefined : LAYERS?.bodies[key]?.[layerId];
    const missing = Boolean(thematic && !thematic.available);
    $(`#map-${slot}`).style.cursor = missing || (earthMode() && key === "earth") ? "default" : "crosshair";
    const img = $<HTMLImageElement>(`#map-${slot} img`);
    img.hidden = missing;
    const unavailable = $<HTMLElement>(`#map-${slot} .unavailable`);
    unavailable.hidden = !missing;
    unavailable.textContent = thematic?.note || "";
    if (!missing) img.src = layerId === "elevation" ? `/map/${src}/${key}.png` : `/map/layer/${key}/${layerId}.png`;
    $(`#title${slot}`).textContent = b.name;
    $(`#min${slot}`).textContent = missing ? "" : thematic?.available ? (thematic.low || "") : fmtM(b.color_min_m);
    $(`#max${slot}`).textContent = missing ? "" : thematic?.available ? (thematic.high || "") : fmtM(b.color_max_m);
    $(`#bar${slot}`).style.background = missing
      ? "transparent"
      : thematic?.available
        ? (RAMPS[thematic.ramp || ""] || GRADIENTS[key])
        : (GRADIENTS[key] || GRADIENTS.moon);
    $(`#range${slot}`).textContent = missing
      ? ""
      : thematic?.available
        ? `${thematic.title} · ${thematic.source}`
        : `${b.source_name} · ${b.width}×${b.height} px · ~${Math.round(b.pixel_km)} km per pixel`;
    $(`#hover-${slot}`).textContent = earthMode()
      ? (key === "earth" ? "Closest places on Earth show up here." : "Hover to read elevation. Click to find the closest places on Earth.")
      : `Hover to read elevation. Click to search ${bodies()[otherOf(key)].name}.`;
    for (const lm of b.landmarks) place(key, lm.lat, lm.lon, "lm", lm.kind === "highest" ? "▲" : "▼", lm.name);
  }
  $("#lmButtons").innerHTML = "";
  const buttonKeys = earthMode() ? Object.values(slots()).filter((k) => k !== "earth") : Object.values(slots());
  for (const key of buttonKeys) {
    for (const lm of bodies()[key].landmarks) {
      const btn = document.createElement("button");
      btn.textContent = LANDMARK_LABELS[`${key},${lm.kind}`] || lm.name;
      btn.title = `${lm.name}: ${fmtM(lm.elevation_m)}`;
      btn.addEventListener("click", () => void runQuery(key, lm.lat, lm.lon, lm.name));
      $("#lmButtons").appendChild(btn);
    }
  }
  const elevationNotes = Object.values(slots()).map((key) => {
    const b = bodies()[key], c = b.calibration;
    return `<p><b>${esc(b.name)} elevation.</b> ${esc(b.source_name)}</p>` + ["highest", "lowest"].map((k) => {
      const e = c[k];
      return `<p>${k === "highest" ? "▲" : "▼"} ${esc(e.landmark)}: real ${fmtM(e.real_elevation_m)}; the elevation map reads ${fmtM(e.map_elevation_at_real_location_m)} there. ` +
        `The map's ${k} point is ${Math.round(e.map_pixel_offset_km).toLocaleString()} km away.</p>`;
    }).join("");
  }).join("");
  const shown = Object.values(slots()).map((key) => {
    const item = LAYERS?.bodies[key]?.[layerId];
    if (layerId === "elevation" || !item) return "";
    const name = bodies()[key].name;
    if (!item.available) return `<p><b>${esc(name)}.</b> ${esc(item.note)}</p>`;
    return `<p><b>${esc(name)} — ${esc(item.title || "")}.</b> ${esc(item.source || "")}</p>` +
      `<p>${esc(item.note)}</p>` +
      (item.citation ? `<p>${esc(item.citation)}</p>` : "");
  }).join("");
  const mapHeading = layerId === "elevation"
    ? ""
    : `<p><b>This map.</b> A click still searches the elevation data below.</p>${shown}`;
  $("#calib").innerHTML = mapHeading + elevationNotes;
  const note = $("#layerNote");
  if (layerId === "elevation") {
    note.textContent = "Elevation. Lightest is highest and darkest is lowest. A click still searches this terrain on every map.";
  } else {
    const shown = Object.values(slots()).map((key) => LAYERS?.bodies[key]?.[layerId]).filter((item): item is LayerInfo => Boolean(item));
    note.textContent = shown.find((item) => item.available)?.note || shown[0]?.note || "";
  }
}

function renderValidation() {
  const v = INFO?.validation;
  if (!v) { $("#validation").innerHTML = `<p class="note">Real elevation data isn't installed, so the images can't be checked.</p>`; return; }
  $("#validation").innerHTML = v.map((c) => {
    const good = c.median_abs_error_m < 600 && c.spearman > 0.9;
    const how = c.source === "images-fit" ? "fitted curve" : "lowest-to-highest scale";
    const lw = c.land_water_agreement != null ? `<span>land vs water</span><span>image says ${(c.underwater_fraction_image! * 100).toFixed(0)}% underwater, reality is ${(c.underwater_fraction_real! * 100).toFixed(0)}%; they agree on ${(c.land_water_agreement * 100).toFixed(0)}% of the surface</span>` : "";
    return `<b>${c.body} image, ${how}</b> <span class="${good ? "good" : "bad"}">${good ? "accurate" : "inaccurate"}</span>
      <div class="metric">
        <span>rank correlation</span><span>${c.spearman.toFixed(3)} (1.0 = brightness perfectly tracks real height)</span>
        <span>typical error</span><span>${fmtM(c.median_abs_error_m)} median, ${fmtM(c.rmse_m)} RMS, bias ${c.bias_m >= 0 ? "+" : ""}${fmtM(c.bias_m)}</span>
        ${lw}
      </div>`;
  }).join("") + `<p class="note">Each image is compared pixel by pixel with the measured data. Earth and Moon are faithful elevation maps (lightest = highest, darkest = lowest). The Mars image is clipped at both ends, so everything above about +7 km (Olympus and the Tharsis volcanoes) is pure white. That breaks a straight lowest-to-highest scale, but the fitted curve handles it.</p>`;
}

async function loadSearches() {
  const box = $("#searches");
  try {
    const rows = await api<SearchRow[]>("/api/searches");
    box.innerHTML = "";
    if (!rows.length) {
      box.textContent = "No saved searches yet. A click on the Moon or Mars is stored here.";
      return;
    }
    for (const row of rows) {
      const btn = document.createElement("button");
      const when = new Date(row.created_at).toLocaleString();
      btn.textContent = `${when} — ${row.summary}`;
      btn.addEventListener("click", () => replay(row));
      box.appendChild(btn);
    }
  } catch {
    box.textContent = "Saved searches are unavailable.";
  }
}

function replay(row: SearchRow) {
  if (!INFO) return;
  $<HTMLSelectElement>("#source").value = row.source || INFO.default_source;
  $<HTMLSelectElement>("#mode").value = row.compare_mode || INFO.default_mode;
  $<HTMLSelectElement>("#surface").value = row.surface || "any";
  fillBodySelects();
  if (row.kind === "query" && row.body_a && row.lat != null && row.lon != null) {
    const onEarth = row.body_b === "earth" && row.body_a !== "earth";
    setSearchMode(onEarth ? "on-earth" : "pair");
    if (!onEarth) {
      $<HTMLSelectElement>("#bodyA").value = row.body_a;
      if (row.body_b) $<HTMLSelectElement>("#bodyB").value = row.body_b;
    }
    renderPair();
    void runQuery(row.body_a, row.lat, row.lon);
    return;
  }
  setSearchMode(row.body_b === "earth" ? "on-earth" : "pair");
  if (row.body_b !== "earth") {
    if (row.body_a) $<HTMLSelectElement>("#bodyA").value = row.body_a;
    if (row.body_b) $<HTMLSelectElement>("#bodyB").value = row.body_b;
  } else if (row.body_a) {
    renderPair();
    $<HTMLSelectElement>("#earthFrom").value = row.body_a;
  }
  renderPair();
  void runTop();
}

function setLayer(id: string) {
  layerId = id;
  document.querySelectorAll<HTMLButtonElement>("#layerPick button").forEach((button) => {
    const on = button.dataset.layer === id;
    button.classList.toggle("on", on);
    button.setAttribute("aria-pressed", on ? "true" : "false");
  });
  renderPair();
}

async function init() {
  INFO = await api<Info>("/api/info");
  LAYERS = await api<LayerCatalog>("/api/layers");
  const pick = $("#layerPick");
  for (const item of LAYERS.layers) {
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.layer = item.id;
    button.textContent = item.label;
    button.addEventListener("click", () => setLayer(item.id));
    pick.appendChild(button);
  }
  document.querySelector<HTMLButtonElement>("#layerPick button")?.classList.add("on");
  document.querySelector<HTMLButtonElement>("#layerPick button")?.setAttribute("aria-pressed", "true");
  const sel = $<HTMLSelectElement>("#source");
  for (const [key, s] of Object.entries(INFO.sources)) sel.add(new Option(s.label, key));
  sel.value = INFO.default_source;
  $<HTMLSelectElement>("#mode").value = INFO.default_mode;
  fillBodySelects();
  $<HTMLSelectElement>("#bodyA").value = "earth";
  $<HTMLSelectElement>("#bodyB").value = "moon";
  wireMap("A"); wireMap("B"); wireMap("C");
  renderPair();
  renderValidation();
  void loadSearches();
  $("#topBtn").addEventListener("click", () => void runTop());
  document.querySelectorAll<HTMLButtonElement>("#searchMode button").forEach((btn) => {
    btn.addEventListener("click", () => {
      if (btn.dataset.mode === searchMode) return;
      setSearchMode(btn.dataset.mode || "on-earth");
      lastQuery = null;
      renderPair();
      $<HTMLElement>("#results").hidden = true;
      $("#steps").innerHTML = "";
      $<HTMLElement>("#breakdownWrap").hidden = true;
      if (earthMode()) $("#status").textContent = "Click a spot on the Moon or Mars to find the closest matches on Earth.";
      else void runTop();
    });
  });
  sel.addEventListener("change", () => { fillBodySelects(); renderPair(); void rerun(); });
  for (const id of ["#bodyA", "#bodyB"]) {
    $(id).addEventListener("change", (ev) => {
      const other = id === "#bodyA" ? "#bodyB" : "#bodyA";
      if (val(other) === (ev.target as HTMLSelectElement).value) {
        $<HTMLSelectElement>(other).value = Object.keys(bodies()).find((k) => k !== (ev.target as HTMLSelectElement).value) || "";
      }
      renderPair();
      void rerun();
    });
  }
  $("#mode").addEventListener("change", () => void rerun());
  $("#surface").addEventListener("change", () => void rerun());
  if (earthMode()) $("#status").textContent = "Click a spot on the Moon or Mars to find the closest matches on Earth.";
  else void runTop();
}

init().catch((e: unknown) => { $("#status").innerHTML = `<span class="err">${esc(e instanceof Error ? e.message : String(e))}</span>`; });
