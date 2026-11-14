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

type Place = { n: string; lat: number; lon: number; z: number; k: string };
type Gazetteer = {
  citation: Record<string, string>;
  earth: Place[];
  ocean: Place[];
  moon: Place[];
  mars: Place[];
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
const shares: Record<string, number> = { elevation: 100 };
let PLACES: Gazetteer | null = null;
let shownMatches: { origin: Site | null; matches: Match[] } = { origin: null, matches: [] };
let cardToken = 0;
const views: Record<string, { zoom: number; panX: number; panY: number }> = {};
const mapImages = new Map<string, Promise<HTMLImageElement>>();
const paintGen: Record<string, number> = {};
let paintQueued = false;

function activeIds() {
  return (LAYERS?.layers ?? []).map((item) => item.id).filter((id) => (shares[id] ?? 0) > 0);
}

function layerLabel(id: string) {
  return LAYERS?.layers.find((item) => item.id === id)?.label || id;
}

function displayPercents(ids: string[]) {
  if (!ids.length) return {};
  const raw = ids.map((id) => shares[id] ?? 0);
  const sum = raw.reduce((total, value) => total + value, 0) || 1;
  const shown = raw.map((value) => Math.floor((value / sum) * 100));
  let left = 100 - shown.reduce((total, value) => total + value, 0);
  const order = raw
    .map((value, index) => ({ index, frac: (value / sum) * 100 - shown[index] }))
    .sort((a, b) => b.frac - a.frac);
  for (let extra = 0; extra < left; extra += 1) shown[order[extra % order.length].index] += 1;
  const out: Record<string, number> = {};
  ids.forEach((id, index) => { out[id] = shown[index]; });
  return out;
}

function usableLayers(key: string) {
  return activeIds().filter((id) => id === "elevation" || LAYERS?.bodies[key]?.[id]?.available !== false);
}

function mapUrl(key: string, id: string) {
  return id === "elevation" ? `/map/${source()}/${key}.png?v=4` : `/map/layer/${key}/${id}.png?v=4`;
}

function viewOf(slot: string) {
  return views[slot] || (views[slot] = { zoom: 1, panX: 0, panY: 0 });
}

function clampPan(pan: number, size: number, zoom: number) {
  return Math.min(0, Math.max(size * (1 - zoom), pan));
}

function applyView(slot: string) {
  const view = viewOf(slot);
  $(`#map-${slot} .mapview`).style.transform = `translate(${view.panX}px, ${view.panY}px) scale(${view.zoom})`;
  $<HTMLButtonElement>(`#map-${slot} .mapreset`).hidden = view.zoom === 1 && view.panX === 0 && view.panY === 0;
  document.querySelectorAll<HTMLElement>(`#map-${slot} .marker`).forEach((marker) => {
    marker.style.transform = `translate(-50%, -50%) scale(${1 / view.zoom})`;
  });
}

function placeLabel(place: Place) {
  const plain = place.k === "city" || place.k === "country" || place.k === "land" || place.k === "crater" || place.k === "satellite feature";
  if (plain || place.n.includes(" ")) return place.n;
  const kind = place.k.replace(/s$/, "");
  if (place.n.toLowerCase().includes(kind)) return place.n;
  return `${place.n} ${kind.charAt(0).toUpperCase()}${kind.slice(1)}`;
}

function nearestPlace(key: string, lat: number, lon: number, underwater: boolean) {
  const groups = key === "earth"
    ? (underwater ? [PLACES?.ocean || [], PLACES?.earth || []] : [PLACES?.earth || [], PLACES?.ocean || []])
    : [PLACES?.[key as "moon" | "mars"] || []];
  const scale = Math.cos((lat * Math.PI) / 180);
  const score = (place: Place) => {
    let dlon = Math.abs(place.lon - lon);
    if (dlon > 180) dlon = 360 - dlon;
    return (place.lat - lat) ** 2 + (dlon * scale) ** 2;
  };
  let best: Place | null = null;
  let bestScore = Infinity;
  for (const group of groups) {
    for (const place of group) {
      const next = score(place);
      if (next < bestScore) {
        best = place;
        bestScore = next;
      }
    }
    if (best && (!underwater || key !== "earth" || bestScore <= 64)) break;
  }
  return best;
}

function placeName(site: Site) {
  const named = nearestPlace(bodyKey(site.body), site.lat, site.lon, site.underwater);
  return named ? placeLabel(named) : fmtLL(site.lat, site.lon);
}

type PlaceInfo = {
  name: string;
  body: string;
  lat: number;
  lon: number;
  description?: string | null;
  context?: string | null;
  origin?: string | null;
  wiki?: { url?: string; title?: string; summary?: string } | null;
  photo?: string | null;
  gazetteer_url?: string | null;
  view_url?: string;
  view_credit?: string;
  source?: string | null;
  diameter_km?: number | null;
};

const placeCache = new Map<string, Promise<PlaceInfo>>();
let twinToken = 0;

function hideCard() {
  cardToken += 1;
  $<HTMLElement>("#placecard").hidden = true;
}

function moveCard(ev: MouseEvent) {
  const card = $<HTMLElement>("#placecard");
  const width = card.offsetWidth || 120;
  const height = card.offsetHeight || 32;
  let left = ev.clientX + 12;
  let top = ev.clientY + 14;
  if (left + width > window.innerWidth - 8) left = Math.max(8, ev.clientX - width - 12);
  if (top + height > window.innerHeight - 8) top = Math.max(8, ev.clientY - height - 12);
  card.style.left = `${left}px`;
  card.style.top = `${top}px`;
}

function viewKm(p: PlaceInfo) {
  if (p.diameter_km) return Math.min(800, Math.max(60, p.diameter_km * 0.8));
  return 150;
}

function renderPlace(p: PlaceInfo, label: string | undefined, elevation: number | undefined) {
  const coords = fmtLL(p.lat, p.lon);
  const wiki = p.wiki;
  const links = [
    wiki && wiki.url ? `<a href="${esc(wiki.url)}" target="_blank" rel="noopener">Wikipedia</a>` : "",
    p.gazetteer_url ? `<a href="${esc(p.gazetteer_url)}" target="_blank" rel="noopener">IAU gazetteer</a>` : "",
  ].filter(Boolean).join(" · ");
  const photo = p.photo ? `<figure><img src="${esc(p.photo)}" alt="Photo of ${esc(p.name)}">
      <figcaption>Photo: ${esc(wiki?.title || p.name)} (Wikimedia)</figcaption></figure>` : "";
  const view = p.view_url ? `<figure><img src="${esc(p.view_url)}&km=${viewKm(p).toFixed(0)}" alt="Orbital view of ${esc(p.name)}">
      <figcaption>Exact spot (red cross): ${esc(p.view_credit || "")}</figcaption></figure>` : "";
  return `<div class="pics">${view}${photo}</div>
    <h3>${esc(p.name)}</h3>
    <div class="sub">${esc(p.body)} · ${coords}${elevation != null ? ` · ${fmtM(elevation)}` : ""}${label && label !== p.name ? ` · ${esc(label)}` : ""}</div>
    ${p.description ? `<div class="desc">${esc(p.description)}${p.context ? `, ${esc(p.context)}` : ""}</div>` : ""}
    ${wiki?.summary ? `<div class="summary">${esc(wiki.summary)}</div>` : p.origin ? `<div class="summary">Named after: ${esc(p.origin)}</div>` : ""}
    ${links ? `<div class="sub" style="margin-top:6px">${links}${p.source ? ` · names: ${esc(p.source)}` : ""}</div>` : p.source ? `<div class="sub" style="margin-top:6px">names: ${esc(p.source)}</div>` : ""}`;
}

function placeFor(body: string, lat: number, lon: number) {
  const key = `${body}:${lat.toFixed(3)}:${lon.toFixed(3)}`;
  let pending = placeCache.get(key);
  if (!pending) {
    pending = api<PlaceInfo>(`/api/place?body=${body}&lat=${lat}&lon=${lon}&source=${source()}`).catch((error) => {
      placeCache.delete(key);
      throw error;
    });
    placeCache.set(key, pending);
  }
  return pending;
}

function showTwin(m: Match | null) {
  const panel = $<HTMLElement>("#twinPanel");
  if (!m) {
    panel.hidden = true;
    return;
  }
  const token = ++twinToken;
  panel.hidden = false;
  $("#twinRoleA").textContent = `Your point on ${m.a.body}`;
  $("#twinRoleB").textContent = `Match on ${m.b.body}`;
  $("#twinSub").textContent = `match #${m.rank}`;
  $("#twinSim").textContent = `${m.similarity.toFixed(1)}%`;
  $("#twinDiff").textContent = `height difference ${m.elevation_difference_m >= 0 ? "+" : ""}${fmtM(m.elevation_difference_m)}`;
  const loading = (p: Site) => `<h3>Looking up this place…</h3><div class="sub">${esc(p.body)} · ${fmtLL(p.lat, p.lon)}</div>`;
  const left = $<HTMLElement>("#twinA");
  const right = $<HTMLElement>("#twinB");
  left.innerHTML = loading(m.a);
  right.innerHTML = loading(m.b);
  const fill = async (el: HTMLElement, site: Site) => {
    try {
      const found = await placeFor(bodyKey(site.body), site.lat, site.lon);
      if (token === twinToken) el.innerHTML = renderPlace(found, placeName(site), site.features.elevation_m);
    } catch (error) {
      if (token === twinToken) el.innerHTML = `<h3>${fmtLL(site.lat, site.lon)}</h3><div class="err">${esc(error instanceof Error ? error.message : "lookup failed")}</div>`;
    }
  };
  void fill(left, m.a);
  void fill(right, m.b);
}

function watchSpot(el: HTMLElement, site: Site, match: Match | null) {
  el.addEventListener("mouseenter", (ev) => openCard(ev, site, match));
  el.addEventListener("mousemove", (ev) => moveCard(ev));
  el.addEventListener("mouseleave", () => hideCard());
}

function openCard(ev: MouseEvent, site: Site, match: Match | null) {
  const token = ++cardToken;
  const card = $<HTMLElement>("#placecard");
  $("#spotName").textContent = placeName(site);
  card.hidden = false;
  moveCard(ev);
  showTwin(match ?? shownMatches.matches[0] ?? null);
  void placeFor(bodyKey(site.body), site.lat, site.lon).then((found) => {
    if (token === cardToken) $("#spotName").textContent = found.name;
  }).catch(() => undefined);
}

function loadMap(url: string) {
  let pending = mapImages.get(url);
  if (!pending) {
    pending = new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error(url));
      img.src = url;
    });
    mapImages.set(url, pending);
  }
  return pending;
}
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
  const slot = slotOf(body);
  const view = slot ? viewOf(slot) : { zoom: 1, panX: 0, panY: 0 };
  const rect = (ev.currentTarget as HTMLElement).getBoundingClientRect();
  const fx = Math.min(1, Math.max(0, (ev.clientX - rect.left - view.panX) / (rect.width * view.zoom)));
  const fy = Math.min(1, Math.max(0, (ev.clientY - rect.top - view.panY) / (rect.height * view.zoom)));
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
  el.style.transform = `translate(-50%, -50%) scale(${1 / viewOf(slot).zoom})`;
  $(`#map-${slot} .mapview`).appendChild(el);
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
  shownMatches = { origin, matches };
  hideCard();
  clearMarkers(false);
  const tbody = $<HTMLTableSectionElement>("#results tbody");
  tbody.innerHTML = "";
  $<HTMLElement>("#breakdownWrap").hidden = true;
  if (!matches.length) {
    $("#status").textContent = "No matches found.";
    $<HTMLElement>("#results").hidden = true;
    showTwin(null);
    return;
  }
  $("#hA").textContent = matches[0].a.body + " location";
  $("#hB").textContent = matches[0].b.body + " location";
  if (origin) {
    const picked = place(bodyKey(origin.body), origin.lat, origin.lon, "origin", "", placeName(origin));
    if (picked) watchSpot(picked, origin, matches[0] ?? null);
  }
  const relief = (p: Site) => p.features.relief_75km_m;

  for (const m of matches) {
    const markers = [];
    if (!origin) markers.push(place(bodyKey(m.a.body), m.a.lat, m.a.lon, "", String(m.rank), placeName(m.a)));
    markers.push(place(bodyKey(m.b.body), m.b.lat, m.b.lon, "", String(m.rank), placeName(m.b)));
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
      showTwin(m);
    };
    tr.addEventListener("click", activate);
    tr.addEventListener("mouseenter", (ev) => { void openCard(ev, m.b, m); });
    tr.addEventListener("mousemove", (ev) => moveCard(ev));
    tr.addEventListener("mouseleave", () => hideCard());
    const points = origin ? [m.b] : [m.a, m.b];
    live.forEach((mk, index) => {
      watchSpot(mk, points[index] ?? m.b, m);
      mk.addEventListener("click", (e) => { e.stopPropagation(); activate(); tr.scrollIntoView({ block: "nearest" }); });
    });
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
    const called = nearestPlace(bodyKey(o.body), o.lat, o.lon, o.underwater);
    $("#status").innerHTML = `Query: <b>${called ? esc(placeLabel(called)) + ", " : ""}${label ? esc(label) + ", " : ""}${o.body} ${fmtLL(o.lat, o.lon)}</b> is ${esc(o.surface)}; ` +
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
  let drag: { x: number; y: number; panX: number; panY: number; moved: boolean; pointer: number } | null = null;
  wrap.addEventListener("mousemove", (ev) => {
    const body = slots()[slot];
    if (!body || drag?.moved) return;
    const { lat, lon } = toLatLon(body, ev);
    clearTimeout(pending);
    pending = setTimeout(async () => {
      try {
        const h = await api<{ surface: string }>(`/api/height?body=${body}&lat=${lat}&lon=${lon}&source=${source()}`);
        $(`#hover-${slot}`).textContent = `${fmtLL(lat, lon)}: ${h.surface}`;
      } catch { /* hover is best-effort */ }
    }, 60);
  });
  wrap.addEventListener("pointerdown", (ev) => {
    if ((ev.target as HTMLElement).closest("button, .marker")) return;
    const view = viewOf(slot);
    drag = { x: ev.clientX, y: ev.clientY, panX: view.panX, panY: view.panY, moved: false, pointer: ev.pointerId };
    wrap.setPointerCapture(ev.pointerId);
  });
  wrap.addEventListener("pointermove", (ev) => {
    if (!drag || drag.pointer !== ev.pointerId) return;
    const dx = ev.clientX - drag.x;
    const dy = ev.clientY - drag.y;
    if (Math.hypot(dx, dy) > 4) drag.moved = true;
    if (!drag.moved) return;
    const view = viewOf(slot);
    const rect = wrap.getBoundingClientRect();
    view.panX = clampPan(drag.panX + dx, rect.width, view.zoom);
    view.panY = clampPan(drag.panY + dy, rect.height, view.zoom);
    applyView(slot);
  });
  wrap.addEventListener("pointerup", (ev) => {
    if (!drag || drag.pointer !== ev.pointerId) return;
    if (drag.moved) wrap.dataset.dragged = "1";
    drag = null;
  });
  wrap.addEventListener("wheel", (ev) => {
    ev.preventDefault();
    const view = viewOf(slot);
    const rect = wrap.getBoundingClientRect();
    const mx = ev.clientX - rect.left;
    const my = ev.clientY - rect.top;
    const next = Math.min(28, Math.max(1, view.zoom * (ev.deltaY < 0 ? 1.18 : 1 / 1.18)));
    const scale = next / view.zoom;
    view.panX = clampPan(mx - (mx - view.panX) * scale, rect.width, next);
    view.panY = clampPan(my - (my - view.panY) * scale, rect.height, next);
    view.zoom = next;
    applyView(slot);
  }, { passive: false });
  $<HTMLButtonElement>(`#map-${slot} .mapreset`).addEventListener("click", (ev) => {
    ev.stopPropagation();
    views[slot] = { zoom: 1, panX: 0, panY: 0 };
    applyView(slot);
  });
  wrap.addEventListener("click", (ev) => {
    if (wrap.dataset.dragged === "1") {
      wrap.dataset.dragged = "";
      return;
    }
    const body = slots()[slot];
    if (!body || (earthMode() && body === "earth")) return;
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
  applyChrome();
  clearMarkers(true);
  for (const [slot, key] of Object.entries(slots())) {
    const b = bodies()[key];
    const panel = $<HTMLElement>(`#slot${slot}`);
    panel.hidden = false;
    panel.classList.toggle("earth-result", earthMode() && key === "earth");
    const drawable = usableLayers(key);
    const missing = drawable.length === 0;
    $(`#map-${slot}`).style.cursor = missing || (earthMode() && key === "earth") ? "default" : "crosshair";
    const unavailable = $<HTMLElement>(`#map-${slot} .unavailable`);
    unavailable.hidden = !missing;
    unavailable.textContent = missing ? "None of the selected maps are available for this body." : "";
    $(`#title${slot}`).textContent = b.name;
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
  writeMixText();
  for (const slot of Object.keys(slots())) applyView(slot);
}

function writeMixText() {
  const elevationNotes = Object.values(slots()).map((key) => {
    const b = bodies()[key], c = b.calibration;
    return `<p><b>${esc(b.name)} elevation.</b> ${esc(b.source_name)}</p>` + ["highest", "lowest"].map((k) => {
      const e = c[k];
      return `<p>${k === "highest" ? "▲" : "▼"} ${esc(e.landmark)}: real ${fmtM(e.real_elevation_m)}; the elevation map reads ${fmtM(e.map_elevation_at_real_location_m)} there. ` +
        `The map's ${k} point is ${Math.round(e.map_pixel_offset_km).toLocaleString()} km away.</p>`;
    }).join("");
  }).join("");
  const percents = displayPercents(activeIds());
  const shown = activeIds().filter((id) => id !== "elevation").map((id) => Object.values(slots()).map((key) => {
    const item = LAYERS?.bodies[key]?.[id];
    if (!item) return "";
    const name = bodies()[key].name;
    if (!item.available) return `<p><b>${esc(name)} — ${esc(layerLabel(id))} ${percents[id]}%.</b> ${esc(item.note)}</p>`;
    return `<p><b>${esc(name)} — ${esc(item.title || "")} ${percents[id]}%.</b> ${esc(item.source || "")}</p>` +
      `<p>${esc(item.note)}</p>` +
      (item.citation ? `<p>${esc(item.citation)}</p>` : "");
  }).join("")).join("");
  const mixLine = activeIds().map((id) => `${layerLabel(id)} ${percents[id]}%`).join(", ");
  const mapHeading = `<p><b>This mix.</b> ${esc(mixLine)}. A click still searches the elevation data below.</p>${shown}`;
  const cited = PLACES
    ? `<p><b>Names on a selected point.</b> ${esc(PLACES.citation.earth)} ${esc(PLACES.citation.ocean)} ${esc(PLACES.citation.moon)}</p>`
    : "";
  $("#calib").innerHTML = mapHeading + elevationNotes + cited;
  $("#layerNote").textContent = `${mixLine}. Hover a point to see its name. The match is under the maps. A click still searches elevation.`;
  for (const [slot, key] of Object.entries(slots())) {
    $(`#legend${slot}`).innerHTML = legendHtml(key);
    const info = bodies()[key];
    $(`#range${slot}`).textContent = usableLayers(key).length
      ? activeIds().map((id) => `${percents[id]}% ${id === "elevation" ? info.source_name : (LAYERS?.bodies[key]?.[id]?.title || layerLabel(id))}`).join(" · ")
      : "";
  }
  schedulePaint();
}

function legendHtml(key: string) {
  const b = bodies()[key];
  const ids = activeIds();
  const percents = displayPercents(ids);
  return ids.map((id) => {
    const percent = `${percents[id]}%`;
    if (id === "elevation") {
      return `<div class="legend-row"><span class="pct">${percent}</span><span class="name">Elevation</span><span class="bar" style="background:${GRADIENTS[key] || GRADIENTS.moon}"></span><span>${esc(fmtM(b.color_min_m))}</span><span>${esc(fmtM(b.color_max_m))}</span></div>`;
    }
    const item = LAYERS?.bodies[key]?.[id];
    if (!item?.available) {
      return `<div class="legend-row"><span class="pct">${percent}</span><span class="name">${esc(layerLabel(id))} is not available for this body</span></div>`;
    }
    const ramp = RAMPS[item.ramp || ""] || GRADIENTS[key] || GRADIENTS.moon;
    const bar = item.ramp === "photo" ? "" : `<span class="bar" style="background:${ramp}"></span><span>${esc(item.low || "")}</span><span>${esc(item.high || "")}</span>`;
    return `<div class="legend-row"><span class="pct">${percent}</span><span class="name">${esc(item.title || layerLabel(id))}</span>${bar}</div>`;
  }).join("");
}

function schedulePaint() {
  if (paintQueued) return;
  paintQueued = true;
  requestAnimationFrame(() => {
    paintQueued = false;
    for (const [slot, key] of Object.entries(slots())) void paintBlend(slot, key);
  });
}

async function paintBlend(slot: string, key: string) {
  const gen = (paintGen[slot] = (paintGen[slot] || 0) + 1);
  const canvas = $<HTMLCanvasElement>(`#map-${slot} canvas.blend`);
  const ids = usableLayers(key);
  if (!ids.length) {
    canvas.hidden = true;
    return;
  }
  const width = Math.max(480, Math.round(canvas.clientWidth * Math.min(window.devicePixelRatio || 1, 2)));
  const height = Math.round(width / 2);
  const loaded = await Promise.all(ids.map((id) => loadMap(mapUrl(key, id)).catch(() => null)));
  if (paintGen[slot] !== gen) return;
  const weights = ids.map((id) => shares[id] ?? 0);
  const off = document.createElement("canvas");
  off.width = width;
  off.height = height;
  const read = off.getContext("2d", { willReadFrequently: true });
  if (!read) return;
  const acc = new Float32Array(width * height * 3);
  let weightSum = 0;
  loaded.forEach((img, index) => {
    if (!img) return;
    const weight = weights[index];
    read.clearRect(0, 0, width, height);
    read.drawImage(img, 0, 0, width, height);
    const data = read.getImageData(0, 0, width, height).data;
    for (let pixel = 0, channel = 0; pixel < data.length; pixel += 4, channel += 3) {
      acc[channel] += data[pixel] * weight;
      acc[channel + 1] += data[pixel + 1] * weight;
      acc[channel + 2] += data[pixel + 2] * weight;
    }
    weightSum += weight;
  });
  if (paintGen[slot] !== gen || weightSum <= 0) return;
  const out = read.createImageData(width, height);
  const scale = 1 / weightSum;
  for (let pixel = 0, channel = 0; pixel < out.data.length; pixel += 4, channel += 3) {
    out.data[pixel] = acc[channel] * scale;
    out.data[pixel + 1] = acc[channel + 1] * scale;
    out.data[pixel + 2] = acc[channel + 2] * scale;
    out.data[pixel + 3] = 255;
  }
  canvas.hidden = false;
  canvas.width = width;
  canvas.height = height;
  canvas.getContext("2d")?.putImageData(out, 0, 0);
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

function syncMixer() {
  const ids = activeIds();
  const percents = displayPercents(ids);
  document.querySelectorAll<HTMLElement>(".mix-item").forEach((item) => {
    const id = item.dataset.layer || "";
    const on = ids.includes(id);
    item.classList.toggle("on", on);
    item.querySelector("button")?.setAttribute("aria-pressed", on ? "true" : "false");
    const range = item.querySelector("input");
    const pct = item.querySelector(".pct");
    if (!(range instanceof HTMLInputElement) || !(pct instanceof HTMLElement)) return;
    range.hidden = !on || ids.length < 2;
    pct.hidden = !on;
    if (!on) return;
    pct.textContent = `${percents[id]}%`;
    if (document.activeElement !== range) range.value = String(Math.min(99, Math.max(1, percents[id])));
  });
}

function setShare(id: string, percent: number) {
  const ids = activeIds();
  if (!ids.includes(id) || ids.length < 2) return;
  const next = Math.min(99, Math.max(1, percent));
  const others = ids.filter((key) => key !== id);
  const rest = 100 - next;
  const otherSum = others.reduce((total, key) => total + shares[key], 0);
  if (otherSum <= 0) {
    const each = rest / others.length;
    others.forEach((key) => { shares[key] = each; });
  } else {
    others.forEach((key) => { shares[key] = (shares[key] / otherSum) * rest; });
  }
  shares[id] = next;
}

function toggleLayer(id: string) {
  const ids = activeIds();
  if (ids.includes(id)) {
    if (ids.length === 1) return;
    const rest = ids.filter((key) => key !== id);
    const sum = rest.reduce((total, key) => total + shares[key], 0) || 1;
    rest.forEach((key) => { shares[key] = (shares[key] / sum) * 100; });
    delete shares[id];
  } else {
    const count = ids.length + 1;
    ids.forEach((key) => { shares[key] = (shares[key] * (count - 1)) / count; });
    shares[id] = 100 / count;
  }
  syncMixer();
  renderPair();
}

async function init() {
  INFO = await api<Info>("/api/info");
  LAYERS = await api<LayerCatalog>("/api/layers");
  const pick = $("#layerPick");
  for (const item of LAYERS.layers) {
    const row = document.createElement("div");
    row.className = "mix-item";
    row.dataset.layer = item.id;
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = item.label;
    button.addEventListener("click", () => toggleLayer(item.id));
    const range = document.createElement("input");
    range.type = "range";
    range.min = "1";
    range.max = "99";
    range.value = "100";
    range.setAttribute("aria-label", `${item.label} percentage`);
    range.addEventListener("input", () => {
      setShare(item.id, Number(range.value));
      syncMixer();
      writeMixText();
    });
    const pct = document.createElement("span");
    pct.className = "pct";
    row.append(button, range, pct);
    pick.appendChild(row);
  }
  syncMixer();
  new ResizeObserver(() => schedulePaint()).observe($("#maps"));
  void fetch("/names.json").then((response) => response.json()).then((raw: { citation: Record<string, string>; earth: [string, number, number, number, string][]; ocean: [string, number, number, number, string][]; moon: [string, number, number, number, string][]; mars: [string, number, number, number, string][] }) => {
    const parse = (rows: [string, number, number, number, string][]) => rows.map(([n, lat, lon, z, k]) => ({ n, lat, lon, z, k }));
    PLACES = {
      citation: raw.citation,
      earth: parse(raw.earth),
      ocean: parse(raw.ocean),
      moon: parse(raw.moon),
      mars: parse(raw.mars),
    };
    writeMixText();
  }).catch(() => { /* names stay off if the file is missing */ });
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
