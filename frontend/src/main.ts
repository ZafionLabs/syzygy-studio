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

type SafetyCell = {
  hazard: number;
  sunlight: number;
  shadow: number;
  radiation: number;
  slope_deg: number;
  roughness_m: number;
  water: boolean;
  dangers: string[];
  high_risk: string[];
  radiation_note: string;
};

type SafetyGrid = {
  w: number;
  h: number;
  hazard: Uint8Array;
  sun: Uint8Array;
  shadow: Uint8Array;
  rad: Uint8Array;
};

type Priorities = { hazard: number; sun: number; shadow: number; rad: number };

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
const fmtCoords = (lat: number, lon: number) => `${Math.abs(lat).toFixed(3)}°${lat >= 0 ? "N" : "S"}, ${Math.abs(lon).toFixed(3)}°${lon >= 0 ? "E" : "W"}`;
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

function usableLayers(key: string) {
  return activeIds().filter((id) => id === "elevation" || LAYERS?.bodies[key]?.[id]?.available !== false);
}

function mapUrl(key: string, id: string) {
  return id === "elevation" ? `/map/${source()}/${key}.png?v=4` : `/map/layer/${key}/${id}.png?v=4`;
}

const PRESETS: Record<string, Priorities> = {
  balanced: { hazard: 25, sun: 25, shadow: 25, rad: 25 },
  flat: { hazard: 100, sun: 0, shadow: 0, rad: 0 },
  sun: { hazard: 0, sun: 100, shadow: 0, rad: 0 },
  shadow: { hazard: 0, sun: 0, shadow: 100, rad: 0 },
  radiation: { hazard: 0, sun: 0, shadow: 0, rad: 100 },
};

function priorities(): Priorities {
  const num = (id: string) => Number((document.querySelector(id) as HTMLInputElement).value) || 0;
  return { hazard: num("#priHazard"), sun: num("#priSun"), shadow: num("#priShadow"), rad: num("#priRad") };
}

const RISK_LABELS: Record<string, string> = { hazard: "Hazards", sun: "Sunlight", shadow: "Shadows", rad: "Radiation" };

function highRisks(cell: { hazard: number; sun: number; shadow: number; rad: number }) {
  const flags: string[] = [];
  if (cell.hazard >= 0.6) flags.push("hazard");
  if (cell.sun < 0.25) flags.push("sun");
  if (cell.shadow >= 0.55) flags.push("shadow");
  if (cell.rad >= 0.75) flags.push("rad");
  return flags;
}

function showHighRisk(flags: string[]) {
  for (const key of Object.keys(RISK_LABELS)) {
    $(`#risk${key.charAt(0).toUpperCase()}${key.slice(1)}`).hidden = !flags.includes(key);
  }
}

function highRiskSentence(flags: string[]) {
  if (!flags.length) return "";
  return flags.map((key) => `${RISK_LABELS[key]}: high risk.`).join(" ");
}

function factorRows(cell: SafetyCell) {
  const rows: [string, boolean, string][] = [
    ["Hazards", cell.high_risk?.includes("hazard"), `slope ${cell.slope_deg.toFixed(0)}°`],
    ["Sunlight", cell.high_risk?.includes("sunlight"), `${Math.round(cell.sunlight * 100)}% of the year`],
    ["Shadows", cell.high_risk?.includes("shadow"), `${Math.round(cell.shadow * 100)}% in shadow`],
    ["Radiation", cell.high_risk?.includes("radiation"), `${Math.round(cell.radiation * 100)}% of this body's range`],
  ];
  return rows.map(([name, high, detail]) =>
    `<div class="part"><span>${name}</span><span>${high ? `<b class="bad">High risk</b> ` : ""}${esc(detail)}</span></div>`).join("");
}

function landingSafety(cell: { hazard: number; sun: number; shadow: number; rad: number }, weights = priorities()) {
  const sum = weights.hazard + weights.sun + weights.shadow + weights.rad;
  if (sum <= 0) return null;
  const bad = (weights.hazard * cell.hazard + weights.sun * (1 - cell.sun) + weights.shadow * cell.shadow + weights.rad * cell.rad) / sum;
  return (1 - bad) * 100;
}

function safetyRgb(safety: number | null): [number, number, number] {
  if (safety == null) return [40, 48, 58];
  const bad = 1 - safety / 100;
  const mix = (a: [number, number, number], b: [number, number, number], t: number): [number, number, number] => [
    Math.round(a[0] + (b[0] - a[0]) * t),
    Math.round(a[1] + (b[1] - a[1]) * t),
    Math.round(a[2] + (b[2] - a[2]) * t),
  ];
  const green: [number, number, number] = [32, 160, 90];
  const yellow: [number, number, number] = [230, 190, 40];
  const red: [number, number, number] = [210, 48, 48];
  return bad < 0.5 ? mix(green, yellow, bad / 0.5) : mix(yellow, red, (bad - 0.5) / 0.5);
}

const safetyGrids = new Map<string, Promise<SafetyGrid>>();
const safetyReady = new Map<string, SafetyGrid>();
const safetyColored = new Map<string, { key: string; canvas: HTMLCanvasElement }>();

function loadSafety(body: string) {
  let pending = safetyGrids.get(body);
  if (!pending) {
    pending = (async () => {
      const img = await loadMap(`/api/safety/${body}.png?v=1`);
      const canvas = document.createElement("canvas");
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      const ctx = canvas.getContext("2d", { willReadFrequently: true });
      if (!ctx) throw new Error("canvas");
      ctx.drawImage(img, 0, 0);
      const data = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
      const n = canvas.width * canvas.height;
      const hazard = new Uint8Array(n);
      const sun = new Uint8Array(n);
      const shadow = new Uint8Array(n);
      const rad = new Uint8Array(n);
      for (let i = 0; i < n; i++) {
        const p = i * 4;
        hazard[i] = data[p];
        sun[i] = data[p + 1];
        shadow[i] = data[p + 2];
        rad[i] = data[p + 3];
      }
      const grid = { w: canvas.width, h: canvas.height, hazard, sun, shadow, rad };
      safetyReady.set(body, grid);
      return grid;
    })();
    safetyGrids.set(body, pending);
  }
  return pending;
}

function sampleSafety(grid: SafetyGrid, body: string, lat: number, lon: number) {
  const info = bodies()[body];
  const fy = (info.lat_range[1] - lat) / (info.lat_range[1] - info.lat_range[0]);
  let fx = (lon - info.lon_range[0]) / (info.lon_range[1] - info.lon_range[0]);
  fx = ((fx % 1) + 1) % 1;
  const row = Math.min(grid.h - 1, Math.max(0, Math.floor(fy * grid.h)));
  const col = Math.min(grid.w - 1, Math.max(0, Math.floor(fx * grid.w)));
  const i = row * grid.w + col;
  return {
    hazard: grid.hazard[i] / 255,
    sun: grid.sun[i] / 255,
    shadow: grid.shadow[i] / 255,
    rad: (grid.rad[i] - 1) / 254,
  };
}

function colorSafety(body: string, grid: SafetyGrid) {
  const weights = priorities();
  const key = `${weights.hazard},${weights.sun},${weights.shadow},${weights.rad}`;
  const cached = safetyColored.get(body);
  if (cached?.key === key) return cached.canvas;
  const canvas = document.createElement("canvas");
  canvas.width = grid.w;
  canvas.height = grid.h;
  const ctx = canvas.getContext("2d");
  if (!ctx) return canvas;
  const image = ctx.createImageData(grid.w, grid.h);
  const px = image.data;
  for (let i = 0; i < grid.w * grid.h; i++) {
    const safety = landingSafety({
      hazard: grid.hazard[i] / 255,
      sun: grid.sun[i] / 255,
      shadow: grid.shadow[i] / 255,
      rad: (grid.rad[i] - 1) / 254,
    }, weights);
    const rgb = safetyRgb(safety);
    const p = i * 4;
    px[p] = rgb[0];
    px[p + 1] = rgb[1];
    px[p + 2] = rgb[2];
    px[p + 3] = 255;
  }
  ctx.putImageData(image, 0, 0);
  safetyColored.set(body, { key, canvas });
  return canvas;
}

function safetyLegend() {
  const weights = priorities();
  const sum = weights.hazard + weights.sun + weights.shadow + weights.rad;
  const note = sum <= 0
    ? "Move a slider to color safer ground green and less safe ground red."
    : `Hazards ${weights.hazard}, sunlight ${weights.sun}, shadows ${weights.shadow}, radiation ${weights.rad}.`;
  return `<div class="legend-row"><span class="name">Landing safety</span><span class="bar" style="background:linear-gradient(90deg,#20a05a,#e6be28,#d23030)"></span><span>safer</span><span>less safe</span></div><div class="note">${esc(note)}</div>`;
}

function viewOf(slot: string) {
  return views[slot] || (views[slot] = { zoom: 1, panX: 0, panY: 0 });
}

function clampPan(pan: number, size: number, zoom: number) {
  return Math.min(0, Math.max(size * (1 - zoom), pan));
}

function applyView(slot: string) {
  const view = viewOf(slot);
  document.querySelectorAll<HTMLElement>(`#map-${slot} .mapview`).forEach((el) => {
    el.style.transform = `translate(${view.panX}px, ${view.panY}px) scale(${view.zoom})`;
  });
  document.querySelectorAll<HTMLButtonElement>(`#map-${slot} .mapreset`).forEach((el) => {
    el.hidden = view.zoom === 1 && view.panX === 0 && view.panY === 0;
  });
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

const RADIUS_KM: Record<string, number> = { earth: 6371, moon: 1737.4, mars: 3389.5 };
const NOT_LANDMARK = new Set(["crater", "satellite feature", "astronaut-named features", "albedo feature"]);

function distanceKm(key: string, lat: number, lon: number, place: Place) {
  const radius = RADIUS_KM[key] ?? 6371;
  const rad = Math.PI / 180;
  const dLat = (place.lat - lat) * rad;
  const dLon = (place.lon - lon) * rad;
  const a = Math.sin(dLat / 2) ** 2 + Math.cos(lat * rad) * Math.cos(place.lat * rad) * Math.sin(dLon / 2) ** 2;
  return 2 * radius * Math.asin(Math.min(1, Math.sqrt(a)));
}

function nearestKind(key: string, lat: number, lon: number, accept: (place: Place) => boolean) {
  const groups = key === "earth" ? [PLACES?.earth || [], PLACES?.ocean || []] : [PLACES?.[key as "moon" | "mars"] || []];
  let best: Place | null = null;
  let bestKm = Infinity;
  for (const group of groups) {
    for (const place of group) {
      if (!accept(place)) continue;
      const km = distanceKm(key, lat, lon, place);
      if (km < bestKm) {
        best = place;
        bestKm = km;
      }
    }
  }
  return best ? { place: best, km: bestKm } : null;
}

function away(km: number, what: string) {
  if (km < 0.05) return `at ${what}`;
  const dist = km < 1 ? `${Math.round(km * 1000)} m` : km < 10 ? `${km.toFixed(1)} km` : `${Math.round(km).toLocaleString()} km`;
  return `${dist} from ${what}`;
}

function howFar(km: number) {
  if (km < 0.05) return "here";
  if (km < 1) return `${Math.round(km * 1000)} m away`;
  if (km < 10) return `${km.toFixed(1)} km away`;
  return `${Math.round(km).toLocaleString()} km away`;
}

function craterTitle(place: Place) {
  return /crater/i.test(place.n) ? place.n : `${place.n} crater`;
}

function fillSpotFacts(site: Site) {
  const key = bodyKey(site.body);
  const landmark = nearestKind(key, site.lat, site.lon, (place) => !NOT_LANDMARK.has(place.k));
  const crater = nearestKind(key, site.lat, site.lon, (place) => place.k === "crater");
  $("#spotCoords").textContent = fmtCoords(site.lat, site.lon);
  $("#spotLandmark").textContent = landmark ? `Nearest landmark: ${placeLabel(landmark.place)}, ${howFar(landmark.km)}` : "";
  $("#spotCrater").textContent = crater ? away(crater.km, `the center of ${craterTitle(crater.place)}`) : "";
}

function parseLatLon(text: string) {
  const tokens = text.trim().toUpperCase().replace(/°/g, "").split(/[,/]+|\s+/).filter(Boolean);
  const nums: { n: number; hemi: string }[] = [];
  for (let i = 0; i < tokens.length; i += 1) {
    const match = tokens[i].match(/^([+-]?\d+(?:\.\d+)?)([NSEW])?$/);
    if (!match) return null;
    let hemi = match[2] || "";
    if (!hemi && tokens[i + 1] && /^[NSEW]$/.test(tokens[i + 1])) {
      hemi = tokens[i + 1];
      i += 1;
    }
    let n = Number(match[1]);
    if (hemi === "S" || hemi === "W") n = -Math.abs(n);
    if (hemi === "N" || hemi === "E") n = Math.abs(n);
    nums.push({ n, hemi });
  }
  if (nums.length !== 2) return null;
  const swapped = nums[0].hemi === "E" || nums[0].hemi === "W" || nums[1].hemi === "N" || nums[1].hemi === "S";
  const lat = swapped ? nums[1].n : nums[0].n;
  const lon = swapped ? nums[0].n : nums[1].n;
  if (lat < -90 || lat > 90 || lon < -180 || lon > 180) return null;
  return { lat, lon };
}

function fillCoordBody() {
  const sel = $<HTMLSelectElement>("#coordBody");
  const prev = sel.value;
  sel.innerHTML = "";
  const keys = earthMode() ? Object.values(slots()).filter((key) => key !== "earth") : Object.values(slots());
  for (const key of keys) sel.add(new Option(bodies()[key].name, key));
  if (Array.from(sel.options).some((option) => option.value === prev)) sel.value = prev;
}

function goToCoordinates() {
  const parsed = parseLatLon(val("#coordText"));
  if (!parsed) {
    $("#status").textContent = "Enter coordinates like 18.1 N, 18.4 W or 18.1, -18.4.";
    return;
  }
  void runQuery(val("#coordBody"), parsed.lat.toFixed(3), parsed.lon.toFixed(3));
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

let cardPoint = { x: 0, y: 0 };

function moveCard(ev: MouseEvent) {
  cardPoint = { x: ev.clientX, y: ev.clientY };
  const card = $<HTMLElement>("#placecard");
  const width = card.offsetWidth || 220;
  const height = card.offsetHeight || 180;
  let left = ev.clientX + 14;
  let top = ev.clientY + 16;
  if (left + width > window.innerWidth - 8) left = Math.max(8, ev.clientX - width - 14);
  if (top + height > window.innerHeight - 8) top = Math.max(8, ev.clientY - height - 16);
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

function featurePercent(row: Breakdown) {
  return 100 * Math.exp(-Math.sqrt(Math.max(0, row.contribution)));
}

function shortFeature(label: string) {
  return label
    .replace("Elevation at the point", "Elevation")
    .replace("Average elevation within ", "Average, ")
    .replace("Roughness (std-dev of elevation) within ", "Roughness, ")
    .replace("Regional slope over ", "Slope, ");
}

function showTwin(m: Match | null) {
  const panel = $<HTMLElement>("#twinPanel");
  const ranks = $<HTMLElement>("#twinRanks");
  const parts = $<HTMLElement>("#twinParts");
  if (!m) {
    panel.hidden = true;
    ranks.innerHTML = "";
    parts.innerHTML = "";
    $<HTMLElement>("#twinSafety").innerHTML = "";
    return;
  }
  const token = ++twinToken;
  panel.hidden = false;
  $("#twinRoleA").textContent = `Your point on ${m.a.body}`;
  $("#twinRoleB").textContent = `Match on ${m.b.body}`;
  $("#twinSub").textContent = `match #${m.rank}`;
  $("#twinSim").textContent = `${m.similarity.toFixed(1)}%`;
  $("#twinDiff").textContent = `height difference ${m.elevation_difference_m >= 0 ? "+" : ""}${fmtM(m.elevation_difference_m)}`;
  parts.innerHTML = m.breakdown.map((row) =>
    `<div class="part"><span>${esc(shortFeature(row.label))}</span><b>${featurePercent(row).toFixed(1)}%</b></div>`).join("");
  ranks.innerHTML = "";
  for (const match of shownMatches.matches) {
    const button = document.createElement("button");
    button.type = "button";
    button.classList.toggle("on", match.rank === m.rank);
    button.dataset.rank = String(match.rank);
    button.textContent = rankSafetyText(match);
    button.addEventListener("click", () => {
      const row = $<HTMLTableSectionElement>("#results tbody").children[match.rank - 1];
      row?.dispatchEvent(new Event("click"));
    });
    ranks.appendChild(button);
  }
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
  void fillTwinSafety(m, token);
}

function rankSafetySuffix(match: Match) {
  const grid = safetyReady.get(bodyKey(match.b.body));
  if (!grid) return "";
  const score = landingSafety(sampleSafety(grid, bodyKey(match.b.body), match.b.lat, match.b.lon));
  return score == null ? "" : ` · ${score.toFixed(0)}% safe`;
}

function rankSafetyText(match: Match) {
  return `#${match.rank} ${match.similarity.toFixed(1)}%${rankSafetySuffix(match)}`;
}

function refreshRankSafety() {
  document.querySelectorAll<HTMLButtonElement>("#twinRanks button").forEach((button) => {
    const match = shownMatches.matches.find((item) => item.rank === Number(button.dataset.rank));
    if (match) button.textContent = rankSafetyText(match);
  });
}

async function fillTwinSafety(match: Match, token: number) {
  const box = $<HTMLElement>("#twinSafety");
  box.innerHTML = "";
  try {
    const [left, right] = await Promise.all([
      api<SafetyCell>(`/api/safety?body=${bodyKey(match.a.body)}&lat=${match.a.lat}&lon=${match.a.lon}`),
      api<SafetyCell>(`/api/safety?body=${bodyKey(match.b.body)}&lat=${match.b.lat}&lon=${match.b.lon}`),
    ]);
    if (token !== twinToken) return;
    const card = (title: string, cell: SafetyCell) => {
      const score = landingSafety({ hazard: cell.hazard, sun: cell.sunlight, shadow: cell.shadow, rad: cell.radiation });
      const heading = score == null ? title : `${title}: ${score.toFixed(0)}% safe with your priorities`;
      return `<div><h3>${esc(heading)}</h3><div class="twin-parts">${factorRows(cell)}</div><ul>${cell.dangers.map((line) => `<li>${esc(line)}</li>`).join("")}</ul><p class="note">${esc(cell.radiation_note)}</p></div>`;
    };
    box.innerHTML = card(match.a.body, left) + card(match.b.body, right);
    showHighRisk(highRisks({ hazard: right.hazard, sun: right.sunlight, shadow: right.shadow, rad: right.radiation }));
  } catch {
    if (token === twinToken) box.innerHTML = "";
  }
}

function watchSpot(el: HTMLElement, site: Site, match: Match | null) {
  el.addEventListener("mouseenter", (ev) => openCard(ev, site, match));
  el.addEventListener("mousemove", (ev) => moveCard(ev));
  el.addEventListener("mouseleave", () => hideCard());
}

function openCard(ev: MouseEvent, site: Site, match: Match | null) {
  const token = ++cardToken;
  const card = $<HTMLElement>("#placecard");
  const img = $<HTMLImageElement>("#spotImg");
  $("#spotName").textContent = placeName(site);
  $("#spotSafety").textContent = "";
  fillSpotFacts(site);
  img.alt = placeName(site);
  img.src = `/api/place-image?body=${bodyKey(site.body)}&lat=${site.lat.toFixed(3)}&lon=${site.lon.toFixed(3)}&km=120`;
  img.onload = () => { if (token === cardToken) moveCard({ clientX: cardPoint.x, clientY: cardPoint.y } as MouseEvent); };
  card.hidden = false;
  moveCard(ev);
  showTwin(match ?? shownMatches.matches[0] ?? null);
  void api<SafetyCell>(`/api/safety?body=${bodyKey(site.body)}&lat=${site.lat}&lon=${site.lon}`).then((cell) => {
    if (token !== cardToken) return;
    const levels = { hazard: cell.hazard, sun: cell.sunlight, shadow: cell.shadow, rad: cell.radiation };
    const score = landingSafety(levels);
    const risks = highRiskSentence(highRisks(levels));
    $("#spotSafety").textContent = `${score == null ? "" : `${score.toFixed(0)}% safe with your priorities. `}${risks}`.trim();
    showHighRisk(highRisks(levels));
  }).catch(() => undefined);
  void placeFor(bodyKey(site.body), site.lat, site.lon).then((found) => {
    if (token !== cardToken) return;
    $("#spotName").textContent = found.name;
    img.alt = found.name;
    if (found.photo) img.src = found.photo;
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
  const choices = Object.keys(bodies()).filter((k) => k !== "earth");
  const chosen = choices.includes(val("#earthFrom")) ? val("#earthFrom") : (choices[0] || "moon");
  return { A: chosen, B: "earth" };
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

function mapWrapOf(ev: Event) {
  const target = ev.target;
  if (!(target instanceof Element)) return null;
  const box = target.closest(".mapwrap");
  return box instanceof HTMLElement ? box : null;
}

function toLatLon(body: string, ev: MouseEvent, box?: HTMLElement | null) {
  const b = bodies()[body];
  const slot = slotOf(body);
  const view = slot ? viewOf(slot) : { zoom: 1, panX: 0, panY: 0 };
  const rect = (box || mapWrapOf(ev))?.getBoundingClientRect();
  if (!rect || rect.width === 0) return null;
  const fx = Math.min(1, Math.max(0, (ev.clientX - rect.left - view.panX) / (rect.width * view.zoom)));
  const fy = Math.min(1, Math.max(0, (ev.clientY - rect.top - view.panY) / (rect.height * view.zoom)));
  return {
    lat: b.lat_range[1] - fy * (b.lat_range[1] - b.lat_range[0]),
    lon: b.lon_range[0] + fx * (b.lon_range[1] - b.lon_range[0]),
  };
}

function place(body: string, lat: number, lon: number, cls: string, text: string, title?: string) {
  const slot = slotOf(body);
  if (!slot) return [];
  const b = bodies()[body];
  const views = document.querySelectorAll<HTMLElement>(`#map-${slot} .mapview`);
  return Array.from(views, (view) => {
    const el = document.createElement("div");
    el.className = `marker ${cls}`;
    el.style.left = `${((lon - b.lon_range[0]) / (b.lon_range[1] - b.lon_range[0])) * 100}%`;
    el.style.top = `${((b.lat_range[1] - lat) / (b.lat_range[1] - b.lat_range[0])) * 100}%`;
    el.textContent = text;
    if (title) el.dataset.label = title;
    el.style.transform = `translate(-50%, -50%) scale(${1 / viewOf(slot).zoom})`;
    view.appendChild(el);
    return el;
  });
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
      <td>${featurePercent(r).toFixed(1)}%</td><td>${(r.weight_share * 100).toFixed(1)}%</td><td>${r.contribution.toFixed(5)}</td></tr>`).join("");
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
    place(bodyKey(origin.body), origin.lat, origin.lon, "origin", "", placeName(origin))
      .forEach((picked) => watchSpot(picked, origin, matches[0] ?? null));
  }
  const relief = (p: Site) => p.features.relief_75km_m;

  for (const m of matches) {
    const groups = [];
    if (!origin) groups.push(place(bodyKey(m.a.body), m.a.lat, m.a.lon, "", String(m.rank), placeName(m.a)));
    groups.push(place(bodyKey(m.b.body), m.b.lat, m.b.lon, "", String(m.rank), placeName(m.b)));
    const live = groups.flat();
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
    groups.forEach((markers, index) => markers.forEach((mk) => {
      watchSpot(mk, points[index] ?? m.b, m);
      mk.addEventListener("click", (e) => { e.stopPropagation(); activate(); tr.scrollIntoView({ block: "nearest" }); });
    }));
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
  let pressed: HTMLElement | null = null;
  let drag: { x: number; y: number; panX: number; panY: number; moved: boolean; pointer: number; box: HTMLElement } | null = null;
  wrap.addEventListener("mousemove", (ev) => {
    const body = slots()[slot];
    const point = !body || drag?.moved ? null : toLatLon(body, ev);
    if (!point) return;
    const { lat, lon } = point;
    clearTimeout(pending);
    pending = setTimeout(async () => {
      try {
        const h = await api<{ surface: string; safety?: SafetyCell }>(`/api/height?body=${body}&lat=${lat}&lon=${lon}&source=${source()}`);
        const safety = h.safety;
        const levels = safety ? { hazard: safety.hazard, sun: safety.sunlight, shadow: safety.shadow, rad: safety.radiation } : null;
        const flags = levels ? highRisks(levels) : [];
        showHighRisk(flags);
        const score = levels ? landingSafety(levels) : null;
        const scoreText = score == null ? "" : ` Landing safety ${score.toFixed(0)}% with your priorities.`;
        const risks = flags.length ? ` ${highRiskSentence(flags)}` : "";
        $(`#hover-${slot}`).textContent = `${fmtLL(lat, lon)}: ${h.surface}.${scoreText}${risks}`;
      } catch { /* hover is best-effort */ }
    }, 60);
  });
  wrap.addEventListener("pointerdown", (ev) => {
    const box = mapWrapOf(ev);
    if (!box || (ev.target as HTMLElement).closest("button, .marker") || pinch) return;
    const view = viewOf(slot);
    pressed = box;
    drag = { x: ev.clientX, y: ev.clientY, panX: view.panX, panY: view.panY, moved: false, pointer: ev.pointerId, box };
    wrap.setPointerCapture(ev.pointerId);
  });
  wrap.addEventListener("pointermove", (ev) => {
    if (!drag || drag.pointer !== ev.pointerId) return;
    const dx = ev.clientX - drag.x;
    const dy = ev.clientY - drag.y;
    if (Math.hypot(dx, dy) > 4) drag.moved = true;
    if (!drag.moved) return;
    const view = viewOf(slot);
    const rect = drag.box.getBoundingClientRect();
    view.panX = clampPan(drag.panX + dx, rect.width, view.zoom);
    view.panY = clampPan(drag.panY + dy, rect.height, view.zoom);
    applyView(slot);
  });
  wrap.addEventListener("pointerup", (ev) => {
    if (!drag || drag.pointer !== ev.pointerId) return;
    if (drag.moved) wrap.dataset.dragged = "1";
    drag = null;
  });
  const zoomAt = (next: number, mx: number, my: number, rect: DOMRect) => {
    const view = viewOf(slot);
    const scale = next / view.zoom;
    view.panX = clampPan(mx - (mx - view.panX) * scale, rect.width, next);
    view.panY = clampPan(my - (my - view.panY) * scale, rect.height, next);
    view.zoom = next;
    applyView(slot);
  };
  // A trackpad pinch arrives as a wheel event with ctrlKey. A two-finger scroll does not, so it moves the page.
  wrap.addEventListener("wheel", (ev) => {
    const box = mapWrapOf(ev);
    if (!ev.ctrlKey || !box) return;
    ev.preventDefault();
    const rect = box.getBoundingClientRect();
    const next = Math.min(28, Math.max(1, viewOf(slot).zoom * (ev.deltaY < 0 ? 1.18 : 1 / 1.18)));
    zoomAt(next, ev.clientX - rect.left, ev.clientY - rect.top, rect);
  }, { passive: false });
  let pinch: { dist: number; zoom: number; mx: number; my: number; rect: DOMRect } | null = null;
  const fingerDistance = (ev: TouchEvent) => Math.hypot(ev.touches[0].clientX - ev.touches[1].clientX, ev.touches[0].clientY - ev.touches[1].clientY);
  wrap.addEventListener("touchstart", (ev) => {
    const box = mapWrapOf(ev);
    if (ev.touches.length !== 2 || !box) return;
    const rect = box.getBoundingClientRect();
    pinch = {
      dist: fingerDistance(ev),
      zoom: viewOf(slot).zoom,
      mx: (ev.touches[0].clientX + ev.touches[1].clientX) / 2 - rect.left,
      my: (ev.touches[0].clientY + ev.touches[1].clientY) / 2 - rect.top,
      rect,
    };
  }, { passive: true });
  wrap.addEventListener("touchmove", (ev) => {
    if (ev.touches.length !== 2 || !pinch) return;
    const dist = fingerDistance(ev);
    if (Math.abs(dist - pinch.dist) < 10) return;
    ev.preventDefault();
    zoomAt(Math.min(28, Math.max(1, pinch.zoom * (dist / pinch.dist))), pinch.mx, pinch.my, pinch.rect);
  }, { passive: false });
  wrap.addEventListener("touchend", () => { pinch = null; });
  wrap.addEventListener("touchcancel", () => { pinch = null; });
  wrap.addEventListener("click", (ev) => {
    if ((ev.target as HTMLElement).closest(".mapreset")) {
      ev.stopPropagation();
      views[slot] = { zoom: 1, panX: 0, panY: 0 };
      applyView(slot);
      return;
    }
    if (wrap.dataset.dragged === "1") {
      wrap.dataset.dragged = "";
      return;
    }
    const body = slots()[slot];
    const point = body ? toLatLon(body, ev, pressed) : null;
    pressed = null;
    if (!point || (earthMode() && body === "earth")) return;
    void runQuery(body, point.lat.toFixed(3), point.lon.toFixed(3));
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
  $<HTMLElement>("#fromBody").hidden = !onEarth;
  $<HTMLElement>("#earthFromLabel").hidden = true;
  $("#topBtn").textContent = onEarth ? "Show closest places on Earth" : "Show global top matches";
  const from = $<HTMLSelectElement>("#earthFrom");
  const prev = from.value;
  from.innerHTML = "";
  Object.keys(bodies()).filter((k) => k !== "earth").forEach((k) => from.add(new Option(bodies()[k].name, k)));
  if (Array.from(from.options).some((o) => o.value === prev)) from.value = prev;
  document.querySelectorAll<HTMLButtonElement>("#fromBody button").forEach((button) => {
    const on = button.dataset.body === from.value;
    button.classList.toggle("on", on);
    button.setAttribute("aria-pressed", on ? "true" : "false");
  });
  for (const slot of ["A", "B", "C"]) $<HTMLElement>(`#slot${slot}`).hidden = !(slot in slots());
  fillCoordBody();
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
    syncFrames(slot, key);
    $(`#title${slot}`).textContent = b.name;
    $(`#hover-${slot}`).textContent = earthMode()
      ? (key === "earth" ? "Closest places on Earth show up here." : "Hover to read elevation. Click to find the closest places on Earth.")
      : `Hover to read elevation. Click to search ${bodies()[otherOf(key)].name}.`;
    for (const lm of b.landmarks) {
      const site = { body: b.name, lat: lm.lat, lon: lm.lon, surface: "", underwater: lm.kind === "lowest" && key === "earth", features: {} };
      place(key, lm.lat, lm.lon, "lm", lm.kind === "highest" ? "▲" : "▼", lm.name).forEach((marker) => watchSpot(marker, site, null));
    }
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
  const shown = activeIds().filter((id) => id !== "elevation").map((id) => Object.values(slots()).map((key) => {
    const item = LAYERS?.bodies[key]?.[id];
    if (!item) return "";
    const name = bodies()[key].name;
    if (!item.available) return `<p><b>${esc(name)} — ${esc(layerLabel(id))}.</b> ${esc(item.note)}</p>`;
    return `<p><b>${esc(name)} — ${esc(item.title || "")}.</b> ${esc(item.source || "")}</p>` +
      `<p>${esc(item.note)}</p>` +
      (item.citation ? `<p>${esc(item.citation)}</p>` : "");
  }).join("")).join("");
  const mixLine = activeIds().map((id) => layerLabel(id)).join(", ");
  const mapHeading = `<p><b>These maps.</b> ${esc(mixLine)}. Each selected map is drawn on its own, one under the other. A click still searches the elevation data.</p>${shown}`;
  const cited = PLACES
    ? `<p><b>Names on a selected point.</b> ${esc(PLACES.citation.earth)} ${esc(PLACES.citation.ocean)} ${esc(PLACES.citation.moon)}</p>`
    : "";
  $("#calib").innerHTML = mapHeading + elevationNotes + cited;
  $("#layerNote").textContent = `${mixLine}. Each map is shown on its own, one under the other. Hover a point to see its name and a picture. A click still searches elevation.`;
  for (const [slot, key] of Object.entries(slots())) {
    syncFrames(slot, key);
    const info = bodies()[key];
    $(`#range${slot}`).textContent = usableLayers(key).length
      ? activeIds().map((id) => (id === "elevation" ? info.source_name : (LAYERS?.bodies[key]?.[id]?.title || layerLabel(id)))).join(" · ")
      : "";
  }
  schedulePaint();
}

function legendHtml(key: string) {
  return activeIds().map((id) => legendOne(key, id)).join("");
}

function legendOne(key: string, id: string) {
  const b = bodies()[key];
  if (id === "elevation") {
    return `<div class="legend-row"><span class="name">Elevation</span><span class="bar" style="background:${GRADIENTS[key] || GRADIENTS.moon}"></span><span>${esc(fmtM(b.color_min_m))}</span><span>${esc(fmtM(b.color_max_m))}</span></div>`;
  }
  const item = LAYERS?.bodies[key]?.[id];
  if (!item?.available) {
    return `<div class="legend-row"><span class="name">${esc(layerLabel(id))} is not available for this body</span></div>`;
  }
  const ramp = RAMPS[item.ramp || ""] || GRADIENTS[key] || GRADIENTS.moon;
  const bar = item.ramp === "photo" ? "" : `<span class="bar" style="background:${ramp}"></span><span>${esc(item.low || "")}</span><span>${esc(item.high || "")}</span>`;
  return `<div class="legend-row"><span class="name">${esc(item.title || layerLabel(id))}</span>${bar}</div>`;
}

function syncFrames(slot: string, key: string) {
  const stack = $(`#map-${slot}`);
  const ids = usableLayers(key);
  stack.querySelectorAll<HTMLElement>(".mapframe").forEach((frame) => {
    const layer = frame.dataset.layer || "";
    if (layer !== "safety" && !ids.includes(layer)) frame.remove();
  });
  stack.querySelector(":scope > .note")?.remove();
  if (!ids.length) {
    stack.insertAdjacentHTML("beforeend", `<p class="note">None of the selected maps are available for this body.</p>`);
    return;
  }
  for (const id of ids) {
    let frame = stack.querySelector<HTMLElement>(`.mapframe[data-layer="${id}"]`);
    if (!frame) {
      frame = document.createElement("div");
      frame.className = "mapframe";
      frame.dataset.layer = id;
      frame.innerHTML = `<div class="frametitle"></div><div class="mapwrap"><div class="mapview"><canvas class="blend"></canvas></div><button type="button" class="mapreset" hidden>Reset view</button></div><div class="legend"></div>`;
      stack.appendChild(frame);
    }
    const title = id === "elevation" ? "Elevation" : (LAYERS?.bodies[key]?.[id]?.title || layerLabel(id));
    const heading = frame.querySelector(".frametitle");
    if (heading) heading.textContent = title;
    const legend = frame.querySelector(".legend");
    if (legend) legend.innerHTML = legendOne(key, id);
    stack.appendChild(frame);
  }
  let safety = stack.querySelector<HTMLElement>(`.mapframe[data-layer="safety"]`);
  if (!safety) {
    safety = document.createElement("div");
    safety.className = "mapframe";
    safety.dataset.layer = "safety";
    safety.innerHTML = `<div class="frametitle">Landing safety</div><div class="mapwrap"><div class="mapview"><canvas class="blend"></canvas></div><button type="button" class="mapreset" hidden>Reset view</button></div><div class="legend"></div>`;
    stack.prepend(safety);
  }
  const legend = safety.querySelector(".legend");
  if (legend) legend.innerHTML = safetyLegend();
  stack.prepend(safety);
  applyView(slot);
}

function schedulePaint() {
  if (paintQueued) return;
  paintQueued = true;
  requestAnimationFrame(() => {
    paintQueued = false;
    for (const [slot, key] of Object.entries(slots())) {
      void paintBlend(slot, key);
      void paintSafety(slot, key);
    }
  });
}

async function paintBlend(slot: string, key: string) {
  const gen = (paintGen[slot] = (paintGen[slot] || 0) + 1);
  const frames = Array.from(document.querySelectorAll<HTMLElement>(`#map-${slot} .mapframe`));
  await Promise.all(frames.map(async (frame) => {
    const id = frame.dataset.layer || "";
    if (id === "safety") return;
    const canvas = frame.querySelector("canvas");
    if (!canvas) return;
    const img = await loadMap(mapUrl(key, id)).catch(() => null);
    if (paintGen[slot] !== gen || !img) return;
    const width = Math.max(480, Math.round((canvas.clientWidth || $(`#map-${slot}`).clientWidth) * Math.min(window.devicePixelRatio || 1, 2)));
    const height = Math.round(width / 2);
    canvas.hidden = false;
    canvas.width = width;
    canvas.height = height;
    canvas.getContext("2d")?.drawImage(img, 0, 0, width, height);
  }));
}

async function paintSafety(slot: string, key: string) {
  const frame = document.querySelector<HTMLElement>(`#map-${slot} .mapframe[data-layer="safety"]`);
  const canvas = frame?.querySelector("canvas");
  if (!frame || !canvas) return;
  const legend = frame.querySelector(".legend");
  if (legend) legend.innerHTML = safetyLegend();
  try {
    const grid = await loadSafety(key);
    const colored = colorSafety(key, grid);
    const width = Math.max(480, Math.round((canvas.clientWidth || $(`#map-${slot}`).clientWidth) * Math.min(window.devicePixelRatio || 1, 2)));
    const height = Math.round(width / 2);
    canvas.hidden = false;
    canvas.width = width;
    canvas.height = height;
    canvas.getContext("2d")?.drawImage(colored, 0, 0, width, height);
    refreshRankSafety();
  } catch {
    canvas.hidden = true;
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

function syncMixer() {
  const ids = activeIds();
  document.querySelectorAll<HTMLElement>(".mix-item").forEach((item) => {
    const on = ids.includes(item.dataset.layer || "");
    item.classList.toggle("on", on);
    item.querySelector("button")?.setAttribute("aria-pressed", on ? "true" : "false");
  });
}

function toggleLayer(id: string) {
  if (shares[id]) {
    if (activeIds().length === 1) return;
    delete shares[id];
  } else {
    shares[id] = 1;
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
    row.append(button);
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
  const writePriority = () => {
    const weights = priorities();
    $("#priHazardOut").textContent = String(weights.hazard);
    $("#priSunOut").textContent = String(weights.sun);
    $("#priShadowOut").textContent = String(weights.shadow);
    $("#priRadOut").textContent = String(weights.rad);
  };
  document.querySelectorAll<HTMLInputElement>("#safetyBar input").forEach((input) => {
    input.addEventListener("input", () => {
      writePriority();
      schedulePaint();
      refreshRankSafety();
    });
  });
  document.querySelectorAll<HTMLButtonElement>("#safetyBar button").forEach((button) => {
    button.addEventListener("click", () => {
      const preset = PRESETS[button.dataset.preset || ""];
      if (!preset) return;
      $<HTMLInputElement>("#priHazard").value = String(preset.hazard);
      $<HTMLInputElement>("#priSun").value = String(preset.sun);
      $<HTMLInputElement>("#priShadow").value = String(preset.shadow);
      $<HTMLInputElement>("#priRad").value = String(preset.rad);
      writePriority();
      schedulePaint();
      refreshRankSafety();
    });
  });
  renderPair();
  renderValidation();
  void loadSearches();
  $("#topBtn").addEventListener("click", () => void runTop());
  document.querySelectorAll<HTMLButtonElement>("#fromBody button").forEach((button) => {
    button.addEventListener("click", () => {
      const body = button.dataset.body || "moon";
      if (body === val("#earthFrom")) return;
      $<HTMLSelectElement>("#earthFrom").value = body;
      lastQuery = null;
      renderPair();
      $<HTMLElement>("#results").hidden = true;
      $("#steps").innerHTML = "";
      $<HTMLElement>("#breakdownWrap").hidden = true;
      $("#status").textContent = `Click a spot on ${bodies()[body].name} to find the closest matches on Earth.`;
    });
  });
  $("#coordGo").addEventListener("click", () => goToCoordinates());
  $<HTMLInputElement>("#coordText").addEventListener("keydown", (ev) => {
    if (ev.key === "Enter") goToCoordinates();
  });
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
