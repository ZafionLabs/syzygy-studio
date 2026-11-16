export type WorldSpot = { body: string; lat: number; lon: number };

let spots: { a: WorldSpot; b: WorldSpot } | null = null;
let side: "a" | "b" = "b";
let wired = false;

function $<T extends Element = HTMLElement>(id: string) {
  const el = document.querySelector(id);
  if (!el) throw new Error(`missing ${id}`);
  return el as T;
}

function bodyKey(name: string) {
  return name.toLowerCase();
}

function fmtLL(lat: number, lon: number) {
  const wrapped = ((lon + 180) % 360 + 360) % 360 - 180;
  return `${Math.abs(lat).toFixed(2)}°${lat >= 0 ? "N" : "S"}, ${Math.abs(wrapped).toFixed(2)}°${wrapped >= 0 ? "E" : "W"}`;
}

function currentSpot() {
  return spots ? spots[side] : null;
}

function mapsQuery(lat: number, lon: number) {
  const wrapped = ((lon + 180) % 360 + 360) % 360 - 180;
  return `${lat.toFixed(5)},${wrapped.toFixed(5)}`;
}

function mapsEmbed(lat: number, lon: number) {
  const query = mapsQuery(lat, lon);
  return `https://www.google.com/maps/embed?origin=mfe&pb=${encodeURIComponent(`!1m3!2m1!1s${query}!6i12`)}`;
}

function mapsUrl(lat: number, lon: number) {
  return `https://www.google.com/maps/search/?api=1&query=${mapsQuery(lat, lon)}`;
}

function showSpot() {
  const panel = $<HTMLElement>("#worldPanel");
  const spot = currentSpot();
  const frame = $<HTMLIFrameElement>("#googleMap");
  const live = $<HTMLAnchorElement>("#earthLive");
  const hint = $<HTMLElement>("#worldHint");
  if (!spot) {
    panel.hidden = true;
    frame.removeAttribute("src");
    return;
  }
  panel.hidden = false;
  $<HTMLElement>("#worldWhere").textContent = `${spot.body} · ${fmtLL(spot.lat, spot.lon)}`;
  if (bodyKey(spot.body) !== "earth") {
    frame.hidden = true;
    frame.removeAttribute("src");
    live.hidden = true;
    hint.textContent = `Google Maps shows Earth. This place is on ${spot.body}.`;
    return;
  }
  frame.hidden = false;
  frame.src = mapsEmbed(spot.lat, spot.lon);
  live.hidden = false;
  live.href = mapsUrl(spot.lat, spot.lon);
  hint.textContent = "This is Google Maps at the selected Earth place.";
}

function wire() {
  if (wired) return;
  wired = true;
  $<HTMLElement>("#worldPlace").addEventListener("click", (event) => {
    const button = (event.target as HTMLElement).closest("button");
    if (!(button instanceof HTMLButtonElement)) return;
    side = button.dataset.side === "a" ? "a" : "b";
    document.querySelectorAll("#worldPlace button").forEach((item) => item.classList.toggle("on", item === button));
    showSpot();
  });
}

export function setPreviewWorld(next: { a: WorldSpot; b: WorldSpot } | null) {
  spots = next;
  wire();
  showSpot();
}

export function refreshWorldMap() {
  showSpot();
}
