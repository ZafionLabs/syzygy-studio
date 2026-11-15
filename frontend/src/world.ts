import * as THREE from "three";

export type WorldSpot = { body: string; lat: number; lon: number };

type Ground = {
  rows: number;
  cols: number;
  north_km: number;
  east_km: number;
  radius_km: number;
  lat: number;
  lon: number;
  low_m: number;
  high_m: number;
  elevation_m: number[];
};

type LayerFact = { title: string; low: string; high: string; ramp: string };

const DRIVE_KM_S = 5;
const SPRINT_KM_S = 16;
const CLEARANCE_KM = 0.004;

let spots: { a: WorldSpot; b: WorldSpot } | null = null;
let side: "a" | "b" = "b";
let layer = "elevation";
let stamp = 4;
let buildToken = 0;
let colorToken = 0;
let renderer: THREE.WebGLRenderer | null = null;
let scene: THREE.Scene | null = null;
let camera: THREE.PerspectiveCamera | null = null;
let mesh: THREE.Mesh | null = null;
let rover: THREE.Group | null = null;
let ground: Ground | null = null;
let heights: Float32Array | null = null;
let facts: Record<string, Record<string, LayerFact>> = {};
let raf = 0;
let yaw = 0;
let orbit = 0.42;
const place = { north: 0, east: 0 };
const keys = new Set<string>();
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

function ensureRenderer() {
  if (renderer) return;
  const canvas = $<HTMLCanvasElement>("#worldCanvas");
  renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x070b12);
  scene.fog = new THREE.Fog(0x070b12, 80, 420);
  camera = new THREE.PerspectiveCamera(62, 1, 0.0004, 400);
  scene.add(new THREE.AmbientLight(0xffffff, 0.72));
  const sun = new THREE.DirectionalLight(0xfff4e0, 1.8);
  sun.position.set(80, 140, 40);
  scene.add(sun);
  rover = makeRover();
  scene.add(rover);
  const resize = () => {
    const width = canvas.clientWidth || 800;
    const height = canvas.clientHeight || 460;
    renderer?.setSize(width, height, false);
    if (camera) {
      camera.aspect = width / Math.max(1, height);
      camera.updateProjectionMatrix();
    }
  };
  new ResizeObserver(resize).observe(canvas);
  resize();
}

function heightMetres(northKm: number, eastKm: number) {
  const field = ground;
  const samples = heights;
  if (!field || !samples) return 0;
  const col = (eastKm / field.east_km + 0.5) * (field.cols - 1);
  const row = (0.5 - northKm / field.north_km) * (field.rows - 1);
  const c0 = Math.max(0, Math.min(field.cols - 2, Math.floor(col)));
  const r0 = Math.max(0, Math.min(field.rows - 2, Math.floor(row)));
  const tc = col - c0;
  const tr = row - r0;
  const at = (r: number, c: number) => samples[r * field.cols + c] ?? 0;
  return at(r0, c0) * (1 - tr) * (1 - tc) + at(r0, c0 + 1) * (1 - tr) * tc + at(r0 + 1, c0) * tr * (1 - tc) + at(r0 + 1, c0 + 1) * tr * tc;
}

function surface(northKm: number, eastKm: number) {
  const field = ground;
  const metres = heightMetres(northKm, eastKm);
  const drop = field ? (northKm * northKm + eastKm * eastKm) / (2 * field.radius_km) : 0;
  return { x: eastKm, y: metres / 1000 - drop, z: -northKm };
}

function makeRover() {
  const group = new THREE.Group();
  const hull = new THREE.Mesh(
    new THREE.BoxGeometry(0.0022, 0.0009, 0.0032),
    new THREE.MeshLambertMaterial({ color: 0xd7dde6 }),
  );
  hull.position.y = 0.00105;
  group.add(hull);
  const deck = new THREE.Mesh(
    new THREE.BoxGeometry(0.0014, 0.00045, 0.0012),
    new THREE.MeshLambertMaterial({ color: 0xc4a46a }),
  );
  deck.position.set(0, 0.0017, -0.0002);
  group.add(deck);
  const mast = new THREE.Mesh(
    new THREE.CylinderGeometry(0.0001, 0.0001, 0.0015, 8),
    new THREE.MeshLambertMaterial({ color: 0x9aa6b5 }),
  );
  mast.position.set(0, 0.0023, -0.0007);
  group.add(mast);
  const wheel = new THREE.CylinderGeometry(0.00042, 0.00042, 0.00032, 10);
  const rubber = new THREE.MeshLambertMaterial({ color: 0x22262e });
  for (const [x, z] of [[-0.00105, -0.00105], [0.00105, -0.00105], [-0.00105, 0.00105], [0.00105, 0.00105]] as const) {
    const meshWheel = new THREE.Mesh(wheel, rubber);
    meshWheel.rotation.z = Math.PI / 2;
    meshWheel.position.set(x, 0.00042, z);
    group.add(meshWheel);
  }
  return group;
}

function here() {
  if (!ground) return "";
  const kmPerDeg = (Math.PI * ground.radius_km) / 180;
  const lat = ground.lat + place.north / kmPerDeg;
  const lon = ground.lon + place.east / (kmPerDeg * Math.max(0.25, Math.cos((ground.lat * Math.PI) / 180)));
  return fmtLL(lat, lon);
}

function applyCamera() {
  if (!camera || !rover) return;
  const foot = surface(place.north, place.east);
  rover.position.set(foot.x, foot.y, foot.z);
  rover.rotation.y = -yaw;
  const faceN = Math.cos(yaw);
  const faceE = Math.sin(yaw);
  const back = 0.028 + orbit * 0.02;
  const camN = place.north - faceN * back;
  const camE = place.east - faceE * back;
  let camY = foot.y + 0.008 + orbit * 0.012;
  for (let stepT = 0.25; stepT <= 1; stepT += 0.25) {
    const groundY = surface(place.north + (camN - place.north) * stepT, place.east + (camE - place.east) * stepT).y;
    camY = Math.max(camY, groundY + CLEARANCE_KM);
  }
  const cam = surface(camN, camE);
  camera.position.set(cam.x, camY, cam.z);
  camera.up.set(0, 1, 0);
  camera.lookAt(foot.x, foot.y + 0.0015, foot.z);
}

function step(dt: number) {
  if (!ground) return;
  let forward = 0;
  let strafe = 0;
  if (keys.has("w") || keys.has("arrowup")) forward += 1;
  if (keys.has("s") || keys.has("arrowdown")) forward -= 1;
  if (keys.has("d") || keys.has("arrowright")) strafe += 1;
  if (keys.has("a") || keys.has("arrowleft")) strafe -= 1;
  if (!forward && !strafe) return;
  const speed = (keys.has("shift") ? SPRINT_KM_S : DRIVE_KM_S) * dt / (Math.hypot(forward, strafe) || 1);
  const faceN = Math.cos(yaw);
  const faceE = Math.sin(yaw);
  place.north += (faceN * forward - faceE * strafe) * speed;
  place.east += (faceE * forward + faceN * strafe) * speed;
  const limitN = ground.north_km * 0.46;
  const limitE = ground.east_km * 0.46;
  place.north = Math.max(-limitN, Math.min(limitN, place.north));
  place.east = Math.max(-limitE, Math.min(limitE, place.east));
}

function frame() {
  raf = requestAnimationFrame(frame);
  const now = performance.now();
  const dt = Math.min(0.05, (now - last) / 1000);
  last = now;
  const canvas = document.querySelector("#worldCanvas");
  const looking = document.pointerLockElement === canvas;
  if (looking || (canvas instanceof HTMLElement && document.activeElement === canvas)) step(dt);
  applyCamera();
  const where = $<HTMLElement>("#worldWhere");
  const spot = currentSpot();
  if (spot && ground) where.textContent = `${spot.body} · ${here()}`;
  if (ground) $<HTMLElement>("#legendHere").textContent = `Ground ${fmtM(heightMetres(place.north, place.east))}`;
  renderer?.render(scene!, camera!);
}

let last = performance.now();

function loop(on: boolean) {
  if (on && !raf) {
    last = performance.now();
    frame();
  }
  if (!on && raf) {
    cancelAnimationFrame(raf);
    raf = 0;
  }
}

function buildMesh(data: Ground, texture: THREE.Texture) {
  if (!scene) return;
  if (mesh) {
    scene.remove(mesh);
    mesh.geometry.dispose();
    const previous = mesh.material as THREE.MeshLambertMaterial;
    previous.map?.dispose();
    previous.dispose();
  }
  const { rows, cols } = data;
  const positions = new Float32Array(rows * cols * 3);
  const uvs = new Float32Array(rows * cols * 2);
  const index: number[] = [];
  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      const i = row * cols + col;
      const north = (0.5 - row / (rows - 1)) * data.north_km;
      const east = (col / (cols - 1) - 0.5) * data.east_km;
      const drop = (north * north + east * east) / (2 * data.radius_km);
      positions[i * 3] = east;
      positions[i * 3 + 1] = data.elevation_m[i] / 1000 - drop;
      positions[i * 3 + 2] = -north;
      uvs[i * 2] = col / (cols - 1);
      uvs[i * 2 + 1] = 1 - row / (rows - 1);
      if (row < rows - 1 && col < cols - 1) {
        const a = i;
        const b = i + 1;
        const c = i + cols;
        const d = c + 1;
        index.push(a, c, b, b, c, d);
      }
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("uv", new THREE.BufferAttribute(uvs, 2));
  geometry.setIndex(index);
  geometry.computeVertexNormals();
  mesh = new THREE.Mesh(geometry, new THREE.MeshLambertMaterial({ map: texture }));
  scene.add(mesh);
  if (scene.fog instanceof THREE.Fog) {
    scene.fog.near = data.north_km * 0.35;
    scene.fog.far = data.north_km * 0.62;
  }
}

async function loadTexture(spot: WorldSpot, generation: number) {
  const url = `/api/ground.png?body=${encodeURIComponent(bodyKey(spot.body))}&lat=${spot.lat}&lon=${spot.lon}&layer=${encodeURIComponent(layer)}&v=${stamp}`;
  const texture = await new THREE.TextureLoader().loadAsync(url);
  if (generation !== colorToken) {
    texture.dispose();
    return null;
  }
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.anisotropy = 8;
  return texture;
}

async function showSpot(reset: boolean) {
  const spot = currentSpot();
  const panel = $<HTMLElement>("#worldPanel");
  if (!spot) {
    panel.hidden = true;
    loop(false);
    return;
  }
  panel.hidden = false;
  ensureRenderer();
  const generation = ++buildToken;
  const paint = ++colorToken;
  const hint = $<HTMLElement>("#worldHint");
  hint.textContent = "Building this place…";
  try {
    const [data, texture] = await Promise.all([
      fetch(`/api/ground?body=${encodeURIComponent(bodyKey(spot.body))}&lat=${spot.lat}&lon=${spot.lon}`).then(async (response) => {
        const payload = await response.json() as Ground & { error?: string };
        if (!response.ok) throw new Error(payload.error || "The ground could not be built.");
        return payload;
      }),
      loadTexture(spot, paint),
    ]);
    if (generation !== buildToken || !texture) return;
    ground = data;
    heights = Float32Array.from(data.elevation_m);
    buildMesh(data, texture);
    if (reset) {
      place.north = 0;
      place.east = 0;
      yaw = 0;
      orbit = 0.42;
    }
    paintLegend();
    hint.textContent = "Click the view, then turn with the mouse. W A S D drives the rover over the real ground. Shift is faster. Esc lets the mouse go.";
    $<HTMLElement>("#worldWhere").textContent = `${spot.body} · ${fmtLL(spot.lat, spot.lon)}`;
    loop(true);
    renderer?.setSize($<HTMLCanvasElement>("#worldCanvas").clientWidth, $<HTMLCanvasElement>("#worldCanvas").clientHeight, false);
  } catch (error) {
    if (generation !== buildToken) return;
    hint.textContent = error instanceof Error ? error.message : "The world could not be shown.";
  }
}

async function swapColor() {
  const spot = currentSpot();
  if (!spot || !mesh) return;
  const generation = ++colorToken;
  const texture = await loadTexture(spot, generation);
  if (!texture || generation !== colorToken || !mesh) return;
  const material = mesh.material as THREE.MeshLambertMaterial;
  material.map?.dispose();
  material.map = texture;
  material.needsUpdate = true;
  paintLegend();
}

function wire() {
  if (wired) return;
  wired = true;
  $<HTMLElement>("#worldPlace").addEventListener("click", (event) => {
    const button = (event.target as HTMLElement).closest("button");
    if (!(button instanceof HTMLButtonElement)) return;
    side = button.dataset.side === "a" ? "a" : "b";
    document.querySelectorAll("#worldPlace button").forEach((item) => item.classList.toggle("on", item === button));
    void showSpot(true);
  });
  $<HTMLElement>("#worldLayers").addEventListener("click", (event) => {
    const button = (event.target as HTMLElement).closest("button");
    if (!(button instanceof HTMLButtonElement) || !button.dataset.layer) return;
    layer = button.dataset.layer;
    document.querySelectorAll("#worldLayers button").forEach((item) => item.classList.toggle("on", item === button));
    void swapColor();
  });
  const canvas = $<HTMLCanvasElement>("#worldCanvas");
  canvas.addEventListener("click", () => {
    canvas.requestPointerLock?.();
    canvas.focus();
  });
  document.addEventListener("mousemove", (event) => {
    if (document.pointerLockElement !== canvas) return;
    yaw -= event.movementX * 0.0022;
    orbit -= event.movementY * 0.0016;
    orbit = Math.max(0.12, Math.min(1.15, orbit));
  });
  const watch = (event: KeyboardEvent, down: boolean) => {
    const key = event.key.toLowerCase();
    if (!["w", "a", "s", "d", "arrowup", "arrowdown", "arrowleft", "arrowright", "shift"].includes(key)) return;
    const canvas = document.querySelector("#worldCanvas");
    if (document.pointerLockElement !== canvas && document.activeElement !== canvas) return;
    if (down) keys.add(key);
    else keys.delete(key);
    if (key.startsWith("arrow") || ["w", "a", "s", "d"].includes(key)) event.preventDefault();
  };
  window.addEventListener("keydown", (event) => watch(event, true));
  window.addEventListener("keyup", (event) => watch(event, false));
  document.addEventListener("pointerlockchange", () => {
    if (document.pointerLockElement !== canvas) keys.clear();
  });
}

function fmtM(metres: number) {
  const sign = metres < 0 ? "−" : "";
  return `${sign}${Math.abs(Math.round(metres)).toLocaleString()} m`;
}

function paintLegend() {
  const box = $<HTMLElement>("#worldLegend");
  const spot = currentSpot();
  if (!spot || !ground) {
    box.hidden = true;
    return;
  }
  box.hidden = false;
  const key = bodyKey(spot.body);
  const fact = facts[key]?.[layer];
  const bar = $<HTMLElement>("#legendBar");
  const low = $<HTMLElement>("#legendLow");
  const high = $<HTMLElement>("#legendHigh");
  if (layer === "elevation") {
    $<HTMLElement>("#legendTitle").textContent = "Elevation";
    bar.hidden = false;
    bar.className = `legend-elev-${key}`;
    low.textContent = fmtM(ground.low_m);
    high.textContent = fmtM(ground.high_m);
  } else if (fact && fact.ramp !== "photo" && fact.low !== fact.high) {
    $<HTMLElement>("#legendTitle").textContent = fact.title;
    bar.hidden = false;
    bar.className = fact.ramp === "rain" ? "legend-rain" : "legend-heat";
    low.textContent = fact.low;
    high.textContent = fact.high;
  } else {
    $<HTMLElement>("#legendTitle").textContent = fact?.title || layer;
    bar.hidden = true;
    bar.className = "";
    low.textContent = fact?.low === fact?.high ? "" : (fact?.low || "");
    high.textContent = fact?.low === fact?.high ? "" : (fact?.high || "");
  }
  $<HTMLElement>("#legendHere").textContent = `Ground ${fmtM(heightMetres(place.north, place.east))}`;
}

async function fillLayers() {
  const box = $<HTMLElement>("#worldLayers");
  if (box.childElementCount) return;
  const catalog = await fetch("/api/layers").then((response) => response.json() as Promise<{
    layers: { id: string; label: string }[];
    bodies: Record<string, Record<string, LayerFact>>;
  }>);
  facts = catalog.bodies;
  for (const item of catalog.layers) {
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.layer = item.id;
    button.textContent = item.label;
    button.classList.toggle("on", item.id === layer);
    box.appendChild(button);
  }
}

export function setPreviewWorld(next: { a: WorldSpot; b: WorldSpot } | null) {
  spots = next;
  wire();
  if (!next) {
    $<HTMLElement>("#worldPanel").hidden = true;
    loop(false);
    return;
  }
  void fillLayers().then(() => showSpot(true));
}

export function refreshWorldMap() {
  stamp += 1;
  if (spots) void swapColor();
}
