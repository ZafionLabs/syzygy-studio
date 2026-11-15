import * as THREE from "three";

export type WorldSpot = { body: string; lat: number; lon: number };

type Ground = {
  body: string;
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
let sky: THREE.Mesh | null = null;
let stars: THREE.Points | null = null;
let sunLight: THREE.DirectionalLight | null = null;
let ambient: THREE.AmbientLight | null = null;
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
  scene.background = new THREE.Color(0x6eaddf);
  camera = new THREE.PerspectiveCamera(62, 1, 0.0004, 400);
  ambient = new THREE.AmbientLight(0xffffff, 0.62);
  scene.add(ambient);
  sunLight = new THREE.DirectionalLight(0xfff4e0, 1.7);
  sunLight.position.set(80, 140, 40);
  scene.add(sunLight);
  sky = makeSky();
  stars = makeStars();
  scene.add(sky);
  scene.add(stars);
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

function paintedTexture(base: string, ink: string, lines: boolean) {
  const canvas = document.createElement("canvas");
  canvas.width = 128;
  canvas.height = 128;
  const draw = canvas.getContext("2d");
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  if (!draw) return texture;
  draw.fillStyle = base;
  draw.fillRect(0, 0, 128, 128);
  for (let i = 0; i < 500; i += 1) {
    draw.fillStyle = ink;
    draw.globalAlpha = 0.15 + Math.random() * 0.35;
    draw.fillRect(Math.random() * 128, Math.random() * 128, 1 + Math.random() * 2, 1 + Math.random() * 2);
  }
  draw.globalAlpha = 1;
  if (lines) {
    draw.strokeStyle = ink;
    draw.globalAlpha = 0.45;
    draw.strokeRect(6, 6, 116, 116);
    draw.beginPath();
    draw.moveTo(6, 64);
    draw.lineTo(122, 64);
    draw.stroke();
  }
  return texture;
}

function panelTexture(base: string, ink: string) {
  return paintedTexture(base, ink, true);
}

function treadTexture() {
  const canvas = document.createElement("canvas");
  canvas.width = 64;
  canvas.height = 64;
  const draw = canvas.getContext("2d");
  const texture = new THREE.CanvasTexture(canvas);
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  texture.colorSpace = THREE.SRGBColorSpace;
  if (!draw) return texture;
  draw.fillStyle = "#241e1a";
  draw.fillRect(0, 0, 64, 64);
  draw.strokeStyle = "#6b6258";
  for (let x = 4; x < 64; x += 8) {
    draw.beginPath();
    draw.moveTo(x, 0);
    draw.lineTo(x + 10, 64);
    draw.stroke();
  }
  return texture;
}

function part(geometry: THREE.BufferGeometry, material: THREE.Material, x: number, y: number, z: number, parent: THREE.Object3D) {
  const mesh = new THREE.Mesh(geometry, material);
  mesh.position.set(x, y, z);
  parent.add(mesh);
  return mesh;
}

function makeSky() {
  const mesh = new THREE.Mesh(
    new THREE.SphereGeometry(120, 28, 18),
    new THREE.MeshBasicMaterial({ side: THREE.BackSide, depthWrite: false, fog: false }),
  );
  mesh.renderOrder = -1;
  mesh.frustumCulled = false;
  return mesh;
}

function makeStars() {
  const count = 220;
  const positions = new Float32Array(count * 3);
  for (let i = 0; i < count; i += 1) {
    const theta = Math.random() * Math.PI * 2;
    const phi = Math.acos(1 - Math.random());
    const radius = 90;
    positions[i * 3] = radius * Math.sin(phi) * Math.cos(theta);
    positions[i * 3 + 1] = radius * Math.cos(phi);
    positions[i * 3 + 2] = radius * Math.sin(phi) * Math.sin(theta);
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  const points = new THREE.Points(geometry, new THREE.PointsMaterial({ color: 0xffffff, size: 1.3, sizeAttenuation: false, fog: false }));
  points.visible = false;
  points.frustumCulled = false;
  return points;
}

function skyTexture(body: string) {
  const canvas = document.createElement("canvas");
  canvas.width = 8;
  canvas.height = 256;
  const draw = canvas.getContext("2d");
  if (!draw) return new THREE.CanvasTexture(canvas);
  const fade = draw.createLinearGradient(0, 0, 0, 256);
  if (body === "mars") {
    fade.addColorStop(0, "#9a5b45");
    fade.addColorStop(0.45, "#c98462");
    fade.addColorStop(0.78, "#e7c3a4");
    fade.addColorStop(1, "#efd0b4");
  } else if (body === "moon") {
    fade.addColorStop(0, "#000000");
    fade.addColorStop(1, "#02030a");
  } else {
    fade.addColorStop(0, "#2f6fbe");
    fade.addColorStop(0.42, "#79b4ea");
    fade.addColorStop(0.78, "#d5e8f7");
    fade.addColorStop(1, "#e7f2fb");
  }
  draw.fillStyle = fade;
  draw.fillRect(0, 0, 8, 256);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

function applyAtmosphere(bodyName: string) {
  if (!scene || !sky || !stars || !sunLight || !ambient) return;
  const body = bodyName.toLowerCase();
  const span = ground?.north_km ?? 100;
  const material = sky.material as THREE.MeshBasicMaterial;
  material.map?.dispose();
  material.map = skyTexture(body);
  material.needsUpdate = true;
  if (body === "moon") {
    scene.fog = null;
    scene.background = new THREE.Color(0x000000);
    stars.visible = true;
    sunLight.color.set(0xffffff);
    sunLight.intensity = 1.9;
    ambient.color.set(0xc8d0dc);
    ambient.intensity = 0.28;
  } else if (body === "mars") {
    scene.background = new THREE.Color(0xc98462);
    scene.fog = new THREE.Fog(0xe6c4a6, span * 0.22, span * 0.7);
    stars.visible = false;
    sunLight.color.set(0xffd2ad);
    sunLight.intensity = 1.45;
    ambient.color.set(0xffe0cc);
    ambient.intensity = 0.58;
  } else {
    scene.background = new THREE.Color(0x6eaddf);
    scene.fog = new THREE.Fog(0xd7e8f6, span * 0.25, span * 0.72);
    stars.visible = false;
    sunLight.color.set(0xfff6e8);
    sunLight.intensity = 1.65;
    ambient.color.set(0xe7f1fb);
    ambient.intensity = 0.62;
  }
}

function makeRover() {
  const group = new THREE.Group();
  group.scale.setScalar(0.001);
  const deckColor = new THREE.MeshLambertMaterial({ map: panelTexture("#9a7048", "#5c4030"), color: 0xffffff });
  const white = new THREE.MeshLambertMaterial({ map: panelTexture("#e6e1d6", "#b7b1a6"), color: 0xffffff });
  const dark = new THREE.MeshLambertMaterial({ map: treadTexture(), color: 0xffffff });
  const lens = new THREE.MeshLambertMaterial({ color: 0x161311 });
  const metal = new THREE.MeshLambertMaterial({ map: panelTexture("#8a8178", "#4e4842"), color: 0xffffff });
  const wheelRadius = 0.26;
  const wheelGeo = new THREE.CylinderGeometry(wheelRadius, wheelRadius, 0.18, 16);
  const hubGeo = new THREE.CylinderGeometry(0.09, 0.09, 0.2, 10);
  for (const x of [-1.2, 1.2]) {
    for (const z of [-1.15, 0, 1.05]) {
      const wheel = part(wheelGeo, dark, x, wheelRadius, z, group);
      wheel.rotation.z = Math.PI / 2;
      const hub = part(hubGeo, metal, x, wheelRadius, z, group);
      hub.rotation.z = Math.PI / 2;
    }
    part(new THREE.BoxGeometry(0.08, 0.08, 2.35), metal, x * 0.92, 0.42, -0.05, group);
  }
  part(new THREE.BoxGeometry(2.15, 0.32, 1.85), deckColor, 0, 0.62, -0.05, group);
  part(new THREE.BoxGeometry(1.15, 0.16, 1.35), new THREE.MeshLambertMaterial({ color: 0xb88958 }), 0, 0.84, 0.05, group);
  part(new THREE.BoxGeometry(0.62, 0.28, 0.48), white, -0.35, 1.02, 0.25, group);
  part(new THREE.BoxGeometry(0.38, 0.22, 0.34), white, 0.48, 0.98, 0.15, group);
  const mast = part(new THREE.CylinderGeometry(0.055, 0.07, 1.15, 10), white, 0, 1.5, -0.42, group);
  mast.position.y = 1.5;
  part(new THREE.BoxGeometry(0.36, 0.2, 0.26), white, 0, 2.12, -0.48, group);
  for (const x of [-0.09, 0.09]) {
    const eye = part(new THREE.CylinderGeometry(0.045, 0.045, 0.08, 10), lens, x, 2.12, -0.62, group);
    eye.rotation.x = Math.PI / 2;
  }
  part(new THREE.BoxGeometry(0.12, 0.12, 0.55), metal, -0.85, 0.78, -1.15, group);
  part(new THREE.BoxGeometry(0.1, 0.1, 0.45), metal, -1.05, 0.62, -1.55, group);
  part(new THREE.BoxGeometry(0.16, 0.16, 0.16), lens, -1.05, 0.55, -1.82, group);
  const dish = part(new THREE.CylinderGeometry(0.28, 0.28, 0.04, 16), white, 0.62, 1.25, 0.55, group);
  dish.rotation.x = 0.5;
  dish.rotation.z = -0.4;
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
  const back = 0.014 + orbit * 0.01;
  const camN = place.north - faceN * back;
  const camE = place.east - faceE * back;
  let camY = foot.y + 0.005 + orbit * 0.007;
  for (let stepT = 0.25; stepT <= 1; stepT += 0.25) {
    const groundY = surface(place.north + (camN - place.north) * stepT, place.east + (camE - place.east) * stepT).y;
    camY = Math.max(camY, groundY + CLEARANCE_KM);
  }
  const cam = surface(camN, camE);
  camera.position.set(cam.x, camY, cam.z);
  camera.up.set(0, 1, 0);
  camera.lookAt(foot.x, foot.y + 0.0013, foot.z);
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
  if (camera) {
    sky?.position.copy(camera.position);
    stars?.position.copy(camera.position);
  }
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

let detail: THREE.Texture | null = null;

function detailTexture() {
  if (detail) return detail;
  const canvas = document.createElement("canvas");
  canvas.width = 256;
  canvas.height = 256;
  const draw = canvas.getContext("2d");
  if (!draw) return new THREE.CanvasTexture(canvas);
  draw.fillStyle = "#f3f1ee";
  draw.fillRect(0, 0, 256, 256);
  for (let i = 0; i < 2200; i += 1) {
    const shade = 150 + Math.floor(Math.random() * 90);
    draw.fillStyle = `rgb(${shade},${shade},${shade})`;
    draw.fillRect(Math.random() * 256, Math.random() * 256, 1 + Math.random() * 2, 1 + Math.random() * 2);
  }
  const texture = new THREE.CanvasTexture(canvas);
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  texture.colorSpace = THREE.SRGBColorSpace;
  detail = texture;
  return texture;
}

function reliefShade(data: Ground) {
  const { rows, cols } = data;
  const shade = new Float32Array(rows * cols);
  const cellNorth = (data.north_km * 1000) / (rows - 1);
  const cellEast = (data.east_km * 1000) / (cols - 1);
  const zenith = Math.PI / 4;
  const azimuth = (Math.PI * 5) / 4;
  const heightAt = (row: number, col: number) => data.elevation_m[Math.max(0, Math.min(rows - 1, row)) * cols + Math.max(0, Math.min(cols - 1, col))];
  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      const eastSlope = (heightAt(row, col + 1) - heightAt(row, col - 1)) / (2 * cellEast);
      const northSlope = (heightAt(row - 1, col) - heightAt(row + 1, col)) / (2 * cellNorth);
      const slope = Math.atan(Math.hypot(eastSlope, northSlope));
      const aspect = Math.atan2(northSlope, -eastSlope);
      const lit = Math.cos(zenith) * Math.cos(slope) + Math.sin(zenith) * Math.sin(slope) * Math.cos(azimuth - aspect);
      shade[row * cols + col] = 0.4 + 0.6 * Math.max(0, lit);
    }
  }
  return shade;
}

function terrainMaterial(topo: THREE.Texture, overlay: THREE.Texture, mix: number) {
  return new THREE.ShaderMaterial({
    uniforms: {
      topoMap: { value: topo },
      overlayMap: { value: overlay },
      detailMap: { value: detailTexture() },
      overlayMix: { value: mix },
      photoBase: { value: 0 },
      sunDir: { value: new THREE.Vector3(80, 140, 40).normalize() },
    },
    vertexShader: `
      attribute float shade;
      varying vec2 vUv;
      varying vec3 vNormal;
      varying vec3 vWorld;
      varying float vShade;
      void main() {
        vUv = uv;
        vShade = shade;
        vec4 world = modelMatrix * vec4(position, 1.0);
        vWorld = world.xyz;
        vNormal = normalize(mat3(modelMatrix) * normal);
        gl_Position = projectionMatrix * viewMatrix * world;
      }
    `,
    fragmentShader: `
      uniform sampler2D topoMap;
      uniform sampler2D overlayMap;
      uniform sampler2D detailMap;
      uniform float overlayMix;
      uniform float photoBase;
      uniform vec3 sunDir;
      varying vec2 vUv;
      varying vec3 vNormal;
      varying vec3 vWorld;
      varying float vShade;
      void main() {
        vec3 topo = texture2D(topoMap, vUv).rgb;
        vec3 over = texture2D(overlayMap, vUv).rgb;
        vec3 grit = texture2D(detailMap, vWorld.xz * 42.0).rgb;
        float lambert = max(dot(normalize(vNormal), normalize(sunDir)), 0.0);
        vec3 color = mix(topo, over, overlayMix * 0.72);
        float gritMix = photoBase > 0.5 ? 0.16 : 0.45;
        color *= mix(vec3(1.0), grit, gritMix);
        float relief = photoBase > 0.5 ? mix(0.94, 1.06, vShade) : mix(0.8, 1.05, vShade);
        color *= relief * (0.9 + 0.1 * lambert);
        gl_FragColor = vec4(color, 1.0);
      }
    `,
  });
}

function buildMesh(data: Ground, topo: THREE.Texture, overlay: THREE.Texture, mix: number) {
  if (!scene) return;
  if (mesh) {
    scene.remove(mesh);
    mesh.geometry.dispose();
    const previous = mesh.material as THREE.ShaderMaterial;
    const topoMap = previous.uniforms.topoMap.value as THREE.Texture;
    const overlayMap = previous.uniforms.overlayMap.value as THREE.Texture;
    topoMap.dispose();
    if (overlayMap !== topoMap) overlayMap.dispose();
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
  geometry.setAttribute("shade", new THREE.BufferAttribute(reliefShade(data), 1));
  geometry.setIndex(index);
  geometry.computeVertexNormals();
  const material = terrainMaterial(topo, overlay, mix);
  material.uniforms.photoBase.value = data.body === "earth" ? 1 : 0;
  mesh = new THREE.Mesh(geometry, material);
  scene.add(mesh);
}

async function loadTexture(spot: WorldSpot, which: string, generation: number) {
  const url = `/api/ground.png?body=${encodeURIComponent(bodyKey(spot.body))}&lat=${spot.lat}&lon=${spot.lon}&layer=${encodeURIComponent(which)}&v=${stamp}`;
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
    const [data, topo, overlay] = await Promise.all([
      fetch(`/api/ground?body=${encodeURIComponent(bodyKey(spot.body))}&lat=${spot.lat}&lon=${spot.lon}`).then(async (response) => {
        const payload = await response.json() as Ground & { error?: string };
        if (!response.ok) throw new Error(payload.error || "The ground could not be built.");
        return payload;
      }),
      loadTexture(spot, "elevation", paint),
      layer === "elevation" ? Promise.resolve(null) : loadTexture(spot, layer, paint),
    ]);
    if (generation !== buildToken || !topo) return;
    ground = data;
    heights = Float32Array.from(data.elevation_m);
    buildMesh(data, topo, overlay ?? topo, overlay ? 1 : 0);
    if (reset) {
      place.north = 0;
      place.east = 0;
      yaw = 0;
      orbit = 0.42;
    }
    applyAtmosphere(spot.body);
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
  const material = mesh.material as THREE.ShaderMaterial;
  if (layer === "elevation") {
    const current = material.uniforms.overlayMap.value as THREE.Texture;
    const topo = material.uniforms.topoMap.value as THREE.Texture;
    if (current !== topo) current.dispose();
    material.uniforms.overlayMap.value = topo;
    material.uniforms.overlayMix.value = 0;
    paintLegend();
    return;
  }
  const texture = await loadTexture(spot, layer, generation);
  if (!texture || generation !== colorToken || !mesh) return;
  const current = material.uniforms.overlayMap.value as THREE.Texture;
  const topo = material.uniforms.topoMap.value as THREE.Texture;
  if (current !== topo) current.dispose();
  material.uniforms.overlayMap.value = texture;
  material.uniforms.overlayMix.value = 1;
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

type Swatch = { color: string; label: string };

const INFERNO: Swatch[] = [
  { color: "#000004", label: "" },
  { color: "#bb3754", label: "" },
  { color: "#fcfea4", label: "" },
];

function heatRows(low: string, high: string): Swatch[] {
  return [
    { color: INFERNO[0].color, label: low },
    { color: INFERNO[1].color, label: "in between" },
    { color: INFERNO[2].color, label: high },
  ];
}

function legendFor(body: string): { title: string; rows: Swatch[] } {
  const named: Record<string, Record<string, { title: string; rows: Swatch[] }>> = {
    earth: {
      elevation: {
        title: "Google Maps",
        rows: [
          { color: "#3d6b3a", label: "Plants and fields" },
          { color: "#c2b48a", label: "Bare ground" },
          { color: "#1d4e89", label: "Water" },
          { color: "#f4f7fb", label: "Snow, ice, or cloud" },
        ],
      },
      temperature: {
        title: "Air temperature",
        rows: [
          { color: "#5e4fa2", label: "Colder" },
          { color: "#e55749", label: "Warm" },
          { color: "#fee593", label: "Warmer" },
        ],
      },
      "ice-water": {
        title: "Ice and water",
        rows: [
          { color: "#0e305c", label: "Ocean" },
          { color: "#5c4a30", label: "Land" },
          { color: "#ffffff", label: "Ice or snow" },
        ],
      },
      geology: {
        title: "Geology",
        rows: [
          { color: "#184e91", label: "Ocean" },
          { color: "#bad6e6", label: "Ice" },
          { color: "#92dcb7", label: "A rock unit" },
        ],
      },
      weather: {
        title: "Rain and snow",
        rows: [
          { color: "#82aaa8", label: "Lighter rain" },
          { color: "#c2da0b", label: "Heavier rain" },
          { color: "#e75623", label: "Heaviest rain" },
        ],
      },
      light: {
        title: "Reflected sunlight",
        rows: [
          { color: "#020514", label: "Ocean and forest" },
          { color: "#b2895d", label: "Desert" },
          { color: "#f1f3f2", label: "Ice" },
        ],
      },
      radiation: {
        title: "Heat leaving Earth",
        rows: [
          { color: "#440154", label: "Less heat leaving" },
          { color: "#2b7100", label: "In between" },
          { color: "#f5fa0a", label: "More heat leaving" },
        ],
      },
    },
    moon: {
      elevation: {
        title: "Elevation",
        rows: [
          { color: "#00224d", label: "Lowest ground" },
          { color: "#7c7b78", label: "Middle" },
          { color: "#fde737", label: "Highest ground" },
        ],
      },
      geology: {
        title: "Geology",
        rows: [
          { color: "#f20d0d", label: "A rock unit" },
          { color: "#fcdc0a", label: "A different unit" },
          { color: "#a6d6ff", label: "A different unit" },
        ],
      },
      light: {
        title: "Reflected sunlight",
        rows: [
          { color: "#222222", label: "Dark maria" },
          { color: "#8a8a8a", label: "Brighter highlands" },
        ],
      },
    },
    mars: {
      elevation: {
        title: "Elevation",
        rows: [
          { color: "#000000", label: "Lowest ground" },
          { color: "#8e5939", label: "Middle" },
          { color: "#ffc77e", label: "Highest ground" },
        ],
      },
      geology: {
        title: "Geology",
        rows: [
          { color: "#a86c44", label: "Southern highlands" },
          { color: "#7eac54", label: "Northern plains" },
          { color: "#dc404e", label: "Volcanic rock" },
          { color: "#ffd640", label: "Volcanic rock" },
        ],
      },
      weather: {
        title: "Dust storm",
        rows: [
          { color: "#280b53", label: "Clearer air" },
          { color: "#fcfea4", label: "Thick dust" },
        ],
      },
      light: {
        title: "Reflected sunlight",
        rows: [
          { color: "#584646", label: "Darker ground" },
          { color: "#d8d0c8", label: "Ice at the poles" },
        ],
      },
    },
  };
  const direct = named[body]?.[layer];
  if (direct) return direct;
  const fact = facts[body]?.[layer];
  if (fact) return { title: fact.title, rows: heatRows(fact.low, fact.high) };
  return { title: layer, rows: [] };
}

function paintLegend() {
  const box = $<HTMLElement>("#worldLegend");
  const spot = currentSpot();
  if (!spot || !ground) {
    box.hidden = true;
    return;
  }
  box.hidden = false;
  const legend = legendFor(bodyKey(spot.body));
  $<HTMLElement>("#legendTitle").textContent = legend.title;
  $<HTMLElement>("#legendItems").innerHTML = legend.rows.map((row) =>
    `<li><i style="background:${row.color}"></i><span>${row.label.replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c] ?? c))}</span></li>`).join("");
  const earth = bodyKey(spot.body) === "earth";
  $<HTMLElement>("#legendNote").textContent = earth
    ? (layer === "elevation" ? "Google Maps satellite. Pick another map to color it." : "These colors sit on the Google Maps photo.")
    : "";
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
