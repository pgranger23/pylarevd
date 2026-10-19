// view3d.js -- Three.js 3-D event view.
//
// Coordinates: all geometry stays in DETECTOR coordinates (x drift, y up, z beam) in the
// GPU buffers (zero copy from the bundle). A root Group carries a fixed rotation into the
// scene frame: scene = (z, y, -x), i.e. screen-right = z (beam), up = y, x (drift) points
// into/out of the screen. The mapping is a proper rotation (no mirror image).
//
// Rendering strategy: one draw call per kind. Control changes (colour-by, scale, cmap,
// theme, overlays, radiologicals) only touch uniforms / visibility / small attributes.
// Radiologicals are filtered in the vertex shader from a per-vertex flag.
// Rendering is on demand (rAF coalesced); the animation loop only runs between
// setAnimating(true) and setAnimating(false).
//
// Public API used by other modules (besides the view interface in main.js):
//   setPdColors(Float32Array rgb|null, alpha?)  per-tile colours (n_tiles*3, tile order of
//                                               geometry 'pd/xyz'); null = neutral walls
//   setAnimating(bool)    keep the render loop running (light playback)
//   requestRender()       schedule one render
//   resetView()           refit the camera to the detector
//   pdTileCount           number of PD tiles (0 if no geometry)
//   setFlashTime(t_us|null)  flash bars (view 'flash3d', or Charge+Light with the "flashes"
//                         box ticked): show only flashes with time <= t_us (ABSOLUTE us, the
//                         clock of oph/time and meta.optical.t0); null = all flashes of the window.
//                         Only a uniform changes (no rebuild). Bars cover the default window
//                         t0 +- 50 us, the 200 brightest, colour = time on the cmap.
//   cameraInfo()          {position,target,up,fov,aspect} in DETECTOR coordinates (cm):
//                         scene (sx,sy,sz) = (z, y, -x)  =>  detector x = -sz, y = sy, z = sx.
//                         The camera looks from position to target, `up` is detector +y,
//                         fov is the vertical field of view in degrees.
// Views: 'flash3d' draws FlashDisplay3D (bars across the drift, PD tiles faint, true vertex,
// charge/reco hidden). Vertices: ov.pvtx = Pandora interaction vertex (role 1), ov.vtx =
// roles 2/3 (a missing pvtx counts as on); the true vertex follows ov.truth.
import * as THREE from './vendor/three.module.js';
import { OrbitControls } from './vendor/OrbitControls.js';
import {
  getLut, makeLutTexture, updateLutTexture, cmapGradientCss, hexToRgb, PDG_CLASSES, pdgClass,
} from './colormaps.js';
import { polylineIndex, expandRagged, percentileRange } from './data.js';

THREE.ColorManagement.enabled = false;   // palette values are used as plain sRGB

// ------------------------------------------------------------------ shaders
const FRAG_DISC = /* glsl */`
varying vec3 vColor;
void main() {
  vec2 c = gl_PointCoord - 0.5;
  if (dot(c, c) > 0.25) discard;
  gl_FragColor = vec4(vColor, 1.0);
}`;

const SELGLSL = /* glsl */`
// selection (select.js): flag texture indexed by group id, 1024 ids per row
uniform sampler2D uSelTex;
uniform float uSelOn, uSelH, uDim;
float selFlag(float id) {
  if (id < -0.5) return 0.0;
  float row = floor((id + 0.5) / 1024.0);
  float col = id + 0.5 - row * 1024.0;
  return texture2D(uSelTex, vec2(col / 1024.0, (row + 0.5) / uSelH)).r;
}
`;

const SP_VERT = /* glsl */`
attribute float aValue;
attribute float aCat;
attribute float aSel;
uniform float uSize, uPx, uMode, uLog, uLo, uInv, uFloor, uAxis, uCmapLo, uCmapHi;
uniform sampler2D uLut;
uniform vec3 uPal[8];
uniform vec3 uUnassoc, uUniform, uBg;
varying vec3 vColor;
varying float vSel;
${SELGLSL}
void main() {
  float t;
  if (uMode < 0.5) {            // value through the LUT
    float x = uLog > 0.5 ? log(max(aValue, uFloor)) : aValue;
    t = clamp((x - uLo) * uInv, 0.0, 1.0);
    vColor = texture2D(uLut, vec2(mix(uCmapLo, uCmapHi, t), 0.5)).rgb;
  } else if (uMode < 1.5) {     // categorical
    vColor = aCat < -0.5 ? uUnassoc : uPal[int(mod(aCat, 8.0))];
  } else if (uMode < 2.5) {     // coordinate (detector x/y/z), linear
    float c = uAxis < 0.5 ? position.x : (uAxis < 1.5 ? position.y : position.z);
    t = clamp((c - uLo) * uInv, 0.0, 1.0);
    vColor = texture2D(uLut, vec2(mix(uCmapLo, uCmapHi, t), 0.5)).rgb;
  } else {
    vColor = uUniform;
  }
  gl_PointSize = uSize * uPx;
  vSel = 0.0;
  if (uSelOn > 0.5) {                         // highlight: selected points big + outlined, the rest faded into the background
    if (selFlag(aSel) > 0.5) { vSel = 1.0; gl_PointSize = max(uSize * 1.8, 5.0) * uPx; }
    else { vSel = -1.0; vColor = mix(uBg, vColor, uDim); }
  }
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}`;

const SP_FRAG = /* glsl */`
varying vec3 vColor;
varying float vSel;
uniform vec3 uHi;
void main() {
  vec2 c = gl_PointCoord - 0.5;
  float r2 = dot(c, c);
  if (r2 > 0.25) discard;
  gl_FragColor = vec4(vSel > 0.5 && r2 > 0.12 ? uHi : vColor, 1.0);
}`;

const DEP_VERT = /* glsl */`
attribute float aFlag;
uniform float uSize, uPx, uShowRad, uHasFlag;
uniform vec3 uUniform;
varying vec3 vColor;
void main() {
  vColor = uUniform;
  gl_PointSize = uSize * uPx;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  if (uHasFlag > 0.5 && aFlag < 0.5 && uShowRad < 0.5) gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
}`;

// Lines: aMeta is cls + 16*flag for truth (modes 0/1), the object id for tracks (mode 2).
const LINE_VERT = /* glsl */`
attribute float aMeta;
attribute float aSel;
uniform float uMode, uHasFlag, uShowRad;
uniform vec3 uColor;
uniform vec3 uPdg[9];
uniform vec3 uPal[8];
uniform vec3 uBg, uHi;
uniform float uDimAll;
varying vec3 vColor;
${SELGLSL}
void main() {
  vColor = uColor;
  float flag = 1.0;
  if (uHasFlag > 0.5) {
    flag = floor(aMeta / 16.0);
    if (uMode > 0.5 && uMode < 1.5) vColor = uPdg[int(aMeta - flag * 16.0)];
  } else if (uMode > 1.5) {
    vColor = uPal[int(mod(aMeta, 8.0))];
  }
  if (uSelOn > 0.5) vColor = selFlag(aSel) > 0.5 ? uHi : mix(uBg, vColor, uDim * 0.9);
  if (uDimAll > 0.5) vColor = mix(uBg, vColor, min(1.0, uDim * 1.33));
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  if (flag < 0.5 && uShowRad < 0.5) gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
}`;
const FRAG_FLAT = /* glsl */`
varying vec3 vColor;
void main() { gl_FragColor = vec4(vColor, 1.0); }`;

const MARK_VERT = /* glsl */`
attribute vec3 aColor;
attribute float aSize;
uniform float uPx;
varying vec3 vColor;
void main() {
  vColor = aColor;
  gl_PointSize = aSize * uPx;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  if (aSize <= 0.0) gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
}`;
const MARK_FRAG = /* glsl */`
varying vec3 vColor;
uniform vec3 uCasing;
void main() {
  vec2 c = (gl_PointCoord - 0.5) * 2.0;
  float r2 = dot(c, c);
  if (r2 > 1.0) discard;
  gl_FragColor = vec4(r2 > 0.6 ? uCasing : vColor, 1.0);
}`;

const PD_VERT = /* glsl */`
varying vec3 vColor;
varying vec2 vUv;
void main() {
  vColor = instanceColor;
  vUv = position.yz;
  gl_Position = projectionMatrix * modelViewMatrix * instanceMatrix * vec4(position, 1.0);
}`;
const PD_FRAG = /* glsl */`
varying vec3 vColor;
varying vec2 vUv;
uniform float uAlpha;
uniform vec3 uEdge;
void main() {
  float e = max(abs(vUv.x), abs(vUv.y));
  vec3 c = e > 0.9 ? uEdge : vColor;
  gl_FragColor = vec4(c, e > 0.9 ? min(1.0, uAlpha + 0.3) : uAlpha);
}`;

// Flash bars: screen-space-width quads (WebGL lines are 1 px). Each flash is 4 vertices: two ends
// x two sides. aInfo = (time rel. to t0 [us], width [css px], alpha, end 0/1).
const FLASH_VERT = /* glsl */`
attribute vec3 aOther;
attribute float aSide;
attribute vec4 aInfo;
uniform vec2 uRes;
uniform float uPx, uTLo, uTHi, uTMax, uCmapLo, uCmapHi;
uniform sampler2D uLut;
varying vec3 vColor;
varying float vAlpha;
void main() {
  vec4 a = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  vec4 b = projectionMatrix * modelViewMatrix * vec4(aOther, 1.0);
  vec2 d = (b.xy / b.w - a.xy / a.w) * uRes * 0.5;
  float len = length(d);
  vec2 dir = len > 1e-3 ? d / len : vec2(1.0, 0.0);
  vec2 nrm = vec2(-dir.y, dir.x);
  float sgn = len > 1e-3 ? -1.0 : (aInfo.w < 0.5 ? -1.0 : 1.0);     // square caps
  float w = aInfo.y * uPx;
  vec2 off = (nrm * aSide + dir * sgn) * w * 0.5;
  gl_Position = a;
  gl_Position.xy += off / (uRes * 0.5) * a.w;
  float t = clamp((aInfo.x - uTLo) / (uTHi - uTLo), 0.0, 1.0);
  vColor = texture2D(uLut, vec2(mix(uCmapLo, uCmapHi, t), 0.5)).rgb;
  vAlpha = aInfo.z;
  if (aInfo.x > uTMax || aInfo.x < uTLo) gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
}`;
const FLASH_FRAG = /* glsl */`
varying vec3 vColor;
varying float vAlpha;
void main() { gl_FragColor = vec4(vColor, vAlpha); }`;

// ------------------------------------------------------------------ module state
const V = {
  ready: false, app: null, container: null, visible: true,
  renderer: null, scene: null, root: null, camera: null, controls: null,
  uni: null,                      // shared uniforms
  lutTex: null, lutName: '',
  geometry: null, bundle: null,
  tpc: null, axes: null, pd: null, pdN: 0, pdAlpha: 0.28, pdCustom: false,
  sp: null, spCat: new Map(), spVal: new Map(), stats: new Map(), spMode: 'value',
  tracks: null, trackCat: null, showers: null, markers: null, mc: null, dep: null,
  flashes: null, flashWin: null, flashTime: null, fbox: null,
  labels: [], labelPool: [], labelLayer: null, axisLabels: [],
  legend: null, title: null,
  selAttrs: new Map(), selTexObj: null, showerPos: null, pk: null, pkRaf: 0, mouse: null, pickMs: [], lastPickObj: null,
  renderQueued: false, animating: false, fitKey: '', pixelRatio: 1, camKey: new Float32Array(9),
  scratch: new THREE.Vector3(),
};

const sceneOf = (x, y, z, out) => out.set(z, y, -x);

function mkUniforms() {
  const pal = Array.from({ length: 8 }, () => new THREE.Vector3());
  const pdg = Array.from({ length: PDG_CLASSES.length }, () => new THREE.Vector3());
  return {
    uLut: { value: null }, uPal: { value: pal }, uPdg: { value: pdg },
    uCmapLo: { value: 0 }, uCmapHi: { value: 1 }, uShowRad: { value: 0 }, uPx: { value: 1 },
    // selection highlight, shared by space points / tracks / showers (uSelOn is per material)
    uSelTex: { value: null }, uSelH: { value: 1 }, uDim: { value: 0.15 }, uBg: { value: new THREE.Vector3() }, uHi: { value: new THREE.Vector3(1, 0.9, 0.2) },
  };
}

// ------------------------------------------------------------------ init
function init(container, app) {
  V.app = app; V.container = container;
  const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
  renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
  V.pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
  renderer.setPixelRatio(V.pixelRatio);
  container.appendChild(renderer.domElement);
  V.renderer = renderer;
  // GPU contexts can be lost (driver resets, and the first draw under SwiftShader): three.js
  // re-uploads everything on restore, we only have to draw again.
  renderer.domElement.addEventListener('webglcontextrestored', () => requestRender());

  V.scene = new THREE.Scene();
  V.root = new THREE.Group();
  V.root.matrixAutoUpdate = false;
  // detector (x,y,z) -> scene (z, y, -x)
  V.root.matrix.set(0, 0, 1, 0,
                    0, 1, 0, 0,
                   -1, 0, 0, 0,
                    0, 0, 0, 1);
  V.root.matrixWorldNeedsUpdate = true;
  V.scene.add(V.root);

  V.camera = new THREE.PerspectiveCamera(40, 1, 1, 1e5);
  V.camera.up.set(0, 1, 0);
  V.controls = new OrbitControls(V.camera, renderer.domElement);
  V.controls.enableDamping = false;
  V.controls.screenSpacePanning = true;
  V.controls.addEventListener('change', () => { requestRender(); schedulePrewarm(); });

  V.uni = mkUniforms();
  V.uni.uPx.value = V.pixelRatio;
  V.lutTex = makeLutTexture('turbo'); V.lutName = 'turbo';
  V.uni.uLut.value = V.lutTex;

  // HUD
  V.labelLayer = document.createElement('div');
  V.labelLayer.style.cssText = 'position:absolute;inset:0;overflow:hidden;pointer-events:none';
  container.appendChild(V.labelLayer);
  V.legend = document.createElement('div'); V.legend.className = 'hud-legend'; container.appendChild(V.legend);
  V.title = document.createElement('div'); V.title.className = 'hud-title'; container.appendChild(V.title);

  app.registerParam('fbars', '0');   // flash bars in Charge+Light (the 3-D panel's own checkbox)
  V.fbox = document.createElement('label');
  V.fbox.style.cssText = 'position:absolute;left:8px;top:26px;z-index:5;font-size:12px;color:var(--muted);display:none;cursor:pointer';
  V.fbox.innerHTML = '<input type="checkbox" style="vertical-align:-2px"> flashes';
  V.fbox.title = 'draw the reconstructed flashes as bars across the drift (window t0 \u00b1 50 \u00b5s)';
  V.fbox.firstChild.addEventListener('change', (e) => app.setState({ fbars: e.target.checked ? '1' : '0' }));
  container.appendChild(V.fbox);
  buildMaterials();
  app.on('selection', () => { applySelection(); });
  bindPicking(renderer.domElement);
  const ro = new ResizeObserver(() => resize());
  ro.observe(container);
  V.ready = true;
  resize();
}

let M = {};   // materials
function buildMaterials() {
  const u = V.uni;
  M.sp = new THREE.ShaderMaterial({
    vertexShader: SP_VERT, fragmentShader: SP_FRAG,
    uniforms: {
      ...u, uSelOn: { value: 0 }, uSize: { value: 2.5 }, uMode: { value: 0 }, uLog: { value: 1 }, uLo: { value: 0 }, uInv: { value: 1 },
      uFloor: { value: 1e-3 }, uAxis: { value: 0 }, uUnassoc: { value: new THREE.Vector3() }, uUniform: { value: new THREE.Vector3(0.6, 0.6, 0.6) },
    },
  });
  M.dep = new THREE.ShaderMaterial({
    vertexShader: DEP_VERT, fragmentShader: FRAG_DISC,
    uniforms: { uSize: { value: 2 }, uPx: u.uPx, uShowRad: u.uShowRad, uHasFlag: { value: 1 }, uUniform: { value: new THREE.Vector3() } },
  });
  const lineUni = () => ({
    uMode: { value: 0 }, uHasFlag: { value: 0 }, uShowRad: u.uShowRad, uColor: { value: new THREE.Vector3() },
    uPdg: u.uPdg, uPal: u.uPal,
    uSelTex: u.uSelTex, uSelH: u.uSelH, uDim: u.uDim, uBg: u.uBg, uHi: u.uHi, uSelOn: { value: 0 }, uDimAll: { value: 0 },
  });
  M.mc = new THREE.ShaderMaterial({ vertexShader: LINE_VERT, fragmentShader: FRAG_FLAT, uniforms: { ...lineUni(), uHasFlag: { value: 1 } } });
  M.tracks = new THREE.ShaderMaterial({ vertexShader: LINE_VERT, fragmentShader: FRAG_FLAT, uniforms: lineUni() });
  M.marker = new THREE.ShaderMaterial({
    vertexShader: MARK_VERT, fragmentShader: MARK_FRAG, depthTest: false, transparent: false,
    uniforms: { uPx: u.uPx, uCasing: { value: new THREE.Vector3() } },
  });
  M.pd = new THREE.ShaderMaterial({
    vertexShader: PD_VERT, fragmentShader: PD_FRAG, transparent: true, depthWrite: false, side: THREE.DoubleSide,
    uniforms: { uAlpha: { value: V.pdAlpha }, uEdge: { value: new THREE.Vector3() } },
  });
  M.flash = new THREE.ShaderMaterial({
    vertexShader: FLASH_VERT, fragmentShader: FLASH_FRAG, transparent: true, depthWrite: false, side: THREE.DoubleSide,
    uniforms: {
      uLut: u.uLut, uCmapLo: u.uCmapLo, uCmapHi: u.uCmapHi, uPx: u.uPx, uRes: { value: new THREE.Vector2(1, 1) },
      uTLo: { value: -50 }, uTHi: { value: 50 }, uTMax: { value: 50 },
    },
  });
  M.tpc = new THREE.LineBasicMaterial({ color: 0x64748b });
  M.shower = new THREE.ShaderMaterial({ vertexShader: LINE_VERT, fragmentShader: FRAG_FLAT, uniforms: lineUni() });
}

// ------------------------------------------------------------------ geometry helpers
function disposeObj(o) {
  if (!o) return;
  V.root.remove(o);
  if (o.geometry) o.geometry.dispose();
  if (o.dispose) o.dispose();
}
function addObj(o, order = 0) { o.frustumCulled = false; o.renderOrder = order; V.root.add(o); return o; }
const attr = (arr, n) => new THREE.BufferAttribute(arr, n);

function boundsOf(geometry) {
  const tpcs = geometry && geometry.meta && geometry.meta.tpcs;
  if (!tpcs || !tpcs.length) return null;
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (const b of tpcs) for (let k = 0; k < 3; k++) { lo[k] = Math.min(lo[k], b[k]); hi[k] = Math.max(hi[k], b[k + 3]); }
  return { lo, hi };
}

function setGeometry(geometry) {
  V.geometry = geometry;
  if (!V.ready) return;
  disposeObj(V.tpc); disposeObj(V.axes); disposeObj(V.pd); V.tpc = V.axes = V.pd = null; V.pdN = 0;
  const bb = boundsOf(geometry);
  if (bb) {
    // TPC wireframes
    const tpcs = geometry.meta.tpcs;
    const pos = new Float32Array(tpcs.length * 24 * 3);
    const E = [[0, 1], [1, 3], [3, 2], [2, 0], [4, 5], [5, 7], [7, 6], [6, 4], [0, 4], [1, 5], [2, 6], [3, 7]];
    let k = 0;
    for (const b of tpcs) {
      const c = (i) => [i & 1 ? b[3] : b[0], i & 2 ? b[4] : b[1], i & 4 ? b[5] : b[2]];
      for (const [a, d] of E) { const p = c(a), q = c(d); pos.set(p, k); pos.set(q, k + 3); k += 6; }
    }
    const g = new THREE.BufferGeometry(); g.setAttribute('position', attr(pos, 3));
    V.tpc = addObj(new THREE.LineSegments(g, M.tpc), -1);
    buildAxes(bb);
    fitCamera(bb, geometry.meta.detector);
  }
  buildPd(geometry);
  applyTheme();
  applyOverlays();
  requestRender();
}

function buildAxes(bb) {
  const ext = Math.max(bb.hi[0] - bb.lo[0], bb.hi[1] - bb.lo[1], bb.hi[2] - bb.lo[2]);
  const L = ext * 0.12;
  const o = bb.lo;
  const pos = new Float32Array([o[0], o[1], o[2], o[0] + L, o[1], o[2],
                                 o[0], o[1], o[2], o[0], o[1] + L, o[2],
                                 o[0], o[1], o[2], o[0], o[1], o[2] + L]);
  const col = new Float32Array([1, .3, .3, 1, .3, .3, .3, 1, .3, .3, 1, .3, .4, .6, 1, .4, .6, 1]);
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', attr(pos, 3)); g.setAttribute('color', attr(col, 3));
  V.axes = addObj(new THREE.LineSegments(g, new THREE.LineBasicMaterial({ vertexColors: true })), 1);
  // end labels (positions in detector coordinates)
  V.axisLabels.forEach((l) => l.el.remove());
  V.axisLabels = [
    mkLabel('x (drift)', [o[0] + L * 1.15, o[1], o[2]]),
    mkLabel('y (up)', [o[0], o[1] + L * 1.15, o[2]]),
    mkLabel('z (beam)', [o[0], o[1], o[2] + L * 1.15]),
  ];
  V.axisLabels.forEach((l, i) => { l.el.style.color = `var(--axis-${'xyz'[i]})`; });
}

function mkLabel(text, pos, color) {
  const el = document.createElement('div');
  el.className = 'hud-label'; el.textContent = text; if (color) el.style.color = color;
  V.labelLayer.appendChild(el);
  return { el, x: pos[0], y: pos[1], z: pos[2], shown: true };
}

function fitCamera(bb, detector, force = false) {
  const key = `${detector}|${bb.lo}|${bb.hi}`;
  if (!force && key === V.fitKey) return;
  V.fitKey = key;
  const c = [(bb.lo[0] + bb.hi[0]) / 2, (bb.lo[1] + bb.hi[1]) / 2, (bb.lo[2] + bb.hi[2]) / 2];
  const D = Math.hypot(bb.hi[0] - bb.lo[0], bb.hi[1] - bb.lo[1], bb.hi[2] - bb.lo[2]);
  const t = sceneOf(c[0], c[1], c[2], new THREE.Vector3());
  V.controls.target.copy(t);
  V.camera.position.set(t.x + 0.5 * D, t.y + 0.55 * D, t.z + 1.25 * D);
  V.camera.near = D / 500; V.camera.far = D * 30;
  V.camera.updateProjectionMatrix();
  V.controls.update();
}

function resetView() {
  const bb = boundsOf(V.geometry);
  if (bb) fitCamera(bb, V.geometry.meta.detector, true);
  requestRender();
}

function buildPd(geometry) {
  const xyz = geometry && geometry.get('pd/xyz');
  const half = geometry && geometry.get('pd/half');
  if (!xyz || !half) return;
  const n = xyz.length / 3;
  const q = new Float32Array([0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1]);
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', attr(q, 3));
  g.setIndex([0, 1, 2, 0, 2, 3]);
  const mesh = new THREE.InstancedMesh(g, M.pd, n);
  const m = mesh.instanceMatrix.array;
  for (let i = 0; i < n; i++) {   // translate(xyz) * scale(1, hy, hz), column-major
    const o = 16 * i;
    m.fill(0, o, o + 16);
    m[o] = 1; m[o + 5] = half[2 * i]; m[o + 10] = half[2 * i + 1]; m[o + 15] = 1;
    m[o + 12] = xyz[3 * i]; m[o + 13] = xyz[3 * i + 1]; m[o + 14] = xyz[3 * i + 2];
  }
  mesh.instanceMatrix.needsUpdate = true;
  mesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(3 * n), 3);
  mesh.instanceColor.setUsage(THREE.DynamicDrawUsage);
  V.pd = addObj(mesh, 2);
  V.pdN = n;
  V.pdCustom = false;
  fillPdNeutral();
}

function fillPdNeutral() {
  if (!V.pd) return;
  const c = hexToRgb(V.app.theme.pd_base);
  const a = V.pd.instanceColor.array;
  for (let i = 0; i < V.pdN; i++) { a[3 * i] = c[0]; a[3 * i + 1] = c[1]; a[3 * i + 2] = c[2]; }
  V.pd.instanceColor.needsUpdate = true;
}

/** rgb: Float32Array(n_tiles*3) in 0..1, or null for neutral walls. */
function setPdColors(rgb, alpha) {
  if (!V.pd) return;
  if (!rgb) { V.pdCustom = false; fillPdNeutral(); M.pd.uniforms.uAlpha.value = V.pdAlpha; }
  else {
    V.pdCustom = true;
    V.pd.instanceColor.array.set(rgb);
    V.pd.instanceColor.needsUpdate = true;
    M.pd.uniforms.uAlpha.value = alpha != null ? alpha : 0.9;
  }
  requestRender();
}

// ------------------------------------------------------------------ event
function clearEvent() {
  for (const k of ['sp', 'tracks', 'showers', 'markers', 'mc', 'dep', 'flashes']) { disposeObj(V[k]); V[k] = null; }
  V.spCat.clear(); V.spVal.clear(); V.stats.clear(); V.trackCat = null;
  V.selAttrs.clear(); V.pk = null; V.showerPos = null; V.lastPickObj = null;
  V.labels = [];
  for (const l of V.labelPool) l.el.style.display = 'none';
}

function setEvent(bundle, geometry) {
  if (!V.ready) return;
  V.bundle = bundle;
  clearEvent();
  if (geometry && geometry !== V.geometry) setGeometry(geometry);
  if (!bundle) { requestRender(); return; }
  buildSpacePoints(bundle);
  buildTracks(bundle);
  buildShowers(bundle);
  buildTruth(bundle);
  buildDeposits(bundle);
  buildLabels(bundle);
  buildMarkers(bundle);
  buildFlashes(bundle);
  V.title.textContent = (bundle.meta.truth && bundle.meta.truth.headline) || '';
  applyAll();
  applySelection();
  // upload the per-object group ids with the first paint so the first hover does not pay for it
  for (const [obj, what] of [[V.sp, 'sp'], [V.tracks, 'tracks'], [V.showers, 'showers']]) {
    const a = obj && !obj.geometry.attributes.aSel ? selAttrFor(what, 'pfparticle') : null;
    if (a) obj.geometry.setAttribute('aSel', a);
  }
  schedulePrewarm();
  render();   // first paint, synchronous
}

function buildSpacePoints(b) {
  const xyz = b.get('sp/xyz');
  if (!xyz || !xyz.length) return;
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', attr(xyz, 3));
  g.setDrawRange(0, xyz.length / 3);
  V.sp = addObj(new THREE.Points(g, M.sp), 0);
}

function buildTracks(b) {
  const pts = b.get('tracks/points'), off = b.get('tracks/offsets');
  if (!pts || !off || off.length < 2) return;
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', attr(pts, 3));
  g.setIndex(new THREE.BufferAttribute(polylineIndex(off), 1));
  V.tracks = addObj(new THREE.LineSegments(g, M.tracks), 1);
}

function buildShowers(b) {
  const st = b.get('showers/start'), dir = b.get('showers/dir'), len = b.get('showers/length');
  if (!st || !dir || !len || !len.length) return;
  const ang = b.get('showers/angle');
  const n = len.length;
  const pos = new Float32Array(n * 5 * 2 * 3);
  let k = 0;
  const d = new THREE.Vector3(), p1 = new THREE.Vector3(), p2 = new THREE.Vector3();
  for (let i = 0; i < n; i++) {
    d.set(dir[3 * i], dir[3 * i + 1], dir[3 * i + 2]).normalize();
    // two unit vectors perpendicular to the axis
    p1.set(0, 1, 0); if (Math.abs(d.y) > 0.9) p1.set(1, 0, 0);
    p1.cross(d).normalize(); p2.copy(d).cross(p1).normalize();
    const L = len[i];
    const r = L * Math.tan(Math.min(ang ? ang[i] : 0.2, 1.2));
    const s = [st[3 * i], st[3 * i + 1], st[3 * i + 2]];
    const e = [s[0] + L * d.x, s[1] + L * d.y, s[2] + L * d.z];
    pos.set(s, k); pos.set(e, k + 3); k += 6;      // axis
    for (const [a, c] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {   // cone edges
      pos.set(s, k);
      pos[k + 3] = e[0] + r * (a * p1.x + c * p2.x);
      pos[k + 4] = e[1] + r * (a * p1.y + c * p2.y);
      pos[k + 5] = e[2] + r * (a * p1.z + c * p2.z);
      k += 6;
    }
  }
  V.showerPos = pos;
  const g = new THREE.BufferGeometry(); g.setAttribute('position', attr(pos, 3));
  V.showers = addObj(new THREE.LineSegments(g, M.shower), 1);
}

function buildTruth(b) {
  const pts = b.get('mc/points'), off = b.get('mc/offsets');
  if (!pts || !off || off.length < 2) return;
  const n = off.length - 1;
  const pdg = b.get('mc/pdg'), fi = b.get('mc/from_interaction');
  const perTraj = new Uint8Array(n);
  for (let i = 0; i < n; i++) perTraj[i] = (pdg ? pdgClass(pdg[i]) : 8) + (fi && !fi[i] ? 0 : 16);
  const meta = expandRagged(off, perTraj, Uint8Array);
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', attr(pts, 3));
  g.setAttribute('aMeta', attr(meta, 1));
  // Index order: segments of from_interaction trajectories first, radiologicals after, so the
  // radiologicals toggle is just a draw range (hidden ones never reach the vertex shader).
  const idx = new Uint32Array(2 * Math.max(0, off[n] - off[0] - n));
  let k = 0;
  const emit = (want) => {
    for (let i = 0; i < n; i++) {
      if (!!(fi ? fi[i] : 1) !== want) continue;
      const e = off[i + 1] - 1;
      for (let j = off[i]; j < e; j++) { idx[k++] = j; idx[k++] = j + 1; }
    }
  };
  emit(true); const nInt = k; emit(false);
  g.setIndex(new THREE.BufferAttribute(idx.subarray(0, k), 1));
  V.mc = addObj(new THREE.LineSegments(g, M.mc), 0);
  V.mc.userData.nInt = nInt; V.mc.userData.nAll = k;
  applyMcRange();
}

function applyMcRange() {
  if (!V.mc) return;
  const u = V.mc.userData;
  V.mc.geometry.setDrawRange(0, V.app.state.ov.rad ? u.nAll : u.nInt);
}

function buildDeposits(b) {
  const xyz = b.get('dep/xyz');
  if (!xyz || !xyz.length) return;
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', attr(xyz, 3));
  const fi = b.get('dep/from_interaction');
  M.dep.uniforms.uHasFlag.value = fi ? 1 : 0;
  if (fi) g.setAttribute('aFlag', attr(fi, 1));
  V.dep = addObj(new THREE.Points(g, M.dep), 0);
}

function buildLabels(b) {
  const pos = b.get('labels/pos');
  if (!pos) return;
  const text = b.get('labels/text');
  const texts = b.meta.label_texts || b.meta.labels || [];
  const n = Math.min(pos.length / 3, 400);
  for (let i = 0; i < n; i++) {
    let l = V.labelPool[i];
    if (!l) { l = mkLabel('', [0, 0, 0]); V.labelPool[i] = l; }
    l.el.textContent = text ? (texts[text[i]] ?? String(text[i])) : String(i);
    l.x = pos[3 * i]; l.y = pos[3 * i + 1]; l.z = pos[3 * i + 2];
    l.el.style.color = 'var(--fg)';
    l.shown = true;
    V.labels.push(l);
  }
}

function buildMarkers(b) {
  const vx = b.get('vtx/xyz'), role = b.get('vtx/role');
  const tv = b.meta.truth && b.meta.truth.vertex;
  const n = (vx ? vx.length / 3 : 0) + (tv ? 1 : 0);
  if (!n) return;
  const pos = new Float32Array(3 * n);
  const col = new Float32Array(3 * n), size = new Float32Array(n);
  const nv = vx ? vx.length / 3 : 0;
  if (vx) pos.set(vx, 0);
  for (let i = 0; i < nv; i++) size[i] = role && role[i] === 1 ? 9 : 5;
  if (tv) { pos.set(tv, 3 * nv); size[nv] = 13; }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', attr(pos, 3));
  g.setAttribute('aColor', attr(col, 3));
  g.setAttribute('aSize', attr(size, 1));
  V.markers = addObj(new THREE.Points(g, M.marker), 3);
  V.markers.userData.nv = nv; V.markers.userData.hasTrue = !!tv;
  V.markers.userData.base = size.slice(); V.markers.userData.role = role || null;
  colorMarkers();
}


// Flash bars (FlashDisplay3D): each flash in the default window (t0 +- 50 us), the 200 brightest,
// is a bar across the drift at its (y, z) -- a square dot if the finder set x. Built once per
// event; the window / "up to time t" test is two uniforms in the vertex shader.
function buildFlashes(b) {
  V.flashWin = null;
  const ft = b.get('flash/time'), fpe = b.get('flash/pe'), fy = b.get('flash/y'), fz = b.get('flash/z');
  if (!ft || !ft.length || !fy || !fz) return;
  const fx = b.get('flash/x');
  const opm = b.meta.optical || {};
  let t0 = Number.isFinite(opm.t0) ? opm.t0 : null;
  if (t0 == null) { let k = 0; for (let i = 1; i < ft.length; i++) if (fpe[i] > fpe[k]) k = i; t0 = ft[k]; }
  const lo = t0 - 50, hi = t0 + 50;
  const sel = [];
  for (let i = 0; i < ft.length; i++) if (Number.isFinite(fy[i]) && Number.isFinite(fz[i]) && ft[i] >= lo && ft[i] <= hi) sel.push(i);
  sel.sort((p, q) => fpe[q] - fpe[p]);
  const total = sel.length;
  sel.length = Math.min(sel.length, 200);
  sel.reverse();                                   // brightest drawn last
  V.flashWin = { t0, lo, hi, shown: sel.length, total };
  const bb = boundsOf(V.geometry);
  const xlo = bb ? bb.lo[0] : -1, xhi = bb ? bb.hi[0] : 1;
  const n = sel.length;
  if (!n) return;
  const pos = new Float32Array(n * 12), oth = new Float32Array(n * 12), side = new Float32Array(n * 4), info = new Float32Array(n * 16);
  const idx = new Uint32Array(n * 6);
  const p0 = 0, p1 = Math.log10(3e5);
  sel.forEach((i, j) => {
    const f = Math.min(1, Math.max(0, (Math.log10(Math.max(fpe[i], 1)) - p0) / (p1 - p0)));
    const w = 2 + 6 * f, alpha = 0.5 + 0.45 * f;
    const known = fx && Number.isFinite(fx[i]);
    const e = [[known ? fx[i] : xlo, fy[i], fz[i]], [known ? fx[i] : xhi, fy[i], fz[i]]];
    for (let end = 0; end < 2; end++) for (let s = 0; s < 2; s++) {
      const v = 4 * j + 2 * end + s;
      pos.set(e[end], 3 * v); oth.set(e[1 - end], 3 * v);
      side[v] = s ? 1 : -1;
      info.set([ft[i] - t0, w, alpha, end], 4 * v);
    }
    idx.set([4 * j, 4 * j + 1, 4 * j + 2, 4 * j + 1, 4 * j + 3, 4 * j + 2], 6 * j);
  });
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', attr(pos, 3)); g.setAttribute('aOther', attr(oth, 3));
  g.setAttribute('aSide', attr(side, 1)); g.setAttribute('aInfo', attr(info, 4));
  g.setIndex(new THREE.BufferAttribute(idx, 1));
  V.flashes = addObj(new THREE.Mesh(g, M.flash), 4);
  M.flash.uniforms.uTLo.value = -50; M.flash.uniforms.uTHi.value = 50;
  applyFlashTime();
}

function applyFlashTime() {
  const w = V.flashWin;
  M.flash.uniforms.uTMax.value = (V.flashTime == null || !w) ? 50 : V.flashTime - w.t0;
}

/** Show flashes with time <= t (absolute us), or all of the window with null. */
function setFlashTime(t) {
  V.flashTime = (t == null || !Number.isFinite(t)) ? null : t;
  applyFlashTime();
  requestRender();
}

function flashesWanted() {
  const st = V.app.state;
  return st.view === 'flash3d' || (st.view === 'charge+light' && st.fbars === '1');
}

function flashLegend() {
  const w = V.flashWin, t = V.app.theme;
  if (!flashesWanted()) return '';
  if (!w) return '<div>no flashes in this event</div>';
  const cap = w.shown < w.total ? `, brightest ${w.shown} of ${w.total}` : '';
  return `<div style="margin-top:6px">flash time [µs], ${w.shown} bars${cap}</div>` +
    `<div class="bar" style="background:${cmapGradientCss(V.app.state.cmap, t.cmap_floor, t.cmap_ceiling)}"></div>` +
    `<div class="ticks"><span>${fmt(w.lo)}</span><span>${fmt(w.hi)}</span></div>` +
    '<div>width, opacity ~ log PE; x unmeasured = bar across drift</div>';
}

function colorMarkers() {
  if (!V.markers) return;
  const t = V.app.theme, a = V.markers.geometry.attributes.aColor.array;
  const nv = V.markers.userData.nv, v = hexToRgb(t.vertex);
  for (let i = 0; i < nv; i++) a.set(v, 3 * i);
  if (V.markers.userData.hasTrue) a.set(hexToRgb(t.truth_vertex), 3 * nv);
  V.markers.geometry.attributes.aColor.needsUpdate = true;
}

// ------------------------------------------------------------------ appearance
function setVec(v, hex) { const c = hexToRgb(hex); v.set(c[0], c[1], c[2]); }

function applyTheme() {
  if (!V.ready) return;
  const t = V.app.theme;
  V.renderer.setClearColor(new THREE.Color(t.axes_bg), 1);
  M.tpc.color.set(t.wire); setVec(M.shower.uniforms.uColor.value, t.shower);
  setVec(V.uni.uBg.value, t.axes_bg); setVec(V.uni.uHi.value, t.name === 'light' ? '#c2410c' : '#fde047');
  setVec(M.tracks.uniforms.uColor.value, t.track);
  setVec(M.mc.uniforms.uColor.value, t.mctrack);
  setVec(M.dep.uniforms.uUniform.value, t.truth);
  setVec(M.marker.uniforms.uCasing.value, t.axes_bg);
  setVec(M.pd.uniforms.uEdge.value, t.pd_edge);
  setVec(M.sp.uniforms.uUnassoc.value, t.unassociated);
  setVec(M.sp.uniforms.uUniform.value, t.unassociated);
  t.categorical.forEach((h, i) => setVec(V.uni.uPal.value[i], h));
  const key = t.name;
  PDG_CLASSES.forEach((c, i) => setVec(V.uni.uPdg.value[i], c[key]));
  V.uni.uCmapLo.value = t.cmap_floor; V.uni.uCmapHi.value = t.cmap_ceiling;
  colorMarkers();
  if (!V.pdCustom) fillPdNeutral();
  M.pd.uniforms.uAlpha.value = V.pdCustom ? M.pd.uniforms.uAlpha.value : V.pdAlpha;
}

function applyOverlays() {
  const st = V.app.state, ov = st.ov;
  const fm = st.view === 'flash3d';
  if (V.sp) V.sp.visible = !fm;
  if (V.tracks) V.tracks.visible = ov.tracks && !fm;
  if (V.showers) V.showers.visible = ov.showers && !fm;
  if (V.mc) V.mc.visible = ov.truth && !fm;
  if (V.dep) V.dep.visible = ov.dep && !fm;
  if (V.markers) {
    // per-vertex show/hide through aSize (<= 0 is clipped in the shader): no rebuild
    const u = V.markers.userData, a = V.markers.geometry.attributes.aSize.array, base = u.base;
    const showP = ov.pvtx !== false, showD = !!ov.vtx;
    let any = false;
    for (let i = 0; i < u.nv; i++) {
      const on = !fm && (u.role && u.role[i] === 1 ? showP : showD);
      a[i] = on ? base[i] * (V.app.selection ? (V.app.vertexSelected(i) ? 1.4 : 0.35) : 1) : 0; any = any || on;   // selection: other vertices shrink
    }
    if (u.hasTrue) { const on = fm || ov.truth; a[u.nv] = on ? base[u.nv] : 0; any = any || on; }
    V.markers.geometry.attributes.aSize.needsUpdate = true;
    V.markers.visible = any;
  }
  if (V.flashes) V.flashes.visible = flashesWanted();
  if (V.pd) V.pd.visible = ov.pd;
  if (!V.pdCustom) M.pd.uniforms.uAlpha.value = fm ? 0.1 : V.pdAlpha;
  V.uni.uShowRad.value = ov.rad ? 1 : 0;
  applyMcRange();
  for (const l of V.labels) l.shown = ov.labels && !fm;
  for (const l of V.labelPool) if (!ov.labels || fm) l.el.style.display = 'none';
  if (V.fbox) {
    V.fbox.style.display = st.view === 'charge+light' ? '' : 'none';
    V.fbox.firstChild.checked = st.fbars === '1';
  }
  V.camKey[0] = NaN;   // force label refresh
}

// value/stat cache ------------------------------------------------
function statsFor(name, arr) {
  let s = V.stats.get(name);
  if (!s) {
    const [lo, hi] = percentileRange(arr, 0.02, 0.98);
    s = { lo, hi, log: lo > 0 && hi / lo > 30 };
    V.stats.set(name, s);
  }
  return s;
}

function coordStats(axis) {
  const key = 'xyz' + axis;
  let s = V.stats.get(key);
  if (!s) {
    const xyz = V.bundle.get('sp/xyz');
    const n = xyz.length / 3, step = Math.max(1, Math.floor(n / 20000));
    const v = [];
    for (let i = 0; i < n; i += step) v.push(xyz[3 * i + axis]);
    v.sort((a, b) => a - b);
    s = { lo: v[Math.floor(0.01 * (v.length - 1))], hi: v[Math.floor(0.99 * (v.length - 1))] };
    if (!(s.hi > s.lo)) s.hi = s.lo + 1;
    V.stats.set(key, s);
  }
  return s;
}

function catAttr(kind) {
  let a = V.spCat.get(kind);
  if (a === undefined) {
    const src = V.bundle.get(`sp/group/${kind}`);
    a = src ? attr(Float32Array.from(src), 1) : null;
    V.spCat.set(kind, a);
  }
  return a;
}

function valAttr(name) {
  let a = V.spVal.get(name);
  if (a === undefined) {
    const src = V.bundle.get(`sp/${name}`);
    a = src ? attr(src, 1) : null;
    V.spVal.set(name, a);
  }
  return a;
}

const CAT_KINDS = ['track', 'shower', 'pfparticle', 'cluster', 'slice'];

function fmt(v) { return Math.abs(v) >= 1000 || (Math.abs(v) < 0.01 && v !== 0) ? v.toExponential(1) : v.toPrecision(3).replace(/\.?0+$/, ''); }

function applyColors() {
  const st = V.app.state;
  const t = V.app.theme;
  if (V.lutName !== st.cmap) { updateLutTexture(V.lutTex, st.cmap); V.lutName = st.cmap; }
  const leg = V.legend;
  leg.innerHTML = '';
  const u = M.sp.uniforms;
  const colour = st.color;
  if (V.sp) {
    const g = V.sp.geometry;
    const axis = ['x', 'y', 'z'].indexOf(colour);
    if (CAT_KINDS.includes(colour)) {
      const a = catAttr(colour);
      if (a) { g.setAttribute('aCat', a); u.uMode.value = 1; leg.innerHTML = `colour: ${colour} id`; }
      else { u.uMode.value = 3; leg.innerHTML = `no ${colour} grouping in this event`; }
    } else if (axis >= 0) {
      const s = coordStats(axis);
      u.uMode.value = 2; u.uAxis.value = axis; u.uLo.value = s.lo; u.uInv.value = 1 / (s.hi - s.lo);
      legendBar(`${colour} (cm)`, fmt(s.lo), fmt(s.hi), st.cmap);
    } else {
      const name = (colour === 'charge' || colour === 'pdg') ? 'charge' : (valAttr(colour) ? colour : 'charge');
      const a = valAttr(name);
      if (a) {
        g.setAttribute('aValue', a);
        const s = statsFor(name, V.bundle.get(`sp/${name}`));
        const log = st.scale === 'log' || (st.scale === 'auto' && s.log);
        u.uMode.value = 0; u.uLog.value = log ? 1 : 0;
        if (log) {
          const lo = s.lo > 0 ? s.lo : Math.max(s.hi * 1e-3, 1e-6);
          u.uFloor.value = lo; u.uLo.value = Math.log(lo); u.uInv.value = 1 / (Math.log(s.hi) - Math.log(lo));
        } else { u.uLo.value = s.lo; u.uInv.value = 1 / (s.hi - s.lo); }
        const missing = (colour !== name && !(colour === 'pdg')) ? ` [no sp/${colour} in this event: showing charge]` : '';
        legendBar(`${name}${log ? ' (log)' : ''}${missing}`, fmt(s.lo), fmt(s.hi), st.cmap);
      } else { u.uMode.value = 3; }
    }
  }
  // tracks
  if (V.tracks) {
    const catTracks = colour === 'track' && V.bundle.get('tracks/offsets');
    if (catTracks) {
      if (!V.trackCat) {
        const off = V.bundle.get('tracks/offsets'), id = V.bundle.get('tracks/id');
        const per = new Float32Array(off.length - 1);
        for (let i = 0; i < per.length; i++) per[i] = id ? id[i] : i;
        V.trackCat = attr(expandRagged(off, per), 1);
      }
      V.tracks.geometry.setAttribute('aMeta', V.trackCat);
    }
    M.tracks.uniforms.uMode.value = catTracks ? 2 : 0;
  }
  // truth
  const pdgMode = colour === 'pdg';
  M.mc.uniforms.uMode.value = pdgMode ? 1 : 0;
  if (pdgMode && V.mc && V.app.state.ov.truth) {
    leg.innerHTML += PDG_CLASSES.map((c) => `<div><span class="sw" style="background:${c[t.name]}"></span>${c.name}</div>`).join('');
  }
  if (st.view === 'flash3d') leg.innerHTML = '';   // charge is hidden: the legend is the flash time bar
  leg.innerHTML += flashLegend();
}

function legendBar(title, lo, hi, cmap) {
  const t = V.app.theme;
  V.legend.innerHTML = `<div>${title}</div><div class="bar" style="background:${cmapGradientCss(cmap, t.cmap_floor, t.cmap_ceiling)}"></div>` +
    `<div class="ticks"><span>${lo}</span><span>${hi}</span></div>`;
}

function applyAll() { applyTheme(); applyColors(); applyOverlays(); }

function onState(state, changed) {
  if (!V.ready) return;
  if (changed.has('theme')) { applyTheme(); }
  if (['theme', 'color', 'scale', 'cmap'].some((k) => changed.has(k))) { if (V.bundle) applyColors(); }
  if (changed.has('ov') || changed.has('theme')) applyOverlays();
  if (changed.has('view') || changed.has('fbars') || (changed.has('ov') && V.bundle)) {
    applyOverlays();
    if (V.bundle) applyColors();   // legend depends on truth visibility and on the flash bars
  }
  if (changed.has('dim')) { V.uni.uDim.value = V.app.selDim ? V.app.selDim() : 0.15; requestRender(); }
  if (changed.has('view')) applySelection();
  requestRender();
}


// ------------------------------------------------------------------ selection highlight (select.js)
/** group id attribute (float) of the selection's group kind, for 'sp' | 'tracks' | 'showers'; null if the data lacks it */
function selAttrFor(what, gk) {
  const key = what + ':' + gk;
  if (V.selAttrs.has(key)) return V.selAttrs.get(key);
  const b = V.bundle;
  let a = null;
  if (what === 'sp') {
    const src = b.get(`sp/group/${gk}`);
    a = src ? attr(Float32Array.from(src), 1) : null;
  } else {
    const own = b.get(what === 'tracks' ? 'tracks/pfp' : 'showers/pfp'), slice = b.get('pfp/slice');
    if (own && (gk === 'pfparticle' || slice)) {
      const per = new Float32Array(own.length);
      for (let i = 0; i < per.length; i++) per[i] = gk === 'pfparticle' ? own[i] : (own[i] >= 0 ? slice[own[i]] : -1);
      if (what === 'tracks') { const off = b.get('tracks/offsets'); a = attr(expandRagged(off, per), 1); }
      else { const v = new Float32Array(per.length * 10); for (let i = 0; i < per.length; i++) v.fill(per[i], 10 * i, 10 * i + 10); a = attr(v, 1); }
    }
  }
  V.selAttrs.set(key, a);
  return a;
}

function applySelection() {
  if (!V.ready) return;
  V.uni.uDim.value = V.app.selDim ? V.app.selDim() : 0.15;
  const sel = V.app.selection, fm = V.app.state.view === 'flash3d';
  const on = !!sel && !fm && !!V.bundle;
  if (on) {
    const W = 1024, H = Math.max(1, Math.ceil(sel.n / W));
    const data = new Uint8Array(W * H); for (let i = 0; i < sel.mask.length; i++) data[i] = sel.mask[i] ? 255 : 0;   // R8 is normalised: 1 would read as 1/255
    const tex = new THREE.DataTexture(data, W, H, THREE.RedFormat, THREE.UnsignedByteType);
    tex.minFilter = tex.magFilter = THREE.NearestFilter; tex.needsUpdate = true;
    if (V.selTexObj) V.selTexObj.dispose();
    V.selTexObj = tex; V.uni.uSelTex.value = tex; V.uni.uSelH.value = H;
  }
  const set = (obj, mat, what) => {
    if (!obj) return;
    const a = on ? selAttrFor(what, sel.groupKind) : null;
    if (a) obj.geometry.setAttribute('aSel', a);
    mat.uniforms.uSelOn.value = a ? 1 : 0;
  };
  set(V.sp, M.sp, 'sp'); set(V.tracks, M.tracks, 'tracks'); set(V.showers, M.shower, 'showers');
  M.mc.uniforms.uDimAll.value = on ? 1 : 0;           // true trajectories fade too
  if (V.markers) applyOverlays();                      // vertex marker sizes follow the selection
  requestRender();
}

/** Recentre the orbit target on the selected object; the camera keeps its distance and direction. */
function focusSelection() {
  const sel = V.app.selection, b = V.bundle;
  if (!sel || !b) return;
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  const add = (x, y, z) => { if (!Number.isFinite(x + y + z)) return; lo[0] = Math.min(lo[0], x); hi[0] = Math.max(hi[0], x); lo[1] = Math.min(lo[1], y); hi[1] = Math.max(hi[1], y); lo[2] = Math.min(lo[2], z); hi[2] = Math.max(hi[2], z); };
  const xyz = b.get('sp/xyz'), g = b.get(`sp/group/${sel.groupKind}`);
  if (xyz && g) for (let i = 0; i < g.length; i++) { const id = g[i]; if (id >= 0 && id < sel.n && sel.mask[id]) add(xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2]); }
  if (!(lo[0] <= hi[0])) {
    const a = selAttrFor('tracks', sel.groupKind), pts = b.get('tracks/points');
    if (a && pts) { const ids = a.array; for (let i = 0; i < ids.length; i++) if (ids[i] >= 0 && sel.mask[ids[i]]) add(pts[3 * i], pts[3 * i + 1], pts[3 * i + 2]); }
  }
  if (!(lo[0] <= hi[0])) return;
  const c = sceneOf((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2, new THREE.Vector3());
  const d = c.clone().sub(V.controls.target);
  V.controls.target.add(d); V.camera.position.add(d);
  V.controls.update();
  requestRender();
}

// ------------------------------------------------------------------ picking (CPU, screen space)
// Space points: projected once per camera/size change into an 8 px bucket grid (only points that
// belong to some PFParticle/slice), then each mouse move is a 3x3 cell lookup; the frontmost point
// within 6 px wins. Track and shower segments: projected once, then a point-segment distance scan.
const CELL = 8, R_SP = 6, R_LINE = 5;

function pickState() {
  const b = V.bundle;
  if (!b) return null;
  if (V.pk && V.pk.bundle === b) return V.pk;
  const xyz = b.get('sp/xyz'), gp = b.get('sp/group/pfparticle'), gs = b.get('sp/group/slice');
  const cand = [];
  if (xyz && (gp || gs)) {
    const n = xyz.length / 3;
    for (let i = 0; i < n; i++) if ((gp && gp[i] >= 0) || (gs && gs[i] >= 0)) cand.push(i);
  }
  const tp = b.get('tracks/points'), to = b.get('tracks/offsets'), tpfp = b.get('tracks/pfp');
  const sp = V.showerPos, spfp = b.get('showers/pfp');
  V.pk = {
    bundle: b, xyz, gp, gs, cand: Int32Array.from(cand),
    sx: new Float32Array(cand.length), sy: new Float32Array(cand.length), sz: new Float32Array(cand.length),
    tp, to, tpfp, tx: tp ? new Float32Array(tp.length / 3) : null, ty: tp ? new Float32Array(tp.length / 3) : null,
    shp: sp, spfp, hx: sp ? new Float32Array(sp.length / 3) : null, hy: sp ? new Float32Array(sp.length / 3) : null,
    key: new Float64Array(10).fill(NaN), grid: null, gw: 0, gh: 0, mat: new THREE.Matrix4(),
  };
  return V.pk;
}

function projectAll(P) {
  const c = V.camera, p = c.position, q = c.quaternion, t = V.controls.target;
  const w = V.container.clientWidth, h = V.container.clientHeight, k = P.key;
  if (k[0] === p.x && k[1] === p.y && k[2] === p.z && k[3] === q.x && k[4] === q.y && k[5] === q.z && k[6] === q.w && k[7] === w && k[8] === h && k[9] === c.fov) return;
  k[0] = p.x; k[1] = p.y; k[2] = p.z; k[3] = q.x; k[4] = q.y; k[5] = q.z; k[6] = q.w; k[7] = w; k[8] = h; k[9] = c.fov;
  c.updateMatrixWorld(); V.root.updateMatrixWorld(true);
  const e = P.mat.multiplyMatrices(c.projectionMatrix, c.matrixWorldInverse).multiply(V.root.matrixWorld).elements;
  const hw = 0.5 * w, hh = 0.5 * h;
  const proj = (x, y, z, ox, oy, oz, i) => {
    const cw = e[3] * x + e[7] * y + e[11] * z + e[15];
    if (!(cw > 1e-6)) { ox[i] = NaN; return; }
    ox[i] = (e[0] * x + e[4] * y + e[8] * z + e[12]) / cw * hw + hw;
    oy[i] = h - ((e[1] * x + e[5] * y + e[9] * z + e[13]) / cw * hh + hh);
    if (oz) oz[i] = (e[2] * x + e[6] * y + e[10] * z + e[14]) / cw;
  };
  const cand = P.cand, xyz = P.xyz;
  const gw = Math.ceil(w / CELL) + 1, gh = Math.ceil(h / CELL) + 1;
  const cellOf = new Int32Array(cand.length), start = new Int32Array(gw * gh + 1);
  for (let j = 0; j < cand.length; j++) {
    const i = cand[j];
    proj(xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2], P.sx, P.sy, P.sz, j);
    const x = P.sx[j], y = P.sy[j];
    if (x >= 0 && x < w && y >= 0 && y < h) { const cc = Math.floor(y / CELL) * gw + Math.floor(x / CELL); cellOf[j] = cc; start[cc + 1]++; } else cellOf[j] = -1;
  }
  for (let i = 0; i < gw * gh; i++) start[i + 1] += start[i];
  const fill = start.slice(0, gw * gh), items = new Int32Array(start[gw * gh]);
  for (let j = 0; j < cand.length; j++) if (cellOf[j] >= 0) items[fill[cellOf[j]]++] = j;
  P.grid = { start, items }; P.gw = gw; P.gh = gh;
  if (P.tp) for (let i = 0, n = P.tx.length; i < n; i++) proj(P.tp[3 * i], P.tp[3 * i + 1], P.tp[3 * i + 2], P.tx, P.ty, null, i);
  if (P.shp) for (let i = 0, n = P.hx.length; i < n; i++) proj(P.shp[3 * i], P.shp[3 * i + 1], P.shp[3 * i + 2], P.hx, P.hy, null, i);
}

function segDist2(px, py, ax, ay, bx, by) {
  const dx = bx - ax, dy = by - ay, l2 = dx * dx + dy * dy;
  let t = l2 > 0 ? ((px - ax) * dx + (py - ay) * dy) / l2 : 0;
  t = t < 0 ? 0 : t > 1 ? 1 : t;
  const ex = ax + t * dx - px, ey = ay + t * dy - py;
  return ex * ex + ey * ey;
}

/** -> {pfp, slice, what, d} for the reco object under the cursor (css px relative to the canvas), or null */
function pickAt(mx, my) {
  const P = pickState();
  if (!P || !V.visible || V.app.state.view === 'flash3d') return null;
  projectAll(P);
  let best = null, bd = Infinity;
  if (V.sp && V.sp.visible && P.grid) {
    const { start, items } = P.grid, gw = P.gw, gh = P.gh;
    const c0 = Math.max(0, Math.floor((mx - R_SP) / CELL)), c1 = Math.min(gw - 1, Math.floor((mx + R_SP) / CELL));
    const r0 = Math.max(0, Math.floor((my - R_SP) / CELL)), r1 = Math.min(gh - 1, Math.floor((my + R_SP) / CELL));
    let bz = Infinity, bj = -1;
    for (let r = r0; r <= r1; r++) for (let c = c0; c <= c1; c++) {
      const cc = r * gw + c;
      for (let a = start[cc]; a < start[cc + 1]; a++) {
        const j = items[a], dx = P.sx[j] - mx, dy = P.sy[j] - my;
        if (dx * dx + dy * dy <= R_SP * R_SP && P.sz[j] < bz) { bz = P.sz[j]; bj = j; }
      }
    }
    if (bj >= 0) {
      const i = P.cand[bj], dx = P.sx[bj] - mx, dy = P.sy[bj] - my;
      bd = Math.sqrt(dx * dx + dy * dy);
      best = { pfp: P.gp ? P.gp[i] : -1, slice: P.gs ? P.gs[i] : -1, what: 'sp', d: bd };
    }
  }
  const sliceOf = (pfp) => { const s = V.bundle.get('pfp/slice'); return pfp >= 0 && s ? s[pfp] : -1; };
  if (V.tracks && V.tracks.visible && P.tp && P.tpfp) {
    const off = P.to, tx = P.tx, ty = P.ty;
    let tb = R_LINE * R_LINE, ti = -1;
    for (let t = 0; t + 1 < off.length; t++) {
      if (P.tpfp[t] < 0) continue;
      for (let j = off[t]; j < off[t + 1] - 1; j++) {
        const ax = tx[j], ay = ty[j], bx = tx[j + 1], by = ty[j + 1];
        if (ax !== ax || bx !== bx) continue;
        if ((ax < mx - R_LINE && bx < mx - R_LINE) || (ax > mx + R_LINE && bx > mx + R_LINE) || (ay < my - R_LINE && by < my - R_LINE) || (ay > my + R_LINE && by > my + R_LINE)) continue;
        const d2 = segDist2(mx, my, ax, ay, bx, by);
        if (d2 < tb) { tb = d2; ti = t; }
      }
    }
    if (ti >= 0 && Math.sqrt(tb) + 1 < bd) { bd = Math.sqrt(tb); best = { pfp: P.tpfp[ti], slice: sliceOf(P.tpfp[ti]), what: 'track', d: bd }; }
  }
  if (V.showers && V.showers.visible && P.shp && P.spfp) {
    const hx = P.hx, hy = P.hy;
    let sb = R_LINE * R_LINE, si = -1;
    for (let s = 0; s < P.spfp.length; s++) {
      if (P.spfp[s] < 0) continue;
      for (let k = 0; k < 5; k++) {
        const a = 10 * s + 2 * k, b2 = a + 1;
        if (hx[a] !== hx[a] || hx[b2] !== hx[b2]) continue;
        const d2 = segDist2(mx, my, hx[a], hy[a], hx[b2], hy[b2]);
        if (d2 < sb) { sb = d2; si = s; }
      }
    }
    if (si >= 0 && Math.sqrt(sb) + 1 < bd) { bd = Math.sqrt(sb); best = { pfp: P.spfp[si], slice: sliceOf(P.spfp[si]), what: 'shower', d: bd }; }
  }
  return best;
}

/** project in the background shortly after the camera settles, so a hover never pays for it */
function schedulePrewarm() {
  clearTimeout(V.pkTimer);
  V.pkTimer = setTimeout(() => {
    if (!V.visible || !V.bundle) return;
    const P = pickState(); if (P) projectAll(P);
  }, 150);
}

function bindPicking(el) {
  const rel = (e) => { const r = el.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
  const run = () => {
    V.pkRaf = 0;
    const m = V.mouse; if (!m) return;
    const t0 = performance.now();
    const res = pickAt(m.x, m.y);
    V.app.hoverObject(res, m.shift);
    V.pickMs.push(performance.now() - t0); if (V.pickMs.length > 200) V.pickMs.shift();
  };
  el.addEventListener('pointermove', (e) => {
    if (e.buttons) return;                                  // orbiting / panning
    const [x, y] = rel(e);
    V.mouse = { x, y, shift: e.shiftKey };
    if (!V.pkRaf) V.pkRaf = requestAnimationFrame(run);
  });
  el.addEventListener('pointerleave', () => { V.mouse = null; V.app.hoverObject(null); });
  let down = null;
  el.addEventListener('pointerdown', (e) => { const [x, y] = rel(e); down = e.button === 0 ? { x, y } : null; });
  el.addEventListener('pointerup', (e) => {
    if (!down) return;
    const [x, y] = rel(e), d = Math.hypot(x - down.x, y - down.y);
    down = null;
    if (d >= 4) return;
    const res = pickAt(x, y);
    V.lastPickObj = res;
    V.app.clickObject(res, e.shiftKey);
  });
  el.addEventListener('dblclick', () => {
    // the two clicks of a double click pinned and unpinned: pin again, then centre on it
    if (!V.app.selection && V.lastPickObj) V.app.clickObject(V.lastPickObj, false);
    focusSelection();
  });
}

// ------------------------------------------------------------------ rendering
function resize() {
  if (!V.ready || !V.visible) return;
  const w = V.container.clientWidth, h = V.container.clientHeight;
  if (w < 2 || h < 2) return;
  V.renderer.setSize(w, h, false);
  V.camera.aspect = w / h;
  V.camera.updateProjectionMatrix();
  V.camKey[0] = NaN;
  requestRender();
}

function setVisible(v) {
  V.visible = v;
  if (v) { resize(); }
}

function requestRender() {
  if (V.renderQueued || V.animating || !V.ready || !V.visible) return;
  V.renderQueued = true;
  requestAnimationFrame(() => { V.renderQueued = false; render(); });
}

function setAnimating(on) {
  if (!V.ready || V.animating === on) return;
  V.animating = on;
  V.renderer.setAnimationLoop(on ? render : null);
  if (!on) requestRender();
}

function updateLabels() {
  const c = V.camera, k = V.camKey, p = c.position, q = c.quaternion;
  const w = V.container.clientWidth, h = V.container.clientHeight;
  if (k[0] === p.x && k[1] === p.y && k[2] === p.z && k[3] === q.x && k[4] === q.y && k[5] === q.z && k[6] === q.w && k[7] === w && k[8] === h) return;
  k[0] = p.x; k[1] = p.y; k[2] = p.z; k[3] = q.x; k[4] = q.y; k[5] = q.z; k[6] = q.w; k[7] = w; k[8] = h;
  c.updateMatrixWorld();
  const place = (l) => {
    const v = sceneOf(l.x, l.y, l.z, V.scratch).project(c);
    const el = l.el;
    if (v.z > 1 || v.z < -1 || Math.abs(v.x) > 1.1 || Math.abs(v.y) > 1.1) { el.style.display = 'none'; return; }
    el.style.display = '';
    el.style.transform = `translate(${((v.x + 1) * 0.5 * w + 4) | 0}px,${((1 - v.y) * 0.5 * h - 6) | 0}px)`;
  };
  for (const l of V.axisLabels) place(l);
  if (V.app.state.ov.labels) for (const l of V.labels) place(l);
}

function render() {
  if (!V.ready || !V.visible) return;
  V.controls.update();
  V.root.updateMatrixWorld(true);
  updateLabels();
  if (V.flashes && V.flashes.visible) V.renderer.getDrawingBufferSize(M.flash.uniforms.uRes.value);
  try { V.renderer.render(V.scene, V.camera); } catch (e) {
    if (!V.renderer.getContext().isContextLost()) throw e;   // lost mid-draw: redraw after restore
  }
}

/** Time n synchronous renders (ms/frame), draining the GPU each time. */
function benchmark(n = 10) {
  const gl = V.renderer.getContext();
  const px = new Uint8Array(4);
  const t0 = performance.now();
  for (let i = 0; i < n; i++) {
    V.camera.position.x += 1e-3 * (i & 1 ? 1 : -1);   // defeat any caching
    V.renderer.render(V.scene, V.camera);
    gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px);
  }
  return (performance.now() - t0) / n;
}

/** Camera in DETECTOR coordinates (cm): scene (sx,sy,sz) = (z, y, -x), so x = -sz, y = sy, z = sx. */
function cameraInfo() {
  const c = V.camera, t = V.controls.target, u = c.up;
  const det = (v) => [-v.z, v.y, v.x];
  return { position: det(c.position), target: det(t), up: det(u), fov: c.fov, aspect: c.aspect };
}

function getCanvas() { render(); return V.renderer.domElement; }

export default {
  name: 'view3d',
  init, setGeometry, setEvent, onState, setVisible, resize, getCanvas,
  setPdColors, setAnimating, requestRender, resetView, benchmark, setFlashTime, cameraInfo, focusSelection,
  /** test hook: pick at css px (x, y) of the canvas -> {res, ms} */
  debugPick(x, y) { const t = performance.now(); const res = pickAt(x, y); return { res, ms: performance.now() - t }; },
  get pickMs() { return V.pickMs; },
  get flashWindow() { return V.flashWin; },
  get pdTileCount() { return V.pdN; },
  get camera() { return V.camera; },
  get scene() { return V.scene; },
  get root() { return V.root; },
  get controls() { return V.controls; },
  get renderer() { return V.renderer; },
};
