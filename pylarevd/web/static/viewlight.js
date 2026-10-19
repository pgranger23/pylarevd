// viewlight.js -- photon-detector light animation: PD walls face-on (z across, y up), time strip,
// playback. Port of pylarevd/light.py (LightAnimation) to the browser.
//
// Rendering: tiles are ONE instanced WebGL draw per wall (a frame change only rewrites a 4-byte
// per-tile colour/outline buffer); axes, TPC outlines and the colour bar live on a static 2-D canvas
// under it; flash ellipses, the vertex star and the clock live on a small dynamic 2-D canvas above
// it. The strip is a cached bitmap plus a playhead. In Charge+Light the same tile colours go to the
// 3-D view through view3d.setPdColors every frame.
//
// URL params (registered): lclock log|linear, lfill afterglow|cumulative, lframes, lfps, lpers,
// lwlo / lwhi (window relative to t0, us), lt0 (absolute us, '' = auto), lt (frame | 'all'), lplay 0|1.

import { getLut } from './colormaps.js';

const PE_FLOOR = 0.5, PE_OFF = 0.2, PE_SIZE_MAX = 3e5, MAX_FLASHES = 50;
const DEF = { lclock: 'log', lfill: 'afterglow', lframes: '120', lfps: '12', lpers: '0.7',
  lwlo: '-0.25', lwhi: '10', lt0: '', lt: 'all', lplay: '0' };
const REBUILD_KEYS = ['lclock', 'lfill', 'lframes', 'lpers', 'lwlo', 'lwhi', 'lt0'];

// ------------------------------------------------------------------ small helpers
const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
function lowerBound(arr, x, n = arr.length) {
  let lo = 0, hi = n;
  while (lo < hi) { const m = (lo + hi) >> 1; if (arr[m] < x) lo = m + 1; else hi = m; }
  return lo;
}
export function fmtDt(dt) {
  const a = Math.abs(dt);
  if (a < 5e-4) return '0';
  const s = dt > 0 ? '+' : '−';
  if (a < 1) return `${s}${(a * 1000).toFixed(0)} ns`;
  if (a < 100) return `${s}${a.toFixed(2)} µs`;
  return `${s}${a.toFixed(0)} µs`;
}
function fmtPe(pe) {
  if (pe >= 1e6) return `${(pe / 1e6).toFixed(1)}M`;
  if (pe >= 1e4) return `${(pe / 1e3).toFixed(0)}k`;
  if (pe >= 1e3) return `${(pe / 1e3).toFixed(1)}k`;
  return pe.toFixed(0);
}
const SUP = { '0':'\u2070','1':'\u00b9','2':'\u00b2','3':'\u00b3','4':'\u2074','5':'\u2075','6':'\u2076','7':'\u2077','8':'\u2078','9':'\u2079','-':'\u207b' };
function pow10(e) { return e <= 3 && e >= 0 ? String(10 ** e) : '10' + String(e).replace(/./g, (c) => SUP[c]); }
function niceTicks(lo, hi, target = 7) {
  const span = hi - lo;
  if (!(span > 0)) return [lo];
  const raw = span / target, mag = 10 ** Math.floor(Math.log10(raw));
  let step = 10 * mag;
  for (const m of [1, 2, 5, 10]) if (m * mag >= raw) { step = m * mag; break; }
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 0.5 * step; v += step) out.push(Math.abs(v) < 1e-9 * step ? 0 : v);
  return out;
}

// ------------------------------------------------------------------ light.py ports
/** Smallest COMMON gap between hit times (frames finer than this repeat). */
function hitTimeQuantum(t) {
  const u = Array.from(new Set(Array.from(t, (x) => Math.round(x * 1e4) / 1e4))).sort((a, b) => a - b);
  const cnt = new Map();
  for (let i = 1; i < u.length; i++) {
    const d = Math.round((u[i] - u[i - 1]) * 1e4) / 1e4;
    if (d > 5e-4) cnt.set(d, (cnt.get(d) || 0) + 1);
  }
  if (!cnt.size) return 0.016;
  const mx = Math.max(...cnt.values());
  let best = Infinity;
  for (const [d, c] of cnt) if (c >= 0.1 * mx && d < best) best = d;
  return clamp(best, 0.001, 0.1);
}

/** Start of the brightest 100 ns burst, half a quantum before its first significant hit. */
function lightOnset(t, pe, quantum, idx = null) {
  const n = idx ? idx.length : t.length;
  let lo = Infinity, tot = 0;
  for (let j = 0; j < n; j++) {
    const i = idx ? idx[j] : j;
    if (Number.isFinite(t[i]) && Number.isFinite(pe[i])) { if (t[i] < lo) lo = t[i]; tot += Math.max(pe[i], 0); }
  }
  if (!n || !(tot > 0)) return null;
  const coarse = new Map();
  for (let j = 0; j < n; j++) {
    const i = idx ? idx[j] : j;
    if (!Number.isFinite(t[i]) || !Number.isFinite(pe[i])) continue;
    const b = Math.floor((t[i] - lo) / 0.1);
    coarse.set(b, (coarse.get(b) || 0) + Math.max(pe[i], 0));
  }
  let bb = 0, bv = -1;
  for (const [b, v] of coarse) if (v > bv || (v === bv && b < bb)) { bv = v; bb = b; }
  const start = lo + bb * 0.1 - 0.3, stop = lo + (bb + 1) * 0.1;
  const fine = new Map();
  for (let j = 0; j < n; j++) {
    const i = idx ? idx[j] : j;
    const ti = t[i];
    if (!(ti >= start && ti < stop)) continue;
    const b = Math.floor((ti - start) / quantum);
    fine.set(b, (fine.get(b) || 0) + Math.max(pe[i], 0));
  }
  let p = 0, pv = -1;
  for (const [b, v] of fine) if (v > pv || (v === pv && b < p)) { pv = v; p = b; }
  const thresh = Math.max(0.02 * pv, 3);
  let k = p;
  const keys = Array.from(fine.keys()).sort((a, b) => a - b);
  for (const b of keys) { if (b <= p && fine.get(b) >= thresh) { k = b; break; } }
  let first = Infinity;
  for (let j = 0; j < n; j++) {
    const i = idx ? idx[j] : j;
    const ti = t[i];
    if (ti >= start + k * quantum && ti < start + (k + 1) * quantum && ti < first) first = ti;
  }
  if (!Number.isFinite(first)) first = start + k * quantum;
  return first - 0.5 * quantum;
}

function linspace(a, b, n) {
  const out = new Float64Array(n + 1);
  for (let i = 0; i <= n; i++) out[i] = a + (b - a) * i / n;
  return out;
}

/** n+1 edges 0..span: step max(quantum, e*(r-1)); r solved so the last edge lands on span. */
function logClock(quantum, span, n) {
  if (n * quantum >= span) return linspace(0, span, n);
  const run = (r) => {
    const e = new Float64Array(n + 1);
    for (let i = 0; i < n; i++) e[i + 1] = e[i] + Math.max(quantum, e[i] * (r - 1));
    return e;
  };
  let lo = 1, hi = 2;
  while (run(hi)[n] < span) hi *= 2;
  for (let i = 0; i < 80; i++) { const mid = 0.5 * (lo + hi); if (run(mid)[n] < span) lo = mid; else hi = mid; }
  const e = run(hi);
  const f = span / e[n];
  for (let i = 0; i <= n; i++) e[i] *= f;
  return e;
}

function frameEdges(t0, lo, hi, n, scale, quantum) {
  if (scale === 'linear' || hi <= t0 + quantum) return linspace(lo, hi, n);
  const pre = t0 - lo;
  const nPre = pre <= 0 ? 0 : Math.max(1, Math.round(0.1 * n));
  const nPost = n - nPre;
  if (nPost < 1) return linspace(lo, hi, n);
  const lc = logClock(quantum, hi - t0, nPost);
  let post = Array.from(lc.subarray(1), (x) => t0 + x);
  let head;
  if (nPre) head = Array.from(linspace(lo, t0, nPre));
  else {
    head = [Math.max(lo, t0)];
    post = post.filter((x) => x > head[0]);
    if (post.length < nPost) return linspace(lo, hi, n);
  }
  return Float64Array.from(head.concat(post));
}

// ------------------------------------------------------------------ the view
const CSS = `
.lt-root { position:absolute; inset:0; display:flex; flex-direction:column; background:var(--axes-bg); color:var(--fg); }
.lt-ctl { padding:4px 8px; background:var(--legend-bg); border-bottom:1px solid var(--legend-edge); display:flex; flex-direction:column; gap:4px; flex:none; }
.lt-row { display:flex; flex-wrap:wrap; align-items:center; gap:4px 10px; }
.lt-row label { display:inline-flex; align-items:center; gap:4px; color:var(--muted); white-space:nowrap; }
.lt-row input[type=number], .lt-row input[type=text] { width:5.4em; }
.lt-row input.wide { width:7.5em; }
.lt-row input[type=range] { flex:1; min-width:120px; accent-color:var(--accent); }
.lt-btn { min-width:2.2em; }
.lt-btn.on { border-color:var(--accent); color:var(--accent); }
.lt-flabel { font:12px ui-monospace,Menlo,monospace; color:var(--fg); min-width:13em; font-variant-numeric:tabular-nums; }
.lt-notes { padding:2px 8px; font:11.5px/1.35 ui-monospace,Menlo,monospace; color:var(--muted); flex:none; border-bottom:1px solid var(--legend-edge); max-height:30%; overflow:auto; }
.lt-notes .warn { color:#f59e0b; white-space:pre-wrap; }
.lt-notes summary { cursor:pointer; white-space:pre-wrap; }
.lt-notes pre { margin:2px 0; font:inherit; white-space:pre-wrap; }
.lt-walls { position:relative; flex:1; min-height:80px; }
.lt-strip { position:relative; flex:none; height:170px; border-top:1px solid var(--legend-edge); }
.lt-walls canvas, .lt-strip canvas { position:absolute; left:0; top:0; width:100%; height:100%; }
.lt-tip { position:absolute; z-index:6; pointer-events:none; display:none; white-space:pre; font:11.5px/1.35 ui-monospace,Menlo,monospace;
  background:var(--legend-bg); border:1px solid var(--legend-edge); border-radius:4px; padding:3px 6px; color:var(--fg); }
.lt-empty { position:absolute; left:12px; top:12px; color:var(--muted); font-size:13px; white-space:pre-wrap; }
#main[data-view="light"] #summary { display:none; }
`;

const V = {
  name: 'viewlight', app: null, container: null, visible: false, ready: false,
  data: null,           // event-derived arrays (see setEvent)
  geo: null,            // geometry-derived arrays (see setGeometry)
  A: null,              // animation arrays built by rebuild()
  k: -1, playing: false, acc: 0, lastT: 0, holdUntil: 0, raf: 0, dirty: false, animFlag: false,
  zoom: 1, u: 0.5, v: 0.5, cells: [], hover: -1, hoverWall: -1, committedLt: 'all', committedPlay: '0',
  W: 0, H: 0, SW: 0, SH: 0, dpr: 1, stats: { last: 0, avg: 0, n: 0 }, lit3d: false, pushed3d: false,
  pal: null, burstCache: null,
};

function T() { return V.app.theme; }
function param(name) { const s = V.app.state[name]; return s == null || s === '' ? DEF[name] : s; }
function num(name, lo, hi) { const x = parseFloat(param(name)); return Number.isFinite(x) ? clamp(x, lo, hi) : parseFloat(DEF[name]); }

// ------------------------------------------------------------------ DOM
function buildDom(container) {
  const st = document.createElement('style'); st.textContent = CSS; document.head.appendChild(st);
  container.innerHTML = `
  <div class="lt-root">
    <div class="lt-ctl">
      <div class="lt-row">
        <button class="lt-btn" data-r="play" title="play / pause (space)">&#9654;</button>
        <button class="lt-btn" data-r="prev" title="previous frame">&#9664;|</button>
        <button class="lt-btn" data-r="next" title="next frame">|&#9654;</button>
        <input type="range" data-r="slider" min="0" max="119" value="0">
        <span class="lt-flabel" data-r="flabel"></span>
        <button class="lt-btn" data-r="all" title="show the whole window: PE collected, every flash">whole window</button>
      </div>
      <div class="lt-row">
        <label>clock <select data-r="clock"><option value="log">log</option><option value="linear">linear</option></select></label>
        <label>fill <select data-r="fill"><option value="afterglow">afterglow</option><option value="cumulative">cumulative</option></select></label>
        <label>frames <input type="number" data-r="frames" min="2" max="1000" step="10"></label>
        <label>fps <input type="number" data-r="fps" min="1" max="120" step="1"></label>
        <label title="afterglow kept per frame">persistence <input type="number" data-r="pers" min="0" max="0.99" step="0.05"></label>
        <label title="window relative to t0 (us)">window <input type="number" data-r="wlo" step="0.05"> .. <input type="number" data-r="whi" step="1"> &micro;s</label>
        <label title="absolute time origin in us; empty = onset of the brightest burst">t0 <input type="text" class="wide" data-r="t0" placeholder="auto"> &micro;s</label>
        <button class="lt-btn" data-r="burstprev" title="jump to the previous light burst">&#9664; burst</button>
        <button class="lt-btn" data-r="burstnext" title="jump to the next light burst">burst &#9654;</button>
        <button class="lt-btn" data-r="t0auto" title="back to the brightest burst">auto t0</button>
      </div>
    </div>
    <div class="lt-notes" data-r="notes"></div>
    <div class="lt-walls" data-r="walls">
      <canvas data-r="static"></canvas><canvas data-r="gl"></canvas><canvas data-r="dyn"></canvas>
      <div class="lt-tip" data-r="tip"></div><div class="lt-empty" data-r="empty"></div>
    </div>
    <div class="lt-strip" data-r="strip"><canvas data-r="stripc"></canvas></div>
  </div>`;
  const R = {};
  for (const el of container.querySelectorAll('[data-r]')) R[el.dataset.r] = el;
  return R;
}

// ------------------------------------------------------------------ GL tiles
const VS = `#version 300 es
in vec2 aCorner; in vec4 aGeo; in vec4 aCol;
uniform vec2 uC; uniform vec2 uH; uniform vec2 uPx;
out vec2 vQ; out vec2 vHalfPx; out vec4 vCol;
void main(){ vec2 p=aGeo.xy+aCorner*aGeo.zw; gl_Position=vec4((p-uC)/uH,0.,1.);
  vQ=aCorner; vHalfPx=aGeo.zw/uH*uPx; vCol=aCol; }`;
const FS = `#version 300 es
precision mediump float; in vec2 vQ; in vec2 vHalfPx; in vec4 vCol;
uniform vec3 uFg; uniform float uEdge; out vec4 o;
void main(){ vec2 d=(1.-abs(vQ))*vHalfPx; float e=min(d.x,d.y); vec3 c=vCol.rgb;
  if(vCol.a>.5 && e<uEdge) c=uFg; o=vec4(c,1.); }`;

function initGL(canvas) {
  const gl = canvas.getContext('webgl2', { alpha: true, premultipliedAlpha: true, antialias: false, preserveDrawingBuffer: false });
  if (!gl) return null;
  const sh = (type, src) => { const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s)); return s; };
  const p = gl.createProgram();
  gl.attachShader(p, sh(gl.VERTEX_SHADER, VS)); gl.attachShader(p, sh(gl.FRAGMENT_SHADER, FS));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
  const G = { gl, p, loc: {}, geoBuf: gl.createBuffer(), colBuf: gl.createBuffer(), cornerBuf: gl.createBuffer(), nInst: 0 };
  for (const a of ['aCorner', 'aGeo', 'aCol']) G.loc[a] = gl.getAttribLocation(p, a);
  for (const u of ['uC', 'uH', 'uPx', 'uFg', 'uEdge']) G.loc[u] = gl.getUniformLocation(p, u);
  gl.bindBuffer(gl.ARRAY_BUFFER, G.cornerBuf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
  return G;
}

// ------------------------------------------------------------------ geometry
function setGeometry(geometry) {
  V.geo = null; V.A = null;
  if (!geometry || !geometry.has('pd/xyz')) { requestFrame(); return; }
  const xyz = geometry.get('pd/xyz'), half = geometry.get('pd/half'), wall = geometry.get('pd/wall');
  const nt = xyz.length / 3;
  const nw = Math.max(1, ...Array.from(wall, (w) => w + 1));
  const ext = [];
  for (let w = 0; w < nw; w++) {
    let z0 = Infinity, z1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (let i = 0; i < nt; i++) {
      if (wall[i] !== w) continue;
      z0 = Math.min(z0, xyz[3 * i + 2] - half[2 * i + 1]); z1 = Math.max(z1, xyz[3 * i + 2] + half[2 * i + 1]);
      y0 = Math.min(y0, xyz[3 * i + 1] - half[2 * i]); y1 = Math.max(y1, xyz[3 * i + 1] + half[2 * i]);
    }
    ext.push({ z0: z0 - 20, z1: z1 + 20, y0: y0 - 20, y1: y1 + 20 });
  }
  const aspects = ext.map((e) => (e.y1 - e.y0) / (e.z1 - e.z0));
  const side = nw > 1 && Math.min(...aspects) > 0.45;
  // instance order: grouped by wall
  const order = new Int32Array(nt), start = new Int32Array(nw + 1);
  let s = 0;
  for (let w = 0; w < nw; w++) { start[w] = s; for (let i = 0; i < nt; i++) if (wall[i] === w) order[s++] = i; }
  start[nw] = s;
  const geoArr = new Float32Array(4 * nt);
  for (let j = 0; j < nt; j++) {
    const i = order[j];
    geoArr[4 * j] = xyz[3 * i + 2]; geoArr[4 * j + 1] = xyz[3 * i + 1];
    geoArr[4 * j + 2] = half[2 * i + 1]; geoArr[4 * j + 3] = half[2 * i];
  }
  const boxes = new Map();
  for (const b of geometry.meta.tpcs || []) {
    const r = [b[2], b[5], b[1], b[4]].map((x) => Math.round(x * 10) / 10);
    boxes.set(r.join(','), r);
  }
  const labels = (geometry.meta.walls || []).map((w) => w.label);
  V.geo = { nt, nw, wall, xyz, half, ext, side, order, start, geoArr, boxes: Array.from(boxes.values()), labels,
    projected: !!geometry.meta.projected, chOff: geometry.get('pd/chan_offsets'), ch: geometry.get('pd/channels'),
    inst: new Uint8Array(4 * nt), inst32: null, rgb: new Float32Array(3 * nt) };
  V.geo.inst32 = new Uint32Array(V.geo.inst.buffer);
  if (V.G) {
    const { gl } = V.G;
    gl.bindBuffer(gl.ARRAY_BUFFER, V.G.geoBuf); gl.bufferData(gl.ARRAY_BUFFER, geoArr, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, V.G.colBuf); gl.bufferData(gl.ARRAY_BUFFER, V.geo.inst, gl.DYNAMIC_DRAW);
  }
  V.zoom = 1; V.u = 0.5; V.v = 0.5;
  layout();
  requestFrame();
}

// ------------------------------------------------------------------ event data
function setEvent(bundle, geometry) {
  V.data = null; V.A = null; V.burstCache = null;
  if (bundle && bundle.has('oph/time') && V.geo) {
    const o = bundle.meta.optical || {};
    const d = { bundle, t: bundle.get('oph/time'), pe: bundle.get('oph/pe'), tile: bundle.get('oph/tile'),
      flash: bundle.get('oph/flash'), meta: o, truth: bundle.meta.truth || {} };
    d.n = d.t.length;
    d.hasLinks = false;
    for (let i = 0; i < d.n; i++) if (d.flash[i] >= 0) { d.hasLinks = true; break; }
    d.quantum = o.quantum > 0 ? o.quantum : hitTimeQuantum(d.t);
    d.nf = bundle.has('flash/time') ? bundle.get('flash/time').length : 0;
    d.nws = d.nf ? bundle.get('flash/wall_share').length / d.nf : 1;
    V.data = d;
  }
  V.pushed3d = false;
  const lt = param('lt');
  V.k = lt === 'all' ? -1 : (parseInt(lt, 10) || 0);
  rebuild();
  if (param('lplay') === '1' && V.data && !V.playing) play(true);
}

/** Bursts of light over the whole readout: onset times, sorted. Cached per event. */
function bursts() {
  const d = V.data;
  if (!d) return [];
  if (V.burstCache) return V.burstCache;
  const bins = new Map();
  let lo = Infinity, mx = 0;
  for (let i = 0; i < d.n; i++) if (Number.isFinite(d.t[i]) && d.t[i] < lo) lo = d.t[i];
  for (let i = 0; i < d.n; i++) {
    if (!Number.isFinite(d.t[i])) continue;
    const b = Math.floor((d.t[i] - lo) / 0.1);
    const v = (bins.get(b) || 0) + Math.max(d.pe[i], 0);
    bins.set(b, v); if (v > mx) mx = v;
  }
  const thr = Math.max(30, 1e-3 * mx);
  const keys = Array.from(bins.keys()).filter((b) => bins.get(b) >= thr).sort((a, b) => a - b);
  const groups = [];
  for (const b of keys) {
    const g = groups[groups.length - 1];
    if (g && b - g[1] <= 10) g[1] = b; else groups.push([b, b]);
  }
  const order = Array.from({ length: d.n }, (_, i) => i).sort((a, b) => d.t[a] - d.t[b]);
  const out = [];
  for (const [b0, b1] of groups) {
    const ta = lo + b0 * 0.1 - 0.3, tb = lo + (b1 + 1) * 0.1;
    const sel = [];
    for (const j of order) { const tj = d.t[j]; if (tj >= ta && tj < tb) sel.push(j); else if (tj >= tb) break; }
    const on = lightOnset(d.t, d.pe, d.quantum, Int32Array.from(sel));
    if (on != null) out.push(on);
  }
  V.burstCache = out.sort((a, b) => a - b);
  return V.burstCache;
}

// ------------------------------------------------------------------ the animation arrays
function rebuild() {
  const d = V.data, g = V.geo;
  const R = V.R;
  if (!d || !g) {
    V.A = null; updateEmpty(); layout(); updateNotes(); requestFrame(); return;
  }
  const n = Math.round(num('lframes', 2, 1000));
  const persistence = num('lpers', 0, 0.99);
  const cumulative = param('lfill') === 'cumulative';
  const t0arg = parseFloat(V.app.state.lt0);
  let t0 = Number.isFinite(t0arg) ? t0arg : d.meta.t0;
  if (!Number.isFinite(t0)) t0 = lightOnset(d.t, d.pe, d.quantum) ?? 0;
  const lo = t0 + num('lwlo', -1e6, 1e6), hi0 = t0 + num('lwhi', -1e6, 1e6);
  const hi = hi0 > lo ? hi0 : lo + 10;
  const scale = param('lclock');
  const edges = frameEdges(t0, lo, hi, n, scale, d.quantum);
  const nt = g.nt;
  const frameOf = (t) => {
    if (!Number.isFinite(t)) return -1;
    let k = lowerBound(edges, t) - 1;
    if (t === edges[0]) k = 0;
    return k >= 0 && k < n ? k : -1;
  };
  const frameCoord = (t) => {
    if (t <= edges[0]) return 0;
    if (t >= edges[n]) return n;
    const i = lowerBound(edges, t);
    const j = Math.max(1, i);
    return (j - 1) + (t - edges[j - 1]) / (edges[j] - edges[j - 1]);
  };
  const arrivals = new Float32Array(n * nt), inside = new Float64Array(n), outside = new Float64Array(n);
  let nWin = 0, peWin = 0, peInFlash = 0, lost = 0, prompt = 0;
  for (let i = 0; i < d.n; i++) {
    const t = d.t[i];
    if (t >= t0 && t <= t0 + 1) prompt += d.pe[i];
    const k = frameOf(t);
    if (k < 0) continue;
    nWin++; peWin += d.pe[i];
    const fl = d.hasLinks && d.flash[i] >= 0;
    if (fl) { inside[k] += d.pe[i]; peInFlash += d.pe[i]; } else outside[k] += d.pe[i];
    const tl = d.tile[i];
    if (tl >= 0 && tl < nt) arrivals[k * nt + tl] += d.pe[i]; else lost++;
  }
  const values = new Float32Array(n * nt);
  const collected = new Float32Array(nt);
  let top = 0;
  const acc = new Float64Array(nt);
  for (let k = 0; k < n; k++) {
    const b = k * nt;
    for (let i = 0; i < nt; i++) {
      const a = arrivals[b + i];
      collected[i] += a;
      acc[i] = cumulative ? acc[i] + a : acc[i] * persistence + a;
      values[b + i] = acc[i];
      if (acc[i] > top) top = acc[i];
    }
  }
  for (let i = 0; i < nt; i++) if (collected[i] > top) top = collected[i];
  const peLo = PE_FLOOR, peHi = Math.max(top, 10 * PE_FLOOR);
  const l0 = Math.log10(peLo), span = Math.log10(peHi) - l0;
  const level = (v) => {
    if (!(v >= PE_OFF)) return 0;
    const u = (Math.log10(clamp(v, peLo, peHi)) - l0) / span;
    return 1 + Math.round(254 * u);
  };
  const lvl = new Uint8Array(n * nt);
  for (let j = 0; j < n * nt; j++) lvl[j] = level(values[j]);
  const lvlAll = new Uint8Array(nt);
  for (let i = 0; i < nt; i++) lvlAll[i] = level(collected[i]);

  // flashes in the window, brightest first
  const B = d.bundle;
  const fT = B.get('flash/time'), fPe = B.get('flash/pe'), fY = B.get('flash/y'), fZ = B.get('flash/z');
  const fYw = B.get('flash/yw'), fZw = B.get('flash/zw'), fSh = B.get('flash/wall_share');
  let cand = [];
  for (let j = 0; j < d.nf; j++) { const k = frameOf(fT[j]); if (k >= 0) cand.push([j, k]); }
  cand.sort((a, b) => fPe[b[0]] - fPe[a[0]]);
  cand = cand.slice(0, MAX_FLASHES);
  const flashes = cand.map(([j, k]) => {
    const yw = Number.isFinite(fYw[j]) && fYw[j] > 0 ? fYw[j] : 30;
    const zw = Number.isFinite(fZw[j]) && fZw[j] > 0 ? fZw[j] : 30;
    const on = [];
    for (let w = 0; w < g.nw; w++) on.push(d.nws === g.nw ? fSh[j * d.nws + w] >= 0.2 : true);
    const f = clamp(Math.log10(clamp(fPe[j], 1, PE_SIZE_MAX)) / Math.log10(PE_SIZE_MAX), 0, 1);
    return { j, k, time: fT[j], pe: fPe[j], y: fY[j], z: fZ[j], yw, zw, on, lw: 0.8 + 2.4 * f };
  });
  const cum = new Float64Array(n);
  let s = 0;
  for (let k = 0; k < n; k++) { s += inside[k] + outside[k]; cum[k] = s; }
  const flashNow = (k) => { let c = 0; for (const f of flashes) if (f.k <= k) c++; return c; };

  V.A = { n, t0, t0Auto: !Number.isFinite(t0arg), lo, hi, edges, scale, nt, arrivals, values, collected, lvl, lvlAll, peLo, peHi,
    inside, outside, cum, flashes, frameOf, frameCoord, persistence, cumulative, nWin, peWin, peInFlash, lost, prompt,
    flashNow, ticks: null };
  V.A.ticks = tickPositions(V.A);
  V.k = clamp(V.k, -1, n - 1);
  buildPalette();
  V.R.slider.max = String(n - 1);
  syncControls();
  updateEmpty(); updateNotes(); layout(); drawStripCache(); requestFrame();
}

function tickPositions(A) {
  let cand;
  if (A.scale === 'log') {
    cand = [-10, -5, -2, -1, -0.5, -0.2, -0.1, -0.05, 0, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
      .filter((c) => A.t0 + c >= A.lo - 1e-9 && A.t0 + c <= A.hi + 1e-9);
  } else cand = niceTicks(A.lo - A.t0, A.hi - A.t0);
  const gap = 0.06 * A.n, p0 = A.frameCoord(A.t0);
  const has0 = cand.some((c) => c === 0);
  const out = [];
  let last = -Infinity;
  for (const c of cand) {
    const p = A.frameCoord(A.t0 + c);
    if (c !== 0 && has0 && Math.abs(p - p0) < gap) continue;
    if (c === 0 || p - last >= gap) { out.push({ p, label: c !== 0 ? fmtDt(c) : 't₀' }); last = p; }
  }
  return out;
}

// ------------------------------------------------------------------ colours
function buildPalette() {
  const th = T(), lut = getLut(V.app.state.cmap || 'turbo');
  const pal8 = new Uint8Array(256 * 4), palf = new Float32Array(256 * 3);
  const hex = (h) => { const x = parseInt(h.slice(1), 16); return [x >> 16 & 255, x >> 8 & 255, x & 255]; };
  const off = hex(th.pd_base);
  const put = (j, c) => { pal8[4 * j] = c[0]; pal8[4 * j + 1] = c[1]; pal8[4 * j + 2] = c[2]; palf[3 * j] = c[0] / 255; palf[3 * j + 1] = c[1] / 255; palf[3 * j + 2] = c[2] / 255; };
  put(0, off);
  for (let j = 1; j < 256; j++) {
    const t = (j - 1) / 254, tt = th.cmap_floor + t * (th.cmap_ceiling - th.cmap_floor);
    const i = clamp(Math.round(tt * 255), 0, 255);
    put(j, [lut[3 * i], lut[3 * i + 1], lut[3 * i + 2]]);
  }
  V.pal = { pal8, palf, pal32: new Uint32Array(pal8.buffer), lut: pal8 };
}

// ------------------------------------------------------------------ frame -> buffers (the hot path)
/** Writes tile colours/outline flags for the current frame. No allocation. */
function applyFrame() {
  const g = V.geo, A = V.A, P = V.pal;
  if (!g || !A || !P) return;
  const nt = g.nt, order = g.order, inst32 = g.inst32, pal32 = P.pal32, palf = P.palf, rgb = g.rgb;
  const k = V.k;
  if (k < 0) {
    const lv = A.lvlAll;
    for (let s = 0; s < nt; s++) {
      const i = order[s], l = lv[i];
      inst32[s] = pal32[l];
      rgb[3 * i] = palf[3 * l]; rgb[3 * i + 1] = palf[3 * l + 1]; rgb[3 * i + 2] = palf[3 * l + 2];
    }
  } else {
    const lv = A.lvl, arr = A.arrivals, b = k * nt;
    for (let s = 0; s < nt; s++) {
      const i = order[s], l = lv[b + i];
      inst32[s] = arr[b + i] > 0 ? (pal32[l] | 0xff000000) : pal32[l];
      rgb[3 * i] = palf[3 * l]; rgb[3 * i + 1] = palf[3 * l + 1]; rgb[3 * i + 2] = palf[3 * l + 2];
    }
  }
}

function drawGL() {
  const G = V.G, g = V.geo;
  if (!G || !g) return;
  const { gl, loc } = G;
  gl.bindBuffer(gl.ARRAY_BUFFER, G.colBuf);
  gl.bufferSubData(gl.ARRAY_BUFFER, 0, g.inst);
  gl.useProgram(G.p);
  gl.enable(gl.SCISSOR_TEST);
  gl.disable(gl.BLEND);
  gl.clearColor(0, 0, 0, 0);
  gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
  gl.scissor(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
  gl.clear(gl.COLOR_BUFFER_BIT);
  const fg = V.fgRgb;
  gl.uniform3f(loc.uFg, fg[0], fg[1], fg[2]);
  gl.uniform1f(loc.uEdge, 1.6 * V.dpr);
  gl.bindBuffer(gl.ARRAY_BUFFER, G.cornerBuf);
  gl.enableVertexAttribArray(loc.aCorner); gl.vertexAttribPointer(loc.aCorner, 2, gl.FLOAT, false, 0, 0); gl.vertexAttribDivisor(loc.aCorner, 0);
  gl.enableVertexAttribArray(loc.aGeo); gl.enableVertexAttribArray(loc.aCol);
  gl.vertexAttribDivisor(loc.aGeo, 1); gl.vertexAttribDivisor(loc.aCol, 1);
  const dpr = V.dpr;
  for (const c of V.cells) {
    const s0 = g.start[c.w], cnt = g.start[c.w + 1] - s0;
    if (!cnt) continue;
    const w = winOf(c);
    const vx = Math.round(c.x * dpr), vw = Math.round(c.cw * dpr), vh = Math.round(c.ch * dpr);
    const vy = Math.round((V.H - c.y - c.ch) * dpr);
    gl.viewport(vx, vy, vw, vh); gl.scissor(vx, vy, vw, vh);
    gl.uniform2f(loc.uC, 0.5 * (w.z0 + w.z1), 0.5 * (w.y0 + w.y1));
    gl.uniform2f(loc.uH, 0.5 * (w.z1 - w.z0), 0.5 * (w.y1 - w.y0));
    gl.uniform2f(loc.uPx, 0.5 * c.cw * dpr, 0.5 * c.ch * dpr);
    gl.bindBuffer(gl.ARRAY_BUFFER, G.geoBuf);
    gl.vertexAttribPointer(loc.aGeo, 4, gl.FLOAT, false, 16, s0 * 16);
    gl.bindBuffer(gl.ARRAY_BUFFER, G.colBuf);
    gl.vertexAttribPointer(loc.aCol, 4, gl.UNSIGNED_BYTE, true, 4, s0 * 4);
    gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, cnt);
  }
  gl.disable(gl.SCISSOR_TEST);
}

// ------------------------------------------------------------------ layout and view window
function layout() {
  const R = V.R;
  if (!R) return;
  const wr = R.walls.getBoundingClientRect();
  const W = Math.max(1, Math.floor(wr.width)), H = Math.max(1, Math.floor(wr.height));
  const dpr = window.devicePixelRatio || 1;
  V.dpr = dpr; V.W = W; V.H = H;
  for (const c of [R.static, R.gl, R.dyn]) {
    const w = Math.round(W * dpr), h = Math.round(H * dpr);
    if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
  }
  const sr = R.strip.getBoundingClientRect();
  V.SW = Math.max(1, Math.floor(sr.width)); V.SH = Math.max(1, Math.floor(sr.height));
  const sw = Math.round(V.SW * dpr), sh = Math.round(V.SH * dpr);
  if (R.stripc.width !== sw || R.stripc.height !== sh) { R.stripc.width = sw; R.stripc.height = sh; }
  V.cells = [];
  const g = V.geo;
  if (!g) { requestFrame(); return; }
  const mL = 48, mR = 78, mT = 52, mB = 36, gapX = 40, gapY = 34;
  const aw = W - mL - mR, ah = H - mT - mB;
  const ex = g.ext.map((e) => ({ w: e.z1 - e.z0, h: e.y1 - e.y0 }));
  const titled = g.nw > 1 || g.projected;
  const titleH = titled ? 16 : 0;
  let s, boxes = [];
  if (g.side) {
    s = Math.min((aw - gapX * (g.nw - 1)) / ex.reduce((a, e) => a + e.w, 0), (ah - titleH) / Math.max(...ex.map((e) => e.h)));
    const tw = ex.reduce((a, e) => a + e.w * s, 0) + gapX * (g.nw - 1);
    let x = mL + Math.max(0, (aw - tw) / 2);
    for (let w = 0; w < g.nw; w++) { boxes.push({ x, y: mT + titleH, cw: ex[w].w * s, ch: ex[w].h * s }); x += ex[w].w * s + gapX; }
  } else {
    s = Math.min(aw / Math.max(...ex.map((e) => e.w)),
      (ah - (gapY + titleH) * (g.nw - 1) - titleH) / ex.reduce((a, e) => a + e.h, 0));
    const tw = Math.max(...ex.map((e) => e.w)) * s;
    let y = mT + titleH;
    for (let w = 0; w < g.nw; w++) {
      boxes.push({ x: mL + Math.max(0, (aw - tw) / 2), y, cw: ex[w].w * s, ch: ex[w].h * s });
      y += ex[w].h * s + gapY + titleH;
    }
  }
  if (!(s > 0)) s = 0.1;
  V.cells = boxes.map((b, w) => ({ w, ...b, s }));
  V.layoutDone = true;
  drawStatic();
  requestFrame();
}

function winOf(c, out = null) {
  const e = V.geo.ext[c.w], z = V.zoom;
  const wz = (e.z1 - e.z0) / z, wy = (e.y1 - e.y0) / z;
  const cz = e.z0 + clamp(V.u, 0.5 / z, 1 - 0.5 / z) * (e.z1 - e.z0);
  const cy = e.y0 + clamp(V.v, 0.5 / z, 1 - 0.5 / z) * (e.y1 - e.y0);
  const o = out || {};
  o.z0 = cz - wz / 2; o.z1 = cz + wz / 2; o.y0 = cy - wy / 2; o.y1 = cy + wy / 2;
  return o;
}

// ------------------------------------------------------------------ static canvas
function fontCss(px, mono = false) { return `${px}px ${mono ? 'ui-monospace, Menlo, monospace' : 'system-ui, -apple-system, "Segoe UI", sans-serif'}`; }

function drawStatic(ctxIn = null, scaleOut = 1) {
  const R = V.R, th = T();
  const cv = R.static;
  const ctx = ctxIn || cv.getContext('2d');
  if (!ctxIn) ctx.setTransform(V.dpr, 0, 0, V.dpr, 0, 0);
  ctx.clearRect(0, 0, V.W, V.H);
  ctx.fillStyle = th.fig_bg; ctx.fillRect(0, 0, V.W, V.H);
  const g = V.geo;
  if (!g) return;
  ctx.font = fontCss(11); ctx.textBaseline = 'middle';
  for (const c of V.cells) {
    const w = winOf(c);
    ctx.fillStyle = th.axes_bg; ctx.fillRect(c.x, c.y, c.cw, c.ch);
    ctx.save(); ctx.beginPath(); ctx.rect(c.x, c.y, c.cw, c.ch); ctx.clip();
    const px = (z) => c.x + (z - w.z0) / (w.z1 - w.z0) * c.cw, py = (y) => c.y + (w.y1 - y) / (w.y1 - w.y0) * c.ch;
    ctx.strokeStyle = th.grid; ctx.globalAlpha = 0.6; ctx.lineWidth = 0.7;
    for (const [z0, z1, y0, y1] of g.boxes) ctx.strokeRect(px(z0), py(y1), px(z1) - px(z0), py(y0) - py(y1));
    ctx.restore(); ctx.globalAlpha = 1;
    ctx.strokeStyle = th.legend_edge; ctx.lineWidth = 1; ctx.strokeRect(c.x + 0.5, c.y + 0.5, c.cw - 1, c.ch - 1);
    // ticks
    ctx.fillStyle = th.fg_muted; ctx.strokeStyle = th.fg_muted;
    ctx.textAlign = 'right';
    for (const y of niceTicks(w.y0, w.y1, Math.max(3, Math.round(c.ch / 55)))) {
      const yy = py(y); if (yy < c.y - 1 || yy > c.y + c.ch + 1) continue;
      ctx.beginPath(); ctx.moveTo(c.x - 4, yy); ctx.lineTo(c.x, yy); ctx.stroke();
      ctx.fillText(String(Math.round(y)), c.x - 6, yy);
    }
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    for (const z of niceTicks(w.z0, w.z1, Math.max(3, Math.round(c.cw / 80)))) {
      const xx = px(z); if (xx < c.x - 1 || xx > c.x + c.cw + 1) continue;
      ctx.beginPath(); ctx.moveTo(xx, c.y + c.ch); ctx.lineTo(xx, c.y + c.ch + 4); ctx.stroke();
      ctx.fillText(String(Math.round(z)), xx, c.y + c.ch + 6);
    }
    ctx.textBaseline = 'middle';
    ctx.fillStyle = th.fg_muted;
    ctx.fillText('z [cm]  (beam)', c.x + c.cw / 2, c.y + c.ch + 24);
    if (g.nw === 1 || g.side ? c.w === 0 : true) {
      ctx.save(); ctx.translate(c.x - 38, c.y + c.ch / 2); ctx.rotate(-Math.PI / 2); ctx.textAlign = 'center';
      ctx.fillText('y [cm]', 0, 0); ctx.restore();
    }
    if (g.nw > 1 || g.projected) {
      ctx.textAlign = 'left'; ctx.fillStyle = th.fg_muted; ctx.font = fontCss(11.5);
      ctx.fillText(g.labels[c.w] || `wall ${c.w}`, c.x, c.y - 9); ctx.font = fontCss(11);
    }
  }
  drawColorbar(ctx);
}

function drawColorbar(ctx) {
  const A = V.A, P = V.pal, th = T();
  if (!A || !P || !V.cells.length) return;
  const top = Math.min(...V.cells.map((c) => c.y)), bot = Math.max(...V.cells.map((c) => c.y + c.ch));
  const right = Math.max(...V.cells.map((c) => c.x + c.cw));
  const x = Math.min(right + 14, V.W - 72), w = 14, h = Math.min(bot - top, V.H - top - 40);
  for (let j = 0; j < h; j++) {
    const lv = 1 + Math.round(254 * (1 - j / Math.max(1, h - 1)));
    ctx.fillStyle = `rgb(${P.pal8[4 * lv]},${P.pal8[4 * lv + 1]},${P.pal8[4 * lv + 2]})`;
    ctx.fillRect(x, top + j, w, 1.2);
  }
  ctx.strokeStyle = th.legend_edge; ctx.strokeRect(x + 0.5, top + 0.5, w - 1, h - 1);
  ctx.fillStyle = th.fg_muted; ctx.strokeStyle = th.fg_muted; ctx.font = fontCss(10.5); ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
  const l0 = Math.log10(A.peLo), span = Math.log10(A.peHi) - l0;
  for (let e = Math.ceil(l0 - 1e-9); e <= Math.floor(Math.log10(A.peHi) + 1e-9); e++) {
    const yy = top + h * (1 - (e - l0) / span);
    ctx.beginPath(); ctx.moveTo(x + w, yy); ctx.lineTo(x + w + 4, yy); ctx.stroke();
    ctx.fillText(pow10(e), x + w + 6, yy);
  }
  ctx.save(); ctx.translate(x + w + 44, top + h / 2); ctx.rotate(-Math.PI / 2); ctx.textAlign = 'center';
  ctx.fillText('PE per detector' + (A.cumulative ? ' (so far)' : ' (afterglow)'), 0, 0); ctx.restore();
}

// ------------------------------------------------------------------ dynamic canvas
function drawDyn(ctx = null) {
  const R = V.R, th = T(), A = V.A, g = V.geo;
  const own = !ctx;
  if (own) { ctx = R.dyn.getContext('2d'); ctx.setTransform(V.dpr, 0, 0, V.dpr, 0, 0); ctx.clearRect(0, 0, V.W, V.H); }
  if (!g || !A) return;
  const k = V.k, pers = A.persistence;
  // HUD
  ctx.textBaseline = 'alphabetic'; ctx.textAlign = 'left';
  const head = V.data.truth && V.data.truth.headline;
  if (head) { ctx.fillStyle = th.mctrack; ctx.font = fontCss(12); ctx.fillText(head, 10, 16); }
  ctx.fillStyle = th.fg; ctx.font = fontCss(20, true);
  ctx.fillText(k < 0 ? 't − t₀: whole window' : `t − t₀ = ${fmtDt(A.edges[k + 1] - A.t0)}`, 10, 42);
  const peSoFar = k < 0 ? A.cum[A.n - 1] : A.cum[k], nfl = k < 0 ? A.flashes.length : A.flashNow(k);
  ctx.textAlign = 'right'; ctx.fillStyle = th.fg_muted; ctx.font = fontCss(12);
  ctx.fillText(`${Math.round(peSoFar).toLocaleString('en-US')} PE ${k < 0 ? 'in the window' : 'so far'}  ·  ${nfl} flash${nfl === 1 ? '' : 'es'}`, V.W - 10, 40);
  ctx.font = fontCss(10.5);
  const key = [];
  if (V.data.truth && V.data.truth.vertex) key.push('★ true vertex');
  if (A.flashes.length) key.push('◯ flash ±1σ');
  key.push('□ outline: light this frame');
  ctx.fillText(key.join('     '), V.W - 10, 14);
  ctx.textAlign = 'left';
  // per wall: flashes, vertex, hovered tile
  const vtx = V.data.truth && V.data.truth.vertex;
  for (const c of V.cells) {
    const w = winOf(c);
    const sx = c.cw / (w.z1 - w.z0), sy = c.ch / (w.y1 - w.y0);
    const px = (z) => c.x + (z - w.z0) * sx, py = (y) => c.y + (w.y1 - y) * sy;
    ctx.save(); ctx.beginPath(); ctx.rect(c.x, c.y, c.cw, c.ch); ctx.clip();
    ctx.strokeStyle = th.track; ctx.fillStyle = th.track;
    for (let f = 0; f < A.flashes.length; f++) {
      const fl = A.flashes[f];
      if (!fl.on[c.w]) continue;
      const age = k < 0 ? 0 : k - fl.k;
      if (k >= 0 && age < 0) continue;
      const alpha = k < 0 ? 0.85 : Math.max(0.45, Math.pow(pers, age));
      ctx.globalAlpha = alpha; ctx.lineWidth = fl.lw;
      const cx = px(fl.z), cy = py(fl.y);
      ctx.beginPath(); ctx.ellipse(cx, cy, fl.zw * sx, fl.yw * sy, 0, 0, 2 * Math.PI); ctx.stroke();
      ctx.lineWidth = 1.4; ctx.beginPath(); ctx.moveTo(cx - 4, cy); ctx.lineTo(cx + 4, cy); ctx.moveTo(cx, cy - 4); ctx.lineTo(cx, cy + 4); ctx.stroke();
      if (f < 6) {
        ctx.globalAlpha = Math.max(0.75, alpha); ctx.font = fontCss(11); ctx.textBaseline = 'top';
        ctx.fillText(` ${fmtPe(fl.pe)} PE`, cx + 0.71 * fl.zw * sx, cy + 0.71 * fl.yw * sy);
      }
    }
    ctx.globalAlpha = 1;
    if (vtx && Number.isFinite(vtx[1]) && Number.isFinite(vtx[2])) {
      const cx = px(vtx[2]), cy = py(vtx[1]);
      ctx.beginPath();
      for (let i = 0; i < 10; i++) { const r = i % 2 ? 4.2 : 9.5, a = -Math.PI / 2 + i * Math.PI / 5; ctx.lineTo(cx + r * Math.cos(a), cy + r * Math.sin(a)); }
      ctx.closePath(); ctx.fillStyle = th.mctrack; ctx.fill(); ctx.strokeStyle = th.axes_bg; ctx.lineWidth = 1; ctx.stroke();
    }
    if (V.hover >= 0 && V.hoverWall === c.w) {
      const i = V.hover, z = V.geo.xyz[3 * i + 2], y = V.geo.xyz[3 * i + 1], hz = V.geo.half[2 * i + 1], hy = V.geo.half[2 * i];
      ctx.strokeStyle = th.fg; ctx.lineWidth = 1.5;
      ctx.strokeRect(px(z - hz) - 1, py(y + hy) - 1, 2 * hz * sx + 2, 2 * hy * sy + 2);
    }
    ctx.restore();
  }
}

// ------------------------------------------------------------------ strip
const stripM = { l: 52, r: 36, t: 8, b: 40 };
function stripY(A, v, floor, topv, h) {
  const a = Math.log10(floor), b = Math.log10(topv);
  return stripM.t + h * (1 - (Math.log10(clamp(v, floor, topv)) - a) / (b - a));
}

function drawStripCache() {
  const R = V.R, A = V.A, th = T();
  if (!A) { const c = R.stripc.getContext('2d'); c.setTransform(1, 0, 0, 1, 0, 0); c.clearRect(0, 0, R.stripc.width, R.stripc.height); return; }
  if (!V.stripCache) V.stripCache = document.createElement('canvas');
  const sc = V.stripCache;
  const w = Math.round(V.SW * V.dpr), h = Math.round(V.SH * V.dpr);
  if (sc.width !== w || sc.height !== h) { sc.width = w; sc.height = h; }
  const ctx = sc.getContext('2d');
  ctx.setTransform(V.dpr, 0, 0, V.dpr, 0, 0);
  ctx.fillStyle = th.fig_bg; ctx.fillRect(0, 0, V.SW, V.SH);
  const pw = V.SW - stripM.l - stripM.r, ph = V.SH - stripM.t - stripM.b;
  if (pw < 20 || ph < 20) return;
  const n = A.n;
  let topPe = 10;
  for (let k = 0; k < n; k++) topPe = Math.max(topPe, A.inside[k], A.outside[k]);
  for (const f of A.flashes) topPe = Math.max(topPe, f.pe);
  const floor = 0.3, topv = topPe * 3;
  const X = (xf) => stripM.l + xf / n * pw, Y = (v) => stripY(A, v, floor, topv, ph);
  ctx.fillStyle = th.axes_bg; ctx.fillRect(stripM.l, stripM.t, pw, ph);
  ctx.save(); ctx.beginPath(); ctx.rect(stripM.l, stripM.t, pw, ph); ctx.clip();
  ctx.font = fontCss(10.5); ctx.textBaseline = 'middle'; ctx.textAlign = 'right';
  ctx.strokeStyle = th.grid; ctx.lineWidth = 0.5; ctx.globalAlpha = 0.5;
  for (let e = Math.ceil(Math.log10(floor)); e <= Math.floor(Math.log10(topv)); e++) {
    const y = Y(10 ** e); ctx.beginPath(); ctx.moveTo(stripM.l, y); ctx.lineTo(stripM.l + pw, y); ctx.stroke();
  }
  for (const tk of A.ticks) { ctx.beginPath(); ctx.moveTo(X(tk.p), stripM.t); ctx.lineTo(X(tk.p), stripM.t + ph); ctx.stroke(); }
  ctx.globalAlpha = 1;
  const stairs = (arr, colour, lw) => {
    ctx.strokeStyle = colour; ctx.lineWidth = lw; ctx.beginPath();
    let prev = false;
    for (let k = 0; k < n; k++) {
      if (!(arr[k] > 0)) { prev = false; continue; }
      const y = Y(arr[k]);
      if (prev) ctx.lineTo(X(k), y); else ctx.moveTo(X(k), y);
      ctx.lineTo(X(k + 1), y); prev = true;
    }
    ctx.stroke();
  };
  const inCol = th.categorical[0], outCol = th.categorical[4];
  const total = new Float64Array(n);
  for (let k = 0; k < n; k++) total[k] = A.inside[k] + A.outside[k];
  if (V.data.hasLinks) { stairs(A.outside, outCol, 1.1); stairs(A.inside, inCol, 1.3); } else stairs(total, inCol, 1.3);
  ctx.strokeStyle = th.track; ctx.fillStyle = th.track; ctx.lineWidth = 0.9;
  for (const f of A.flashes) {
    const x = X(A.frameCoord(f.time)), y = Y(f.pe);
    ctx.globalAlpha = 0.8; ctx.beginPath(); ctx.moveTo(x, Y(floor)); ctx.lineTo(x, y); ctx.stroke();
    ctx.globalAlpha = 1; ctx.beginPath(); ctx.arc(x, y, 2.6, 0, 2 * Math.PI); ctx.fill();
  }
  const x0 = X(A.frameCoord(A.t0));
  ctx.strokeStyle = th.fg_muted; ctx.lineWidth = 0.9; ctx.setLineDash([2, 3]);
  ctx.beginPath(); ctx.moveTo(x0, stripM.t); ctx.lineTo(x0, stripM.t + ph); ctx.stroke(); ctx.setLineDash([]);
  ctx.restore();
  // axes
  ctx.strokeStyle = th.legend_edge; ctx.lineWidth = 1; ctx.strokeRect(stripM.l + 0.5, stripM.t + 0.5, pw - 1, ph - 1);
  ctx.fillStyle = th.fg_muted; ctx.strokeStyle = th.fg_muted; ctx.textAlign = 'right';
  for (let e = Math.ceil(Math.log10(floor)); e <= Math.floor(Math.log10(topv)); e++) {
    const y = Y(10 ** e);
    ctx.beginPath(); ctx.moveTo(stripM.l - 4, y); ctx.lineTo(stripM.l, y); ctx.stroke();
    ctx.fillText(pow10(e), stripM.l - 6, y);
  }
  ctx.textAlign = 'center'; ctx.textBaseline = 'top';
  for (const tk of A.ticks) {
    ctx.beginPath(); ctx.moveTo(X(tk.p), stripM.t + ph); ctx.lineTo(X(tk.p), stripM.t + ph + 4); ctx.stroke();
    ctx.fillText(tk.label, X(tk.p), stripM.t + ph + 6);
  }
  ctx.font = fontCss(11);
  const clockTxt = A.scale === 'log' ? 'frames log-spaced after t₀' : 'uniform frames';
  ctx.fillText(`t − t₀   (${clockTxt};  t₀ = ${A.t0.toFixed(3)} µs)`, stripM.l + pw / 2, stripM.t + ph + 22);
  ctx.save(); ctx.translate(12, stripM.t + ph / 2); ctx.rotate(-Math.PI / 2); ctx.textBaseline = 'middle'; ctx.fillText('PE per frame', 0, 0); ctx.restore();
  // legend
  const items = V.data.hasLinks ? [['hits in a flash', inCol, 'l'], ['hits in no flash', outCol, 'l'], ['flash (total PE)', th.track, 'd']] : [['OpHit PE', inCol, 'l'], ['flash (total PE)', th.track, 'd']];
  ctx.font = fontCss(10.5);
  const wTxt = items.map((it) => ctx.measureText(it[0]).width + 26);
  const lw = wTxt.reduce((a, b) => a + b, 0) + 8, lx = stripM.l + pw - lw - 4, ly = stripM.t + 4;
  ctx.fillStyle = th.legend_bg; ctx.globalAlpha = 0.85; ctx.fillRect(lx, ly, lw, 18); ctx.globalAlpha = 1;
  ctx.strokeStyle = th.legend_edge; ctx.strokeRect(lx + 0.5, ly + 0.5, lw - 1, 17);
  let xx = lx + 6;
  ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
  items.forEach((it, i) => {
    ctx.strokeStyle = it[1]; ctx.fillStyle = it[1]; ctx.lineWidth = 1.5;
    if (it[2] === 'l') { ctx.beginPath(); ctx.moveTo(xx, ly + 9); ctx.lineTo(xx + 14, ly + 9); ctx.stroke(); }
    else { ctx.beginPath(); ctx.arc(xx + 7, ly + 9, 2.6, 0, 2 * Math.PI); ctx.fill(); }
    ctx.fillStyle = th.fg; ctx.fillText(it[0], xx + 18, ly + 9);
    xx += wTxt[i];
  });
}

function drawStrip() {
  const R = V.R, A = V.A, th = T();
  const ctx = R.stripc.getContext('2d');
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  if (!A || !V.stripCache) return;
  ctx.drawImage(V.stripCache, 0, 0);
  ctx.setTransform(V.dpr, 0, 0, V.dpr, 0, 0);
  const pw = V.SW - stripM.l - stripM.r, ph = V.SH - stripM.t - stripM.b;
  if (V.k >= 0) {
    const x = stripM.l + (V.k + 1) / A.n * pw;
    ctx.fillStyle = th.fg; ctx.globalAlpha = 0.08; ctx.fillRect(stripM.l, stripM.t, x - stripM.l, ph); ctx.globalAlpha = 1;
    ctx.strokeStyle = th.fg; ctx.lineWidth = 1.4; ctx.beginPath(); ctx.moveTo(x, stripM.t); ctx.lineTo(x, stripM.t + ph); ctx.stroke();
  }
}

// ------------------------------------------------------------------ rendering
function requestFrame() {
  V.dirty = true;
  if (V.raf || V.playing || !V.visible) return;
  V.raf = requestAnimationFrame(() => { V.raf = 0; if (V.dirty) renderNow(); });
}

function renderNow(sync = false) {
  if (!V.ready || !V.visible) return;
  const t0 = performance.now();
  V.dirty = false;
  if (V.A && V.geo) {
    applyFrame();
    drawGL();
    drawDyn_();
    drawStrip();
    push3d();
  } else {
    clearCanvases();
  }
  updateLabels();
  const dt = performance.now() - t0;
  const s = V.stats; s.last = dt; s.n++; s.avg += (dt - s.avg) / Math.min(s.n, 30);
  if (sync && V.G) { const px = new Uint8Array(4); V.G.gl.readPixels(0, 0, 1, 1, V.G.gl.RGBA, V.G.gl.UNSIGNED_BYTE, px); }
}

function drawDyn_() {
  const ctx = V.R.dyn.getContext('2d');
  ctx.setTransform(V.dpr, 0, 0, V.dpr, 0, 0);
  ctx.clearRect(0, 0, V.W, V.H);
  drawDyn(ctx);
}

function clearCanvases() {
  const dc = V.R.dyn.getContext('2d'); dc.setTransform(1, 0, 0, 1, 0, 0); dc.clearRect(0, 0, V.R.dyn.width, V.R.dyn.height);
  if (V.G) { const { gl } = V.G; gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT); }
  drawStatic();
  drawStripCache();
}

function push3d() {
  const v3 = V.app.views.view3d;
  const want = V.lit3d && V.visible && V.A && V.geo && v3 && v3.pdTileCount === V.geo.nt;
  if (want) { v3.setPdColors(V.geo.rgb); V.pushed3d = true; }
  else if (V.pushed3d && v3) { v3.setPdColors(null); V.pushed3d = false; }
}

function sync3dAnimating() {
  const v3 = V.app.views.view3d;
  const want = !!(V.playing && V.lit3d && V.visible && v3);
  if (want !== V.animFlag) { V.animFlag = want; try { v3.setAnimating(want); } catch (e) { console.error(e); } }
}

// ------------------------------------------------------------------ playback
function play(on) {
  if (on && (!V.A)) on = false;
  if (on === V.playing) return;
  V.playing = on;
  if (on) {
    if (V.k < 0 || V.k >= V.A.n - 1) setK(0);
    V.lastT = performance.now(); V.acc = 0; V.holdUntil = 0;
    if (V.raf) { cancelAnimationFrame(V.raf); V.raf = 0; }
    V.raf = requestAnimationFrame(tick);
  } else {
    if (V.raf) { cancelAnimationFrame(V.raf); V.raf = 0; }
    commit();
  }
  sync3dAnimating();
  setPlayState(on);
  V.R.play.innerHTML = on ? '&#10074;&#10074;' : '&#9654;';
  V.R.play.classList.toggle('on', on);
  if (!on) requestFrame();
}

function setPlayState(on) {
  const s = V.app.state;
  s.lplay = V.committedPlay;
  V.committedPlay = on ? '1' : '0';
  V.app.setState({ lplay: V.committedPlay });
}

function tick(now) {
  V.raf = 0;
  if (!V.playing) return;
  if (!V.visible || !V.A) { V.raf = requestAnimationFrame(tick); V.lastT = now; return; }
  const fps = num('lfps', 1, 120);
  const dt = Math.min(250, now - V.lastT); V.lastT = now;
  if (V.holdUntil) { if (now >= V.holdUntil) { V.holdUntil = 0; V.acc = 0; V.k = -1; setK(0); } }
  else {
    V.acc += dt * fps / 1000;
    const adv = Math.floor(V.acc);
    if (adv > 0) {
      V.acc -= adv;
      const nk = V.k + adv;
      if (nk >= V.A.n - 1) { V.k = V.A.n - 1; V.dirty = true; V.holdUntil = now + 1200; }
      else { V.k = nk; V.dirty = true; }
    }
  }
  if (V.dirty) { V.app.state.lt = String(V.k); renderNow(); }
  V.raf = requestAnimationFrame(tick);
}

/** Set the frame (k = -1: whole window). Silent in the URL until commit(). */
function setK(k) {
  V.k = V.A ? clamp(k, -1, V.A.n - 1) : -1;
  V.app.state.lt = V.k < 0 ? 'all' : String(V.k);
  requestFrame();
}

let commitTimer = 0;
function commit() {
  clearTimeout(commitTimer);
  const cur = V.k < 0 ? 'all' : String(V.k);
  if (cur === V.committedLt) return;
  V.app.state.lt = V.committedLt;
  V.committedLt = cur;
  V.app.setState({ lt: cur });
}
function commitSoon() { clearTimeout(commitTimer); commitTimer = setTimeout(commit, 350); }

// ------------------------------------------------------------------ DOM sync
function updateLabels() {
  const A = V.A, R = V.R;
  if (!A) { R.flabel.textContent = ''; return; }
  const txt = V.k < 0 ? `whole window (${A.n} frames)` : `frame ${V.k + 1}/${A.n}  t−t₀ ${fmtDt(A.edges[V.k + 1] - A.t0)}`;
  if (R.flabel.textContent !== txt) R.flabel.textContent = txt;
  const sv = V.k < 0 ? '0' : String(V.k);
  if (R.slider.value !== sv && document.activeElement !== R.slider) R.slider.value = sv;
  R.all.classList.toggle('on', V.k < 0);
}

function syncControls() {
  const R = V.R;
  if (!R) return;
  const set = (el, v) => { if (document.activeElement !== el && el.value !== String(v)) el.value = String(v); };
  set(R.clock, param('lclock')); set(R.fill, param('lfill')); set(R.frames, param('lframes'));
  set(R.fps, param('lfps')); set(R.pers, param('lpers')); set(R.wlo, param('lwlo')); set(R.whi, param('lwhi'));
  const t0 = V.app.state.lt0;
  if (document.activeElement !== R.t0) R.t0.value = t0 || '';
  if (V.A && !t0) R.t0.placeholder = `auto (${V.A.t0.toFixed(3)})`;
  R.slider.disabled = !V.A;
}

function updateEmpty() {
  const R = V.R;
  if (V.A) { R.empty.textContent = ''; return; }
  const notes = (V.data ? [] : ((V.app.bundle && V.app.bundle.meta.optical && V.app.bundle.meta.optical.notes) || []));
  const geoNotes = V.app.geometry ? (V.app.geometry.meta.notes || []) : [];
  R.empty.textContent = V.app.bundle || V.app.geometry
    ? ['No light data for this event.', ...notes, ...geoNotes.filter((x) => /photon|optical|pd/i.test(x))].join('\n') : 'loading...';
}

function updateNotes() {
  const A = V.A, d = V.data, R = V.R;
  if (!A || !d) { R.notes.innerHTML = ''; return; }
  const esc = (s) => String(s).replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
  const e = A.edges;
  const k0 = lowerBound(e, A.t0 + 1e-12);
  const first = e[Math.min(k0 + 1, e.length - 1)] - e[Math.min(k0, e.length - 2)];
  const lines = [];
  lines.push(`t0 = ${A.t0.toFixed(3)} µs ${A.t0Auto ? '(onset of the brightest OpHit light burst)' : '(given)'}; window t0 ${fmtDt(A.lo - A.t0)} .. ${fmtDt(A.hi - A.t0)}, ${A.n} frames, ${A.scale === 'log' ? 'log-spaced after t0' : 'uniform'} (first after t0 ${(first * 1000).toFixed(0)} ns, last ${((e[e.length - 1] - e[e.length - 2]) * 1000).toFixed(0)} ns); tiles: ${A.cumulative ? 'cumulative' : 'afterglow'}`);
  lines.push(`${A.nWin.toLocaleString('en-US')} hits, ${Math.round(A.peWin).toLocaleString('en-US')} PE in the window` + (d.hasLinks && A.peWin > 0 ? `, ${Math.round(100 * A.peInFlash / A.peWin)}% of it in a flash` : ''));
  if (A.lost) lines.push(`${A.lost} hits on channels with no detector position are not drawn`);
  const prod = ((d.meta.products || [])[0] || '').split('_')[1] || '';
  const f0 = A.flashes[0];
  lines.push(f0 ? `${A.flashes.length} flashes${prod ? ` (${prod})` : ''}; brightest ${Math.round(f0.pe).toLocaleString('en-US')} PE at t0 ${fmtDt(f0.time - A.t0)}, y = ${f0.y.toFixed(0)}, z = ${f0.z.toFixed(0)} cm`
    : `no flashes in the window${prod ? ` (${prod})` : ''}`);
  const tv = d.truth.vertex;
  if (tv && tv.every(Number.isFinite)) lines.push(`true vertex x = ${tv[0].toFixed(0)}, y = ${tv[1].toFixed(0)}, z = ${tv[2].toFixed(0)} cm (star)`);
  const warns = [];
  // exporter notes (association status, repaired positions); the missing-flash warning is recomputed for the current t0
  for (const n of d.meta.notes || []) {
    if (/PE of hits arrive within 1 us/.test(n)) continue;
    (/association|repaired|recomputed|unusable/i.test(n) ? warns : lines).push(n.trim());
  }
  if (A.prompt >= 100) {
    const fT = d.bundle.get('flash/time'); let has = false;
    for (let j = 0; j < d.nf; j++) if (fT[j] >= A.t0 - 0.5 && fT[j] <= A.t0 + 1) { has = true; break; }
    if (!has) {
      const alt = (V.app.bundle.meta.flash_tags || []).filter((t) => t !== V.app.bundle.meta.flash_tag);
      warns.unshift(`! ${Math.round(A.prompt).toLocaleString('en-US')} PE of hits arrive within 1 us of t0 but ${prod || 'the flash finder'} has no flash there -- the interaction's flash was not reconstructed${alt.length ? `; compare flash producer ${alt.join(', ')}` : ''}`);
    }
  }
  R.notes.innerHTML = (warns.length ? warns.map((w) => `<div class="warn">${esc(w)}</div>`).join('') : '')
    + `<details><summary>${esc(lines[0])}</summary><pre>${esc(lines.slice(1).join('\n'))}</pre></details>`;
  R.notes.style.display = '';
}

// ------------------------------------------------------------------ interaction
function cellAt(x, y) {
  for (const c of V.cells) if (x >= c.x && x <= c.x + c.cw && y >= c.y && y <= c.y + c.ch) return c;
  return null;
}
function tileAt(c, x, y) {
  const g = V.geo, w = winOf(c);
  const sx = c.cw / (w.z1 - w.z0), sy = c.ch / (w.y1 - w.y0);
  const z = w.z0 + (x - c.x) / sx, yy = w.y1 - (y - c.y) / sy;
  let best = -1, bd = Infinity;
  const tz = 2 / sx, ty = 2 / sy;
  for (let s = g.start[c.w]; s < g.start[c.w + 1]; s++) {
    const i = g.order[s];
    const dz = Math.abs(z - g.xyz[3 * i + 2]) - g.half[2 * i + 1], dy = Math.abs(yy - g.xyz[3 * i + 1]) - g.half[2 * i];
    if (dz <= tz && dy <= ty) { const d = Math.max(dz, dy); if (d < bd) { bd = d; best = i; } }
  }
  return best;
}

function bindInteraction() {
  const R = V.R, dyn = R.dyn;
  const pos = (e) => { const r = dyn.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
  let drag = null;
  dyn.addEventListener('wheel', (e) => {
    if (!V.geo) return;
    e.preventDefault();
    const [x, y] = pos(e), c = cellAt(x, y) || V.cells[0];
    if (!c) return;
    const w = winOf(c), e0 = V.geo.ext[c.w];
    const fx = clamp((x - c.x) / c.cw, 0, 1), fy = clamp((y - c.y) / c.ch, 0, 1);
    const pz = w.z0 + fx * (w.z1 - w.z0), py = w.y1 - fy * (w.y1 - w.y0);
    const z2 = clamp(V.zoom * Math.exp(-e.deltaY * 0.0015), 1, 60);
    const wz = (e0.z1 - e0.z0) / z2, wy = (e0.y1 - e0.y0) / z2;
    const nz0 = pz - fx * wz, ny1 = py + fy * wy;
    V.zoom = z2;
    V.u = (nz0 + wz / 2 - e0.z0) / (e0.z1 - e0.z0);
    V.v = (ny1 - wy / 2 - e0.y0) / (e0.y1 - e0.y0);
    V.u = clamp(V.u, 0.5 / z2, 1 - 0.5 / z2); V.v = clamp(V.v, 0.5 / z2, 1 - 0.5 / z2);
    drawStatic(); requestFrame();
  }, { passive: false });
  dyn.addEventListener('pointerdown', (e) => {
    const [x, y] = pos(e), c = cellAt(x, y);
    if (!c) return;
    drag = { x, y, c, u: V.u, v: V.v }; dyn.setPointerCapture(e.pointerId);
  });
  dyn.addEventListener('pointermove', (e) => {
    const [x, y] = pos(e);
    if (drag) {
      const e0 = V.geo.ext[drag.c.w], s = drag.c.cw / ((e0.z1 - e0.z0) / V.zoom);
      V.u = drag.u - (x - drag.x) / s / (e0.z1 - e0.z0);
      V.v = drag.v + (y - drag.y) / s / (e0.y1 - e0.y0);
      V.u = clamp(V.u, 0.5 / V.zoom, 1 - 0.5 / V.zoom); V.v = clamp(V.v, 0.5 / V.zoom, 1 - 0.5 / V.zoom);
      drawStatic(); requestFrame(); R.tip.style.display = 'none'; return;
    }
    const c = V.geo && cellAt(x, y);
    const i = c ? tileAt(c, x, y) : -1;
    if (i !== V.hover || (c && c.w !== V.hoverWall)) { V.hover = i; V.hoverWall = c ? c.w : -1; requestFrame(); }
    if (i < 0) { R.tip.style.display = 'none'; return; }
    R.tip.textContent = tipText(i);
    R.tip.style.display = 'block';
    const tw = R.tip.offsetWidth, th = R.tip.offsetHeight;
    R.tip.style.left = `${Math.min(x + 14, V.W - tw - 4)}px`; R.tip.style.top = `${Math.min(y + 14, V.H - th - 4)}px`;
  });
  const up = (e) => { drag = null; try { dyn.releasePointerCapture(e.pointerId); } catch (err) { /* not captured */ } };
  dyn.addEventListener('pointerup', up); dyn.addEventListener('pointercancel', up);
  dyn.addEventListener('pointerleave', () => { if (V.hover >= 0) { V.hover = -1; requestFrame(); } R.tip.style.display = 'none'; });
  dyn.addEventListener('dblclick', () => { V.zoom = 1; V.u = 0.5; V.v = 0.5; drawStatic(); requestFrame(); });

  // strip scrubbing
  const sc = R.stripc;
  let scrub = false;
  const frameAt = (e) => {
    const r = sc.getBoundingClientRect(), pw = V.SW - stripM.l - stripM.r;
    return clamp(Math.floor((e.clientX - r.left - stripM.l) / pw * V.A.n), 0, V.A.n - 1);
  };
  sc.addEventListener('pointerdown', (e) => { if (!V.A) return; scrub = true; sc.setPointerCapture(e.pointerId); if (V.playing) play(false); setK(frameAt(e)); });
  sc.addEventListener('pointermove', (e) => { if (scrub) setK(frameAt(e)); });
  const sup = (e) => { if (!scrub) return; scrub = false; try { sc.releasePointerCapture(e.pointerId); } catch (err) { /* */ } commit(); };
  sc.addEventListener('pointerup', sup); sc.addEventListener('pointercancel', sup);

  // controls
  R.play.addEventListener('click', () => play(!V.playing));
  R.prev.addEventListener('click', () => { play(false); setK((V.k < 0 ? V.A.n : V.k) - 1); commit(); });
  R.next.addEventListener('click', () => { play(false); setK(V.k + 1); commit(); });
  R.all.addEventListener('click', () => { play(false); setK(V.k < 0 ? 0 : -1); commit(); });
  R.slider.addEventListener('input', () => { if (V.playing) play(false); setK(parseInt(R.slider.value, 10)); commitSoon(); });
  const bindParam = (el, key, tf = (x) => x) => el.addEventListener('change', () => V.app.setState({ [key]: tf(el.value) }));
  bindParam(R.clock, 'lclock'); bindParam(R.fill, 'lfill'); bindParam(R.frames, 'lframes');
  bindParam(R.fps, 'lfps'); bindParam(R.pers, 'lpers'); bindParam(R.wlo, 'lwlo'); bindParam(R.whi, 'lwhi');
  bindParam(R.t0, 'lt0', (x) => { const f = parseFloat(x); return Number.isFinite(f) ? String(f) : ''; });
  R.t0auto.addEventListener('click', () => V.app.setState({ lt0: '' }));
  const jump = (dir) => {
    const b = bursts(), t0 = V.A ? V.A.t0 : 0;
    const cand = dir > 0 ? b.find((t) => t > t0 + 2e-4) : [...b].reverse().find((t) => t < t0 - 2e-4);
    if (cand != null) V.app.setState({ lt0: cand.toFixed(5) });
    else V.app.status(`no ${dir > 0 ? 'later' : 'earlier'} light burst`);
  };
  R.burstnext.addEventListener('click', () => jump(1));
  R.burstprev.addEventListener('click', () => jump(-1));
  window.addEventListener('keydown', (e) => {
    if (!V.visible || e.key !== ' ') return;
    const t = e.target && e.target.tagName;
    if (t === 'INPUT' || t === 'SELECT' || t === 'TEXTAREA' || t === 'BUTTON') return;
    e.preventDefault(); play(!V.playing);
  });
}

function tipText(i) {
  const g = V.geo, A = V.A;
  const chans = [];
  if (g.chOff && g.ch) for (let j = g.chOff[i]; j < g.chOff[i + 1]; j++) chans.push(g.ch[j]);
  let sofar = 0;
  const nt = g.nt;
  if (A) { const kk = V.k < 0 ? A.n - 1 : V.k; for (let k = 0; k <= kk; k++) sofar += A.arrivals[k * nt + i]; }
  return `tile ${i}  (wall ${g.wall[i]})  y ${g.xyz[3 * i + 1].toFixed(0)}  z ${g.xyz[3 * i + 2].toFixed(0)} cm\n`
    + `channels: ${chans.length > 10 ? chans.slice(0, 10).join(', ') + `, ... (${chans.length})` : (chans.join(', ') || '-')}\n`
    + `PE ${V.k < 0 ? 'in window' : 'so far'}: ${sofar.toFixed(1)}\nPE in window: ${A ? A.collected[i].toFixed(1) : '-'}`;
}

// ------------------------------------------------------------------ module interface
export default {
  name: 'viewlight',
  init(container, app) {
    V.app = app; V.container = container;
    for (const [k, v] of Object.entries(DEF)) app.registerParam(k, v);
    V.R = buildDom(container);
    V.fgRgb = [1, 1, 1];
    try { V.G = initGL(V.R.gl); } catch (e) { console.error('viewlight: WebGL', e); V.G = null; app.status('light view: WebGL unavailable', true); }
    V.ready = true;
    bindInteraction();
    new ResizeObserver(() => { if (V.visible) { layout(); drawStripCache(); requestFrame(); } }).observe(V.R.walls);
    new ResizeObserver(() => { if (V.visible) { layout(); drawStripCache(); requestFrame(); } }).observe(V.R.strip);
    this.updateTheme();
    window.__evdLight = this;
  },
  updateTheme() {
    const th = V.app.theme, x = parseInt(th.fg.slice(1), 16);
    V.fgRgb = [(x >> 16 & 255) / 255, (x >> 8 & 255) / 255, (x & 255) / 255];
  },
  setGeometry(geometry) { setGeometry(geometry); },
  setEvent(bundle, geometry) { if (!V.geo && geometry) setGeometry(geometry); setEvent(bundle, geometry); },
  onState(state, changed) {
    V.lit3d = state.view === 'charge+light';
    if (changed.has('theme') || changed.has('cmap')) {
      this.updateTheme(); buildPalette();
      if (V.visible) { drawStatic(); drawStripCache(); }
      V.pushed3d = false;
    }
    if (REBUILD_KEYS.some((k) => changed.has(k)) && V.data) rebuild();
    if (changed.has('lt')) {
      const k = state.lt === 'all' || state.lt === '' ? -1 : (parseInt(state.lt, 10) || 0);
      if (V.A && k !== V.k) { V.k = clamp(k, -1, V.A.n - 1); V.committedLt = state.lt; requestFrame(); }
      V.committedLt = state.lt;
    }
    if (changed.has('lplay') && !changed.has('lt0')) {
      const want = state.lplay === '1';
      if (want !== V.playing && V.A) { V.committedPlay = state.lplay; play(want); }
    }
    if (changed.has('view') || changed.has('theme') || changed.has('cmap')) { sync3dAnimating(); requestFrame(); }
    syncControls();
  },
  setVisible(vis) {
    V.visible = !!vis;
    if (!vis) { if (V.playing) play(false); V.app.views.view3d && V.pushed3d && (V.app.views.view3d.setPdColors(null), V.pushed3d = false); return; }
    requestFrame();
  },
  resize() { layout(); drawStripCache(); requestFrame(); },
  getCanvas() {
    if (!V.A || !V.visible) return null;
    renderNow();
    const out = document.createElement('canvas');
    out.width = Math.round(V.W * V.dpr); out.height = Math.round((V.H + V.SH) * V.dpr);
    const c = out.getContext('2d');
    c.drawImage(V.R.static, 0, 0);
    c.drawImage(V.R.gl, 0, 0);
    c.drawImage(V.R.dyn, 0, 0);
    c.drawImage(V.R.stripc, 0, V.H * V.dpr);
    return out;
  },
  // testing helpers
  _V: V,
  setFrame(k) { setK(k); commit(); renderNow(); },
  /** Time n frame updates (CPU submit + forced GL sync); returns ms per frame. */
  benchmark(n = 120) {
    if (!V.A) return null;
    const keep = V.k, t0 = performance.now();
    let cpu = 0;
    for (let i = 0; i < n; i++) { V.k = i % V.A.n; V.app.state.lt = String(V.k); renderNow(true); cpu += V.stats.last; }
    const wall = (performance.now() - t0) / n;
    V.k = keep; requestFrame();
    return { msPerFrame: wall, cpuMs: cpu / n, nTiles: V.geo.nt };
  },
};
