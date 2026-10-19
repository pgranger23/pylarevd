// view2d.js -- wire-plane panels (hits + projected overlays), WebGL2 + Canvas2D overlay.
//
// One WebGL canvas holds every panel (viewport + scissor per panel); a second, transparent
// Canvas2D on top draws axes, titles, reference lines, vertex markers, labels, the hover ring
// and receives the pointer events. Hits and deposits are gl.POINTS (value -> LUT in the vertex
// shader, categorical colour x marker shape in the fragment shader); tracks, shower outlines
// and true trajectories are GL_LINES (widened by offset passes, dashed in the shader); shower
// cones also get a triangle-fan fill.
// Every control change (colour-by, scale, cmap, theme, overlays) only changes uniforms or which
// already-built attribute buffer is bound; geometry is built once per event (or lazily on first use).
//
// Spec: docs/WEB_VIEWER.md (WP-A notes for the arrays, WP-C notes for the view interface).
// Zoom/pan per panel persists across state changes and event steps; it is reset by a new
// geometry (file/detector), a double click, or the "reset" button.

import { getLut, cmapGradientCss, hexToRgb, PDG_CLASSES, pdgClass } from './colormaps.js';

const CSS = `
.v2d-bar { position:absolute; left:6px; top:3px; z-index:6; display:flex; gap:10px; align-items:center;
  font-size:12px; color:var(--muted); }
.v2d-bar label { display:inline-flex; align-items:center; gap:4px; }
.v2d-bar select, .v2d-bar button { padding:0 5px; font-size:12px; }
.v2d-bar input[type=range] { width:70px; }
.v2d-canvas { position:absolute; left:0; top:0; width:100%; height:100%; display:block; }
.v2d-ov { touch-action:none; cursor:crosshair; }
.v2d-tip { position:absolute; z-index:7; pointer-events:none; display:none; white-space:pre; font:11px/1.35 ui-monospace,Menlo,monospace;
  background:color-mix(in srgb, var(--legend-bg) 92%, transparent); border:1px solid var(--legend-edge); border-radius:4px;
  padding:3px 6px; color:var(--fg); }
.v2d-legend { top:26px !important; max-width:150px; }
.v2d-legend .ln { display:inline-block; width:14px; height:0; margin-right:4px; vertical-align:3px; border-top:2px solid; }
`;

const MARGIN = { l: 58, r: 8, t: 20, b: 34 };
const TOPSTRIP = 24;
const GUTTER = 138;               // right-hand strip for the legend on wide containers
const ALPHA = 0.75;               // matplotlib _ALPHA
const N_SHAPES = 5;               // o ^ s D v

const V = {
  app: null, container: null, ready: false, visible: false,
  gl: null, glc: null, ovc: null, ov: null, W: 0, H: 0, dpr: 1, cw: 0, ch: 0,
  progs: {}, lutTex: null, lutName: null,
  S: null,                       // per-event state
  views: {},                     // space -> [{cx,cy,hx,hy,link,init:{hx,hy}}]
  rects: [],                     // layout per panel (css px)
  space: 'wire', linkW: true, size: 3,
  bar: {}, legend: null, tip: null, hover: null,
  needGL: false, needOv: false, raf: 0, drag: null, shot: null,
  lastRenderMs: 0, lastLegend: '',
};

// ---------------------------------------------------------------------------- shaders
const POINT_VS = `#version 300 es
precision highp float;
in vec2 aPos; in float aVal; in float aCat; in float aFlag;
uniform vec2 uCenter, uHalf;
uniform float uSize, uMode, uLog, uLo, uInv, uFloor, uCmapLo, uCmapHi, uShowRad;
uniform sampler2D uLut;
uniform vec3 uPal[8];
uniform vec3 uUnassoc, uUniform;
out vec3 vColor; flat out float vShape;
void main() {
  vec2 n = (aPos - uCenter) / uHalf;
  float z = 0.0; vShape = 0.0;
  if (uMode < 0.5) {
    float x = uLog > 0.5 ? log(max(aVal, uFloor)) : aVal;
    float t = clamp((x - uLo) * uInv, 0.0, 1.0);
    vColor = texture(uLut, vec2(mix(uCmapLo, uCmapHi, t), 0.5)).rgb;
    z = 0.9 - 1.8 * t;                       // brighter hits in front, as matplotlib's sorted scatter
  } else if (uMode < 1.5) {
    if (aCat < -0.5) { vColor = uUnassoc; z = 0.5; }
    else { vColor = uPal[int(mod(aCat, 8.0))]; vShape = mod(aCat, 5.0); }
  } else { vColor = uUniform; }
  gl_PointSize = uSize;
  gl_Position = vec4(n, z, 1.0);
  if (aFlag < 0.5 && uShowRad < 0.5) gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
}`;
const POINT_FS = `#version 300 es
precision mediump float;
in vec3 vColor; flat in float vShape;
uniform float uAlpha;
out vec4 o;
void main() {
  vec2 q = gl_PointCoord * 2.0 - 1.0; q.y = -q.y;
  float s = vShape; bool ok;
  if (s < 0.5) ok = dot(q, q) <= 1.0;
  else if (s < 1.5) ok = q.y > -0.8 && abs(q.x) <= (0.9 - q.y) * 0.53;
  else if (s < 2.5) ok = max(abs(q.x), abs(q.y)) <= 0.8;
  else if (s < 3.5) ok = abs(q.x) + abs(q.y) <= 1.0;
  else { q.y = -q.y; ok = q.y > -0.8 && abs(q.x) <= (0.9 - q.y) * 0.53; }
  if (!ok) discard;
  o = vec4(vColor, uAlpha);
}`;

const LINE_VS = `#version 300 es
precision highp float;
in vec2 aPos; in vec2 aExtra;
uniform vec2 uCenter, uHalf, uOff;
uniform float uMode, uHasFlag, uShowRad;
uniform vec3 uColor, uPal[8], uPdg[9], uUnassoc;
out vec3 vColor; out float vCum;
void main() {
  gl_Position = vec4((aPos - uCenter) / uHalf + uOff, 0.0, 1.0);
  float cls = mod(aExtra.y, 16.0);
  if (uMode < 0.5) vColor = uColor;
  else if (uMode < 1.5) vColor = aExtra.y < -0.5 ? uUnassoc : uPal[int(mod(aExtra.y, 8.0))];
  else vColor = uPdg[int(cls)];
  if (uHasFlag > 0.5 && aExtra.y > 15.5 && uShowRad < 0.5) gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
  vCum = aExtra.x;
}`;
const LINE_FS = `#version 300 es
precision highp float;
in vec3 vColor; in float vCum;
uniform float uAlpha, uDash, uPxPerCm, uPeriod;
out vec4 o;
void main() {
  if (uDash > 0.5 && fract(vCum * uPxPerCm / uPeriod) > 0.55) discard;
  o = vec4(vColor, uAlpha);
}`;
const FILL_VS = `#version 300 es
precision highp float;
in vec2 aPos; uniform vec2 uCenter, uHalf, uOff;
void main() { gl_Position = vec4((aPos - uCenter) / uHalf + uOff, 0.0, 1.0); }`;
const FILL_FS = `#version 300 es
precision mediump float;
uniform vec3 uColor; uniform float uAlpha; out vec4 o;
void main() { o = vec4(uColor, uAlpha); }`;

class Prog {
  constructor(gl, vs, fs, attribs) {
    const mk = (type, src) => {
      const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error('view2d shader: ' + gl.getShaderInfoLog(s));
      return s;
    };
    this.gl = gl; this.p = gl.createProgram();
    gl.attachShader(this.p, mk(gl.VERTEX_SHADER, vs)); gl.attachShader(this.p, mk(gl.FRAGMENT_SHADER, fs));
    attribs.forEach((a, i) => gl.bindAttribLocation(this.p, i, a));
    gl.linkProgram(this.p);
    if (!gl.getProgramParameter(this.p, gl.LINK_STATUS)) throw new Error('view2d link: ' + gl.getProgramInfoLog(this.p));
    this.u = {};
  }
  use() { this.gl.useProgram(this.p); return this; }
  l(n) { let v = this.u[n]; if (v === undefined) v = this.u[n] = this.gl.getUniformLocation(this.p, n); return v; }
  f(n, x) { this.gl.uniform1f(this.l(n), x); }
  v2(n, x, y) { this.gl.uniform2f(this.l(n), x, y); }
  v3(n, a) { this.gl.uniform3f(this.l(n), a[0], a[1], a[2]); }
}

// GPU buffer that survives context loss (CPU array kept)
class Res {
  constructor(data) { this.data = data; this.buf = null; }
  bind(gl) {
    if (!this.buf || this.ctx !== gl) {
      this.buf = gl.createBuffer(); this.ctx = gl;
      gl.bindBuffer(gl.ARRAY_BUFFER, this.buf);
      gl.bufferData(gl.ARRAY_BUFFER, this.data, gl.STATIC_DRAW);
    } else gl.bindBuffer(gl.ARRAY_BUFFER, this.buf);
    return this;
  }
  free() { if (this.buf && this.ctx) { try { this.ctx.deleteBuffer(this.buf); } catch (e) { /* lost */ } } this.buf = null; }
}

function attrF(gl, loc, res, size, byteOff = 0, div = 0) {
  res.bind(gl);
  gl.enableVertexAttribArray(loc);
  gl.vertexAttribPointer(loc, size, gl.FLOAT, false, 0, byteOff);
  gl.vertexAttribDivisor(loc, div);
}
function attrOff(gl, loc, def) {
  gl.disableVertexAttribArray(loc);
  gl.vertexAttribDivisor(loc, 0);
  gl.vertexAttrib4f(loc, def, def, def, def);
}

// ---------------------------------------------------------------------------- init
function initGL() {
  const gl = V.glc.getContext('webgl2', { antialias: false, alpha: false, depth: true, preserveDrawingBuffer: false });
  if (!gl) throw new Error('WebGL2 unavailable');
  V.gl = gl;
  V.progs.point = new Prog(gl, POINT_VS, POINT_FS, ['aPos', 'aVal', 'aCat', 'aFlag']);
  V.progs.line = new Prog(gl, LINE_VS, LINE_FS, ['aPos', 'aExtra']);
  V.progs.fill = new Prog(gl, FILL_VS, FILL_FS, ['aPos']);
  V.lutTex = gl.createTexture(); V.lutName = null;
  gl.bindTexture(gl.TEXTURE_2D, V.lutTex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.enable(gl.BLEND);
  gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
  gl.depthFunc(gl.LEQUAL);
}

function uploadLut() {
  const st = V.app.state;
  if (V.lutName === st.cmap || !V.gl) return;
  const lut = getLut(st.cmap), d = new Uint8Array(256 * 4);
  for (let i = 0; i < 256; i++) { d[4 * i] = lut[3 * i]; d[4 * i + 1] = lut[3 * i + 1]; d[4 * i + 2] = lut[3 * i + 2]; d[4 * i + 3] = 255; }
  const gl = V.gl;
  gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, V.lutTex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 256, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, d);
  V.lutName = st.cmap;
}

function init(container, app) {
  V.app = app; V.container = container;
  const st = document.createElement('style'); st.textContent = CSS; document.head.appendChild(st);
  app.registerParam('space2d', 'wire');
  container.innerHTML = '';
  V.glc = document.createElement('canvas'); V.glc.className = 'v2d-canvas';
  V.ovc = document.createElement('canvas'); V.ovc.className = 'v2d-canvas v2d-ov';
  container.append(V.glc, V.ovc);
  V.ov = V.ovc.getContext('2d');
  // controls (own DOM: no change to the main toolbar)
  const bar = document.createElement('div'); bar.className = 'v2d-bar';
  bar.innerHTML = '<label>space <select class="sp"><option value="wire">wire (w, x)</option><option value="readout">readout (channel, tick)</option></select></label>' +
    '<label title="share the w axis across panels (pan/zoom together)"><input type="checkbox" class="lk" checked> link w</label>' +
    '<label>size <input type="range" class="sz" min="1" max="10" step="0.5" value="3"></label>' +
    '<button class="rs" title="reset zoom of all panels (or double-click)">reset zoom</button>';
  container.appendChild(bar);
  V.bar = { sel: bar.querySelector('.sp'), lk: bar.querySelector('.lk'), sz: bar.querySelector('.sz'), rs: bar.querySelector('.rs') };
  V.bar.sel.addEventListener('change', () => app.setState({ space2d: V.bar.sel.value }));
  V.bar.lk.addEventListener('change', () => { V.linkW = V.bar.lk.checked; if (V.linkW) syncLinked(-1); schedule(true, true); });
  V.bar.sz.addEventListener('input', () => { V.size = parseFloat(V.bar.sz.value); schedule(true, false); });
  V.bar.rs.addEventListener('click', () => resetViews());
  V.legend = document.createElement('div'); V.legend.className = 'hud-legend v2d-legend'; container.appendChild(V.legend);
  V.tip = document.createElement('div'); V.tip.className = 'v2d-tip'; container.appendChild(V.tip);
  try { initGL(); } catch (e) { console.error(e); app.status('2D view: ' + e.message, true); return; }
  V.glc.addEventListener('webglcontextlost', (e) => { e.preventDefault(); V.gl = null; });
  V.glc.addEventListener('webglcontextrestored', () => {
    try { initGL(); if (V.S) for (const r of V.S.res) { r.buf = null; } uploadLut(); schedule(true, true); } catch (e) { console.error(e); }
  });
  bindPointer();
  if (typeof ResizeObserver !== 'undefined') new ResizeObserver(() => { if (V.visible) resize(); }).observe(container);
  V.ready = true;
}

// ---------------------------------------------------------------------------- data (per event)
const HIT_NUM = ['charge', 'amplitude', 'tick', 'multiplicity'];
const CAT_KINDS = ['track', 'shower', 'pfparticle', 'cluster', 'slice'];

function panelSpecs(space) {
  if (!V.S) return [];
  const m = V.S.bundle.meta;
  return (space === 'readout' ? m.panels_readout : m.panels) || [];
}

function buildLayout(space) {
  const b = V.S.bundle;
  const specs = panelSpecs(space), np = specs.length;
  const n = b.rows('hits/charge') || b.rows('hits/h');
  let pan, h, v;
  if (space === 'readout') {
    pan = b.get('hits/chan_view'); const ch = b.get('hits/channel'); h = ch ? Float32Array.from(ch) : null; v = b.get('hits/tick');
  } else { pan = b.get('hits/panel'); h = b.get('hits/h'); v = b.get('hits/v'); }
  const L = { space, np, n: 0, ranges: [], orig: null, pos: null, vsrc: v, bufs: new Map(), grids: [], specs };
  if (!pan || !h || !v) { L.ranges = specs.map(() => [0, 0]); L.pos = new Res(new Float32Array(0)); L.orig = new Uint32Array(0); return L; }
  const cnt = new Uint32Array(np + 1);
  for (let i = 0; i < n; i++) { const p = pan[i]; if (p >= 0 && p < np && Number.isFinite(h[i]) && Number.isFinite(v[i])) cnt[p + 1]++; }
  for (let p = 0; p < np; p++) cnt[p + 1] += cnt[p];
  const m = cnt[np], cur = cnt.slice(0, np);
  const orig = new Uint32Array(m), pos = new Float32Array(2 * m);
  for (let i = 0; i < n; i++) {
    const p = pan[i];
    if (p >= 0 && p < np && Number.isFinite(h[i]) && Number.isFinite(v[i])) {
      const k = cur[p]++; orig[k] = i; pos[2 * k] = h[i]; pos[2 * k + 1] = v[i];
    }
  }
  L.n = m; L.orig = orig; L.pos = new Res(pos);
  for (let p = 0; p < np; p++) L.ranges.push([cnt[p], cnt[p + 1]]);
  V.S.res.push(L.pos);
  return L;
}

function layout() {
  const S = V.S;
  if (!S.layouts[V.space]) S.layouts[V.space] = buildLayout(V.space);
  return S.layouts[V.space];
}

function layoutBuf(L, key, make) {
  let r = L.bufs.get(key);
  if (!r) { const d = make(); r = d ? new Res(d) : null; if (r) V.S.res.push(r); L.bufs.set(key, r); }
  return r;
}

function gather(src, orig) {
  const out = new Float32Array(orig.length);
  for (let i = 0; i < orig.length; i++) out[i] = src[orig[i]];
  return out;
}

/** rank (colour/marker index) per original hit for a grouping kind, plus counts. */
function catRanks(kind) {
  const S = V.S;
  if (S.cats.has(kind)) return S.cats.get(kind);
  const g = S.bundle.get(`hits/group/${kind}`);
  let info = null;
  if (g) {
    const ids = new Set();
    for (let i = 0; i < g.length; i++) if (g[i] >= 0) ids.add(g[i]);
    const sorted = Array.from(ids).sort((a, b) => a - b);
    const map = new Map(sorted.map((id, r) => [id, r]));
    const rank = new Float32Array(g.length);
    let un = 0;
    for (let i = 0; i < g.length; i++) { const r = g[i] >= 0 ? map.get(g[i]) : -1; rank[i] = r; if (r < 0) un++; }
    info = { rank, map, nObj: sorted.length, nUn: un };
  }
  S.cats.set(kind, info);
  return info;
}

function valueSource(L, name) {
  const b = V.S.bundle;
  if (name === 'x') return L.vsrc;
  return b.get(`hits/${name}`);
}

function valueStats(L, name) {
  const key = `${L.space}:${name}`;
  const S = V.S;
  if (S.stats.has(key)) return S.stats.get(key);
  const src = valueSource(L, name);
  let s = null;
  if (src && src.length) {
    const step = Math.max(1, Math.floor(src.length / 60000));
    const pos = [], all = [];
    for (let i = 0; i < src.length; i += step) { const x = src[i]; if (Number.isFinite(x)) { all.push(x); if (x > 0) pos.push(x); } }
    all.sort((a, b) => a - b); pos.sort((a, b) => a - b);
    const q = (a, f) => a[Math.min(a.length - 1, Math.max(0, Math.floor(f * (a.length - 1))))];
    s = { lin: all.length ? [q(all, 0.005), q(all, 0.995)] : [0, 1], pos: pos.length ? [q(pos, 0.005), q(pos, 0.995)] : [1, 10],
      logOk: name === 'charge' || name === 'amplitude' };
  }
  S.stats.set(key, s);
  return s;
}

function colorSpec() {
  const st = V.app.state, L = layout();
  let c = st.color;
  let note = '';
  if (CAT_KINDS.includes(c)) {
    const info = catRanks(c);
    if (info) return { kind: 'cat', name: c, info };
    note = `no ${c} grouping in this event; showing charge`; c = 'charge';
  } else if (c === 'y' || c === 'z' || c === 'pdg') {
    note = `${c} is not available per hit in 2D; showing charge`; c = 'charge';
  }
  if (c === 'x') { if (!L.vsrc) c = 'charge'; } else if (!HIT_NUM.includes(c) || !V.S.bundle.get(`hits/${c}`)) { note = note || `no ${c} for hits; showing charge`; c = 'charge'; }
  const s = valueStats(L, c);
  if (!s) return { kind: 'none', name: c, note: 'no hits' };
  const log = st.scale === 'log' || (st.scale === 'auto' && s.logOk);
  let lo, hi;
  if (log) {
    hi = s.pos[1]; lo = Math.max(s.pos[0], hi / 1e4); if (!(hi > lo)) hi = lo * 10;
  } else {
    [lo, hi] = s.lin;
    if (s.logOk || c === 'multiplicity') lo = Math.max(0, lo);
    if (!(hi > lo)) hi = lo + 1;
  }
  return { kind: 'val', name: c === 'x' && L.space === 'readout' ? 'tick' : c, key: c, log, lo, hi, note };
}

// ---- line sets (tracks / showers / mc), sorted by panel -------------------------------------------
function buildLines(pts, offs, panel, np, metaOf, closed) {
  const nr = offs.length - 1, cnt = new Uint32Array(np + 1);
  const segs = (r) => { const len = offs[r + 1] - offs[r]; return len > 1 ? len - 1 + (closed ? 1 : 0) : 0; };
  for (let r = 0; r < nr; r++) { const p = panel[r]; if (p < np) cnt[p + 1] += segs(r); }
  for (let p = 0; p < np; p++) cnt[p + 1] += cnt[p];
  const tot = cnt[np], cur = cnt.slice(0, np);
  const seg = new Float32Array(4 * tot), ext = new Float32Array(4 * tot);   // ext: (cum, meta) per vertex
  for (let r = 0; r < nr; r++) {
    const p = panel[r]; if (p >= np) continue;
    const a = offs[r], e = offs[r + 1];
    if (e - a < 2) continue;
    const m = metaOf(r);
    let cum = 0;
    const push = (i, j) => {
      const k = cur[p]++;
      const x0 = pts[2 * i], y0 = pts[2 * i + 1], x1 = pts[2 * j], y1 = pts[2 * j + 1];
      seg[4 * k] = x0; seg[4 * k + 1] = y0; seg[4 * k + 2] = x1; seg[4 * k + 3] = y1;
      ext[4 * k] = cum; ext[4 * k + 1] = m;
      cum += Math.hypot(x1 - x0, y1 - y0);
      ext[4 * k + 2] = cum; ext[4 * k + 3] = m;
    };
    for (let i = a; i < e - 1; i++) push(i, i + 1);
    if (closed) push(e - 1, a);
  }
  const ranges = []; for (let p = 0; p < np; p++) ranges.push([cnt[p], cnt[p + 1] - cnt[p]]);
  const set = { seg: new Res(seg), ext: new Res(ext), ranges, total: tot };
  V.S.res.push(set.seg, set.ext);
  return set;
}

function buildFans(pts, offs, panel, np) {
  const nr = offs.length - 1, cnt = new Uint32Array(np + 1);
  for (let r = 0; r < nr; r++) { const p = panel[r]; if (p < np && offs[r + 1] - offs[r] > 2) cnt[p + 1] += 3 * (offs[r + 1] - offs[r]); }
  for (let p = 0; p < np; p++) cnt[p + 1] += cnt[p];
  const cur = cnt.slice(0, np), out = new Float32Array(2 * cnt[np]);
  for (let r = 0; r < nr; r++) {
    const p = panel[r]; if (p >= np) continue;
    const a = offs[r], e = offs[r + 1], n = e - a;
    if (n < 3) continue;
    let cx = 0, cy = 0;
    for (let i = a; i < e; i++) { cx += pts[2 * i]; cy += pts[2 * i + 1]; }
    cx /= n; cy /= n;
    for (let i = 0; i < n; i++) {
      const i0 = a + i, i1 = a + (i + 1) % n;
      let k = cur[p]; cur[p] += 3; k *= 2;
      out[k] = cx; out[k + 1] = cy; out[k + 2] = pts[2 * i0]; out[k + 3] = pts[2 * i0 + 1]; out[k + 4] = pts[2 * i1]; out[k + 5] = pts[2 * i1 + 1];
    }
  }
  const ranges = []; for (let p = 0; p < np; p++) ranges.push([cnt[p], cnt[p + 1] - cnt[p]]);
  const set = { pos: new Res(out), ranges };
  V.S.res.push(set.pos);
  return set;
}

function overlaySet(kind) {
  const S = V.S, b = S.bundle, np = (b.meta.panels || []).length;
  if (S.sets[kind] !== undefined) return S.sets[kind];
  let set = null;
  try {
    if (kind === 'tracks') {
      const pts = b.get('tracks2d/points'), off = b.get('tracks2d/offsets'), pan = b.get('tracks2d/panel'), id = b.get('tracks2d/track');
      if (pts && off && pan) {
        const info = catRanks('track');
        set = buildLines(pts, off, pan, np, (r) => {
          const t = id ? id[r] : r; if (!info) return t % 8;
          const k = info.map.get(t); return k === undefined ? t % 8 : k;
        }, false);
      }
    } else if (kind === 'showers') {
      const pts = b.get('showers2d/points'), off = b.get('showers2d/offsets'), pan = b.get('showers2d/panel');
      if (pts && off && pan) { set = buildLines(pts, off, pan, np, () => 0, true); set.fill = buildFans(pts, off, pan, np); }
    } else if (kind === 'mc') {
      const pts = b.get('mc2d/points'), off = b.get('mc2d/offsets'), pan = b.get('mc2d/panel'), idx = b.get('mc2d/index');
      const pdg = b.get('mc/pdg'), fi = b.get('mc/from_interaction');
      if (pts && off && pan) {
        set = buildLines(pts, off, pan, np, (r) => {
          const i = idx ? idx[r] : -1;
          return (pdg && i >= 0 ? pdgClass(pdg[i]) : 8) + (fi && i >= 0 && !fi[i] ? 16 : 0);
        }, false);
        set.hasFlag = !!(fi && b.meta.truth && b.meta.truth.has_split);
      }
    }
  } catch (e) { console.error('view2d overlay', kind, e); }
  S.sets[kind] = set;
  return set;
}

// ---- face projection (deposits, labels) ----------------------------------------------------------
function projectPoint(panel, x, y, z) {
  for (const f of panel.faces || []) {
    for (const bx of f.boxes) {
      if (x >= bx[0] && x <= bx[3] && y >= bx[1] && y <= bx[4] && z >= bx[2] && z <= bx[5]) return [f.dir[0] * y + f.dir[1] * z, x];
    }
  }
  return null;
}

function depositSet() {
  const S = V.S, b = S.bundle;
  if (S.sets.dep !== undefined) return S.sets.dep;
  let set = null;
  const xyz = b.get('dep/xyz'), panels = b.meta.panels || [], np = panels.length;
  if (xyz && np) {
    const n = xyz.length / 3, fi = b.get('dep/from_interaction');
    const pos = panels.map(() => new Float32Array(2 * n)), flag = panels.map(() => new Float32Array(n)), cnt = new Uint32Array(np);
    // flatten boxes per panel for a tight loop
    const fb = panels.map((p) => {
      const arr = [];
      for (const f of p.faces || []) for (const bx of f.boxes) arr.push(bx[0], bx[1], bx[2], bx[3], bx[4], bx[5], f.dir[0], f.dir[1]);
      return Float32Array.from(arr);
    });
    for (let i = 0; i < n; i++) {
      const x = xyz[3 * i], y = xyz[3 * i + 1], z = xyz[3 * i + 2];
      for (let p = 0; p < np; p++) {
        const B = fb[p];
        for (let k = 0; k < B.length; k += 8) {
          if (x >= B[k] && x <= B[k + 3] && y >= B[k + 1] && y <= B[k + 4] && z >= B[k + 2] && z <= B[k + 5]) {
            const j = cnt[p]++; pos[p][2 * j] = B[k + 6] * y + B[k + 7] * z; pos[p][2 * j + 1] = x; flag[p][j] = fi ? fi[i] : 1;
            break;
          }
        }
      }
    }
    set = { pos: [], flag: [], n: [] };
    for (let p = 0; p < np; p++) {
      // shuffle so that a prefix is a uniform subsample (drawn while panning/zooming)
      const m = cnt[p], P = pos[p], F = flag[p];
      let seed = 12345 + p;
      for (let i = m - 1; i > 0; i--) {
        seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
        const j = seed % (i + 1);
        const x = P[2 * i], y = P[2 * i + 1], f0 = F[i];
        P[2 * i] = P[2 * j]; P[2 * i + 1] = P[2 * j + 1]; F[i] = F[j]; P[2 * j] = x; P[2 * j + 1] = y; F[j] = f0;
      }
      const a = new Res(pos[p].slice(0, 2 * cnt[p])), f = new Res(flag[p].slice(0, cnt[p]));
      S.res.push(a, f); set.pos.push(a); set.flag.push(f); set.n.push(cnt[p]);
    }
  }
  S.sets.dep = set;
  return set;
}

// ---------------------------------------------------------------------------- views (zoom/pan)
function dataExtent(L, p) {
  const [s, e] = L.ranges[p], pos = L.pos.data;
  if (e <= s) return null;
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (let i = s; i < e; i++) {
    const x = pos[2 * i], y = pos[2 * i + 1];
    if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
  }
  return [x0, x1, y0, y1];
}

function frameViews(space) {
  const L = layout(), specs = L.specs;
  const ext = specs.map((s, p) => {
    let e = dataExtent(L, p);
    if (!e) { const hr = s.hrange || [0, 1], vr = s.vrange || [0, 1]; e = [hr[0], hr[1], vr[0], vr[1]]; }
    return e;
  });
  const pad = (lo, hi) => { const d = (hi - lo) || 1; return [lo - 0.03 * d, hi + 0.03 * d]; };
  let pl = Infinity, ph = -Infinity;
  for (const e of ext) { pl = Math.min(pl, e[0]); ph = Math.max(ph, e[1]); }
  const pooled = ph - pl;
  return specs.map((s, p) => {
    const e = ext[p];
    const own = e[1] - e[0];
    const share = space === 'wire' && specs.length > 1 && !(pooled > 0 && own < 0.35 * pooled);
    const [x0, x1] = share ? pad(pl, ph) : pad(e[0], e[1]);
    const [y0, y1] = pad(e[2], e[3]);
    const hx = (x1 - x0) / 2, hy = (y1 - y0) / 2;
    return { cx: (x0 + x1) / 2, cy: (y0 + y1) / 2, hx, hy, link: share, init: { hx, hy } };
  });
}

function ensureViews() {
  const specs = panelSpecs(V.space);
  const cur = V.views[V.space];
  if (cur && cur.length === specs.length) return cur;
  if (!specs.length) return (V.views[V.space] = []);
  return (V.views[V.space] = frameViews(V.space));
}

function resetViews() {
  if (!V.S) return;
  V.views[V.space] = frameViews(V.space);
  schedule(true, true);
}

function syncLinked(src) {
  const vs = V.views[V.space]; if (!vs || !V.linkW) return;
  let ref = src >= 0 ? vs[src] : vs.find((v) => v.link);
  if (!ref) return;
  for (const v of vs) if (v.link && v !== ref) { v.cx = ref.cx; v.hx = ref.hx; }
}

// ---------------------------------------------------------------------------- layout / geometry of the canvas
function computeRects() {
  const n = (V.views[V.space] || []).length || panelSpecs(V.space).length;
  V.rects = [];
  if (!n) return;
  const gut = V.cw >= 640 ? GUTTER : 0;
  const ncols = n <= 3 ? 1 : 2, nrows = Math.ceil(n / ncols);
  const aw = V.cw - gut, ah = V.ch - TOPSTRIP;
  const cwid = aw / ncols, chei = ah / nrows;
  for (let i = 0; i < n; i++) {
    const c = i % ncols, r = Math.floor(i / ncols);
    const x = c * cwid, y = TOPSTRIP + r * chei;
    V.rects.push({ x, y, w: cwid, h: chei,
      px: x + MARGIN.l, py: y + MARGIN.t, pw: Math.max(10, cwid - MARGIN.l - MARGIN.r), ph: Math.max(10, chei - MARGIN.t - MARGIN.b) });
  }
}

function resize() {
  if (!V.ready) return;
  const r = V.container.getBoundingClientRect();
  V.cw = Math.max(1, Math.round(r.width)); V.ch = Math.max(1, Math.round(r.height));
  V.dpr = window.devicePixelRatio || 1;
  V.W = Math.round(V.cw * V.dpr); V.H = Math.round(V.ch * V.dpr);
  for (const c of [V.glc, V.ovc]) { if (c.width !== V.W) c.width = V.W; if (c.height !== V.H) c.height = V.H; }
  computeRects();
  schedule(true, true);
}

// ---------------------------------------------------------------------------- rendering
/** Called on wheel/drag: heavy layers (deposits) are subsampled until the interaction stops. */
function interacting() {
  V.interact = true;
  clearTimeout(V.interactTimer);
  V.interactTimer = setTimeout(() => { V.interact = false; schedule(true, false); }, 160);
}

function schedule(gl, ov) {
  if (gl) V.needGL = true;
  if (ov || gl) V.needOv = true;
  if (V.raf || !V.visible) return;
  V.raf = requestAnimationFrame(() => {
    V.raf = 0;
    if (!V.visible) return;
    try { renderAll(); } catch (e) { console.error('view2d render', e); }
  });
}

function renderAll() {
  if (V.needGL) { V.needGL = false; renderGL(); }
  if (V.needOv) { V.needOv = false; renderOv(); }
}

function rgb(hex) { return hexToRgb(hex); }

function pointsUniforms(P, mode, cs, size, t) {
  P.f('uMode', mode); P.f('uSize', size);
  P.f('uCmapLo', t.cmap_floor); P.f('uCmapHi', t.cmap_ceiling);
  P.v3('uUnassoc', rgb(t.unassociated));
  const pal = new Float32Array(24);
  t.categorical.forEach((h, i) => { const c = rgb(h); pal.set(c, 3 * i); });
  V.gl.uniform3fv(P.l('uPal[0]'), pal);
  if (cs && cs.kind === 'val') { P.f('uLog', cs.log ? 1 : 0); P.f('uFloor', cs.log ? cs.lo : 0); P.f('uLo', cs.log ? Math.log(cs.lo) : cs.lo); P.f('uInv', 1 / ((cs.log ? Math.log(cs.hi) - Math.log(cs.lo) : cs.hi - cs.lo) || 1)); }
}

function zoomFactor(view) { return Math.sqrt((view.init.hx / view.hx) * (view.init.hy / view.hy)); }

function renderGL() {
  const gl = V.gl; if (!gl || !V.S || !V.W) return;
  const t0 = performance.now();
  const S = V.S, st = V.app.state, th = V.app.theme, ov = st.ov;
  const views = ensureViews(); if (V.rects.length !== views.length) computeRects();
  uploadLut();
  const L = layout(), cs = colorSpec();
  gl.disable(gl.SCISSOR_TEST); gl.viewport(0, 0, V.W, V.H);
  const bg = rgb(th.fig_bg), ax = rgb(th.axes_bg);
  gl.clearColor(bg[0], bg[1], bg[2], 1); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  gl.enable(gl.SCISSOR_TEST);
  const dpr = V.dpr, wire = V.space === 'wire';
  const sets = wire ? { tracks: ov.tracks && overlaySet('tracks'), showers: ov.showers && overlaySet('showers'), mc: ov.truth && overlaySet('mc'), dep: ov.dep && depositSet() } : {};
  // hits buffers for the current colour
  let valRes = null, catRes = null;
  if (cs.kind === 'val') {
    valRes = layoutBuf(L, `v:${cs.key}`, () => gather(valueSource(L, cs.key), L.orig));
  } else if (cs.kind === 'cat') {
    catRes = layoutBuf(L, `c:${cs.name}`, () => gather(cs.info.rank, L.orig));
  }
  for (let p = 0; p < views.length; p++) {
    const r = V.rects[p], v = views[p];
    const x = Math.round(r.px * dpr), w = Math.round(r.pw * dpr), h = Math.round(r.ph * dpr), y = V.H - Math.round((r.py + r.ph) * dpr);
    gl.viewport(x, y, w, h); gl.scissor(x, y, w, h);
    gl.clearColor(ax[0], ax[1], ax[2], 1); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    const pxPerCmX = w / (2 * v.hx), pxPerCmY = h / (2 * v.hy);
    const zf = zoomFactor(v);
    const baseSize = V.size * dpr * Math.min(2.5, Math.max(1, Math.pow(zf, 0.4)));
    // 1. deposits (faint underlay)
    const dep = sets.dep;
    if (dep && dep.n[p]) {
      gl.disable(gl.DEPTH_TEST);
      const P = V.progs.point.use();
      P.v2('uCenter', v.cx, v.cy); P.v2('uHalf', v.hx, v.hy); P.f('uAlpha', 0.35);
      P.f('uShowRad', ov.rad ? 1 : 0);
      pointsUniforms(P, 2, null, Math.max(1, 1.2 * dpr), th); P.v3('uUniform', rgb(th.truth));
      attrF(gl, 0, dep.pos[p], 2); attrOff(gl, 1, 0); attrOff(gl, 2, 0); attrF(gl, 3, dep.flag[p], 1);
      gl.drawArrays(gl.POINTS, 0, V.interact ? Math.ceil(dep.n[p] * 0.12) : dep.n[p]);
    }
    // 2. shower fill
    const sh = sets.showers;
    gl.disable(gl.DEPTH_TEST);
    if (sh && sh.fill && sh.fill.ranges[p][1]) {
      const P = V.progs.fill.use(); P.v2('uCenter', v.cx, v.cy); P.v2('uHalf', v.hx, v.hy); P.v3('uColor', rgb(th.shower)); P.f('uAlpha', 0.22);
      for (let i = 1; i < 4; i++) { gl.disableVertexAttribArray(i); gl.vertexAttribDivisor(i, 0); }
      attrF(gl, 0, sh.fill.pos, 2, sh.fill.ranges[p][0] * 8);
      gl.drawArrays(gl.TRIANGLES, 0, sh.fill.ranges[p][1]);
    }
    // 3. hits
    const [hs, he] = L.ranges[p];
    if (he > hs && cs.kind !== 'none') {
      gl.enable(gl.DEPTH_TEST);
      const P = V.progs.point.use();
      P.v2('uCenter', v.cx, v.cy); P.v2('uHalf', v.hx, v.hy); P.f('uAlpha', ALPHA); P.f('uShowRad', 1);
      const mode = cs.kind === 'cat' ? 1 : 0;
      pointsUniforms(P, mode, cs, mode === 1 ? baseSize * 1.5 : baseSize, th); P.v3('uUniform', rgb(th.unassociated));
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, V.lutTex); gl.uniform1i(P.l('uLut'), 0);
      attrF(gl, 0, L.pos, 2, hs * 8);
      if (valRes) attrF(gl, 1, valRes, 1, hs * 4); else attrOff(gl, 1, 0);
      if (catRes) attrF(gl, 2, catRes, 1, hs * 4); else attrOff(gl, 2, 0);
      attrOff(gl, 3, 1);
      gl.drawArrays(gl.POINTS, 0, he - hs);
      gl.disable(gl.DEPTH_TEST);
    }
    // 4. lines: mc (dashed), shower outline, tracks
    // lines: GL_LINES from the segment buffer (consecutive vertex pairs), widened by repeating the pass
    // with 1 device-px offsets. (Instanced screen-space quads were 5x slower under software GL.)
    const line = (set, o) => {
      if (!set || !set.ranges[p][1]) return;
      const P = V.progs.line.use();
      P.v2('uCenter', v.cx, v.cy); P.v2('uHalf', v.hx, v.hy);
      P.f('uAlpha', o.alpha); P.f('uMode', o.mode);
      P.f('uHasFlag', o.hasFlag ? 1 : 0); P.f('uShowRad', ov.rad ? 1 : 0);
      P.f('uDash', o.dash ? 1 : 0); P.f('uPxPerCm', 0.5 * (pxPerCmX + pxPerCmY)); P.f('uPeriod', 7 * dpr);
      P.v3('uColor', rgb(o.color)); P.v3('uUnassoc', rgb(th.unassociated));
      const pal = new Float32Array(24); th.categorical.forEach((hx, i) => pal.set(rgb(hx), 3 * i)); gl.uniform3fv(P.l('uPal[0]'), pal);
      if (o.mode === 2) { const pd = new Float32Array(27); PDG_CLASSES.forEach((c, i) => pd.set(rgb(c[th.name]), 3 * i)); gl.uniform3fv(P.l('uPdg[0]'), pd); }
      attrF(gl, 0, set.seg, 2, set.ranges[p][0] * 16);
      attrF(gl, 1, set.ext, 2, set.ranges[p][0] * 16);
      const n = Math.max(1, Math.round(o.width * dpr)), passes = [[0, 0], [1, 0], [0, 1], [1, 1]].slice(0, Math.min(4, n === 3 ? 3 : n));
      for (const [ox, oy] of passes) { P.v2('uOff', ox * 2 / w, oy * 2 / h); gl.drawArrays(gl.LINES, 0, 2 * set.ranges[p][1]); }
    };
    line(sets.mc, { width: 1, alpha: 0.7, color: th.mctrack, mode: st.color === 'pdg' ? 2 : 0, dash: true, hasFlag: sets.mc && sets.mc.hasFlag });
    line(sh, { width: 1, alpha: 0.8, color: th.shower, mode: 0 });
    line(sets.tracks, { width: 2, alpha: 0.9, color: th.track, mode: st.color === 'track' ? 1 : 0 });
  }
  gl.disable(gl.SCISSOR_TEST);
  V.lastRenderMs = performance.now() - t0;
}

// ---------------------------------------------------------------------------- overlay (axes etc.)
function niceTicks(lo, hi, target) {
  const span = hi - lo; if (!(span > 0)) return { ticks: [lo], dec: 0 };
  const raw = span / target, mag = Math.pow(10, Math.floor(Math.log10(raw))), nrm = raw / mag;
  const step = (nrm < 1.5 ? 1 : nrm < 3 ? 2 : nrm < 7 ? 5 : 10) * mag;
  const out = []; for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-6; v += step) out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  return { ticks: out, dec: Math.min(6, Math.max(0, -Math.floor(Math.log10(step) + 1e-9))) };
}

function fmt(v) { return Math.abs(v) >= 1000 || (Math.abs(v) < 0.01 && v !== 0) ? v.toExponential(1) : v.toPrecision(3).replace(/\.?0+$/, ''); }

function toPx(p, hx, hy) {
  const r = V.rects[p], v = V.views[V.space][p];
  return [r.px + (hx - v.cx) / (2 * v.hx) * r.pw + r.pw / 2, r.py + r.ph / 2 - (hy - v.cy) / (2 * v.hy) * r.ph];
}
function toData(p, px, py) {
  const r = V.rects[p], v = V.views[V.space][p];
  return [v.cx + (px - r.px - r.pw / 2) / r.pw * 2 * v.hx, v.cy - (py - r.py - r.ph / 2) / r.ph * 2 * v.hy];
}

function renderOv() {
  const g = V.ov; if (!g) return;
  const th = V.app.theme, st = V.app.state, S = V.S;
  g.setTransform(V.dpr, 0, 0, V.dpr, 0, 0);
  g.clearRect(0, 0, V.cw, V.ch);
  const views = V.views[V.space] || [], specs = S ? panelSpecs(V.space) : [];
  if (!S) { g.fillStyle = th.fg_muted; g.font = '12px system-ui, sans-serif'; g.fillText('2D: no event loaded', 12, 44); return; }
  if (V.rects.length !== views.length) computeRects();
  const wire = V.space === 'wire', ov = st.ov;
  const roles = S.bundle.get('vtx/role');
  g.textBaseline = 'alphabetic';
  for (let p = 0; p < views.length; p++) {
    const r = V.rects[p], v = views[p], sp = specs[p] || {};
    g.save();
    g.beginPath(); g.rect(r.px, r.py, r.pw, r.ph); g.clip();
    // reference lines
    if (wire) {
      g.strokeStyle = th.fg_muted; g.lineWidth = 1;
      g.globalAlpha = 0.8; g.setLineDash([6, 4]);
      for (const xa of sp.anodes || []) { const y = toPx(p, 0, xa)[1]; g.beginPath(); g.moveTo(r.px, y); g.lineTo(r.px + r.pw, y); g.stroke(); }
      g.globalAlpha = 0.6; g.setLineDash([2, 3]);
      for (const xc of sp.edges || []) { const y = toPx(p, 0, xc)[1]; g.beginPath(); g.moveTo(r.px, y); g.lineTo(r.px + r.pw, y); g.stroke(); }
      g.setLineDash([]); g.globalAlpha = 1;
      // vertices
      if (ov.vtx && S.bundle.get('vtx2d/hv')) {
        const hv = S.bundle.get('vtx2d/hv'), pa = S.bundle.get('vtx2d/panel'), ix = S.bundle.get('vtx2d/index');
        const by = { x: [], D: [], o: [] };
        for (let k = 0; k < pa.length; k++) {
          if (pa[k] !== p) continue;
          const [x, y] = toPx(p, hv[2 * k], hv[2 * k + 1]);
          if (x < r.px - 10 || x > r.px + r.pw + 10 || y < r.py - 10 || y > r.py + r.ph + 10) continue;
          const role = roles && ix ? roles[ix[k]] : 3;
          by[role === 1 ? 'x' : role === 2 ? 'D' : 'o'].push(x, y);
        }
        for (const kind of ['o', 'D', 'x']) drawMarkers(g, kind, by[kind], th.vertex, th.axes_bg);
      }
      // true vertex
      const tv = S.bundle.meta.truth && S.bundle.meta.truth.vertex2d && S.bundle.meta.truth.vertex2d[p];
      if (ov.truth && tv) { const [x, y] = toPx(p, tv[0], tv[1]); drawMarkers(g, 'X', [x, y], th.mctrack, th.axes_bg); }
    }
    // hover ring
    if (V.hover && V.hover.p === p) {
      const [x, y] = toPx(p, V.hover.h, V.hover.v);
      g.strokeStyle = th.fg; g.lineWidth = 1.5; g.beginPath(); g.arc(x, y, 6, 0, 2 * Math.PI); g.stroke();
    }
    g.restore();
    // labels (not clipped hard: boxes may overhang)
    if (wire && ov.labels) drawLabels(g, p, r, th);
    // frame + ticks
    g.strokeStyle = th.fg_muted; g.lineWidth = 1; g.strokeRect(r.px + 0.5, r.py + 0.5, r.pw - 1, r.ph - 1);
    g.fillStyle = th.fg; g.strokeStyle = th.fg_muted; g.font = '10px system-ui, sans-serif';
    const xt = niceTicks(v.cx - v.hx, v.cx + v.hx, Math.max(3, Math.round(r.pw / 80)));
    g.textAlign = 'center'; g.textBaseline = 'top';
    for (const t of xt.ticks) {
      const x = toPx(p, t, 0)[0]; if (x < r.px - 1 || x > r.px + r.pw + 1) continue;
      g.beginPath(); g.moveTo(x, r.py + r.ph); g.lineTo(x, r.py + r.ph + 4); g.stroke();
      g.fillText(t.toFixed(xt.dec), x, r.py + r.ph + 5);
    }
    const yt = niceTicks(v.cy - v.hy, v.cy + v.hy, Math.max(3, Math.round(r.ph / 36)));
    g.textAlign = 'right'; g.textBaseline = 'middle';
    for (const t of yt.ticks) {
      const y = toPx(p, 0, t)[1]; if (y < r.py - 1 || y > r.py + r.ph + 1) continue;
      g.beginPath(); g.moveTo(r.px - 4, y); g.lineTo(r.px, y); g.stroke();
      g.fillText(t.toFixed(yt.dec), r.px - 6, y);
    }
    g.fillStyle = th.fg_muted; g.textBaseline = 'alphabetic';
    g.textAlign = 'center'; g.fillText(sp.hlabel || '', r.px + r.pw / 2, r.y + r.h - 4);
    g.save(); g.translate(r.x + 11, r.py + r.ph / 2); g.rotate(-Math.PI / 2); g.fillText(sp.vlabel || '', 0, 0); g.restore();
    g.fillStyle = th.fg; g.font = '11px system-ui, sans-serif'; g.textAlign = 'left';
    g.fillText(`${sp.title || ''}   (${(L_count(p)).toLocaleString()} hits)`, r.px, r.py - 6);
    if (v.link && V.linkW && wire && views.length > 1) { g.fillStyle = th.fg_muted; g.font = '9px system-ui, sans-serif'; g.textAlign = 'right'; g.fillText('w linked', r.px + r.pw, r.py - 6); }
  }
}

function L_count(p) { const L = layout(); const [s, e] = L.ranges[p] || [0, 0]; return e - s; }

function markerPath(g, kind, xy) {
  g.beginPath();
  for (let i = 0; i < xy.length; i += 2) {
    const x = xy[i], y = xy[i + 1];
    if (kind === 'x' || kind === 'X') { g.moveTo(x - 5, y - 5); g.lineTo(x + 5, y + 5); g.moveTo(x - 5, y + 5); g.lineTo(x + 5, y - 5); }
    else if (kind === 'D') { g.moveTo(x, y - 5); g.lineTo(x + 5, y); g.lineTo(x, y + 5); g.lineTo(x - 5, y); g.closePath(); }
    else { g.moveTo(x + 3.5, y); g.arc(x, y, 3.5, 0, 2 * Math.PI); }
  }
}

/** All markers of one kind in a single path: one casing stroke + one fill/stroke (Canvas2D is slow per-call). */
function drawMarkers(g, kind, xy, col, edge) {
  if (!xy.length) return;
  g.lineJoin = 'miter'; g.lineCap = 'round';
  markerPath(g, kind, xy);
  if (kind === 'x' || kind === 'X') {
    g.strokeStyle = edge; g.lineWidth = kind === 'x' ? 4 : 5; g.stroke();
    g.strokeStyle = col; g.lineWidth = kind === 'x' ? 1.8 : 2.8; g.stroke();
  } else {
    g.fillStyle = col; g.strokeStyle = edge; g.lineWidth = 1; g.fill(); g.stroke();
  }
}

function labelProjection(p) {
  const S = V.S;
  if (!S.labelPts) S.labelPts = {};
  if (S.labelPts[p]) return S.labelPts[p];
  const b = S.bundle, pos = b.get('labels/pos'), tx = b.get('labels/text'), texts = b.meta.label_texts || [];
  const out = [];
  const panel = (b.meta.panels || [])[p];
  if (pos && panel) {
    for (let k = 0; k < pos.length / 3; k++) {
      const pr = projectPoint(panel, pos[3 * k], pos[3 * k + 1], pos[3 * k + 2]);
      if (pr) out.push({ h: pr[0], v: pr[1], text: texts[tx[k]] || '' });
    }
  }
  return (S.labelPts[p] = out);
}

function drawLabels(g, p, r, th) {
  g.save(); g.beginPath(); g.rect(r.px, r.py, r.pw, r.ph); g.clip();
  g.font = 'bold 10px system-ui, sans-serif'; g.textAlign = 'left'; g.textBaseline = 'middle';
  for (const l of labelProjection(p)) {
    const [x, y] = toPx(p, l.h, l.v);
    if (x < r.px || x > r.px + r.pw || y < r.py || y > r.py + r.ph) continue;
    g.fillStyle = '#38bdf8'; g.strokeStyle = th.axes_bg; g.lineWidth = 1;
    g.beginPath(); g.arc(x, y, 4, 0, 2 * Math.PI); g.fill(); g.stroke();
    const w = g.measureText(l.text).width;
    g.fillStyle = th.legend_bg; g.globalAlpha = 0.85; g.strokeStyle = th.legend_edge;
    g.fillRect(x + 6, y - 8, w + 6, 16); g.strokeRect(x + 6.5, y - 7.5, w + 6, 16);
    g.globalAlpha = 1; g.fillStyle = th.fg; g.fillText(l.text, x + 9, y);
  }
  g.restore();
}

// ---------------------------------------------------------------------------- legend
function updateLegend() {
  const st = V.app.state, th = V.app.theme;
  let html = '';
  if (V.S) {
    const cs = colorSpec();
    if (cs.kind === 'val') {
      const lo = cs.log ? Math.log(cs.lo) : cs.lo, hi = cs.log ? Math.log(cs.hi) : cs.hi;
      const tk = [0, 0.5, 1].map((f) => fmt(cs.log ? Math.exp(lo + f * (hi - lo)) : lo + f * (hi - lo)));
      html += `<div>${cs.name}${cs.log ? ' (log)' : ''}</div><div class="bar" style="background:${cmapGradientCss(st.cmap, th.cmap_floor, th.cmap_ceiling)}"></div>` +
        `<div class="ticks"><span>${tk[0]}</span><span>${tk[1]}</span><span>${tk[2]}</span></div>`;
      if (cs.note) html += `<div style="color:var(--muted)">${cs.note}</div>`;
    } else if (cs.kind === 'cat') {
      html += `<div>colour x marker: ${cs.name} id</div><div style="color:var(--muted)">${cs.info.nObj} objects, ${cs.info.nUn.toLocaleString()} unassociated (grey)</div>`;
      const sw = th.categorical.slice(0, 8).map((c) => `<span class="sw" style="background:${c}"></span>`).join('');
      html += `<div>${sw}</div><div style="color:var(--muted)">o ^ s D v cycle</div>`;
    } else html += '<div>no hits</div>';
    if (V.space === 'wire') {
      const ov = st.ov, b = V.S.bundle, av = (n) => b.has(n);
      const line = (c, name, dash) => `<div><span class="ln" style="border-color:${c};${dash ? 'border-top-style:dashed' : ''}"></span>${name}</div>`;
      if (ov.tracks && av('tracks2d/points')) html += line(th.track, 'reco tracks');
      if (ov.showers && av('showers2d/points')) html += line(th.shower, 'shower cones');
      if (ov.truth && av('mc2d/points')) html += line(th.mctrack, 'true trajectories', true);
      if (ov.dep && av('dep/xyz')) html += `<div><span class="sw" style="background:${th.truth}"></span>true deposits</div>`;
      if (ov.vtx && av('vtx2d/hv')) html += `<div><span class="sw" style="background:${th.vertex}"></span>vertices (x D o)</div>`;
    }
  }
  if (html !== V.lastLegend) { V.legend.innerHTML = html; V.lastLegend = html; }
}

// ---------------------------------------------------------------------------- pointer interaction
function panelAt(px, py) {
  for (let i = 0; i < V.rects.length; i++) {
    const r = V.rects[i];
    if (px >= r.x && px < r.x + r.w && py >= r.y && py < r.y + r.h) return i;
  }
  return -1;
}

function region(p, px, py) {
  const r = V.rects[p];
  const inX = px >= r.px && px <= r.px + r.pw, inY = py >= r.py && py <= r.py + r.ph;
  if (inX && inY) return 'plot';
  if (inX && py > r.py + r.ph) return 'xaxis';
  if (inY && px < r.px) return 'yaxis';
  return 'none';
}

function zoomAt(p, px, py, fx, fy) {
  const vs = V.views[V.space], v = vs[p];
  const [dx, dy] = toData(p, px, py);
  const lim = (view, f, h) => Math.min(view.init[h] * 30, Math.max(view.init[h] / 2000, f));
  if (fx !== 1) {
    const ap = vs.filter((q, i) => i === p || (V.linkW && v.link && q.link));
    for (const q of ap) { const nh = lim(q, q.hx / fx, 'hx'); const f = q.hx / nh; q.cx = dx - (dx - q.cx) / f; q.hx = nh; }
  }
  if (fy !== 1) { const nh = lim(v, v.hy / fy, 'hy'); const f = v.hy / nh; v.cy = dy - (dy - v.cy) / f; v.hy = nh; }
  syncLinked(p);
}

function bindPointer() {
  const c = V.ovc;
  const pos = (e) => { const r = c.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
  c.addEventListener('wheel', (e) => {
    if (!V.S) return;
    e.preventDefault();
    const [px, py] = pos(e), p = panelAt(px, py); if (p < 0) return;
    const reg = region(p, px, py); if (reg === 'none') return;
    const dy = e.deltaMode === 1 ? e.deltaY * 33 : e.deltaY;
    const f = Math.exp(-Math.max(-300, Math.min(300, dy)) * 0.0018);
    let fx = f, fy = f;
    if (e.shiftKey || reg === 'xaxis') fy = 1;
    if (e.altKey || reg === 'yaxis') fx = 1;
    interacting(); zoomAt(p, px, py, fx, fy);
    V.hover = null; hideTip(); schedule(true, true);
  }, { passive: false });
  c.addEventListener('pointerdown', (e) => {
    if (!V.S || e.button !== 0) return;
    const [px, py] = pos(e), p = panelAt(px, py); if (p < 0 || region(p, px, py) === 'none') return;
    V.drag = { p, px, py }; c.setPointerCapture(e.pointerId); c.style.cursor = 'grabbing'; hideTip();
  });
  c.addEventListener('pointermove', (e) => {
    if (!V.S) return;
    const [px, py] = pos(e);
    if (V.drag) {
      const d = V.drag, r = V.rects[d.p], vs = V.views[V.space], v = vs[d.p];
      const dx = px - d.px, dy = py - d.py; d.px = px; d.py = py;
      v.cx -= dx / r.pw * 2 * v.hx; v.cy += dy / r.ph * 2 * v.hy;
      interacting(); syncLinked(d.p);
      V.hover = null; schedule(true, true);
      return;
    }
    updateHover(px, py, e);
  });
  const end = (e) => { if (V.drag) { V.drag = null; c.style.cursor = 'crosshair'; try { c.releasePointerCapture(e.pointerId); } catch (x) { /* none */ } } };
  c.addEventListener('pointerup', end); c.addEventListener('pointercancel', end);
  c.addEventListener('pointerleave', () => { if (V.hover) { V.hover = null; schedule(false, true); } hideTip(); });
  c.addEventListener('dblclick', () => resetViews());
}

function hideTip() { V.tip.style.display = 'none'; }

// ---- spatial grid for hover --------------------------------------------------------------------------
function gridFor(L, p) {
  let G = L.grids[p]; if (G) return G;
  const [s, e] = L.ranges[p], n = e - s, pos = L.pos.data;
  if (n <= 0) return (L.grids[p] = { n: 0 });
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (let i = s; i < e; i++) { const x = pos[2 * i], y = pos[2 * i + 1]; if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; }
  const gd = Math.max(1, Math.min(256, Math.round(Math.sqrt(n / 3))));
  const cw = ((x1 - x0) || 1) / gd * 1.000001, ch = ((y1 - y0) || 1) / gd * 1.000001;
  const start = new Uint32Array(gd * gd + 1), cell = new Uint32Array(n);
  for (let i = 0; i < n; i++) {
    const cx = Math.floor((pos[2 * (s + i)] - x0) / cw), cy = Math.floor((pos[2 * (s + i) + 1] - y0) / ch);
    cell[i] = cy * gd + cx; start[cell[i] + 1]++;
  }
  for (let i = 0; i < gd * gd; i++) start[i + 1] += start[i];
  const cur = start.slice(0, gd * gd), items = new Uint32Array(n);
  for (let i = 0; i < n; i++) items[cur[cell[i]]++] = s + i;
  return (L.grids[p] = { n, x0, y0, cw, ch, gd, start, items });
}

function nearestHit(L, p, hx, hy, rx, ry) {
  const G = gridFor(L, p); if (!G.n) return -1;
  const pos = L.pos.data;
  const c0 = Math.max(0, Math.floor((hx - rx - G.x0) / G.cw)), c1 = Math.min(G.gd - 1, Math.floor((hx + rx - G.x0) / G.cw));
  const r0 = Math.max(0, Math.floor((hy - ry - G.y0) / G.ch)), r1 = Math.min(G.gd - 1, Math.floor((hy + ry - G.y0) / G.ch));
  let best = -1, bd = 1;
  for (let r = r0; r <= r1; r++) {
    for (let c = c0; c <= c1; c++) {
      const k = r * G.gd + c;
      for (let j = G.start[k]; j < G.start[k + 1]; j++) {
        const i = G.items[j], dx = (pos[2 * i] - hx) / rx, dy = (pos[2 * i + 1] - hy) / ry, d = dx * dx + dy * dy;
        if (d < bd) { bd = d; best = i; }
      }
    }
  }
  return best;
}

function updateHover(px, py, e) {
  const p = panelAt(px, py);
  if (p < 0 || region(p, px, py) !== 'plot') { if (V.hover) { V.hover = null; schedule(false, true); } hideTip(); return; }
  const L = layout(), v = V.views[V.space][p], r = V.rects[p];
  const [hx, hy] = toData(p, px, py);
  const rad = 7;   // px
  const i = nearestHit(L, p, hx, hy, rad / r.pw * 2 * v.hx, rad / r.ph * 2 * v.hy);
  if (i < 0) { if (V.hover) { V.hover = null; schedule(false, true); } hideTip(); return; }
  const o = L.orig[i], b = V.S.bundle, wire = V.space === 'wire';
  const at = (n) => { const a = b.get(n); return a ? a[o] : null; };
  const hh = L.pos.data[2 * i], vv = L.pos.data[2 * i + 1];
  V.hover = { p, h: hh, v: vv };
  const lines = [];
  const ch = at('hits/channel'), tick = at('hits/tick');
  if (ch != null) lines.push(`channel   ${ch}`);
  if (tick != null) lines.push(`tick      ${tick.toFixed(1)}`);
  for (const n of ['charge', 'amplitude']) { const x = at(`hits/${n}`); if (x != null) lines.push(`${n.padEnd(10)}${fmt(x)}`); }
  const mu = at('hits/multiplicity'); if (mu != null) lines.push(`mult      ${mu}`);
  lines.push(wire ? `w ${hh.toFixed(1)} cm   x ${vv.toFixed(1)} cm` : `channel ${hh.toFixed(0)}  tick ${vv.toFixed(1)}`);
  const objs = CAT_KINDS.map((k) => { const a = at(`hits/group/${k}`); return a != null && a >= 0 ? `${k} ${a}` : null; }).filter(Boolean);
  lines.push(objs.length ? objs.join('  ') : 'no reco object');
  V.tip.textContent = lines.join('\n');
  V.tip.style.display = 'block';
  const cr = V.container.getBoundingClientRect();
  let tx = px + 14, ty = py + 14;
  const tw = V.tip.offsetWidth, th = V.tip.offsetHeight;
  if (tx + tw > cr.width - 4) tx = px - tw - 10;
  if (ty + th > cr.height - 4) ty = py - th - 10;
  V.tip.style.left = `${Math.max(2, tx)}px`; V.tip.style.top = `${Math.max(2, ty)}px`;
  schedule(false, true);
}

// ---------------------------------------------------------------------------- module interface
function freeEvent() {
  if (V.S) for (const r of V.S.res) r.free();
  V.S = null; V.hover = null; hideTip();
}

function setGeometry() { V.views = {}; }

function setEvent(bundle) {
  if (!V.ready) return;
  freeEvent();
  if (bundle) {
    V.S = { bundle, res: [], layouts: {}, stats: new Map(), cats: new Map(), sets: {}, labelPts: null };
    V.bar.sel.value = V.space;
    const t0 = performance.now();
    ensureViews();
    V.lastBuildMs = performance.now() - t0;
  }
  computeRects();
  updateLegend();
  schedule(true, true);
}

function onState(state, changed) {
  if (!V.ready) return;
  if (changed.has('space2d')) {
    const s = state.space2d === 'readout' ? 'readout' : 'wire';
    if (s !== V.space) { V.space = s; V.hover = null; hideTip(); if (V.S) { ensureViews(); computeRects(); } }
    V.bar.sel.value = V.space;
  }
  if (['color', 'scale', 'cmap', 'theme', 'ov', 'space2d'].some((k) => changed.has(k))) { if (V.S) updateLegend(); schedule(true, true); }
}

function setVisible(vis) {
  V.visible = !!vis;
  if (vis) { resize(); }
}

function getCanvas() {
  if (!V.ready || !V.S || !V.W) return null;
  V.needGL = false; V.needOv = false;
  renderGL(); renderOv();
  if (!V.shot) V.shot = document.createElement('canvas');
  V.shot.width = V.W; V.shot.height = V.H;
  const g = V.shot.getContext('2d');
  g.drawImage(V.glc, 0, 0);
  g.drawImage(V.ovc, 0, 0);
  // legend text is DOM: draw a compact copy
  return V.shot;
}

export default {
  name: 'view2d',
  init, setGeometry, setEvent, onState, setVisible, resize, getCanvas,
  resetView: resetViews,
  /** test hook: synchronous render with GPU sync; returns ms */
  debugRender() { const t = performance.now(); renderGL(); V.gl.finish(); return performance.now() - t; },
  debugHitPx(p, k) { const L = layout(), i = L.ranges[p][0] + k; return toPx(p, L.pos.data[2 * i], L.pos.data[2 * i + 1]); },
  debugState() { return { views: V.views, rects: V.rects, space: V.space, lastRenderMs: V.lastRenderMs }; },
};
