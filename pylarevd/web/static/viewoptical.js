// viewoptical.js -- Optical (PDS) view: OpHits and flashes against time.
//
// Port of display.OpticalDisplay. Two panels share one time axis:
//   (a) OpHits: time vs photon-detector z (tile position from the geometry via oph/tile;
//       the channel index when no hit has a position), colour = PE on the FIXED log range
//       PE_SIZE_RANGE (1..3e5), brighter hits drawn on top;
//   (b) flashes: time vs flash z, marker size ~ log PE (fixed range), colour = flash y over the
//       TPC y extent; the true vertex z is a dashed line when the event has a true vertex.
// Default window = meta.optical.t0 +- 50 us (the _light_centre rule: onset of the brightest
// OpHit burst, else the brightest flash); wheel zooms the time axis about the cursor, drag pans,
// double click (or the "reset" button) restores the window, "whole readout" shows everything.
//
// Rendering: ONE WebGL2 canvas (a viewport + scissor per panel, one gl.POINTS draw per panel)
// under a transparent Canvas2D that draws axes, colour bars, the vertex line and the hover
// ring, and receives the pointer events. Times are uploaded once per event as float32
// relative to t0, so pan/zoom only changes two uniforms (no buffer traffic).
// Data: oph/{time,pe,channel,tile,flash}, flash/{time,pe,y,z,yw,zw,x}, meta.optical, pd/xyz,
// meta.tpcs, meta.truth.vertex.
//
// Params (registerParam): owin = half-width of the default window in us, or 'all'.
import { getLut, hexToRgb } from './colormaps.js';

const PE_RANGE = [1.0, 3e5];        // display.PE_SIZE_RANGE
const WINDOWS = ['1', '5', '20', '50', '200', '1000', 'all'];
const M = { l: 66, r: 92, t: 20, gap: 56, b: 40 };

const CSS = `
.vopt-head { position:absolute; left:0; right:0; top:0; z-index:6; padding:3px 8px; font-size:12px; color:var(--muted);
  background:color-mix(in srgb, var(--axes-bg) 80%, transparent); }
.vopt-head .row { display:flex; gap:10px; align-items:center; flex-wrap:wrap; }
.vopt-head select, .vopt-head button { padding:0 5px; font-size:12px; }
.vopt-head .note { margin-top:2px; white-space:pre-wrap; line-height:1.3; max-height:5.2em; overflow:hidden; }
.vopt-head .warn { color:#f59e0b; }
.vopt-stage { position:absolute; left:0; right:0; bottom:0; }
.vopt-stage canvas { position:absolute; left:0; top:0; width:100%; height:100%; display:block; }
.vopt-ov { touch-action:none; cursor:crosshair; }
.vopt-tip { position:absolute; z-index:7; pointer-events:none; display:none; white-space:pre;
  font:11px/1.35 ui-monospace,Menlo,monospace; background:color-mix(in srgb, var(--legend-bg) 92%, transparent);
  border:1px solid var(--legend-edge); border-radius:4px; padding:3px 6px; color:var(--fg); }
.vopt-msg { position:absolute; inset:0; display:flex; align-items:center; justify-content:center; color:var(--muted); z-index:5; padding:20px; text-align:center; }
`;

const VS = `#version 300 es
precision highp float;
in float aT; in float aY; in float aV; in float aS;
uniform vec2 uX, uY;           // visible range (time rel. to t0, y)
uniform float uCLog, uCLo, uCHi, uCmapLo, uCmapHi, uSizeMode, uSize, uPx;
uniform sampler2D uLut;
out vec3 vColor;
void main() {
  float v = uCLog > 0.5 ? log(max(aV, 1e-3)) * 0.4342944819 : aV;
  float t = clamp((v - uCLo) / (uCHi - uCLo), 0.0, 1.0);
  vColor = texture(uLut, vec2(mix(uCmapLo, uCmapHi, t), 0.5)).rgb;
  float f = clamp(log(max(aS, 1.0)) * 0.4342944819 / 5.4771212547, 0.0, 1.0);   // log10(3e5) = 5.477
  gl_PointSize = (uSizeMode > 0.5 ? 5.0 + 14.0 * f : uSize) * uPx;
  gl_Position = vec4(2.0 * (aT - uX.x) / (uX.y - uX.x) - 1.0, 2.0 * (aY - uY.x) / (uY.y - uY.x) - 1.0, 0.0, 1.0);
}`;
const FS = `#version 300 es
precision mediump float;
in vec3 vColor;
uniform float uAlpha;
out vec4 o;
void main() {
  vec2 c = gl_PointCoord - 0.5;
  if (dot(c, c) > 0.25) discard;
  o = vec4(vColor, uAlpha);
}`;

const O = {
  app: null, container: null, ready: false, visible: false,
  head: null, bar: null, note: null, stage: null, glc: null, ovc: null, ov: null, tip: null, msg: null, sel: null, span: null,
  gl: null, prog: null, loc: {}, lutTex: null, lutName: '', dpr: 1, W: 0, H: 0,
  E: null,                         // per-event data (null = nothing to draw)
  x0: 0, x1: 1,                    // visible time range, relative to E.tref
  rects: [], hover: null, drag: null, raf: 0, lastMs: 0,
};

// ----------------------------------------------------------------------------- helpers
function niceTicks(lo, hi, target = 7) {
  const span = hi - lo;
  if (!(span > 0)) return { ticks: [lo], step: 1 };
  const raw = span / target, mag = 10 ** Math.floor(Math.log10(raw));
  let step = 10 * mag;
  for (const m of [1, 2, 5, 10]) if (m * mag >= raw) { step = m * mag; break; }
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 0.5 * step; v += step) out.push(v);
  return { ticks: out, step };
}
const fmtTick = (v, step) => (Math.abs(v) < 1e-9 * step ? 0 : v).toFixed(Math.max(0, Math.min(6, Math.ceil(-Math.log10(step) - 1e-9))));
function fmtPe(pe) { return pe >= 1e4 ? `${(pe / 1e3).toFixed(0)}k` : pe >= 1e3 ? `${(pe / 1e3).toFixed(1)}k` : pe.toFixed(pe < 10 ? 1 : 0); }
const esc = (s) => String(s).replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));

// ----------------------------------------------------------------------------- init
function init(container, app) {
  O.app = app; O.container = container;
  app.registerParam('owin', '50');
  const st = document.createElement('style'); st.textContent = CSS; document.head.appendChild(st);

  O.head = document.createElement('div'); O.head.className = 'vopt-head';
  O.bar = document.createElement('div'); O.bar.className = 'row';
  O.bar.innerHTML = `<label>window &plusmn;<select class="vopt-win">${WINDOWS.map((w) =>
    `<option value="${w}">${w === 'all' ? 'whole readout' : w + ' µs'}</option>`).join('')}</select></label>` +
    '<button class="vopt-reset" title="back to the default window (double click)">reset</button>' +
    '<span class="vopt-span"></span>' +
    '<span style="margin-left:auto" title="wheel: zoom time, drag: pan, double click: reset">wheel zoom &middot; drag pan &middot; dbl-click reset</span>';
  O.note = document.createElement('div'); O.note.className = 'note';
  O.head.append(O.bar, O.note);
  O.sel = O.bar.querySelector('.vopt-win'); O.span = O.bar.querySelector('.vopt-span');
  O.sel.value = app.state.owin;
  O.sel.addEventListener('change', () => { app.setState({ owin: O.sel.value }); });
  O.bar.querySelector('.vopt-reset').addEventListener('click', () => resetWindow());

  O.stage = document.createElement('div'); O.stage.className = 'vopt-stage';
  O.glc = document.createElement('canvas'); O.ovc = document.createElement('canvas'); O.ovc.className = 'vopt-ov';
  O.stage.append(O.glc, O.ovc);
  O.tip = document.createElement('div'); O.tip.className = 'vopt-tip';
  O.msg = document.createElement('div'); O.msg.className = 'vopt-msg'; O.msg.style.display = 'none';
  container.append(O.stage, O.head, O.tip, O.msg);
  O.ov = O.ovc.getContext('2d');

  const gl = O.gl = O.glc.getContext('webgl2', { antialias: false, alpha: false, preserveDrawingBuffer: false });
  if (!gl) { showMsg('WebGL2 is not available in this browser.'); return; }
  O.glc.addEventListener('webglcontextlost', (e) => e.preventDefault());
  O.glc.addEventListener('webglcontextrestored', () => { buildGL(); uploadEvent(); schedule(); });
  buildGL();

  O.ovc.addEventListener('wheel', onWheel, { passive: false });
  O.ovc.addEventListener('pointerdown', onDown);
  O.ovc.addEventListener('pointermove', onMove);
  O.ovc.addEventListener('pointerup', onUp);
  O.ovc.addEventListener('pointerleave', () => { O.hover = null; O.tip.style.display = 'none'; schedule(); });
  O.ovc.addEventListener('dblclick', () => resetWindow());
  new ResizeObserver(() => resize()).observe(container);
  O.ready = true;
}

function showMsg(t) { O.msg.textContent = t; O.msg.style.display = t ? 'flex' : 'none'; }

function buildGL() {
  const gl = O.gl;
  const mk = (type, src) => { const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s)); return s; };
  const p = gl.createProgram();
  gl.attachShader(p, mk(gl.VERTEX_SHADER, VS)); gl.attachShader(p, mk(gl.FRAGMENT_SHADER, FS));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
  O.prog = p; O.loc = {};
  for (const n of ['aT', 'aY', 'aV', 'aS']) O.loc[n] = gl.getAttribLocation(p, n);
  for (const n of ['uX', 'uY', 'uCLog', 'uCLo', 'uCHi', 'uCmapLo', 'uCmapHi', 'uSizeMode', 'uSize', 'uPx', 'uLut', 'uAlpha']) O.loc[n] = gl.getUniformLocation(p, n);
  O.lutTex = gl.createTexture(); O.lutName = '';
  gl.bindTexture(gl.TEXTURE_2D, O.lutTex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
}

function uploadLut() {
  const name = O.app.state.cmap;
  if (O.lutName === name) return;
  const lut = getLut(name), d = new Uint8Array(256 * 4);
  for (let i = 0; i < 256; i++) { d[4 * i] = lut[3 * i]; d[4 * i + 1] = lut[3 * i + 1]; d[4 * i + 2] = lut[3 * i + 2]; d[4 * i + 3] = 255; }
  const gl = O.gl;
  gl.bindTexture(gl.TEXTURE_2D, O.lutTex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 256, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, d);
  O.lutName = name;
}

// ----------------------------------------------------------------------------- event data
function setGeometry() { O.geomKey = null; }

function setEvent(bundle, geometry) {
  O.E = null; O.hover = null;
  if (!bundle) { schedule(); return; }
  const meta = bundle.meta, opm = meta.optical || {};
  const time = bundle.get('oph/time'), pe = bundle.get('oph/pe');
  const ftime = bundle.get('flash/time');
  const nh = time ? time.length : 0, nf = ftime ? ftime.length : 0;
  if (!nh && !nf) {
    O.E = null;
    showMsg('No optical data in this event' + ((meta.notes || []).length ? ': ' + meta.notes.filter((n) => /optical|flash|ophit/i.test(n)).join('; ') : ''));
    O.note.textContent = ''; schedule(); return;
  }
  showMsg('');
  const E = { nh, nf, bundle, meta, opm, geometry };
  // time reference: t0 from the exporter (onset of the brightest OpHit burst), else brightest flash
  let tref = Number.isFinite(opm.t0) ? opm.t0 : null;
  const fpe = bundle.get('flash/pe');
  if (tref == null && nf) { let b = 0; for (let i = 1; i < nf; i++) if (fpe[i] > fpe[b]) b = i; tref = ftime[b]; }
  if (tref == null && nh) { let b = 0; for (let i = 1; i < nh; i++) if (pe[i] > pe[b]) b = i; tref = time[b]; }
  E.tref = tref || 0;

  // hit positions: z of the tile each hit sits on (NaN where unknown)
  const tile = bundle.get('oph/tile'), chan = bundle.get('oph/channel');
  const pdxyz = geometry && geometry.get('pd/xyz');
  const hz = new Float32Array(nh);
  let positioned = false;
  if (tile && pdxyz) {
    const nt = pdxyz.length / 3;
    for (let i = 0; i < nh; i++) {
      const k = tile[i];
      if (k >= 0 && k < nt) { hz[i] = pdxyz[3 * k + 2]; positioned = true; } else hz[i] = NaN;
    }
  }
  if (!positioned) for (let i = 0; i < nh; i++) hz[i] = chan ? chan[i] : i;
  E.positioned = positioned;
  // draw order: faint first, bright on top
  const order = new Uint32Array(nh);
  for (let i = 0; i < nh; i++) order[i] = i;
  order.sort((a, b) => pe[a] - pe[b]);
  E.order = order;
  E.hT = new Float32Array(nh); E.hY = new Float32Array(nh); E.hP = new Float32Array(nh);
  for (let j = 0; j < nh; j++) {
    const i = order[j];
    E.hT[j] = time[i] - E.tref; E.hY[j] = hz[i]; E.hP[j] = pe[i];
  }
  E.hz = hz;

  // flashes
  const fy = bundle.get('flash/y'), fz = bundle.get('flash/z');
  E.fsel = [];
  for (let i = 0; i < nf; i++) if (fy && fz && Number.isFinite(fy[i]) && Number.isFinite(fz[i])) E.fsel.push(i);
  // brightest last so it sits on top
  E.fsel.sort((a, b) => fpe[a] - fpe[b]);
  const nfs = E.fsel.length;
  E.fT = new Float32Array(nfs); E.fZ = new Float32Array(nfs); E.fY = new Float32Array(nfs); E.fP = new Float32Array(nfs);
  E.fsel.forEach((i, j) => { E.fT[j] = ftime[i] - E.tref; E.fZ[j] = fz[i]; E.fY[j] = fy[i]; E.fP[j] = fpe[i]; });

  // ranges
  const tpcs = meta.tpcs && meta.tpcs.length ? meta.tpcs : (geometry && geometry.meta.tpcs) || [];
  E.yRange = tpcs.length ? [Math.min(...tpcs.map((b) => b[1])), Math.max(...tpcs.map((b) => b[4]))] : [-1, 1];
  let zlo = Infinity, zhi = -Infinity;
  const acc = (v) => { if (Number.isFinite(v)) { if (v < zlo) zlo = v; if (v > zhi) zhi = v; } };
  for (let i = 0; i < nh; i++) acc(hz[i]);
  if (positioned) for (let j = 0; j < nfs; j++) acc(E.fZ[j]);
  if (positioned) for (const b of tpcs) { acc(b[2]); acc(b[5]); }
  if (!(zhi > zlo)) { zlo = 0; zhi = 1; }
  const pad = 0.04 * (zhi - zlo);
  E.hRange = [zlo - pad, zhi + pad];
  let flo = Infinity, fhi = -Infinity;
  for (let j = 0; j < nfs; j++) { flo = Math.min(flo, E.fZ[j]); fhi = Math.max(fhi, E.fZ[j]); }
  if (positioned) for (const b of tpcs) { flo = Math.min(flo, b[2]); fhi = Math.max(fhi, b[5]); }
  if (!(fhi > flo)) { flo = E.hRange[0]; fhi = E.hRange[1]; }
  const fp = 0.04 * (fhi - flo);
  E.fRange = positioned ? E.hRange : [flo - fp, fhi + fp];
  // full time extent
  let tlo = Infinity, thi = -Infinity;
  for (let i = 0; i < nh; i++) { const t = time[i] - E.tref; if (t < tlo) tlo = t; if (t > thi) thi = t; }
  for (let j = 0; j < nfs; j++) { if (E.fT[j] < tlo) tlo = E.fT[j]; if (E.fT[j] > thi) thi = E.fT[j]; }
  if (!(thi > tlo)) { tlo = -1; thi = 1; }
  const tp = 0.02 * (thi - tlo);
  E.full = [tlo - tp, thi + tp];
  const tv = meta.truth && meta.truth.vertex;
  E.vz = tv && Number.isFinite(tv[2]) ? tv[2] : null;
  O.E = E;
  uploadEvent();
  setHead();
  resetWindow(true);
}

function uploadEvent() {
  const E = O.E, gl = O.gl;
  if (!E || !gl) return;
  const mkBuf = (arr) => { const b = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, b); gl.bufferData(gl.ARRAY_BUFFER, arr, gl.STATIC_DRAW); return b; };
  const L = O.loc;
  const mkVao = (T, Y, V, S, n) => {
    const vao = gl.createVertexArray(); gl.bindVertexArray(vao);
    for (const [name, buf] of [['aT', mkBuf(T)], ['aY', mkBuf(Y)], ['aV', mkBuf(V)], ['aS', mkBuf(S)]]) {
      gl.bindBuffer(gl.ARRAY_BUFFER, buf);
      gl.enableVertexAttribArray(L[name]); gl.vertexAttribPointer(L[name], 1, gl.FLOAT, false, 0, 0);
    }
    gl.bindVertexArray(null);
    return { vao, n };
  };
  E.gHits = mkVao(E.hT, E.hY, E.hP, E.hP, E.nh);
  E.gFl = mkVao(E.fT, E.fZ, E.fY, E.fP, E.fT.length);
}

function setHead() {
  const E = O.E; if (!E) return;
  const b = E.bundle, fpe = b.get('flash/pe'), ft = b.get('flash/time'), fy = b.get('flash/y'), fz = b.get('flash/z');
  const lines = [];
  const bits = [`${E.nf} flashes, ${E.nh} optical hits`];
  if (E.nf) {
    let k = 0; for (let i = 1; i < E.nf; i++) if (fpe[i] > fpe[k]) k = i;
    const med = Array.from(fpe).sort((a, c) => a - c)[Math.floor(E.nf / 2)];
    bits.push(`brightest flash t = ${ft[k].toFixed(2)} µs, ${fpe[k].toFixed(0)} PE (${(fpe[k] / Math.max(med, 1e-9)).toFixed(0)}x the median flash), y = ${fy[k].toFixed(0)}, z = ${fz[k].toFixed(0)} cm`);
  }
  if (Number.isFinite(E.opm.t0)) bits.push(`t0 = ${E.opm.t0.toFixed(2)} µs`);
  lines.push(bits.join('   -   '));
  const pe = b.get('oph/pe');
  let nonpos = 0; for (let i = 0; i < E.nh; i++) if (!(pe[i] > 0)) nonpos++;
  if (nonpos) lines.push(`${nonpos} OpHits have PE <= 0 (baseline undershoot); they sit at the bottom of the log colour scale`);
  if (!E.positioned && E.nh) lines.push('photon-detector positions unavailable: the hit panel shows the optical channel index, not z');
  for (const n of E.opm.notes || []) lines.push(String(n));
  E.noteLines = lines;
  O.head.title = lines.join('\n');
  resize();
}

function updateNote() {
  const E = O.E; if (!E) { O.note.textContent = ''; return; }
  const t0 = E.tref;
  const lo = O.x0 + t0, hi = O.x1 + t0;
  let nh = 0, nf = 0;
  const time = E.bundle.get('oph/time');
  for (let i = 0; i < E.nh; i++) { const t = time[i]; if (t >= lo && t <= hi) nh++; }
  const ft = E.bundle.get('flash/time');
  for (let i = 0; i < E.nf; i++) { const t = ft[i]; if (t >= lo && t <= hi) nf++; }
  O.span.textContent = `t in [${lo.toFixed(2)}, ${hi.toFixed(2)}] µs: ${nh} of ${E.nh} hits, ${nf} of ${E.nf} flashes`;
  O.note.textContent = E.noteLines.join('\n');
  layoutStage();
}

// ----------------------------------------------------------------------------- window
function defaultWindow() {
  const E = O.E, w = O.app.state.owin;
  if (!E) return [-50, 50];
  if (w === 'all') return E.full.slice();
  const hw = Math.max(0.01, parseFloat(w) || 50);
  return [-hw, hw];
}
function resetWindow(noSchedule) {
  const [a, b] = defaultWindow();
  O.x0 = a; O.x1 = b;
  if (O.sel && O.sel.value !== O.app.state.owin) O.sel.value = O.app.state.owin;
  updateNote();
  if (!noSchedule) schedule();
}

function onState(state, changed) {
  if (!O.ready) return;
  if (changed.has('owin')) resetWindow();
  else if (['theme', 'cmap'].some((k) => changed.has(k))) schedule();
}

function setVisible(v) { O.visible = v; if (v) { resize(); } }

// ----------------------------------------------------------------------------- layout / render
function layoutStage() {
  const h = O.head.offsetHeight;
  if (O.stage.style.top !== h + 'px') { O.stage.style.top = h + 'px'; return true; }
  return false;
}

function resize() {
  if (!O.ready || !O.visible) return;
  layoutStage();
  const w = O.stage.clientWidth, h = O.stage.clientHeight;
  if (w < 4 || h < 4) return;
  O.dpr = Math.min(window.devicePixelRatio || 1, 2);
  O.W = w; O.H = h;
  for (const c of [O.glc, O.ovc]) { c.width = Math.round(w * O.dpr); c.height = Math.round(h * O.dpr); }
  const top0 = M.t, ph = (h - M.t - M.gap - M.b) / 2;
  const px = M.l, pw = Math.max(10, w - M.l - M.r);
  O.rects = [{ x: px, y: top0, w: pw, h: Math.max(10, ph) },
             { x: px, y: top0 + ph + M.gap, w: pw, h: Math.max(10, ph) }];
  schedule();
}

function schedule() {
  if (O.raf || !O.ready || !O.visible) return;
  O.raf = requestAnimationFrame(() => { O.raf = 0; render(); });
}

function render() {
  if (!O.ready || !O.visible || !O.W) return;
  const t0 = performance.now();
  drawGL();
  drawOverlay();
  O.lastMs = performance.now() - t0;
}

function drawGL() {
  const gl = O.gl; if (!gl || gl.isContextLost()) return;
  const th = O.app.theme, bg = hexToRgb(th.axes_bg);
  gl.disable(gl.SCISSOR_TEST);
  gl.viewport(0, 0, O.glc.width, O.glc.height);
  gl.clearColor(bg[0], bg[1], bg[2], 1); gl.clear(gl.COLOR_BUFFER_BIT);
  const E = O.E; if (!E) return;
  uploadLut();
  gl.useProgram(O.prog);
  const L = O.loc, d = O.dpr;
  gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, O.lutTex); gl.uniform1i(L.uLut, 0);
  gl.uniform1f(L.uCmapLo, th.cmap_floor); gl.uniform1f(L.uCmapHi, th.cmap_ceiling);
  gl.uniform1f(L.uPx, d);
  gl.uniform2f(L.uX, O.x0, O.x1);
  gl.enable(gl.SCISSOR_TEST);
  gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
  const vp = (r) => { const x = Math.round(r.x * d), y = Math.round((O.H - r.y - r.h) * d), w = Math.round(r.w * d), h = Math.round(r.h * d);
    gl.viewport(x, y, w, h); gl.scissor(x, y, w, h); };
  // (a) hits
  vp(O.rects[0]);
  gl.uniform2f(L.uY, E.hRange[0], E.hRange[1]);
  gl.uniform1f(L.uCLog, 1); gl.uniform1f(L.uCLo, Math.log10(PE_RANGE[0])); gl.uniform1f(L.uCHi, Math.log10(PE_RANGE[1]));
  gl.uniform1f(L.uSizeMode, 0); gl.uniform1f(L.uSize, 2.6); gl.uniform1f(L.uAlpha, 1);
  if (E.gHits.n) { gl.bindVertexArray(E.gHits.vao); gl.drawArrays(gl.POINTS, 0, E.gHits.n); }
  // (b) flashes
  vp(O.rects[1]);
  gl.uniform2f(L.uY, E.fRange[0], E.fRange[1]);
  gl.uniform1f(L.uCLog, 0); gl.uniform1f(L.uCLo, E.yRange[0]); gl.uniform1f(L.uCHi, E.yRange[1]);
  gl.uniform1f(L.uSizeMode, 1); gl.uniform1f(L.uAlpha, 0.85);
  if (E.gFl.n) { gl.bindVertexArray(E.gFl.vao); gl.drawArrays(gl.POINTS, 0, E.gFl.n); }
  gl.bindVertexArray(null);
  gl.disable(gl.SCISSOR_TEST);
}

const xOf = (r, t) => r.x + (t - O.x0) / (O.x1 - O.x0) * r.w;
const yOf = (r, rng, v) => r.y + r.h - (v - rng[0]) / (rng[1] - rng[0]) * r.h;

function drawOverlay() {
  const c = O.ov, d = O.dpr, th = O.app.theme;
  c.setTransform(d, 0, 0, d, 0, 0);
  c.clearRect(0, 0, O.W, O.H);
  const E = O.E; if (!E) return;
  c.font = '11px system-ui, sans-serif';
  const ticksX = niceTicks(O.x0 + E.tref, O.x1 + E.tref, Math.max(3, O.rects[0].w / 90));
  const panels = [
    { r: O.rects[0], rng: E.hRange, title: 'optical hits (brightest on top)', ylab: E.positioned ? 'photon detector z [cm]' : 'optical channel (index, not position)' },
    { r: O.rects[1], rng: E.fRange, title: 'flashes (size ~ PE, colour = y)', ylab: 'flash z [cm]' },
  ];
  panels.forEach((p, k) => {
    const r = p.r;
    c.save(); c.beginPath(); c.rect(r.x, r.y, r.w, r.h); c.clip();
    // t0 marker and true vertex
    c.strokeStyle = th.fg_muted; c.globalAlpha = 0.5; c.setLineDash([2, 4]); c.lineWidth = 1;
    const xt0 = xOf(r, 0);
    if (xt0 > r.x && xt0 < r.x + r.w) { c.beginPath(); c.moveTo(xt0, r.y); c.lineTo(xt0, r.y + r.h); c.stroke(); }
    c.restore();
    if (E.vz != null && (k === 1 || E.positioned)) {
      const y = yOf(r, p.rng, E.vz);
      if (y >= r.y && y <= r.y + r.h) {
        c.save(); c.strokeStyle = th.truth_vertex; c.lineWidth = 1.4; c.setLineDash([7, 4]);
        c.beginPath(); c.moveTo(r.x, y); c.lineTo(r.x + r.w, y); c.stroke();
        c.fillStyle = th.truth_vertex; c.textAlign = 'right'; c.textBaseline = 'bottom';
        c.fillText(`true vertex z = ${E.vz.toFixed(0)} cm`, r.x + r.w - 4, y - 2);
        c.restore();
      }
    }
    // frame, ticks
    c.strokeStyle = th.grid; c.fillStyle = th.fg; c.lineWidth = 1; c.setLineDash([]);
    c.strokeRect(r.x + 0.5, r.y + 0.5, r.w, r.h);
    c.textBaseline = 'top'; c.textAlign = 'center';
    for (const v of ticksX.ticks) {
      const x = xOf(r, v - E.tref);
      if (x < r.x - 1 || x > r.x + r.w + 1) continue;
      c.beginPath(); c.moveTo(x + 0.5, r.y + r.h); c.lineTo(x + 0.5, r.y + r.h + 4); c.stroke();
      c.fillText(fmtTick(v, ticksX.step), x, r.y + r.h + 6);
    }
    const ty = niceTicks(p.rng[0], p.rng[1], Math.max(3, r.h / 36));
    c.textAlign = 'right'; c.textBaseline = 'middle';
    for (const v of ty.ticks) {
      const y = yOf(r, p.rng, v);
      if (y < r.y - 1 || y > r.y + r.h + 1) continue;
      c.beginPath(); c.moveTo(r.x - 4, y + 0.5); c.lineTo(r.x, y + 0.5); c.stroke();
      c.fillText(fmtTick(v, ty.step), r.x - 7, y);
    }
    c.save(); c.translate(14, r.y + r.h / 2); c.rotate(-Math.PI / 2); c.textAlign = 'center'; c.textBaseline = 'middle';
    c.fillStyle = th.fg_muted; c.fillText(p.ylab, 0, 0); c.restore();
    c.fillStyle = th.fg; c.textAlign = 'left'; c.textBaseline = 'bottom';
    c.fillText(p.title, r.x, r.y - 4);
    c.fillStyle = th.fg_muted; c.textAlign = 'center'; c.textBaseline = 'top';
    if (k === 1) c.fillText(`time [µs]   (dotted line: t0 = ${E.tref.toFixed(2)} µs)`, r.x + r.w / 2, r.y + r.h + 20);
  });
  // colour bars
  colourBar(c, th, O.rects[0], 'PE', Math.log10(PE_RANGE[0]), Math.log10(PE_RANGE[1]), true);
  colourBar(c, th, O.rects[1], 'flash y [cm]', E.yRange[0], E.yRange[1], false);
  // hover ring
  const h = O.hover;
  if (h) { c.save(); c.strokeStyle = th.fg; c.lineWidth = 1.5; c.beginPath(); c.arc(h.px, h.py, h.rad + 3, 0, 2 * Math.PI); c.stroke(); c.restore(); }
}

function colourBar(c, th, r, label, lo, hi, log) {
  const x = r.x + r.w + 16, w = 12, y = r.y + 4, h = Math.max(20, r.h - 8);
  const lut = getLut(O.app.state.cmap);
  for (let i = 0; i < h; i++) {
    const t = 1 - i / (h - 1), idx = Math.round((th.cmap_floor + (th.cmap_ceiling - th.cmap_floor) * t) * 255);
    c.fillStyle = `rgb(${lut[3 * idx]},${lut[3 * idx + 1]},${lut[3 * idx + 2]})`;
    c.fillRect(x, y + i, w, 1.5);
  }
  c.strokeStyle = th.grid; c.strokeRect(x + 0.5, y + 0.5, w, h);
  c.fillStyle = th.fg; c.textAlign = 'left'; c.textBaseline = 'middle';
  const ticks = log ? [] : niceTicks(lo, hi, 4).ticks;
  if (log) for (let e = Math.ceil(lo); e <= hi; e++) ticks.push(e);
  for (const v of ticks) {
    const yy = y + (1 - (v - lo) / (hi - lo)) * (h - 1);
    c.fillText(log ? (e10(v)) : String(Math.round(v)), x + w + 4, yy);
  }
  c.fillStyle = th.fg_muted; c.textBaseline = 'bottom'; c.fillText(label, x - 2, y - 4);
}
const e10 = (e) => (e <= 3 ? String(10 ** e) : `1e${e}`);

// ----------------------------------------------------------------------------- interaction
function panelAt(x, y) {
  for (let k = 0; k < 2; k++) { const r = O.rects[k]; if (r && x >= r.x && x <= r.x + r.w && y >= r.y && y <= r.y + r.h) return k; }
  return -1;
}
function mouse(e) { const b = O.ovc.getBoundingClientRect(); return [e.clientX - b.left, e.clientY - b.top]; }

function setX(a, b) {
  const E = O.E; if (!E) return;
  const minSpan = 0.002, maxSpan = (E.full[1] - E.full[0]) * 1.2;
  let span = b - a;
  if (span < minSpan) { const m = (a + b) / 2; a = m - minSpan / 2; b = m + minSpan / 2; span = minSpan; }
  if (span > maxSpan) { const m = (a + b) / 2; a = m - maxSpan / 2; b = m + maxSpan / 2; }
  O.x0 = a; O.x1 = b;
  updateNote(); schedule();
}

function onWheel(e) {
  if (!O.E || !O.rects.length) return;
  e.preventDefault();
  const [mx] = mouse(e), r = O.rects[0];
  const tc = O.x0 + (mx - r.x) / r.w * (O.x1 - O.x0);
  const f = Math.exp(Math.max(-200, Math.min(200, e.deltaY)) * 0.0015);
  setX(tc - (tc - O.x0) * f, tc + (O.x1 - tc) * f);
}
function onDown(e) {
  const [mx, my] = mouse(e);
  if (panelAt(mx, my) < 0 && mx < O.rects[0].x) return;
  O.drag = { mx, x0: O.x0, x1: O.x1, moved: false };
  O.ovc.setPointerCapture(e.pointerId);
}
function onMove(e) {
  const [mx, my] = mouse(e);
  if (O.drag) {
    const r = O.rects[0], dt = (mx - O.drag.mx) / r.w * (O.drag.x1 - O.drag.x0);
    if (Math.abs(mx - O.drag.mx) > 2) O.drag.moved = true;
    setX(O.drag.x0 - dt, O.drag.x1 - dt);
    O.tip.style.display = 'none';
    return;
  }
  pick(mx, my);
}
function onUp(e) { O.drag = null; try { O.ovc.releasePointerCapture(e.pointerId); } catch (_) { /* not captured */ } }

function pick(mx, my) {
  const E = O.E, k = E ? panelAt(mx, my) : -1;
  if (k < 0) { if (O.hover) { O.hover = null; schedule(); } O.tip.style.display = 'none'; return; }
  const r = O.rects[k];
  const tx = O.x0 + (mx - r.x) / r.w * (O.x1 - O.x0);
  const pxPerT = r.w / (O.x1 - O.x0);
  const T = k === 0 ? E.hT : E.fT, Y = k === 0 ? E.hY : E.fZ, rng = k === 0 ? E.hRange : E.fRange;
  const reach = 9, dtMax = reach / pxPerT;
  let best = -1, bd = reach * reach;
  const n = T.length;
  for (let j = 0; j < n; j++) {
    const dt = T[j] - tx;
    if (dt > dtMax || dt < -dtMax) continue;
    const dx = dt * pxPerT, dy = yOf(r, rng, Y[j]) - my;
    const dd = dx * dx + dy * dy;
    if (dd <= bd) { bd = dd; best = j; }       // ties: later = brighter
  }
  if (best < 0) { if (O.hover) { O.hover = null; schedule(); } O.tip.style.display = 'none'; return; }
  const b = E.bundle;
  let txt;
  if (k === 0) {
    const i = E.order[best];
    const time = b.get('oph/time')[i], chan = b.get('oph/channel'), tile = b.get('oph/tile'), fl = b.get('oph/flash');
    txt = `OpHit\nt = ${time.toFixed(3)} µs  (t0 ${(time - E.tref >= 0 ? '+' : '')}${(time - E.tref).toFixed(3)})\nchannel ${chan ? chan[i] : '-'}` +
      `${tile && tile[i] >= 0 ? `   tile ${tile[i]}` : ''}\nPE = ${b.get('oph/pe')[i].toFixed(2)}\n${E.positioned ? 'z' : 'index'} = ${Number.isFinite(E.hz[i]) ? E.hz[i].toFixed(1) : '-'}` +
      `${fl && fl[i] >= 0 ? `\nin flash #${fl[i]}` : ''}`;
    O.hover = { px: xOf(r, T[best]), py: yOf(r, rng, Y[best]), rad: 3 };
  } else {
    const i = E.fsel[best];
    const g = (n) => { const a = b.get('flash/' + n); return a ? a[i] : NaN; };
    const x = g('x');
    txt = `flash #${i}\nt = ${g('time').toFixed(3)} µs  (t0 ${(g('time') - E.tref >= 0 ? '+' : '')}${(g('time') - E.tref).toFixed(3)})\nPE = ${g('pe').toFixed(0)}\n` +
      `y = ${g('y').toFixed(1)} ± ${g('yw').toFixed(1)} cm\nz = ${g('z').toFixed(1)} ± ${g('zw').toFixed(1)} cm` +
      (Number.isFinite(x) ? `\nx = ${x.toFixed(1)} cm` : '\nx not reconstructed');
    O.hover = { px: xOf(r, T[best]), py: yOf(r, rng, Y[best]), rad: 5 + 7 * Math.min(1, Math.log10(Math.max(1, E.fP[best])) / 5.477) };
  }
  O.tip.textContent = txt;
  O.tip.style.display = 'block';
  const sb = O.stage.getBoundingClientRect(), cb = O.container.getBoundingClientRect();
  let lx = sb.left - cb.left + O.hover.px + 14, ly = sb.top - cb.top + O.hover.py + 10;
  const tw = O.tip.offsetWidth, th = O.tip.offsetHeight;
  if (lx + tw > cb.width - 4) lx = lx - tw - 28;
  if (ly + th > cb.height - 4) ly = cb.height - th - 4;
  O.tip.style.left = Math.max(2, lx) + 'px'; O.tip.style.top = Math.max(2, ly) + 'px';
  schedule();
}

// ----------------------------------------------------------------------------- screenshot / debug
function getCanvas() {
  if (!O.ready || !O.W) return null;
  drawGL();
  drawOverlay();
  const out = document.createElement('canvas');
  out.width = O.glc.width; out.height = O.glc.height;
  const g = out.getContext('2d');
  g.drawImage(O.glc, 0, 0);
  g.drawImage(O.ovc, 0, 0);
  // header text (summary / notes) on top of the image
  g.setTransform(O.dpr, 0, 0, O.dpr, 0, 0);
  g.fillStyle = O.app.theme.fg_muted; g.font = '11px system-ui, sans-serif'; g.textBaseline = 'top';
  if (O.E) (O.E.noteLines || []).slice(0, 2).forEach((l, i) => g.fillText(l.slice(0, 220), 8, 2 + 13 * i));
  return out;
}

/** Test hook: time n full redraws (ms/frame, GPU drained). */
function benchmark(n = 10) {
  const gl = O.gl, px = new Uint8Array(4), t0 = performance.now();
  const x0 = O.x0, x1 = O.x1;
  for (let i = 0; i < n; i++) {
    const s = 1e-4 * (i & 1 ? 1 : -1);
    O.x0 = x0 + s; O.x1 = x1 + s;
    drawGL(); drawOverlay();
    gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px);
  }
  O.x0 = x0; O.x1 = x1;
  return (performance.now() - t0) / n;
}

export default {
  name: 'viewoptical',
  init, setGeometry, setEvent, onState, setVisible, resize, getCanvas, benchmark,
  debugState: () => ({ x0: O.x0, x1: O.x1, rects: O.rects, tref: O.E && O.E.tref, nh: O.E && O.E.nh, nf: O.E && O.E.nf, positioned: O.E && O.E.positioned }),
};
