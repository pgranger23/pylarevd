// viewtrigger.js -- the Trigger view: trigger primitives (TPs), activities (TAs) and candidates
// (TCs) in channel x time, time horizontal.
//
// Layouts (param tlay):
//   timeline  a shared time axis on top (TC and TA windows as bars, TP rate stacked by plane),
//             one channel x time panel per plane below, details of the selection at the bottom;
//   grid      one row per detector unit (APA), one column per plane, a thin timeline on top;
//   explorer  a TC > TA tree on the left, a zoom on the selection, histograms of the
//             selection's TPs against the rest of their plane(s);
//   reco      one plane (param tplane) next to the 2-D wire view ("Trigger + 2D"): hovering a
//             TP marks it there, selecting a TA or TC zooms it there (app event 'trigger').
// The channel axis is a ROW axis: the channels of one plane, numbered in channel order, so the
// interleaved U/V/Z blocks of each APA do not leave two thirds of a panel empty. Labels show
// the real channel numbers; unit boundaries are drawn. Rows come from the geometry bundle's
// ro/chan_view + ro/chan_unit (meta.units); without them, the row is the channel.
//
// Rendering: one WebGL2 canvas (viewport + scissor per panel). Each TP is ONE instance drawn
// twice: a GL_LINES bar over its time over threshold and a square point at its peak.
// Per-instance buffers are uploaded once per event in (plane, unit, in-activity) order, so a
// panel is a contiguous instance range and the TPs of activities are drawn on top. Colour
// mode, selection emphasis, zoom and pan only change uniforms or one small buffer.
// A transparent Canvas2D on top draws axes, TA boxes, TC bands, the timeline, histograms and
// the hover marker, and takes the pointer events.
//
// Keys (view visible, no form field focused): n / p next / previous candidate (activity when
// the event has no candidate), z zoom to the selection, Escape clears it.
import { getLut, hexToRgb, cmapGradientCss } from './colormaps.js';

const LAYOUTS = [['timeline', 'timeline + planes'], ['grid', 'detector-unit grid'],
  ['explorer', 'hierarchy explorer'], ['reco', 'trigger vs reco']];
const COLOURS = [['adc', 'ADC sum'], ['tot', 'time over threshold'], ['peak', 'ADC peak'], ['ta', 'activity']];
const VIEWS = ['U', 'V', 'Z'];
const PM = { l: 74, r: 10, t: 16, b: 6, xlab: 22 };   // panel margins (css px); xlab under the bottom panels
const HEAD = 30;

const CSS = `
.vtrg-head { position:absolute; left:0; right:0; top:0; height:${HEAD}px; z-index:6; display:flex; gap:10px; align-items:center;
  padding:0 8px; font-size:12px; color:var(--muted); background:color-mix(in srgb, var(--axes-bg) 80%, transparent); white-space:nowrap; overflow:hidden; }
.vtrg-head select, .vtrg-head button { padding:0 5px; font-size:12px; }
.vtrg-head .info { margin-left:auto; overflow:hidden; text-overflow:ellipsis; }
.vtrg-stage { position:absolute; left:0; right:0; top:${HEAD}px; bottom:0; }
.vtrg-stage canvas { position:absolute; left:0; top:0; width:100%; height:100%; display:block; }
.vtrg-ov { touch-action:none; cursor:crosshair; }
.vtrg-tip { position:absolute; z-index:8; pointer-events:none; display:none; white-space:pre; font:11px/1.35 ui-monospace,Menlo,monospace;
  background:color-mix(in srgb, var(--legend-bg) 94%, transparent); border:1px solid var(--legend-edge); border-radius:4px; padding:3px 6px; color:var(--fg); }
.vtrg-msg { position:absolute; inset:${HEAD}px 0 0 0; display:none; align-items:center; justify-content:center; color:var(--muted); z-index:5; padding:20px; text-align:center; }
.vtrg-tree { position:absolute; left:0; top:${HEAD}px; bottom:0; width:250px; z-index:4; overflow:auto; display:none; font-size:12px;
  border-right:1px solid var(--legend-edge); background:var(--axes-bg); color:var(--fg); }
.vtrg-tree .row { padding:3px 8px; cursor:pointer; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.vtrg-tree .row:hover { background:color-mix(in srgb, var(--fg) 10%, transparent); }
.vtrg-tree .row.sel { background:color-mix(in srgb, #e879f9 30%, transparent); }
.vtrg-tree .row.ta { padding-left:24px; color:var(--muted); }
.vtrg-tree .row.head { cursor:default; color:var(--muted); font-weight:600; background:none; }
.vtrg-tree .row .k { color:var(--fg); font-weight:600; }
.vtrg-details { position:absolute; left:0; right:0; bottom:0; z-index:4; display:none; padding:3px 8px; font:11px/1.4 ui-monospace,Menlo,monospace;
  color:var(--fg); background:color-mix(in srgb, var(--axes-bg) 90%, transparent); border-top:1px solid var(--legend-edge); white-space:pre; overflow:hidden; }
`;

const VS = `#version 300 es
precision highp float;
in float aT0; in float aT1; in float aPk; in float aRow; in float aVal; in float aCat; in float aEmph;
uniform vec2 uX, uY;
uniform float uPass, uMode, uLog, uLo, uHi, uCmapLo, uCmapHi, uSize, uDim, uAlpha;
uniform sampler2D uLut;
uniform vec3 uPal[8], uGrey, uUni, uHi3;
out vec4 vCol;
void main() {
  float t = uPass < 0.5 ? (gl_VertexID == 0 ? aT0 : aT1) : aPk;
  gl_Position = vec4(2.0 * (t - uX.x) / (uX.y - uX.x) - 1.0, 2.0 * (aRow - uY.x) / (uY.y - uY.x) - 1.0, 0.0, 1.0);
  vec3 c;
  if (uMode < 0.5) {
    float v = uLog > 0.5 ? log(max(aVal, 1e-6)) : aVal;
    float f = clamp((v - uLo) / max(uHi - uLo, 1e-9), 0.0, 1.0);
    c = texture(uLut, vec2(mix(uCmapLo, uCmapHi, f), 0.5)).rgb;
  } else if (uMode < 1.5) {
    c = aCat < -0.5 ? uGrey : uPal[int(mod(aCat, 8.0))];
  } else c = uUni;
  float a = uAlpha;
  gl_PointSize = uSize;
  if (aEmph > 0.5) { a = 1.0; gl_PointSize = uSize + 2.0; if (uMode > 0.5 && uMode < 1.5) c = uHi3; }
  else if (aEmph < -0.5) a *= uDim;
  vCol = vec4(c, a);
}`;
const FS = `#version 300 es
precision mediump float;
in vec4 vCol; out vec4 o;
void main() { o = vCol; }`;

const T = {
  app: null, container: null, ready: false, visible: false,
  head: null, sel: {}, info: null, stage: null, glc: null, ovc: null, ov: null, tip: null, msg: null, tree: null, details: null,
  gl: null, prog: null, loc: {}, lutTex: null, lutName: '', buf: {}, dpr: 1, W: 0, H: 0, cw: 0, ch: 0,
  E: null,                       // per-event data
  R: null,                       // per-geometry channel rows
  x0: 0, x1: 1, ys: {},          // shared time range; per-panel row ranges by panel key
  L: null,                       // current layout: {kind, panels, timeline, hists}
  hover: null, drag: null, raf: 0, selKey: '', lastMs: 0,
};

// ----------------------------------------------------------------------------- helpers
const rgb = (h) => hexToRgb(h);
const esc = (s) => String(s).replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
function niceTicks(lo, hi, target) {
  const span = hi - lo; if (!(span > 0)) return { ticks: [lo], step: 1 };
  const raw = span / Math.max(1, target), mag = 10 ** Math.floor(Math.log10(raw)), n = raw / mag;
  const step = (n < 1.5 ? 1 : n < 3 ? 2 : n < 7 ? 5 : 10) * mag;
  const out = []; for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-6; v += step) out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  return { ticks: out, step };
}
const fmt = (v) => (Math.abs(v) >= 1e4 || (Math.abs(v) < 0.01 && v !== 0) ? v.toExponential(2) : (+v.toPrecision(4)).toString());
function trigColours(th) {
  const dark = th.name !== 'light';
  return { ta: dark ? '#e879f9' : '#a21caf', tc: dark ? '#fbbf24' : '#b45309', views: dark ? ['#5b9bd5', '#59a14f', '#f28e2b'] : ['#1f6fb2', '#2e7d32', '#d2691e'] };
}
function effLayout() {
  const st = T.app.state;
  if (st.view === 'trigger+2d') return 'reco';
  return LAYOUTS.some(([k]) => k === st.tlay) && st.tlay !== 'reco' ? st.tlay : 'timeline';
}
function parseSel() {
  const m = /^(tc|ta):(\d+)$/.exec(T.app.state.tsel || '');
  if (!m || !T.E) return null;
  const i = +m[2], n = m[1] === 'tc' ? T.E.ntc : T.E.nta;
  return i < n ? { kind: m[1], i } : null;
}

// ----------------------------------------------------------------------------- init
function init(container, app) {
  T.app = app; T.container = container;
  app.registerParam('tlay', 'timeline');
  app.registerParam('tcol', 'adc');
  app.registerParam('thits', '0');
  app.registerParam('tplane', '2');
  app.registerParam('tsel', '');
  const st = document.createElement('style'); st.textContent = CSS; document.head.appendChild(st);

  T.head = document.createElement('div'); T.head.className = 'vtrg-head';
  T.head.innerHTML =
    `<label>layout <select class="lay">${LAYOUTS.map(([k, l]) => `<option value="${k}">${l}</option>`).join('')}</select></label>` +
    `<label>colour TPs by <select class="col">${COLOURS.map(([k, l]) => `<option value="${k}">${l}</option>`).join('')}</select></label>` +
    `<label class="plane-l">plane <select class="plane">${VIEWS.map((v, i) => `<option value="${i}">${v}</option>`).join('')}</select></label>` +
    '<label title="reconstructed hits, faint, under the TPs"><input type="checkbox" class="hits"> hits</label>' +
    '<button class="fit" title="zoom to the selected candidate / activity (z)">zoom to selection</button>' +
    '<button class="rst" title="whole readout (double click)">reset</button>' +
    '<span class="info"></span>';
  T.sel = { lay: T.head.querySelector('.lay'), col: T.head.querySelector('.col'), plane: T.head.querySelector('.plane'),
    planeL: T.head.querySelector('.plane-l'), hits: T.head.querySelector('.hits') };
  T.info = T.head.querySelector('.info');
  T.sel.lay.addEventListener('change', () => {
    const v = T.sel.lay.value;
    app.setState(v === 'reco' ? { tlay: 'reco', view: 'trigger+2d' } : { tlay: v, view: 'trigger' });
  });
  T.sel.col.addEventListener('change', () => app.setState({ tcol: T.sel.col.value }));
  T.sel.plane.addEventListener('change', () => app.setState({ tplane: T.sel.plane.value }));
  T.sel.hits.addEventListener('change', () => app.setState({ thits: T.sel.hits.checked ? '1' : '0' }));
  T.head.querySelector('.fit').addEventListener('click', () => fitSelection());
  T.head.querySelector('.rst').addEventListener('click', () => resetView());

  T.stage = document.createElement('div'); T.stage.className = 'vtrg-stage';
  T.glc = document.createElement('canvas'); T.ovc = document.createElement('canvas'); T.ovc.className = 'vtrg-ov';
  T.stage.append(T.glc, T.ovc);
  T.tip = document.createElement('div'); T.tip.className = 'vtrg-tip';
  T.msg = document.createElement('div'); T.msg.className = 'vtrg-msg';
  T.tree = document.createElement('div'); T.tree.className = 'vtrg-tree';
  T.details = document.createElement('div'); T.details.className = 'vtrg-details';
  container.append(T.stage, T.head, T.tip, T.msg, T.tree, T.details);
  T.ov = T.ovc.getContext('2d');
  T.tree.addEventListener('click', (e) => {
    const r = e.target.closest('.row[data-sel]'); if (r) select(r.dataset.sel, true);
  });

  const gl = T.gl = T.glc.getContext('webgl2', { antialias: false, alpha: false, preserveDrawingBuffer: false });
  if (!gl) { showMsg('WebGL2 is not available in this browser.'); return; }
  T.glc.addEventListener('webglcontextlost', (e) => e.preventDefault());
  T.glc.addEventListener('webglcontextrestored', () => { buildGL(); T.lutName = ''; uploadEvent(); schedule(); });
  buildGL();
  T.ovc.addEventListener('wheel', onWheel, { passive: false });
  T.ovc.addEventListener('pointerdown', onDown);
  T.ovc.addEventListener('pointermove', onMove);
  T.ovc.addEventListener('pointerup', onUp);
  T.ovc.addEventListener('pointerleave', () => { setHover(null); });
  T.ovc.addEventListener('dblclick', () => resetView());
  window.addEventListener('keydown', onKey);
  if (typeof ResizeObserver !== 'undefined') new ResizeObserver(() => { if (T.visible) resize(); }).observe(container);
  T.ready = true;
}

function showMsg(t) { T.msg.textContent = t || ''; T.msg.style.display = t ? 'flex' : 'none'; }

function buildGL() {
  const gl = T.gl;
  const mk = (type, src) => { const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error('viewtrigger shader: ' + gl.getShaderInfoLog(s)); return s; };
  const p = gl.createProgram();
  gl.attachShader(p, mk(gl.VERTEX_SHADER, VS)); gl.attachShader(p, mk(gl.FRAGMENT_SHADER, FS));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error('viewtrigger link: ' + gl.getProgramInfoLog(p));
  T.prog = p; T.loc = {};
  for (const n of ['aT0', 'aT1', 'aPk', 'aRow', 'aVal', 'aCat', 'aEmph']) T.loc[n] = gl.getAttribLocation(p, n);
  for (const n of ['uX', 'uY', 'uPass', 'uMode', 'uLog', 'uLo', 'uHi', 'uCmapLo', 'uCmapHi', 'uSize', 'uDim', 'uAlpha', 'uLut', 'uPal[0]', 'uGrey', 'uUni', 'uHi3'])
    T.loc[n] = gl.getUniformLocation(p, n);
  T.lutTex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, T.lutTex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  T.buf = {};
  gl.enable(gl.BLEND);
  gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
}

function uploadLut() {
  const name = T.app.state.cmap;
  if (T.lutName === name || !T.gl) return;
  const lut = getLut(name), d = new Uint8Array(256 * 4);
  for (let i = 0; i < 256; i++) { d[4 * i] = lut[3 * i]; d[4 * i + 1] = lut[3 * i + 1]; d[4 * i + 2] = lut[3 * i + 2]; d[4 * i + 3] = 255; }
  const gl = T.gl;
  gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, T.lutTex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 256, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, d);
  T.lutName = name;
}

// ----------------------------------------------------------------------------- geometry rows
function setGeometry(geom) {
  T.R = null; T.ys = {};
  const cv = geom && geom.get('ro/chan_view'), cu = geom && geom.get('ro/chan_unit');
  if (!cv || !cu) return;
  const nch = cv.length, rowOf = new Int32Array(nch).fill(-1), chOfRow = [[], [], []];
  for (let c = 0; c < nch; c++) { const v = cv[c]; if (v >= 0 && v < 3) { rowOf[c] = chOfRow[v].length; chOfRow[v].push(c); } }
  const units = (geom.meta.units || []).map((u) => u.name);
  // per view: [{unit, r0, r1}] contiguous unit blocks along the row axis
  const blocks = chOfRow.map((list) => {
    const out = [];
    for (let r = 0; r < list.length; r++) {
      const u = cu[list[r]];
      if (!out.length || out[out.length - 1].unit !== u) out.push({ unit: u, r0: r, r1: r + 1 });
      else out[out.length - 1].r1 = r + 1;
    }
    return out;
  });
  T.R = { nch, rowOf, chOfRow: chOfRow.map((l) => Int32Array.from(l)), unitOf: cu, viewOf: cv, units, blocks };
}

// ----------------------------------------------------------------------------- event data
function setEvent(bundle, geometry) {
  T.E = null; T.hover = null; hideTip();
  if (geometry && !T.R) setGeometry(geometry);
  if (!bundle) { renderTree(); schedule(); return; }
  const b = bundle, ch = b.get('tp/channel');
  if (!ch || !ch.length) {
    showMsg('No trigger primitives in this event' + (b.meta.available && b.meta.available.trigger === false ?
      ' (the file has no dunetrigger TriggerPrimitive product)' : ''));
    T.info.textContent = ''; renderTree(); setDetails(); schedule(); return;
  }
  showMsg('');
  const n = ch.length, R = T.R;
  const t0 = b.get('tp/tick0'), tot = b.get('tp/tot'), pk = b.get('tp/tick'), adc = b.get('tp/adc'), apk = b.get('tp/peak');
  const ta = b.get('tp/ta'), cvTp = b.get('tp/chan_view');
  const view = new Int8Array(n), unit = new Int16Array(n), row = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const c = ch[i];
    if (R && c >= 0 && c < R.nch && R.rowOf[c] >= 0) { view[i] = R.viewOf[c]; unit[i] = R.unitOf[c]; row[i] = R.rowOf[c]; }
    else { view[i] = cvTp ? cvTp[i] : 2; unit[i] = 0; row[i] = c; }
  }
  // instance order: plane, unit, in-activity last (drawn on top)
  const order = new Uint32Array(n); for (let i = 0; i < n; i++) order[i] = i;
  const key = (i) => (view[i] * 4096 + (unit[i] + 1)) * 2 + (ta[i] >= 0 ? 1 : 0);
  order.sort((a, c) => key(a) - key(c) || a - c);
  const vr = [[0, 0], [0, 0], [0, 0]], ur = [new Map(), new Map(), new Map()];
  for (let j = 0; j < n; j++) {
    const i = order[j], v = view[i], u = unit[i];
    if (v < 0 || v > 2) continue;
    if (vr[v][1] === 0) vr[v] = [j, j + 1]; else vr[v][1] = j + 1;
    const r = ur[v].get(u); if (r) r[1] = j + 1; else ur[v].set(u, [j, j + 1]);
  }
  for (let v = 0; v < 3; v++) if (vr[v][1] === 0) vr[v] = [0, 0];
  const E = { b, n, ch, t0, tot, pk, adc, apk, ta, view, unit, row, order, vr, ur };
  // activities and candidates
  const meta = b.meta.trigger || {};
  E.nta = b.rows('ta/tick0'); E.ntc = b.rows('tc/tick0');
  E.taT0 = b.get('ta/tick0'); E.taT1 = b.get('ta/tick1'); E.taPk = b.get('ta/tick_peak'); E.taAdc = b.get('ta/adc');
  E.taN = b.get('ta/n_tp'); E.taTc = b.get('ta/tc'); E.taAlg = meta.ta_algorithm_names || [];
  E.tcT0 = b.get('tc/tick0'); E.tcT1 = b.get('tc/tick1'); E.tcT = b.get('tc/tick'); E.tcType = meta.tc_type_names || []; E.tcNta = b.get('tc/n_ta');
  E.meta = meta;
  E.tickUs = meta.ticks_per_dts ? 1 / (62.5 * meta.ticks_per_dts) : 0.5;
  // per activity: its TPs' rows per (view, unit), and its majority (view, unit)
  E.taTps = Array.from({ length: E.nta }, () => []);
  for (let i = 0; i < n; i++) if (ta[i] >= 0 && ta[i] < E.nta) E.taTps[ta[i]].push(i);
  E.taBox = E.taTps.map((list) => {
    const m = new Map();
    for (const i of list) {
      const k = view[i] * 4096 + unit[i] + 1, r = row[i], c = m.get(k);
      if (c) { c.r0 = Math.min(c.r0, r); c.r1 = Math.max(c.r1, r); c.n++; c.c0 = Math.min(c.c0, ch[i]); c.c1 = Math.max(c.c1, ch[i]); }
      else m.set(k, { view: view[i], unit: unit[i], r0: r, r1: r, n: 1, c0: ch[i], c1: ch[i] });
    }
    return [...m.values()].sort((a, c) => c.n - a.n);
  });
  E.tcTas = Array.from({ length: E.ntc }, () => []);
  for (let a = 0; a < E.nta; a++) if (E.taTc[a] >= 0 && E.taTc[a] < E.ntc) E.tcTas[E.taTc[a]].push(a);
  // hits (optional underlay), in plane order
  const hc = b.get('hits/channel'), ht = b.get('hits/tick');
  if (hc && ht) {
    const nh = hc.length, hv = new Int8Array(nh), hr = new Float32Array(nh);
    for (let i = 0; i < nh; i++) {
      const c = hc[i];
      if (R && c < R.nch && R.rowOf[c] >= 0) { hv[i] = R.viewOf[c]; hr[i] = R.rowOf[c]; } else { hv[i] = -1; hr[i] = c; }
    }
    const ho = []; for (let v = 0; v < 3; v++) for (let i = 0; i < nh; i++) if (hv[i] === v) ho.push(i);
    const hvr = [[0, 0], [0, 0], [0, 0]]; let j = 0;
    for (let v = 0; v < 3; v++) { const s = j; while (j < ho.length && hv[ho[j]] === v) j++; hvr[v] = [s, j]; }
    E.hits = { order: ho, vr: hvr, t: Float32Array.from(ho, (i) => ht[i]), r: Float32Array.from(ho, (i) => hr[i]) };
  }
  // full time range
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < n; i++) { if (t0[i] < lo) lo = t0[i]; const e = t0[i] + tot[i]; if (e > hi) hi = e; }
  for (let k = 0; k < E.ntc; k++) { lo = Math.min(lo, E.tcT0[k]); hi = Math.max(hi, E.tcT1[k]); }
  const pad = 0.01 * (hi - lo || 1);
  E.full = [lo - pad, hi + pad];
  E.vals = {};
  T.E = E;
  T.selKey = '';
  uploadEvent();
  T.x0 = E.full[0]; T.x1 = E.full[1]; T.ys = {};
  T.info.textContent = `${n.toLocaleString()} TPs · ${E.nta} TA · ${E.ntc} TC`;
  renderTree();
  applySelection(true, false);
  layout();
  if (effLayout() === 'explorer' && parseSel()) fitSelection(true);
  schedule();
}

/** colour values for the current mode, in instance order, with their display range */
function colourValues(mode) {
  const E = T.E;
  if (E.vals[mode]) return E.vals[mode];
  const src = mode === 'tot' ? E.tot : mode === 'peak' ? E.apk : E.adc;
  const log = mode !== 'tot';
  const v = Float32Array.from(E.order, (i) => src[i]);
  const s = Array.from(v).filter((x) => Number.isFinite(x) && (!log || x > 0)).sort((a, c) => a - c);
  const q = (f) => s.length ? s[Math.min(s.length - 1, Math.max(0, Math.floor(f * (s.length - 1))))] : 1;
  let lo = q(0.01), hi = q(0.99);
  if (!(hi > lo)) hi = lo + 1;
  return (E.vals[mode] = { v, log, lo: log ? Math.log(Math.max(lo, 1e-6)) : lo, hi: log ? Math.log(hi) : hi, rlo: lo, rhi: hi });
}

function glBuf(name, data) {
  const gl = T.gl;
  let b = T.buf[name];
  if (!b) b = T.buf[name] = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, b);
  gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
}

function uploadEvent() {
  const E = T.E, gl = T.gl; if (!E || !gl) return;
  const o = E.order, n = E.n;
  const f = (src) => Float32Array.from(o, (i) => src[i]);
  glBuf('t0', f(E.t0));
  glBuf('t1', Float32Array.from(o, (i) => E.t0[i] + Math.max(E.tot[i], 0.5)));
  glBuf('pk', f(E.pk));
  glBuf('row', Float32Array.from(o, (i) => E.row[i]));
  glBuf('cat', Float32Array.from(o, (i) => E.ta[i]));
  glBuf('emph', new Float32Array(n));
  E.valMode = null;
  if (E.hits) {
    glBuf('ht', E.hits.t); glBuf('hr', E.hits.r);
    glBuf('hz', new Float32Array(E.hits.t.length)); glBuf('hcat', new Float32Array(E.hits.t.length).fill(-1));
  }
}

// ----------------------------------------------------------------------------- selection
function select(key, fit) {
  T.app.setState({ tsel: key || '' });
  if (fit) setTimeout(() => fitSelection(true), 0);
}

/** recompute emphasis (selected activity TPs bright, the rest dim) and tell the 2-D view */
function applySelection(emit = true, fit = true) {
  const E = T.E, s = parseSel();
  const key = s ? `${s.kind}:${s.i}` : '';
  if (E && T.gl) {
    const tas = new Set(s ? (s.kind === 'ta' ? [s.i] : E.tcTas[s.i]) : []);
    const em = new Float32Array(E.n);
    if (s) for (let j = 0; j < E.n; j++) { const a = E.ta[E.order[j]]; em[j] = tas.has(a) ? 1 : -1; }
    glBuf('emph', em);
  }
  T.selKey = key;
  for (const r of T.tree.querySelectorAll('.row[data-sel]')) r.classList.toggle('sel', r.dataset.sel === key);
  setDetails();
  if (emit && T.visible) T.app.emit('trigger', { type: 'select', sel: s, fit });
}

function stepSelection(d) {
  const E = T.E; if (!E) return;
  const s = parseSel();
  if (E.ntc) {
    const i = s && s.kind === 'tc' ? (s.i + d + E.ntc) % E.ntc : (d > 0 ? 0 : E.ntc - 1);
    select(`tc:${i}`, true);
  } else if (E.nta) {
    const i = s && s.kind === 'ta' ? (s.i + d + E.nta) % E.nta : (d > 0 ? 0 : E.nta - 1);
    select(`ta:${i}`, true);
  }
}

function selWindow(s) {
  const E = T.E;
  if (s.kind === 'tc') return [E.tcT0[s.i], E.tcT1[s.i]];
  return [E.taT0[s.i], E.taT1[s.i]];
}
function selBoxes(s) {
  const E = T.E;
  const tas = s.kind === 'ta' ? [s.i] : E.tcTas[s.i];
  const m = new Map();
  for (const a of tas) for (const bx of E.taBox[a]) {
    const k = bx.view * 4096 + bx.unit + 1, c = m.get(k);
    if (c) { c.r0 = Math.min(c.r0, bx.r0); c.r1 = Math.max(c.r1, bx.r1); c.n += bx.n; } else m.set(k, { ...bx });
  }
  return [...m.values()].sort((a, c) => c.n - a.n);
}

function fitSelection(quiet) {
  const E = T.E, s = parseSel();
  if (!E || !s) { if (!quiet) resetView(); return; }
  if (!quiet && T.visible) T.app.emit('trigger', { type: 'select', sel: s, fit: true });
  const [w0, w1] = selWindow(s), span = Math.max(w1 - w0, 20);
  T.x0 = w0 - 0.25 * span; T.x1 = w1 + 0.25 * span;
  layout();
  for (const p of T.L.panels) {
    const bx = selBoxes(s).filter((x) => x.view === p.view && (p.unit < 0 || x.unit === p.unit));
    if (!bx.length) continue;
    const r0 = Math.min(...bx.map((x) => x.r0)), r1 = Math.max(...bx.map((x) => x.r1)), rs = Math.max(r1 - r0, 20);
    T.ys[p.key] = [Math.max(r0 - 0.2 * rs, p.rows[0] - 0.5), Math.min(r1 + 0.2 * rs, p.rows[1] - 0.5)];
  }
  schedule();
}

function resetView() {
  if (!T.E) return;
  T.x0 = T.E.full[0]; T.x1 = T.E.full[1]; T.ys = {};
  schedule();
}

// ----------------------------------------------------------------------------- side DOM
function renderTree() {
  const E = T.E;
  if (!E) { T.tree.innerHTML = ''; return; }
  const ta = (a) => {
    const bx = E.taBox[a][0];
    const where = bx ? `${T.R && T.R.units[bx.unit] ? T.R.units[bx.unit] : 'unit ' + (bx.unit + 1)} ${VIEWS[bx.view] || '?'}` : '';
    return `<div class="row ta" data-sel="ta:${a}" title="activity ${a}: ${E.taAlg[a] || ''}"><span class="k">TA ${a}</span> · ${where} · ${E.taN[a]} TPs · ADC ${fmt(E.taAdc[a])}</div>`;
  };
  let html = `<div class="row head">${E.ntc} candidate${E.ntc === 1 ? '' : 's'}, ${E.nta} activit${E.nta === 1 ? 'y' : 'ies'}</div>`;
  for (let c = 0; c < E.ntc; c++) {
    html += `<div class="row" data-sel="tc:${c}" title="candidate ${c}"><span class="k">TC ${c}</span> · ${esc(E.tcType[c] || '')} · t ${E.tcT[c].toFixed(0)} · ${E.tcT0[c].toFixed(0)}–${E.tcT1[c].toFixed(0)}</div>`;
    for (const a of E.tcTas[c]) html += ta(a);
  }
  const orphans = []; for (let a = 0; a < E.nta; a++) if (!(E.taTc[a] >= 0)) orphans.push(a);
  if (orphans.length) { html += '<div class="row head">activities without a candidate</div>'; for (const a of orphans) html += ta(a); }
  if (!E.nta) html += '<div class="row head" style="font-weight:400">no activity in this event: the TPs alone</div>';
  T.tree.innerHTML = html;
  for (const r of T.tree.querySelectorAll('.row[data-sel]')) r.classList.toggle('sel', r.dataset.sel === T.selKey);
}

function describe(s) {
  const E = T.E;
  const us = (t) => `${(t * E.tickUs).toFixed(1)} µs`;
  if (s.kind === 'tc') {
    const i = s.i;
    return `TC ${i}  ${E.tcType[i] || ''} (${(E.meta.tc_algorithm_names || [])[i] || ''})  candidate tick ${E.tcT[i].toFixed(0)} (${us(E.tcT[i])})` +
      `  window ${E.tcT0[i].toFixed(0)}–${E.tcT1[i].toFixed(0)} (${(E.tcT1[i] - E.tcT0[i]).toFixed(0)} ticks)  ${E.tcTas[i].length} TA: ` +
      E.tcTas[i].map((a) => `TA ${a} (${E.taN[a]} TPs)`).join(', ');
  }
  const a = s.i, bx = E.taBox[a][0];
  const where = bx ? `${T.R && T.R.units[bx.unit] ? T.R.units[bx.unit] : 'unit ' + (bx.unit + 1)} ${VIEWS[bx.view]}, channels ${bx.c0}–${bx.c1}` : '';
  return `TA ${a}  ${E.taAlg[a] || ''}  ${E.taN[a]} TPs, ADC sum ${fmt(E.taAdc[a])}  window ${E.taT0[a].toFixed(0)}–${E.taT1[a].toFixed(0)}` +
    `  peak ${E.taPk[a].toFixed(0)}  ${where}  ${E.taTc[a] >= 0 ? '→ TC ' + E.taTc[a] : 'no candidate'}`;
}
function setDetails() {
  const E = T.E, s = parseSel();
  if (!E) { T.details.textContent = ''; return; }
  T.details.textContent = s ? describe(s) :
    'click a TP, a TA box or a TC band to select it  ·  n / p: next / previous candidate  ·  z: zoom to it  ·  wheel: zoom time (shift: channels)  ·  drag: pan  ·  double click: reset';
}

// ----------------------------------------------------------------------------- layout
function layout() {
  const E = T.E, kind = effLayout();
  T.L = { kind, panels: [], timeline: null, hists: [] };
  if (!T.cw || !T.ch) return;
  const W = T.cw, H = T.ch;
  const showTree = kind === 'explorer';
  const showDetails = kind === 'timeline';
  T.tree.style.display = showTree ? 'block' : 'none';
  T.details.style.display = showDetails ? 'block' : 'none';
  T.sel.planeL.style.display = kind === 'reco' ? '' : 'none';
  if (!E) return;
  const left = showTree ? 250 : 0, bottom = showDetails ? 22 : 0;
  const R = T.R;
  const rowsOf = (v, u) => {
    if (!R) { let m = 0; for (const c of T.E.ch) if (c > m) m = c; return [0, m + 1]; }
    if (u < 0) return [0, R.chOfRow[v].length];
    const bl = R.blocks[v].find((x) => x.unit === u);
    return bl ? [bl.r0, bl.r1] : [0, 1];
  };
  const panel = (key, view, unit, x, y, w, h, title, xlab) => {
    const plot = { x: x + PM.l, y: y + PM.t, w: Math.max(10, w - PM.l - PM.r), h: Math.max(10, h - PM.t - PM.b - (xlab ? PM.xlab : 0)) };
    T.L.panels.push({ key, view, unit, cell: { x, y, w, h }, plot, rows: rowsOf(view, unit), title, xlab });
  };
  const views = [0, 1, 2].filter((v) => E.vr[v][1] > E.vr[v][0]);
  let top = 0;
  if (kind === 'timeline' || kind === 'grid' || kind === 'reco') {
    const lanes = packLanes(), hist = kind !== 'grid';
    const th = 10 + 4 + 14 * (1 + Math.max(1, lanes.n)) + (hist ? 58 : 2);
    T.L.timeline = { x: left, y: 0, w: W - left, h: th, lanes, hist: kind !== 'grid', plot: { x: left + PM.l, y: 6, w: W - left - PM.l - PM.r, h: th - 10 } };
    top = th + 4;
  }
  const avail = H - top - bottom;
  if (kind === 'timeline') {
    const h = avail / Math.max(1, views.length);
    views.forEach((v, k) => panel(`p${v}`, v, -1, left, top + k * h, W - left, h, `${VIEWS[v]} plane${v === 2 ? ' (collection)' : ''}`, k === views.length - 1));
  } else if (kind === 'reco') {
    const v = +T.app.state.tplane;
    panel(`r${v}`, v, -1, left, top, W - left, avail, `${VIEWS[v]} plane`, true);
  } else if (kind === 'grid') {
    const units = R ? R.units.map((_, u) => u).filter((u) => views.some((v) => E.ur[v].has(u) || R.blocks[v].some((b) => b.unit === u))) : [0];
    const nu = Math.max(1, units.length), nv = Math.max(1, views.length);
    const cw = (W - left) / nv, chh = avail / nu;
    units.forEach((u, ri) => views.forEach((v, ci) => panel(`g${u}_${v}`, v, u, left + ci * cw, top + ri * chh, cw, chh,
      `${R ? R.units[u] : 'unit'} · ${VIEWS[v]}`, ri === nu - 1)));
  } else if (kind === 'explorer') {
    const s = parseSel();
    const hh = Math.min(170, Math.max(110, 0.3 * avail));
    const ph = avail - hh;
    if (s) {
      const bx = selBoxes(s);
      const cells = bx.length ? bx.slice(0, 3) : [{ view: 2, unit: -1 }];
      const w = (W - left) / cells.length;
      cells.forEach((c, k) => panel(`e${c.view}_${c.unit}`, c.view, c.unit, left + k * w, top, w, ph,
        `${c.unit >= 0 && R ? R.units[c.unit] + ' · ' : ''}${VIEWS[c.view]}  (selection bright, the rest faint)`, true));
    } else {
      const h = ph / Math.max(1, views.length);
      views.forEach((v, k) => panel(`p${v}`, v, -1, left, top + k * h, W - left, h, `${VIEWS[v]} plane — pick a candidate or activity on the left`, k === views.length - 1));
    }
    const hw = (W - left) / 4;
    ['time over threshold [ticks]', 'ADC sum (log10)', 'ADC peak', 'TPs per channel'].forEach((t, k) =>
      T.L.hists.push({ title: t, k, x: left + k * hw, y: top + ph + 4, w: hw, h: hh - 4 }));
  }
}

/** greedy lanes for the TA bars of the timeline: overlapping activities stack */
function packLanes() {
  const E = T.E, ends = [], lane = new Int16Array(E.nta);
  const order = Array.from({ length: E.nta }, (_, a) => a).sort((a, c) => E.taT0[a] - E.taT0[c]);
  for (const a of order) {
    let k = ends.findIndex((e) => e <= E.taT0[a]);
    if (k < 0) { k = ends.length; ends.push(0); }
    ends[k] = E.taT1[a]; lane[a] = k;
  }
  return { lane, n: ends.length };
}

// ----------------------------------------------------------------------------- render
function schedule() {
  if (T.raf || !T.visible) return;
  T.raf = requestAnimationFrame(() => { T.raf = 0; try { render(); } catch (e) { console.error('viewtrigger render', e); } });
}

function yRange(p) {
  const r = T.ys[p.key];
  if (r) return r;
  const pad = Math.max(1, 0.01 * (p.rows[1] - p.rows[0]));
  return [p.rows[0] - pad, p.rows[1] + pad];
}

function render() {
  const t0 = performance.now();
  renderGL();
  renderOverlay();
  T.lastMs = performance.now() - t0;
}

function bindInst(name, loc) {
  const gl = T.gl;
  gl.bindBuffer(gl.ARRAY_BUFFER, T.buf[name]);
  gl.enableVertexAttribArray(loc);
  gl.vertexAttribPointer(loc, 1, gl.FLOAT, false, 0, 0);
  gl.vertexAttribDivisor(loc, 1);
}

function renderGL() {
  const gl = T.gl; if (!gl) return;
  const th = T.app.theme, E = T.E, st = T.app.state;
  const bg = rgb(th.fig_bg), ax = rgb(th.axes_bg);
  gl.disable(gl.SCISSOR_TEST); gl.viewport(0, 0, T.W, T.H);
  gl.clearColor(bg[0], bg[1], bg[2], 1); gl.clear(gl.COLOR_BUFFER_BIT);
  if (!E || !T.L) return;
  uploadLut();
  const P = T.loc, mode = st.tcol === 'ta' ? 1 : 0;
  gl.useProgram(T.prog);
  gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, T.lutTex); gl.uniform1i(P.uLut, 0);
  gl.uniform1f(P.uCmapLo, th.cmap_floor); gl.uniform1f(P.uCmapHi, th.cmap_ceiling);
  const pal = new Float32Array(24); th.categorical.forEach((h, i) => pal.set(rgb(h), 3 * i)); gl.uniform3fv(P['uPal[0]'], pal);
  gl.uniform3fv(P.uGrey, rgb(th.unassociated)); gl.uniform3fv(P.uHi3, rgb(trigColours(th).ta));
  if (mode === 0) {
    const cv = colourValues(st.tcol);
    if (E.valMode !== st.tcol) { glBuf('val', cv.v); E.valMode = st.tcol; }
    gl.uniform1f(P.uLog, cv.log ? 1 : 0); gl.uniform1f(P.uLo, cv.lo); gl.uniform1f(P.uHi, cv.hi);
  } else if (!T.buf.val) glBuf('val', new Float32Array(E.n));
  const selOn = !!T.selKey;
  gl.enable(gl.SCISSOR_TEST);
  const dpr = T.dpr;
  for (const p of T.L.panels) {
    const r = p.plot;
    const x = Math.round(r.x * dpr), w = Math.round(r.w * dpr), h = Math.round(r.h * dpr), y = T.H - Math.round((r.y + r.h) * dpr);
    gl.viewport(x, y, w, h); gl.scissor(x, y, w, h);
    gl.clearColor(ax[0], ax[1], ax[2], 1); gl.clear(gl.COLOR_BUFFER_BIT);
    const [y0, y1] = yRange(p);
    gl.uniform2f(P.uX, T.x0, T.x1); gl.uniform2f(P.uY, y0, y1);
    // hits underlay
    if (st.thits === '1' && E.hits) {
      const [s, e] = E.hits.vr[p.view];
      if (e > s) {
        gl.uniform1f(P.uMode, 2); gl.uniform3fv(P.uUni, rgb(th.fg_muted)); gl.uniform1f(P.uAlpha, 0.35);
        gl.uniform1f(P.uSize, Math.max(1.5, 2 * dpr)); gl.uniform1f(P.uDim, 1); gl.uniform1f(P.uPass, 1);
        for (const [nm, lc] of [['ht', P.aT0], ['ht', P.aT1], ['ht', P.aPk], ['hr', P.aRow], ['hz', P.aVal], ['hcat', P.aCat], ['hz', P.aEmph]]) bindInst(nm, lc);
        setOffsets(s);
        gl.drawArraysInstanced(gl.POINTS, 0, 1, e - s);
      }
    }
    const [s, e] = p.unit < 0 ? E.vr[p.view] : (E.ur[p.view].get(p.unit) || [0, 0]);
    if (e <= s) continue;
    for (const [nm, lc] of [['t0', P.aT0], ['t1', P.aT1], ['pk', P.aPk], ['row', P.aRow], ['val', P.aVal], ['cat', P.aCat], ['emph', P.aEmph]]) bindInst(nm, lc);
    setOffsets(s);
    gl.uniform1f(P.uMode, mode); gl.uniform1f(P.uDim, selOn ? 0.18 : 1); gl.uniform1f(P.uAlpha, 0.85);
    gl.uniform1f(P.uPass, 0); gl.uniform1f(P.uSize, 1);
    gl.drawArraysInstanced(gl.LINES, 0, 2, e - s);
    const zoom = (T.E.full[1] - T.E.full[0]) / Math.max(1e-6, T.x1 - T.x0);
    gl.uniform1f(P.uPass, 1); gl.uniform1f(P.uSize, Math.min(6, (2 + 0.6 * Math.log2(Math.max(1, zoom))) * dpr));
    gl.drawArraysInstanced(gl.POINTS, 0, 1, e - s);
  }
  gl.disable(gl.SCISSOR_TEST);
}

/** point every bound per-instance attribute at instance `s` */
function setOffsets(s) {
  const gl = T.gl;
  for (const nm of ['aT0', 'aT1', 'aPk', 'aRow', 'aVal', 'aCat', 'aEmph']) {
    const loc = T.loc[nm];
    const b = gl.getVertexAttrib(loc, gl.VERTEX_ATTRIB_ARRAY_BUFFER_BINDING);
    gl.bindBuffer(gl.ARRAY_BUFFER, b);
    gl.vertexAttribPointer(loc, 1, gl.FLOAT, false, 0, 4 * s);
  }
}

// ---- coordinates
function px(p, t) { const r = p.plot || p; return r.x + (t - T.x0) / (T.x1 - T.x0) * r.w; }
function pyRow(p, row) { const [y0, y1] = yRange(p); return p.plot.y + p.plot.h - (row - y0) / (y1 - y0) * p.plot.h; }
function tAt(p, x) { const r = p.plot || p; return T.x0 + (x - r.x) / r.w * (T.x1 - T.x0); }
function rowAt(p, y) { const [y0, y1] = yRange(p); return y0 + (p.plot.y + p.plot.h - y) / p.plot.h * (y1 - y0); }

function renderOverlay() {
  const g = T.ov; if (!g) return;
  const th = T.app.theme, E = T.E, C = trigColours(th);
  g.setTransform(T.dpr, 0, 0, T.dpr, 0, 0);
  g.clearRect(0, 0, T.cw, T.ch);
  if (!E || !T.L) return;
  const s = parseSel();
  const selTas = new Set(s ? (s.kind === 'ta' ? [s.i] : E.tcTas[s.i]) : []);
  if (T.L.timeline) drawTimeline(g, th, C, s, selTas);
  for (const p of T.L.panels) drawPanel(g, th, C, p, s, selTas);
  for (const h of T.L.hists) drawHist(g, th, C, h, s, selTas);
  drawLegend(g, th);
  if (T.hover && T.hover.mark) {
    const m = T.hover.mark;
    g.strokeStyle = th.fg; g.lineWidth = 1.5; g.beginPath(); g.arc(m[0], m[1], 6, 0, 2 * Math.PI); g.stroke();
  }
}

function drawPanel(g, th, C, p, s, selTas) {
  const E = T.E, r = p.plot, R = T.R;
  g.save(); g.beginPath(); g.rect(r.x, r.y, r.w, r.h); g.clip();
  // TC bands: vertical, across the whole panel
  for (let k = 0; k < E.ntc; k++) {
    const xa = px(p, E.tcT0[k]), xb = px(p, E.tcT1[k]), xc = px(p, E.tcT[k]);
    if (xb < r.x || xa > r.x + r.w) continue;
    const on = s && s.kind === 'tc' && s.i === k;
    g.fillStyle = C.tc; g.globalAlpha = on ? 0.16 : 0.07; g.fillRect(xa, r.y, xb - xa, r.h);
    g.globalAlpha = on ? 1 : 0.7; g.strokeStyle = C.tc; g.lineWidth = on ? 2 : 1; g.setLineDash([4, 3]);
    g.beginPath(); g.moveTo(xa, r.y); g.lineTo(xa, r.y + r.h); g.moveTo(xb, r.y); g.lineTo(xb, r.y + r.h); g.stroke();
    g.setLineDash([]); g.lineWidth = on ? 2 : 1.2; g.beginPath(); g.moveTo(xc, r.y); g.lineTo(xc, r.y + r.h); g.stroke();
    g.globalAlpha = 1; g.fillStyle = C.tc; g.font = '10px system-ui, sans-serif'; g.textBaseline = 'top'; g.textAlign = 'left';
    g.fillText(`TC ${k}`, Math.max(xa, r.x) + 3, r.y + 2);
  }
  // TA boxes: the activity's window x its TPs' rows, in this panel's plane (and unit)
  for (let a = 0; a < E.nta; a++) {
    for (const bx of E.taBox[a]) {
      if (bx.view !== p.view || (p.unit >= 0 && bx.unit !== p.unit)) continue;
      const xa = px(p, E.taT0[a]), xb = px(p, E.taT1[a]), ya = pyRow(p, bx.r1 + 0.5), yb = pyRow(p, bx.r0 - 0.5);
      if (xb < r.x || xa > r.x + r.w) continue;
      const on = selTas.has(a);
      g.fillStyle = C.ta; g.globalAlpha = on ? 0.12 : 0.05; g.fillRect(xa, ya, xb - xa, yb - ya);
      g.globalAlpha = on ? 1 : 0.8; g.strokeStyle = C.ta; g.lineWidth = on ? 2.5 : 1.3; g.strokeRect(xa, ya, xb - xa, yb - ya);
      g.globalAlpha = 1; g.fillStyle = C.ta; g.font = 'bold 10px system-ui, sans-serif'; g.textAlign = 'right'; g.textBaseline = 'top';
      g.fillText(`TA ${a}`, Math.min(xb, r.x + r.w) - 3, Math.max(ya, r.y) + 2);
    }
  }
  // unit boundaries along the row axis (plane panels span every unit)
  if (R && p.unit < 0) {
    g.strokeStyle = th.fg_muted; g.lineWidth = 1; g.globalAlpha = 0.6; g.setLineDash([2, 3]);
    for (const bl of R.blocks[p.view]) {
      const y = pyRow(p, bl.r0 - 0.5);
      if (y > r.y && y < r.y + r.h) { g.beginPath(); g.moveTo(r.x, y); g.lineTo(r.x + r.w, y); g.stroke(); }
    }
    g.setLineDash([]); g.globalAlpha = 1;
  }
  g.restore();
  // frame, title, axes
  g.strokeStyle = th.fg_muted; g.lineWidth = 1; g.strokeRect(r.x + 0.5, r.y + 0.5, r.w - 1, r.h - 1);
  g.fillStyle = th.fg; g.font = '11px system-ui, sans-serif'; g.textAlign = 'left'; g.textBaseline = 'alphabetic';
  const [s0, s1] = p.unit < 0 ? E.vr[p.view] : (E.ur[p.view].get(p.unit) || [0, 0]);
  g.fillText(`${p.title}   (${(s1 - s0).toLocaleString()} TPs)`, r.x, r.y - 4);
  g.font = '10px system-ui, sans-serif'; g.fillStyle = th.fg; g.strokeStyle = th.fg_muted;
  // y: channel numbers at nice row positions; unit names at block centres
  const [y0, y1] = yRange(p), list = R ? R.chOfRow[p.view] : null;
  const yt = niceTicks(y0, y1, Math.max(2, Math.round(r.h / 34)));
  g.textAlign = 'right'; g.textBaseline = 'middle';
  for (const t of yt.ticks) {
    const ri = Math.round(t), y = pyRow(p, ri);
    if (y < r.y - 1 || y > r.y + r.h + 1 || ri < p.rows[0] || ri >= p.rows[1] || (list && ri >= list.length)) continue;
    g.beginPath(); g.moveTo(r.x - 4, y); g.lineTo(r.x, y); g.stroke();
    g.fillText(String(list ? list[ri] : ri), r.x - 6, y);
  }
  if (R && p.unit < 0) {
    g.fillStyle = th.fg_muted; g.font = '9px system-ui, sans-serif';
    for (const bl of R.blocks[p.view]) {
      const ya = pyRow(p, bl.r1), yb = pyRow(p, bl.r0), ym = 0.5 * (ya + yb);
      if (ym < r.y + 6 || ym > r.y + r.h - 6 || yb - ya < 22) continue;
      g.save(); g.translate(p.cell.x + 9, ym); g.rotate(-Math.PI / 2); g.textAlign = 'center'; g.fillText(R.units[bl.unit] || '', 0, 0); g.restore();
    }
  } else {
    g.save(); g.fillStyle = th.fg_muted; g.translate(p.cell.x + 9, r.y + r.h / 2); g.rotate(-Math.PI / 2); g.textAlign = 'center';
    g.fillText('channel', 0, 0); g.restore();
  }
  if (p.xlab) drawTimeAxis(g, th, r, r.y + r.h);
}

function drawTimeAxis(g, th, r, y) {
  const xt = niceTicks(T.x0, T.x1, Math.max(3, Math.round(r.w / 90)));
  g.strokeStyle = th.fg_muted; g.fillStyle = th.fg; g.font = '10px system-ui, sans-serif'; g.textAlign = 'center'; g.textBaseline = 'top';
  for (const t of xt.ticks) {
    const x = px({ plot: r }, t); if (x < r.x - 1 || x > r.x + r.w + 1) continue;
    g.beginPath(); g.moveTo(x, y); g.lineTo(x, y + 4); g.stroke();
    g.fillText(String(Math.round(t)), x, y + 5);
  }
  g.fillStyle = th.fg_muted; g.textAlign = 'right'; g.textBaseline = 'bottom';
  const us = T.E ? T.E.tickUs : 0.5;
  g.fillText(`time [ticks of ${us} µs]`, r.x + r.w, y + PM.xlab + 2);
}

function drawTimeline(g, th, C, s, selTas) {
  const E = T.E, TL = T.L.timeline, r = TL.plot;
  const bar = 11, gap = 3;
  g.save();
  g.fillStyle = th.axes_bg; g.fillRect(r.x, r.y, r.w, r.h);
  g.beginPath(); g.rect(r.x, r.y, r.w, r.h); g.clip();
  let y = r.y + 4;
  // candidates
  g.font = '10px system-ui, sans-serif'; g.textBaseline = 'middle';
  for (let k = 0; k < E.ntc; k++) {
    const xa = px(TL, E.tcT0[k]), xb = px(TL, E.tcT1[k]), on = s && s.kind === 'tc' && s.i === k;
    g.fillStyle = C.tc; g.globalAlpha = on ? 0.9 : 0.55; g.fillRect(xa, y, Math.max(2, xb - xa), bar);
    g.globalAlpha = 1; g.fillStyle = th.axes_bg; g.fillRect(px(TL, E.tcT[k]) - 1, y, 2, bar);
    if (on) { g.strokeStyle = th.fg; g.lineWidth = 1.5; g.strokeRect(xa, y, Math.max(2, xb - xa), bar); }
    g.fillStyle = th.fg; g.textAlign = 'left'; g.fillText(`TC ${k}`, Math.max(xa, r.x) + 3, y + bar / 2);
  }
  TL.tcY = [y, y + bar];
  y += bar + gap;
  // activities, one lane per overlap level
  TL.taY0 = y;
  for (let a = 0; a < E.nta; a++) {
    const ly = y + TL.lanes.lane[a] * (bar + gap), xa = px(TL, E.taT0[a]), xb = px(TL, E.taT1[a]), on = selTas.has(a);
    g.fillStyle = C.ta; g.globalAlpha = on ? 0.9 : 0.5; g.fillRect(xa, ly, Math.max(2, xb - xa), bar);
    g.globalAlpha = 1;
    if (on) { g.strokeStyle = th.fg; g.lineWidth = 1.5; g.strokeRect(xa, ly, Math.max(2, xb - xa), bar); }
    const bx = E.taBox[a][0];
    g.fillStyle = th.fg; g.textAlign = 'left';
    g.fillText(`TA ${a}${bx && T.R ? ' · ' + T.R.units[bx.unit] + ' ' + VIEWS[bx.view] : ''}`, Math.max(xa, r.x) + 3, ly + bar / 2);
  }
  y += Math.max(1, TL.lanes.n) * (bar + gap);
  // TP rate: per pixel column bin, stacked by plane; activity TPs as a line
  if (TL.hist) {
    const hy0 = y + 2, hh = r.y + r.h - hy0 - 2;
    const nb = Math.max(10, Math.floor(r.w / 3)), cnt = [new Float32Array(nb), new Float32Array(nb), new Float32Array(nb)], inTa = new Float32Array(nb);
    const sc = nb / (T.x1 - T.x0);
    for (let i = 0; i < E.n; i++) {
      const k = Math.floor((E.pk[i] - T.x0) * sc);
      if (k < 0 || k >= nb) continue;
      const v = E.view[i]; if (v >= 0 && v < 3) cnt[v][k]++;
      if (E.ta[i] >= 0) inTa[k]++;
    }
    let mx = 1; for (let k = 0; k < nb; k++) mx = Math.max(mx, cnt[0][k] + cnt[1][k] + cnt[2][k]);
    const bw = r.w / nb;
    for (let k = 0; k < nb; k++) {
      let acc = 0;
      for (let v = 0; v < 3; v++) {
        const c = cnt[v][k]; if (!c) continue;
        const h = c / mx * hh;
        g.fillStyle = C.views[v]; g.globalAlpha = 0.7; g.fillRect(r.x + k * bw, hy0 + hh - acc - h, Math.max(1, bw - 0.5), h);
        acc += h;
      }
    }
    g.globalAlpha = 1; g.strokeStyle = C.ta; g.lineWidth = 1.5; g.beginPath();
    for (let k = 0; k < nb; k++) { const yy = hy0 + hh - inTa[k] / mx * hh; if (k) g.lineTo(r.x + (k + 0.5) * bw, yy); else g.moveTo(r.x + 0.5 * bw, yy); }
    g.stroke();
    g.fillStyle = th.fg_muted; g.textAlign = 'right'; g.textBaseline = 'top'; g.font = '9px system-ui, sans-serif';
    g.fillText(`TPs per ${(1 / sc).toFixed(1)} ticks (max ${mx})`, r.x + r.w - 3, hy0);
    TL.histY = [hy0, hy0 + hh];
  }
  g.restore();
  g.fillStyle = th.fg_muted; g.font = '10px system-ui, sans-serif'; g.textAlign = 'right'; g.textBaseline = 'middle';
  g.fillText('TC', r.x - 6, TL.tcY[0] + bar / 2);
  g.fillText('TA', r.x - 6, TL.taY0 + bar / 2);
  if (TL.histY) {
    g.fillText('TP rate', r.x - 6, 0.5 * (TL.histY[0] + TL.histY[1]));
    let lx = r.x + 4;
    g.textAlign = 'left'; g.textBaseline = 'top'; g.font = '9px system-ui, sans-serif';
    for (let v = 0; v < 3; v++) { g.fillStyle = C.views[v]; g.fillRect(lx, TL.histY[0] + 1, 8, 8); g.fillStyle = th.fg_muted; g.fillText(VIEWS[v], lx + 10, TL.histY[0]); lx += 26; }
    g.strokeStyle = C.ta; g.lineWidth = 1.5; g.beginPath(); g.moveTo(lx, TL.histY[0] + 5); g.lineTo(lx + 10, TL.histY[0] + 5); g.stroke();
    g.fillStyle = th.fg_muted; g.fillText('in a TA', lx + 13, TL.histY[0]);
  }
  g.strokeStyle = th.fg_muted; g.lineWidth = 1; g.strokeRect(r.x + 0.5, r.y + 0.5, r.w - 1, r.h - 1);
}

/** explorer histograms: the selection's TPs (dark) against all TPs of the same plane(s) (light) */
function drawHist(g, th, C, h, s, selTas) {
  const E = T.E;
  const r = { x: h.x + 46, y: h.y + 16, w: h.w - 56, h: h.h - 34 };
  g.fillStyle = th.fg; g.font = '11px system-ui, sans-serif'; g.textAlign = 'left'; g.textBaseline = 'alphabetic';
  g.fillText(h.title, r.x, h.y + 12);
  g.fillStyle = th.axes_bg; g.fillRect(r.x, r.y, r.w, r.h);
  g.strokeStyle = th.fg_muted; g.lineWidth = 1; g.strokeRect(r.x + 0.5, r.y + 0.5, r.w - 1, r.h - 1);
  const views = new Set(s ? selBoxes(s).map((b) => b.view) : [0, 1, 2]);
  const all = [], sel = [];
  const val = (i) => h.k === 0 ? E.tot[i] : h.k === 1 ? Math.log10(Math.max(E.adc[i], 1)) : E.apk[i];
  if (h.k < 3) {
    for (let i = 0; i < E.n; i++) {
      if (!views.has(E.view[i])) continue;
      const v = val(i); all.push(v); if (selTas.has(E.ta[i])) sel.push(v);
    }
  } else {               // TPs per channel, over the channels the selection (or everything) touches
    const ca = new Map(), cs = new Map();
    for (let i = 0; i < E.n; i++) {
      if (!views.has(E.view[i])) continue;
      ca.set(E.ch[i], (ca.get(E.ch[i]) || 0) + 1);
      if (selTas.has(E.ta[i])) cs.set(E.ch[i], (cs.get(E.ch[i]) || 0) + 1);
    }
    for (const [c, k] of (s ? cs : ca)) { sel.push(k); all.push(ca.get(c)); }
  }
  if (!all.length) return;
  const srt = all.slice().sort((a, b) => a - b);
  let lo = srt[0], hi = srt[Math.max(0, Math.floor(0.995 * (srt.length - 1)))];
  if (!(hi > lo)) hi = lo + 1;
  const nb = 40, ha = new Float32Array(nb), hs = new Float32Array(nb);
  const bin = (v) => Math.min(nb - 1, Math.max(0, Math.floor((v - lo) / (hi - lo) * nb)));
  for (const v of all) ha[bin(v)]++;
  for (const v of sel) hs[bin(v)]++;
  const ma = Math.max(...ha) || 1, ms = Math.max(...hs) || 1;
  const step = (arr, m, col, alpha, fill) => {
    g.globalAlpha = alpha; g.fillStyle = col; g.strokeStyle = col; g.lineWidth = 1.5; g.beginPath();
    for (let k = 0; k < nb; k++) {
      const x0 = r.x + k / nb * r.w, x1 = r.x + (k + 1) / nb * r.w, y = r.y + r.h - arr[k] / m * (r.h - 2);
      if (fill) g.fillRect(x0, y, x1 - x0 - 0.5, r.y + r.h - y);
      else { if (k) g.lineTo(x0, y); else g.moveTo(x0, y); g.lineTo(x1, y); }
    }
    if (!fill) g.stroke();
    g.globalAlpha = 1;
  };
  step(ha, ma, th.fg_muted, 0.35, true);
  if (s && sel.length) step(hs, ms, C.ta, 1, false);
  g.fillStyle = th.fg_muted; g.font = '9px system-ui, sans-serif'; g.textBaseline = 'top';
  g.textAlign = 'left'; g.fillText(fmt(lo), r.x, r.y + r.h + 2);
  g.textAlign = 'right'; g.fillText(fmt(hi), r.x + r.w, r.y + r.h + 2);
  g.fillText(s ? `sel ${sel.length} / all ${all.length}` : `${all.length}`, r.x + r.w - 2, r.y + 2);
  g.textAlign = 'right'; g.textBaseline = 'middle'; g.fillText('shape', r.x - 4, r.y + r.h / 2);
}

function drawLegend(g, th) {
  const st = T.app.state, E = T.E;
  const x = T.cw - 200, y = 4;
  if (st.tcol === 'ta') return;
  const cv = colourValues(st.tcol);
  const grd = g.createLinearGradient(x, 0, x + 120, 0);
  const css = cmapGradientCss(st.cmap, th.cmap_floor, th.cmap_ceiling);
  const stops = css.match(/#[0-9a-fA-F]{6}|rgb\([^)]*\)/g) || [];
  stops.forEach((c, i) => grd.addColorStop(i / Math.max(1, stops.length - 1), c));
  const top = T.L.timeline ? T.L.timeline.y + T.L.timeline.h + 2 : (T.L.panels[0] ? T.L.panels[0].cell.y + 2 : y);
  g.fillStyle = grd; g.fillRect(x, top, 120, 6);
  g.fillStyle = th.fg_muted; g.font = '9px system-ui, sans-serif'; g.textBaseline = 'top'; g.textAlign = 'left';
  g.fillText(fmt(cv.rlo), x, top + 7); g.textAlign = 'right'; g.fillText(fmt(cv.rhi), x + 120, top + 7);
  g.textAlign = 'left'; g.fillText(COLOURS.find((c) => c[0] === st.tcol)[1] + (cv.log ? ' (log)' : ''), x + 124, top);
  void E;
}

// ----------------------------------------------------------------------------- pointer
function panelAt(x, y) {
  for (const p of T.L ? T.L.panels : []) { const r = p.plot; if (x >= r.x && x <= r.x + r.w && y >= r.y && y <= r.y + r.h) return p; }
  return null;
}
function inTimeline(x, y) {
  const TL = T.L && T.L.timeline; if (!TL) return false;
  const r = TL.plot; return x >= r.x && x <= r.x + r.w && y >= r.y && y <= r.y + r.h;
}

function pick(x, y) {
  const E = T.E; if (!E || !T.L) return null;
  if (inTimeline(x, y)) {
    const TL = T.L.timeline, t = tAt(TL, x);
    if (y >= TL.tcY[0] && y <= TL.tcY[1]) for (let k = 0; k < E.ntc; k++) if (t >= E.tcT0[k] && t <= E.tcT1[k]) return { kind: 'tc', i: k };
    for (let a = 0; a < E.nta; a++) {
      const ly = TL.taY0 + TL.lanes.lane[a] * 14;
      if (y >= ly && y <= ly + 11 && t >= E.taT0[a] && t <= E.taT1[a]) return { kind: 'ta', i: a };
    }
    return null;
  }
  const p = panelAt(x, y); if (!p) return null;
  // nearest TP within 6 px
  const [s, e] = p.unit < 0 ? E.vr[p.view] : (E.ur[p.view].get(p.unit) || [0, 0]);
  const kx = p.plot.w / (T.x1 - T.x0), [y0, y1] = yRange(p), ky = p.plot.h / (y1 - y0);
  const t = tAt(p, x), rw = rowAt(p, y);
  let best = -1, bd = 36;
  for (let j = s; j < e; j++) {
    const i = E.order[j];
    const a = E.t0[i], b = a + E.tot[i];
    const dx = (t < a ? a - t : t > b ? t - b : 0) * kx, dy = (E.row[i] - rw) * ky;
    if (dx > 6 || dy > 6 || dy < -6) continue;
    const d = dx * dx + dy * dy;
    if (d < bd) { bd = d; best = i; }
  }
  if (best >= 0) return { kind: 'tp', i: best, p, mark: [px(p, E.pk[best]), pyRow(p, E.row[best])] };
  for (let a = 0; a < E.nta; a++) for (const bx of E.taBox[a]) {
    if (bx.view !== p.view || (p.unit >= 0 && bx.unit !== p.unit)) continue;
    if (t >= E.taT0[a] && t <= E.taT1[a] && rw >= bx.r0 - 0.5 && rw <= bx.r1 + 0.5) return { kind: 'ta', i: a, p };
  }
  for (let k = 0; k < E.ntc; k++) if (t >= E.tcT0[k] && t <= E.tcT1[k]) return { kind: 'tc', i: k, p };
  return { kind: 'none', p };
}

function tipFor(h) {
  const E = T.E;
  if (h.kind === 'tp') {
    const i = h.i, v = VIEWS[E.view[i]] || '?', place = ['unique wire', 'wire of the matching hit', 'ambiguous wire', 'no active wire'];
    const pl = E.b.get('tp/place');
    const lines = [`TP  channel ${E.ch[i]} (${v}${T.R ? ', ' + T.R.units[E.unit[i]] : ''})`,
      `start ${E.t0[i].toFixed(1)}  peak ${E.pk[i].toFixed(1)}  ToT ${E.tot[i].toFixed(1)} ticks`,
      `${(E.pk[i] * E.tickUs).toFixed(2)} µs   DTS ${E.meta.ticks_per_dts ? Math.round(E.pk[i] / E.meta.ticks_per_dts) : '?'}`,
      `ADC sum ${fmt(E.adc[i])}   peak ${fmt(E.apk[i])}`];
    if (pl) lines.push(`2-D placement: ${place[pl[i]] || '?'}`);
    lines.push(E.ta[i] >= 0 ? `in TA ${E.ta[i]}${E.taTc[E.ta[i]] >= 0 ? '  →  TC ' + E.taTc[E.ta[i]] : ''}` : 'in no activity');
    return lines.join('\n');
  }
  if (h.kind === 'ta' || h.kind === 'tc') return describe(h).replace(/  +/g, '\n');
  return '';
}

function hideTip() { if (T.tip) T.tip.style.display = 'none'; }

function setHover(h, ex, ey) {
  const prev = T.hover;
  T.hover = h && h.kind !== 'none' ? h : null;
  if (T.hover) {
    T.tip.textContent = tipFor(T.hover);
    T.tip.style.display = 'block';
    const cr = T.container.getBoundingClientRect();
    let tx = ex + 14, ty = ey + HEAD + 14;
    const tw = T.tip.offsetWidth, tht = T.tip.offsetHeight;
    if (tx + tw > cr.width - 4) tx = ex - tw - 10;
    if (ty + tht > cr.height - 4) ty = ey + HEAD - tht - 10;
    T.tip.style.left = `${Math.max(2, tx)}px`; T.tip.style.top = `${Math.max(2, ty)}px`;
  } else hideTip();
  const tp = T.hover && T.hover.kind === 'tp' ? T.hover.i : null, ptp = prev && prev.kind === 'tp' ? prev.i : null;
  if (tp !== ptp && effLayout() === 'reco') T.app.emit('trigger', { type: 'hover', tp });
  schedule();
}

function localXY(e) { const r = T.ovc.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; }

function onWheel(e) {
  if (!T.E) return;
  e.preventDefault();
  const [x, y] = localXY(e), f = Math.exp(Math.sign(e.deltaY) * 0.15);
  const p = panelAt(x, y);
  if (e.shiftKey && p) {           // channel zoom of this panel
    const [y0, y1] = yRange(p), c = rowAt(p, y);
    T.ys[p.key] = [c - (c - y0) * f, c + (y1 - c) * f];
  } else {
    const ref = p || (inTimeline(x, y) ? T.L.timeline : null);
    if (!ref) return;
    const c = tAt(ref, x);
    T.x0 = c - (c - T.x0) * f; T.x1 = c + (T.x1 - c) * f;
  }
  schedule();
}
function onDown(e) {
  if (!T.E) return;
  const [x, y] = localXY(e);
  T.drag = { x, y, x0: T.x0, x1: T.x1, p: panelAt(x, y), moved: false };
  if (T.drag.p) T.drag.ys = yRange(T.drag.p).slice();
  T.ovc.setPointerCapture(e.pointerId);
}
function onMove(e) {
  const [x, y] = localXY(e);
  const d = T.drag;
  if (d) {
    const ref = d.p || (T.L.timeline && inTimeline(d.x, d.y) ? T.L.timeline : null);
    if (Math.abs(x - d.x) + Math.abs(y - d.y) > 3) d.moved = true;
    if (ref && d.moved) {
      const r = ref.plot || ref, dt = (x - d.x) / r.w * (d.x1 - d.x0);
      T.x0 = d.x0 - dt; T.x1 = d.x1 - dt;
      if (d.p && e.shiftKey) { const dr = (y - d.y) / d.p.plot.h * (d.ys[1] - d.ys[0]); T.ys[d.p.key] = [d.ys[0] + dr, d.ys[1] + dr]; }
      schedule();
    }
    return;
  }
  setHover(pick(x, y), x, y);
}
function onUp(e) {
  const d = T.drag; T.drag = null;
  try { T.ovc.releasePointerCapture(e.pointerId); } catch (err) { /* not captured */ }
  if (!d || d.moved) return;
  const [x, y] = localXY(e), h = pick(x, y);
  if (!h) return;
  if (h.kind === 'tp') select(T.E.ta[h.i] >= 0 ? `ta:${T.E.ta[h.i]}` : '', false);
  else if (h.kind === 'ta' || h.kind === 'tc') select(`${h.kind}:${h.i}`, effLayout() === 'explorer');
  else select('', false);
}
function onKey(e) {
  if (!T.visible || !T.E) return;
  const tag = (document.activeElement && document.activeElement.tagName) || '';
  if (['INPUT', 'SELECT', 'TEXTAREA'].includes(tag)) return;
  if (e.key === 'n') { stepSelection(1); e.preventDefault(); }
  else if (e.key === 'p') { stepSelection(-1); e.preventDefault(); }
  else if (e.key === 'z') { fitSelection(); e.preventDefault(); }
  else if (e.key === 'Escape' && T.selKey) select('', false);
}

// ----------------------------------------------------------------------------- module interface
function resize() {
  const r = T.container.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  T.cw = Math.max(1, Math.round(r.width)); T.ch = Math.max(1, Math.round(r.height - HEAD));
  T.dpr = dpr; T.W = Math.round(T.cw * dpr); T.H = Math.round(T.ch * dpr);
  for (const c of [T.glc, T.ovc]) { c.width = T.W; c.height = T.H; }
  layout();
  schedule();
}

function onState(state, changed) {
  if (!T.ready) return;
  T.sel.lay.value = effLayout();
  T.sel.col.value = state.tcol; T.sel.plane.value = state.tplane; T.sel.hits.checked = state.thits === '1';
  if (changed.has('tsel')) {
    applySelection(true);
    if (effLayout() === 'explorer') { layout(); fitSelection(true); }
  }
  if (['tlay', 'view', 'tplane'].some((k) => changed.has(k))) { layout(); if (effLayout() === 'explorer' && parseSel()) fitSelection(true); }
  if (['tsel', 'tlay', 'view', 'tplane', 'tcol', 'thits', 'theme', 'cmap'].some((k) => changed.has(k))) schedule();
}

function setVisible(v) {
  T.visible = !!v;
  if (v) { resize(); if (T.selKey) T.app.emit('trigger', { type: 'select', sel: parseSel(), fit: false }); }
  else hideTip();
}

function getCanvas() {
  if (!T.ready || !T.W) return null;
  render();
  const c = document.createElement('canvas'); c.width = T.W; c.height = T.H + Math.round(HEAD * T.dpr);
  const g = c.getContext('2d');
  g.fillStyle = T.app.theme.fig_bg; g.fillRect(0, 0, c.width, c.height);
  g.drawImage(T.glc, 0, Math.round(HEAD * T.dpr)); g.drawImage(T.ovc, 0, Math.round(HEAD * T.dpr));
  return c;
}

export default {
  name: 'viewtrigger',
  init, setGeometry, setEvent, onState, setVisible, resize, getCanvas,
  /** test hooks */
  debugState() { return { layout: effLayout(), panels: T.L ? T.L.panels.map((p) => ({ key: p.key, title: p.title, plot: p.plot })) : [], x: [T.x0, T.x1], sel: T.selKey, lastMs: T.lastMs, n: T.E ? T.E.n : 0 }; },
  /** css px (relative to the stage) of TP i in panel k, or null */
  debugTpPx(k, i) { const p = T.L && T.L.panels[k]; if (!p || !T.E) return null; return [px(p, T.E.pk[i]), pyRow(p, T.E.row[i]) + HEAD]; },
  debugPick(x, y) { const h = pick(x, y - HEAD); return h ? { kind: h.kind, i: h.i } : null; },
};
