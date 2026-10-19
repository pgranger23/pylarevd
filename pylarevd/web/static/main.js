// main.js -- pylarevd web viewer: app state, navigation, URL state, view registry.
//
// ============================== VIEW MODULE INTERFACE ==============================
// A view is an ES module whose default export is a plain object:
//
//   export default {
//     name: 'view2d',
//     init(container, app) {},          // ONCE at startup. container = the <section class="panel">
//                                       // reserved for this view. Build DOM / canvas here.
//     setGeometry(geometry) {},         // optional. A geometry Bundle (kind "geometry") became
//                                       // current (new file or first load). Called before setEvent.
//     setEvent(bundle, geometry) {},    // a new event Bundle (kind "event") is current.
//                                       // bundle === null means "clear" (loading / error).
//     onState(state, changed) {},       // ANY state change (including the first, with every key in
//                                       // `changed`). `changed` is a Set of top-level state keys:
//                                       // file entry hits flash view color scale cmap theme ov +
//                                       // registered params. Must be cheap and must not rebuild
//                                       // geometry unless a data-defining key changed.
//     setVisible(visible) {},           // optional. Panel shown/hidden by the view selector.
//                                       // main calls resize() right after showing it.
//     resize() {},                      // optional. Container size changed.
//     getCanvas() {},                   // optional. Canvas used by the screenshot button
//                                       // (render synchronously before returning).
//   };
//
// `app` (passed to init, also imported as the `app` export of this file) offers:
//   app.state                         current state object (read-only for views; use setState)
//   app.setState(patch)               merge, update URL, notify every view and subscriber
//   app.registerParam(name, default)  declare an extra string state key that is saved in / restored
//                                     from the URL (e.g. viewlight registers 'tmode', 't', 'play').
//                                     Call from init(); the value is then in app.state[name].
//   app.on(type, fn) -> unsubscribe   type 'state' fn(state, changed) | 'event' fn(bundle, geometry)
//                                     | 'geometry' fn(geometry)
//   app.bundle / app.geometry         the current event / geometry Bundles (may be null)
//   app.theme                         current theme colours (colormaps.js THEMES entry)
//   app.views                         registry {view3d, view2d, viewlight} -> module objects, so
//                                     viewlight can call app.views.view3d.setPdColors(rgb)
//   app.status(text, isError)         set the status line
//   app.requestLayout()               ask main to re-run resize() on visible views
// Bundle API (data.js): bundle.meta, bundle.get(name) -> typed array|null, bundle.has(name),
// bundle.rows(name), bundle.names(prefix). Typed arrays are zero-copy views.
// =====================================================================================

import { serverApi, sampleApi } from './data.js';
import { THEMES, CMAP_NAMES } from './colormaps.js';
import view3d from './view3d.js';
import view2d from './view2d.js';
import viewlight from './viewlight.js';
import viewoptical from './viewoptical.js';
import './exports.js';
import { initSelection } from './select.js';

const OVERLAYS = ['truth', 'rad', 'dep', 'tracks', 'showers', 'pvtx', 'vtx', 'labels', 'pd'];
const OV_DEFAULT = ['truth', 'tracks', 'showers', 'pvtx', 'vtx', 'pd'];
const VIEW_PANELS = {
  '3d': ['view3d'], '2d': ['view2d'], 'light': ['viewlight'],
  'charge+light': ['view3d', 'viewlight'], '2d+3d': ['view2d', 'view3d'],
  'optical': ['viewoptical'], 'flash3d': ['view3d'],
};
const PANEL_IDS = { view3d: 'panel-3d', view2d: 'panel-2d', viewlight: 'panel-light', viewoptical: 'panel-optical' };

const DEFAULTS = {
  file: '', entry: 0, hits: '', flash: '', view: '3d', color: 'charge', scale: 'auto',
  cmap: 'turbo', theme: 'dark',
};

const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
const sampleName = params.get('sample');
const api = sampleName ? sampleApi(sampleName) : serverApi();

const listeners = { state: [], event: [], geometry: [], selection: [] };
const extraParams = {};   // name -> default (string)

const state = { ...DEFAULTS, ov: Object.fromEntries(OVERLAYS.map((o) => [o, OV_DEFAULT.includes(o)])) };

export const app = {
  state, api,
  bundle: null, geometry: null,
  views: { view3d, view2d, viewlight, viewoptical },
  get theme() { return THEMES[state.theme] || THEMES.dark; },
  setState, registerParam, status, requestLayout,
  emit(type, ...args) {
    for (const fn of (listeners[type] || []).slice()) { try { fn(...args); } catch (e) { console.error(`listener ${type}:`, e); } }
  },
  on(type, fn) {
    listeners[type].push(fn);
    return () => { const i = listeners[type].indexOf(fn); if (i >= 0) listeners[type].splice(i, 1); };
  },
};

// ------------------------------------------------------------------ URL state
function readUrl() {
  const p = new URLSearchParams(location.search);
  for (const k of ['file', 'hits', 'flash']) if (p.has(k)) state[k] = p.get(k);
  if (p.has('entry')) { const e = parseInt(p.get('entry'), 10); if (Number.isFinite(e) && e >= 0) state.entry = e; }
  if (p.has('view') && VIEW_PANELS[p.get('view')]) state.view = p.get('view');
  if (p.has('color')) state.color = p.get('color');
  if (p.has('scale') && ['auto', 'log', 'linear'].includes(p.get('scale'))) state.scale = p.get('scale');
  if (p.has('cmap') && CMAP_NAMES.includes(p.get('cmap'))) state.cmap = p.get('cmap');
  if (p.has('theme') && THEMES[p.get('theme')]) state.theme = p.get('theme');
  if (p.has('ov')) {
    const on = new Set(p.get('ov').split(',').filter(Boolean));
    for (const o of OVERLAYS) state.ov[o] = on.has(o);
  }
  for (const [name] of Object.entries(extraParams)) if (p.has(name)) state[name] = p.get(name);
}

function urlFor() {
  const p = new URLSearchParams();
  if (sampleName) p.set('sample', sampleName);
  else if (state.file) p.set('file', state.file);
  p.set('entry', String(state.entry));
  for (const k of ['hits', 'flash']) if (state[k]) p.set(k, state[k]);
  for (const k of ['view', 'color', 'scale', 'cmap', 'theme']) if (state[k] !== DEFAULTS[k]) p.set(k, state[k]);
  const on = OVERLAYS.filter((o) => state.ov[o]);
  if (on.join() !== OV_DEFAULT.join()) p.set('ov', on.join(','));
  for (const [name, def] of Object.entries(extraParams)) if (state[name] !== def) p.set(name, state[name]);
  return `${location.pathname}?${p.toString()}`;
}

function writeUrl() {
  try { history.replaceState(null, '', urlFor()); } catch (e) { /* sandboxed contexts */ }
}

// ------------------------------------------------------------------ state core
function registerParam(name, def) {
  extraParams[name] = String(def);
  if (!(name in state)) state[name] = String(def);
  const p = new URLSearchParams(location.search);
  if (p.has(name)) state[name] = p.get(name);
}

function setState(patch) {
  const changed = new Set();
  for (const [k, v] of Object.entries(patch)) {
    if (k === 'ov') {
      for (const [o, val] of Object.entries(v)) if (state.ov[o] !== !!val) { state.ov[o] = !!val; changed.add('ov'); }
    } else if (state[k] !== v) { state[k] = v; changed.add(k); }
  }
  if (!changed.size) return;
  afterChange(changed);
}

function afterChange(changed) {
  if (changed.has('theme')) document.documentElement.dataset.theme = state.theme;
  if (changed.has('view')) applyLayout();
  writeUrl();
  syncControls();
  for (const v of Object.values(app.views)) guard(v, 'onState', state, changed);
  for (const fn of listeners.state.slice()) fn(state, changed);
}

function guard(mod, fn, ...args) {
  if (!mod || typeof mod[fn] !== 'function') return;
  try { mod[fn](...args); } catch (e) { console.error(`${mod.name || 'view'}.${fn}:`, e); status(`${mod.name || 'view'}.${fn}: ${e.message}`, true); }
}

function status(text, isError = false) {
  $('status-text').textContent = text;
  $('status').classList.toggle('error', !!isError);
}

// -------------------------------------------------------------------- layout
let layoutQueued = false;
function requestLayout() {
  if (layoutQueued) return;
  layoutQueued = true;
  requestAnimationFrame(() => {
    layoutQueued = false;
    const vis = VIEW_PANELS[state.view];
    for (const name of vis) guard(app.views[name], 'resize');
  });
}

function applyLayout() {
  $('main').dataset.view = state.view;
  const vis = new Set(VIEW_PANELS[state.view]);
  for (const [name, mod] of Object.entries(app.views)) guard(mod, 'setVisible', vis.has(name));
  requestLayout();
}
window.addEventListener('resize', requestLayout);

// ------------------------------------------------------------------ controls
function fillSelect(sel, items, current) {
  // items: [value, label]; keep `current` selectable even if the list lacks it
  sel.replaceChildren();
  const have = new Set(items.map((i) => i[0]));
  const all = (current && !have.has(current)) ? [[current, current], ...items] : items;
  for (const [v, l] of all) { const o = document.createElement('option'); o.value = v; o.textContent = l; sel.appendChild(o); }
  sel.value = current;
}

function syncControls() {
  $('sel-view').value = state.view;
  $('sel-color').value = state.color;
  $('sel-scale').value = state.scale;
  $('sel-cmap').value = state.cmap;
  $('sel-theme').value = state.theme;
  for (const cb of document.querySelectorAll('[data-ov]')) cb.checked = !!state.ov[cb.dataset.ov];
  if (state.file && $('sel-file').value !== state.file) $('sel-file').value = state.file;
  if (state.hits && $('sel-hits').value !== state.hits) $('sel-hits').value = state.hits;
  if (state.flash && $('sel-flash').value !== state.flash) $('sel-flash').value = state.flash;
}

function bindControls() {
  fillSelect($('sel-cmap'), CMAP_NAMES.map((n) => [n, n]), state.cmap);
  const bind = (id, key) => $(id).addEventListener('change', (e) => setState({ [key]: e.target.value }));
  bind('sel-view', 'view'); bind('sel-color', 'color'); bind('sel-scale', 'scale');
  bind('sel-cmap', 'cmap'); bind('sel-theme', 'theme');
  for (const cb of document.querySelectorAll('[data-ov]')) {
    cb.addEventListener('change', () => setState({ ov: { [cb.dataset.ov]: cb.checked } }));
  }
  $('sel-file').addEventListener('change', (e) => openFile(e.target.value, 0));
  $('sel-hits').addEventListener('change', (e) => { setState({ hits: e.target.value }); loadEvent(); });
  $('sel-flash').addEventListener('change', (e) => { setState({ flash: e.target.value }); loadEvent(); });
  $('btn-prev').addEventListener('click', () => step(-1));
  $('btn-next').addEventListener('click', () => step(1));
  $('inp-jump').addEventListener('keydown', (e) => {
    if (e.key !== 'Enter') return;
    const i = parseInt(e.target.value, 10);
    if (Number.isFinite(i)) gotoIndex(i);
  });
  $('inp-open').addEventListener('keydown', (e) => { if (e.key === 'Enter') openByPath(); });
  $('btn-open').addEventListener('click', openByPath);
  $('btn-reset').addEventListener('click', () => guard(app.views.view3d, 'resetView'));
  $('btn-shot').addEventListener('click', screenshot);
  $('btn-link').addEventListener('click', async () => {
    const url = location.origin + urlFor();
    try { await navigator.clipboard.writeText(url); status('link copied: ' + url); } catch (e) { status(url); }
  });
  window.addEventListener('keydown', (e) => {
    const t = e.target && e.target.tagName;
    if (t === 'INPUT' || t === 'SELECT' || t === 'TEXTAREA' || e.ctrlKey || e.metaKey || e.altKey) return;
    if (e.key === 'ArrowLeft') { step(-1); e.preventDefault(); }
    else if (e.key === 'ArrowRight') { step(1); e.preventDefault(); }
  });
}

// ------------------------------------------------------------- open by path
function openMsg(text, isError) {
  const el = $('open-msg');
  el.textContent = text || '';
  el.classList.toggle('error', !!isError);
  el.hidden = !text;
}

async function openByPath() {
  const path = $('inp-open').value.trim();
  if (!path || $('btn-open').disabled) return;
  openMsg('opening ...', false);
  try {
    const r = await fetch('/api/open', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ path }),
    });
    let body = null;
    try { body = await r.json(); } catch (e) { /* not JSON */ }
    if (!r.ok) {
      const msg = (body && body.error) || `HTTP ${r.status}`;
      if (r.status === 403) setOpenAllowed(false, msg);
      openMsg(msg, true);
      return;
    }
    files = await api.files();
    fillSelect($('sel-file'), files.map((f) => [f.path, f.name || f.path]), body.path);
    $('inp-open').value = '';
    openMsg('', false);
    await openFile(body.path, 0);
  } catch (e) {
    openMsg(String(e.message || e), true);
  }
}

function setOpenAllowed(ok, why) {
  $('inp-open').disabled = !ok; $('btn-open').disabled = !ok;
  const tip = ok ? 'absolute path of a .root file on the server, Enter to open' : (why || 'opening files by path is disabled on this server');
  $('inp-open').title = tip; $('btn-open').title = tip;
}

async function initOpenBox() {
  if (sampleName) { $('open-group').hidden = true; return; }
  try {
    const r = await fetch('/api/status');
    if (r.ok) {
      const st = await r.json();
      if (st.allow_open === false) setOpenAllowed(false, 'opening files by path is disabled when the server is bound beyond localhost (restart with --allow-remote-open)');
    }
  } catch (e) { /* attempt anyway; a 403 will disable the box */ }
}

// ---------------------------------------------------------------- navigation
let files = [];
let events = [];
let loadToken = 0;
let abortCtl = null;

function currentIndex() { return events.findIndex((e) => e.entry === state.entry); }

function updateEventLabel() {
  const i = currentIndex();
  const ev = events[i];
  $('ev-label').textContent = ev
    ? `${ev.run}:${ev.subrun}:${ev.event}  (${i + 1}/${events.length})` : '-';
  $('inp-jump').max = Math.max(0, events.length - 1);
  if (i >= 0) $('inp-jump').value = i;
}

function step(d) { gotoIndex(currentIndex() + d); }

function gotoIndex(i) {
  if (!events.length) return;
  i = Math.max(0, Math.min(events.length - 1, i));
  const entry = events[i].entry;
  if (entry === state.entry && app.bundle) return;
  setState({ entry });
  loadEvent();
}

async function openFile(path, entry) {
  const tok = ++loadToken;
  status(`opening ${path} ...`);
  try {
    if (abortCtl) abortCtl.abort();
    abortCtl = new AbortController();
    const sig = abortCtl.signal;
    const [evs, geo] = await Promise.all([api.events(path, sig), api.geometry(path, sig)]);
    if (tok !== loadToken) return;
    events = evs;
    app.geometry = geo;
    const wantEntry = entry != null ? entry : state.entry;
    const e = events.find((x) => x.entry === wantEntry) || events[0];
    // changing file resets the per-file selections
    const fileChanged = !!state.file && state.file !== path;
    setState({ file: path, entry: e ? e.entry : 0, ...(fileChanged && entry === 0 ? { hits: '', flash: '' } : {}) });
    for (const fn of listeners.geometry.slice()) fn(geo);
    for (const v of Object.values(app.views)) guard(v, 'setGeometry', geo);
    updateEventLabel();
    await loadEvent();
  } catch (e) {
    if (e.name === 'AbortError') return;
    console.error(e); status(String(e.message || e), true);
  }
}

async function loadEvent() {
  const tok = ++loadToken;
  if (abortCtl) abortCtl.abort();
  abortCtl = new AbortController();
  const sig = abortCtl.signal;
  const t0 = performance.now();
  status(`loading entry ${state.entry} ...`);
  try {
    const bundle = await api.event(state.file, state.entry, state.hits, state.flash, sig);
    if (tok !== loadToken) return;
    const t1 = performance.now();
    app.bundle = bundle;
    guard(app, 'resetSelection', bundle);       // selection.js: new tables, drop/restore the selection
    updateEventLabel();
    const meta = bundle.meta;
    fillSelect($('sel-hits'), (meta.hit_tags || []).map((t) => [t, t]), meta.hit_tag || state.hits);
    fillSelect($('sel-flash'), (meta.flash_tags || []).map((t) => [t, t]), meta.flash_tag || state.flash);
    // the server resolves defaults; the state follows what we actually got
    const patch = {};
    if (meta.hit_tag && meta.hit_tag !== state.hits) patch.hits = meta.hit_tag;
    if (meta.flash_tag && meta.flash_tag !== state.flash) patch.flash = meta.flash_tag;
    if (Object.keys(patch).length) setState(patch);
    setSummary(meta.summary || []);
    const t2 = performance.now();
    for (const v of Object.values(app.views)) guard(v, 'setEvent', bundle, app.geometry);
    for (const fn of listeners.event.slice()) fn(bundle, app.geometry);
    const t3 = performance.now();
    requestAnimationFrame(() => requestAnimationFrame(() => {
      const t4 = performance.now();
      window.__evdTiming = {
        fetch: bundle.timing ? bundle.timing.fetch : t1 - t0, parse: bundle.timing ? bundle.timing.parse : 0,
        setEvent: t3 - t2, firstPaintAfterLoad: t4 - t1, MB: bundle.byteLength / 1e6,
      };
      status(`entry ${state.entry}: fetch ${window.__evdTiming.fetch.toFixed(0)} ms, parse ${window.__evdTiming.parse.toFixed(0)} ms, ` +
        `build ${window.__evdTiming.setEvent.toFixed(0)} ms, paint ${window.__evdTiming.firstPaintAfterLoad.toFixed(0)} ms ` +
        `(${window.__evdTiming.MB.toFixed(1)} MB)`);
    }));
  } catch (e) {
    if (e.name === 'AbortError') return;
    console.error(e); status(String(e.message || e), true);
  }
}

// ------------------------------------------------------------------ summary bar
function applySummaryOpen() {
  const open = state.sum === '1';
  $('summary-bar').classList.toggle('collapsed', !open);
  $('sum-toggle').firstChild.textContent = open ? '\u25be ' : '\u25b8 ';
  $('sum-toggle').title = open ? 'hide the full event summary' : 'show the full event summary';
  requestLayout();
}
function setSummary(lines) {
  $('summary').textContent = lines.join('\n');
  $('sum-head').textContent = lines.length ? lines[0] : '';
  $('summary-bar').classList.toggle('empty', !lines.length);
  applySummaryOpen();
}

// ---------------------------------------------------------------- screenshot
function screenshot() {
  const main = $('main').getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const out = document.createElement('canvas');
  out.width = Math.round(main.width * dpr); out.height = Math.round(main.height * dpr);
  const g = out.getContext('2d');
  g.fillStyle = app.theme.fig_bg; g.fillRect(0, 0, out.width, out.height);
  let any = false;
  for (const name of VIEW_PANELS[state.view]) {
    const mod = app.views[name];
    let c = null;
    try { c = mod && mod.getCanvas && mod.getCanvas(); } catch (e) { console.error(e); }
    if (!c) continue;
    const r = $(PANEL_IDS[name]).getBoundingClientRect();
    g.drawImage(c, (r.left - main.left) * dpr, (r.top - main.top) * dpr, r.width * dpr, r.height * dpr);
    any = true;
  }
  if (!any) { status('nothing to capture', true); return; }
  out.toBlob((blob) => {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    const ev = events[currentIndex()];
    a.download = `pylarevd_${state.view.replace('+', '_')}_${ev ? `${ev.run}_${ev.subrun}_${ev.event}` : 'event'}.png`;
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 5000);
  }, 'image/png');
}

// ---------------------------------------------------------------------- boot
async function boot() {
  for (const [name, mod] of Object.entries(app.views)) {
    if (!mod.name) mod.name = name;
    guard(mod, 'init', $(PANEL_IDS[name]), app);   // modules may registerParam here
  }
  readUrl();                                        // after registerParam so extra keys restore
  document.documentElement.dataset.theme = state.theme;
  bindControls();
  initOpenBox();
  $('sum-toggle').addEventListener('click', () => { setState({ sum: state.sum === '1' ? '0' : '1' }); });
  app.on('state', (s, ch) => { if (ch.has('sum')) applySummaryOpen(); });
  applySummaryOpen();
  syncControls();
  applyLayout();
  const all = new Set(['file', 'entry', 'hits', 'flash', 'view', 'color', 'scale', 'cmap', 'theme', 'ov', ...Object.keys(extraParams)]);
  for (const v of Object.values(app.views)) guard(v, 'onState', state, all);
  try {
    files = await api.files();
    fillSelect($('sel-file'), files.map((f) => [f.path, f.name || f.path]), '');
    if (!files.length) { status('no files available', true); return; }
    const f = files.find((x) => x.path === state.file) || files[0];
    $('sel-file').value = f.path;
    await openFile(f.path, f.path === state.file ? state.entry : 0);
  } catch (e) {
    console.error(e); status(String(e.message || e), true);
  }
}

registerParam('sum', '0');   // summary bar open (1) / collapsed (0)
initSelection(app);          // selection model + info card (select.js); registers selby/fam/pfp/slice
boot();
window.__evd = app;   // debugging / tests
