// exports.js -- "PDF" and "GIF" toolbar buttons (WP-H).
//
// Both call GET /api/export, which renders with the Python matplotlib renderers (the same
// ones the old Dash app used) and returns the file: a 300 dpi vector PDF of the current view,
// or an animated GIF (3-D: 360-degree rotation from the current camera; Light and
// Charge+Light: the light animation with the current light settings).
//
// The module is imported by main.js before `window.__evd` exists, so it waits for the app
// and the toolbar, injects the buttons next to #btn-shot, and reads everything from
// app.state (plus app.views.view3d.cameraInfo?.() for the camera).

const GIF_VIEWS = ['3d', 'light', 'charge+light'];
const LIGHT_KEYS = ['lclock', 'lfill', 'lframes', 'lfps', 'lpers', 'lwlo', 'lwhi', 'lt0', 'lt'];

let app = null;
let busy = false;
let pdfBtn = null, gifBtn = null;

function buildQuery(kind) {
  const s = app.state;
  const q = new URLSearchParams();
  q.set('file', s.file);
  q.set('entry', String(s.entry));
  if (s.hits) q.set('hits', s.hits);
  if (s.flash) q.set('flash', s.flash);
  q.set('kind', kind);
  q.set('view', s.view);
  if (kind === 'gif' && s.view === '3d') q.set('rotstep', '4');
  for (const k of ['color', 'scale', 'cmap', 'theme']) if (s[k]) q.set(k, s[k]);
  // The Python renderers know only the old on/off "radiologicals" switch (the truth filters of the
  // viewer are not applied to PDF/PNG/GIF exports): any visible non-interaction origin turns it on.
  const ovOn = Object.keys(s.ov || {}).filter((k) => s.ov[k]);
  if (app.truth && app.truth.radiologicalsOn()) ovOn.push('rad');
  q.set('ov', ovOn.join(','));
  for (const k of ['merge2d', 'space2d']) if (s[k]) q.set(k, s[k]);
  for (const k of LIGHT_KEYS) if (s[k] !== undefined && s[k] !== '') q.set(k, s[k]);
  let cam = null;
  try {
    const v3 = app.views && app.views.view3d;
    cam = v3 && v3.cameraInfo ? v3.cameraInfo() : null;
  } catch (e) { cam = null; }
  if (cam && cam.position && cam.target) {
    q.set('campos', cam.position.map((x) => +x.toFixed(2)).join(','));
    q.set('camtgt', cam.target.map((x) => +x.toFixed(2)).join(','));
    if (cam.up) q.set('camup', cam.up.map((x) => +x.toFixed(4)).join(','));
  }
  return q;
}

function filenameOf(resp, fallback) {
  const cd = resp.headers.get('Content-Disposition') || '';
  const m = /filename="?([^";]+)"?/.exec(cd);
  return m ? m[1] : fallback;
}

function expectedSeconds(kind) {
  const s = app.state;
  if (kind === 'gif') return s.view === '3d' ? 15 : 20;
  return 5;
}

async function run(kind) {
  if (busy || !app) return;
  busy = true; refresh();
  const label = kind.toUpperCase();
  const t0 = performance.now();
  const spin = ['|', '/', '-', '\\'];
  let i = 0;
  const tick = () => app.status(`${spin[i++ % 4]} rendering ${label} on the server ... `
    + `${((performance.now() - t0) / 1000).toFixed(0)} s (typically ~${expectedSeconds(kind)} s)`);
  tick();
  const timer = setInterval(tick, 400);
  try {
    const resp = await fetch('/api/export?' + buildQuery(kind).toString(), { cache: 'no-store' });
    if (!resp.ok) {
      let msg = `HTTP ${resp.status}`;
      try { msg = (await resp.json()).error || msg; } catch (e) { /* not JSON */ }
      throw new Error(msg);
    }
    const blob = await resp.blob();
    const name = filenameOf(resp, `pylarevd_export.${kind}`);
    const note = resp.headers.get('X-Export-Note');
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = name;
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 10000);
    app.status(`${label} saved: ${name} (${(blob.size / 1e6).toFixed(2)} MB, `
      + `${((performance.now() - t0) / 1000).toFixed(1)} s)` + (note ? ` - ${note}` : ''));
  } catch (e) {
    console.error(e);
    app.status(`${label} export failed: ${e.message || e}`, true);
  } finally {
    clearInterval(timer);
    busy = false; refresh();
  }
}

function refresh() {
  if (!app || !pdfBtn) return;
  const sample = new URLSearchParams(location.search).has('sample');
  const view = app.state.view;
  const noServer = sample ? 'exports need the Python server (not available in static sample mode)' : '';
  pdfBtn.disabled = busy || !!noServer;
  pdfBtn.title = noServer || (view === '2d+3d'
    ? 'vector PDF (300 dpi) of the 2-D panels (2D+3D exports the 2-D side)'
    : 'vector PDF (300 dpi) of the current view, rendered on the server');
  const gifOk = GIF_VIEWS.includes(view) && !sample;
  gifBtn.disabled = busy || !gifOk;
  gifBtn.title = noServer || (!GIF_VIEWS.includes(view)
    ? 'GIF is available for the 3D (rotation), Light and Charge+Light views'
    : view === '3d' ? 'rotating GIF from the current camera (about 15 s)'
      : 'light animation GIF with the current light settings (10-30 s)');
}

function install() {
  const shot = document.getElementById('btn-shot');
  if (!shot || !window.__evd || !window.__evd.state) return false;
  app = window.__evd;
  pdfBtn = document.createElement('button');
  pdfBtn.id = 'btn-pdf'; pdfBtn.textContent = 'PDF';
  gifBtn = document.createElement('button');
  gifBtn.id = 'btn-gif'; gifBtn.textContent = 'GIF';
  shot.after(pdfBtn, gifBtn);
  pdfBtn.addEventListener('click', () => run('pdf'));
  gifBtn.addEventListener('click', () => run('gif'));
  app.on('state', (s, changed) => { if (changed.has('view')) refresh(); });
  refresh();
  return true;
}

(function waitForApp(tries = 0) {
  if (install()) return;
  if (tries < 400) setTimeout(() => waitForApp(tries + 1), 50);
})();
