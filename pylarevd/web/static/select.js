// select.js -- shared selection model + info card (WP-I).
//
// One selection for every view: app.state.sel = {kind:'pfp'|'slice', id, pinned, family} or null.
//   * hover (views call app.hoverObject({pfp, slice}, shiftKey)) sets a TRANSIENT selection,
//     debounced ~30 ms, cleared 120 ms after the pointer leaves every object (so the pointer can
//     travel to the info card);
//   * click (app.clickObject) PINS it; clicking the same object again, clicking nothing, or Esc unpins;
//   * "select by" particle|slice|none (toolbar, URL param `selby`) decides what a hover/click selects;
//     holding Shift while hovering selects the other kind; "none" turns highlighting off entirely;
//   * "family" (URL param `fam`) extends a PFParticle selection to its whole hierarchy: from the
//     top-level parent down through all daughters;
//   * keys (pinned particle): [ parent, ] first daughter, , / . previous/next sibling, f family, Esc clear.
// Only a PINNED selection is in the URL: `pfp=<row>` or `slice=<row>` (rows of the pfp/* and slice/*
// tables, the same ids as hits/group/pfparticle and hits/group/slice).
//
// Views read `app.selection` (null or {kind, groupKind:'pfparticle'|'slice', mask:Uint8Array,
// n, count, id, stamp}) and listen with app.on('selection', fn). The mask is indexed by the group id of
// `groupKind`, so a view only needs a per-object group id attribute and a small id->flag texture.
//
// Tables come from the pfp/* and slice/* arrays of the bundle. If the PFParticle collection was
// unreadable (available.pfp false) a minimal table is synthesised from the hit/space-point group ids.

const HOVER_MS = 30, CLEAR_MS = 120;
const CARD_VIEWS = new Set(['2d', '3d', '2d+3d', 'charge+light']);
const PLANES = ['U', 'V', 'Z'];
const SYM = {
  11: 'e-', '-11': 'e+', 13: 'mu-', '-13': 'mu+', 22: 'gamma', 211: 'pi+', '-211': 'pi-', 111: 'pi0',
  2212: 'p', 2112: 'n', 321: 'K+', '-321': 'K-', 130: 'K0L', 12: 'nu_e', 14: 'nu_mu', 16: 'nu_tau',
  '-12': 'anti-nu_e', '-14': 'anti-nu_mu', '-16': 'anti-nu_tau', 1000180400: 'Ar40', 1000180390: 'Ar39',
};
export function pdgSymbol(p) {
  if (SYM[p] !== undefined) return SYM[p];
  if (p > 1000000000) return `nucleus ${p}`;
  return `pdg ${p}`;
}

const esc = (s) => String(s).replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));

export function initSelection(app) {
  const $ = (id) => document.getElementById(id);
  const state = app.state;
  state.sel = null;
  app.selection = null;
  app.registerParam('selby', 'particle');
  app.registerParam('fam', '0');
  app.registerParam('pfp', '');
  app.registerParam('slice', '');
  app.registerParam('dim', '30');      // visibility of non-selected objects, percent

  let T = null;               // tables of the current bundle
  let hoverTimer = 0, clearTimer = 0, pendKey = '';
  let last = { ids: null, shift: false };
  let firstEvent = true, stamp = 0, cardHeld = false;

  // ------------------------------------------------------------------ tables
  function buildTables(b) {
    if (!b) return null;
    const g = (n) => b.get(n);
    const meta = b.meta || {};
    const t = { b, hasPfp: !!g('pfp/id'), hasSlice: !!g('slice/id') };
    const gp = g('hits/group/pfparticle'), gsp = g('sp/group/pfparticle');
    const gs = g('hits/group/slice'), gss = g('sp/group/slice');
    const maxOf = (...arrs) => { let m = -1; for (const a of arrs) if (a) for (let i = 0; i < a.length; i++) if (a[i] > m) m = a[i]; return m + 1; };
    t.groupsPfp = !!(gp || gsp); t.groupsSlice = !!(gs || gss);
    if (t.hasPfp) {
      t.nPfp = g('pfp/id').length;
      for (const k of ['id', 'pdg', 'parent', 'top', 'slice', 'track', 'shower', 'n_hits', 'n_sp', 'score', 'vertex',
        'true_mc', 'true_pdg', 'true_tid', 'n_hits_plane', 'daughter_offsets', 'daughters']) t[k] = g('pfp/' + k);
    } else {
      t.nPfp = Math.max(maxOf(gp, gsp), meta.groups && meta.groups.pfparticle ? meta.groups.pfparticle : 0);
    }
    t.nSlice = t.hasSlice ? g('slice/id').length : Math.max(maxOf(gs, gss), meta.groups && meta.groups.slice ? meta.groups.slice : 0);
    if (t.hasSlice) for (const k of ['id', 'n_hits', 'n_hits_plane', 'n_sp', 'n_pfp', 'top', 'nu', 'vertex', 'true_mc', 'true_pdg', 'true_tid']) t['s_' + k] = g('slice/' + k);
    // synthesised counts when the exporter had no tables
    const count = (ids, n) => { const c = new Int32Array(n); if (ids) for (let i = 0; i < ids.length; i++) if (ids[i] >= 0 && ids[i] < n) c[ids[i]]++; return c; };
    if (!t.hasPfp && t.nPfp) { t.n_hits = count(gp, t.nPfp); t.n_sp = count(gsp, t.nPfp); }
    if (!t.hasSlice && t.nSlice) { t.s_n_hits = count(gs, t.nSlice); t.s_n_sp = count(gss, t.nSlice); }
    // PFParticles per slice
    t.slicePfps = null;
    if (t.hasPfp && t.hasSlice) {
      t.slicePfps = Array.from({ length: t.nSlice }, () => []);
      for (let p = 0; p < t.nPfp; p++) { const s = t.slice[p]; if (s >= 0 && s < t.nSlice) t.slicePfps[s].push(p); }
    }
    return t;
  }

  const daughters = (p) => (T.hasPfp ? Array.from(T.daughters.subarray(T.daughter_offsets[p], T.daughter_offsets[p + 1])) : []);
  const typeOf = (p) => {
    if (!T.hasPfp) return 'particle';
    const a = Math.abs(T.pdg[p]);
    return a === 13 ? 'track-like' : a === 11 ? 'shower-like' : (a === 12 || a === 14 || a === 16) ? 'neutrino' : `pdg ${T.pdg[p]}`;
  };

  // ------------------------------------------------------------- resolution
  function resolve(sel) {
    if (!sel || !T) return null;
    const isPfp = sel.kind === 'pfp';
    const n = isPfp ? T.nPfp : T.nSlice;
    if (!(sel.id >= 0 && sel.id < n)) return null;
    const mask = new Uint8Array(n);
    let count = 0, root = sel.id;
    if (isPfp && sel.family && T.hasPfp) {
      root = T.top[sel.id];
      const stack = [root];
      while (stack.length) {
        const p = stack.pop();
        if (mask[p]) continue;
        mask[p] = 1; count++;
        for (const d of daughters(p)) stack.push(d);
      }
    } else { mask[sel.id] = 1; count = 1; }
    return { sel, kind: sel.kind, groupKind: isPfp ? 'pfparticle' : 'slice', mask, n, count, id: sel.id, root, stamp: ++stamp };
  }

  function emit() {
    app.emit('selection', app.selection);
  }

  // ------------------------------------------------------------------ apply
  function apply(sel, { silentUrl = false } = {}) {
    state.sel = sel;
    app.selection = resolve(sel);
    if (!app.selection) state.sel = sel = null;
    renderCard();
    emit();
    if (!silentUrl) {
      const pinned = sel && sel.pinned;
      app.setState({ pfp: pinned && sel.kind === 'pfp' ? String(sel.id) : '', slice: pinned && sel.kind === 'slice' ? String(sel.id) : '' });
    }
  }

  const sameSel = (a, b) => a === b || (a && b && a.kind === b.kind && a.id === b.id && a.pinned === b.pinned && a.family === b.family);
  const familyOn = () => state.fam === '1';
  const selOff = () => state.selby === 'none';
  /** Visibility (0..1) of everything outside the selection: URL/toolbar `dim`, in percent. */
  app.selDim = () => { const d = parseFloat(state.dim); return Number.isFinite(d) ? Math.min(1, Math.max(0, d / 100)) : 0.3; };
  const byKind = (shift) => (selOff() ? null : (state.selby === 'slice') !== !!shift ? 'slice' : 'pfp');
  const selbyValue = (v) => (v === 'slice' || v === 'none' ? v : 'particle');

  function target(ids, shift) {
    if (!ids || !T) return null;
    const kind = byKind(shift);
    if (!kind) return null;
    const id = kind === 'pfp' ? ids.pfp : ids.slice;
    if (id === undefined || id === null || id < 0) return null;
    return { kind, id };
  }

  // ------------------------------------------------------------- public API
  /** Select an object (programmatic / card links). sel = {kind, id, pinned?, family?}. */
  app.select = (sel) => {
    clearTimeout(hoverTimer); clearTimeout(clearTimer); pendKey = '';
    if (!sel) return app.clearSelection();
    apply({ kind: sel.kind, id: sel.id, pinned: !!sel.pinned, family: sel.family != null ? !!sel.family : familyOn() });
  };
  app.clearSelection = () => {
    clearTimeout(hoverTimer); clearTimeout(clearTimer); pendKey = '';
    if (state.sel || state.pfp || state.slice) apply(null);
  };
  app.selectionTables = () => T;
  /** true if reco vertex `v` (row of vtx/*) belongs to the selection, or there is no selection / no way to tell */
  app.vertexSelected = (v) => {
    const s = app.selection;
    if (!s || !T || !T.hasPfp || !T.vertex) return true;
    if (!T.vtxPfp) {
      const n = T.b.rows('vtx/xyz');
      T.vtxPfp = new Int32Array(n).fill(-1);
      for (let p = T.nPfp - 1; p >= 0; p--) { const q = T.vertex[p]; if (q >= 0 && q < n) T.vtxPfp[q] = p; }
    }
    const p = T.vtxPfp[v];
    if (p < 0) return false;
    return s.groupKind === 'pfparticle' ? !!s.mask[p] : (T.slice[p] >= 0 && !!s.mask[T.slice[p]]);
  };

  app.hoverObject = (ids, shift = false) => {
    last = { ids, shift };
    if (state.sel && state.sel.pinned) return;
    const t = target(ids, shift);
    if (!t) {
      pendKey = '';
      clearTimeout(hoverTimer);
      if (state.sel && !clearTimer && !cardHeld) clearTimer = setTimeout(() => { clearTimer = 0; if (!state.sel || !state.sel.pinned) apply(null, { silentUrl: true }); }, CLEAR_MS);
      return;
    }
    clearTimeout(clearTimer); clearTimer = 0;
    const key = `${t.kind}:${t.id}:${familyOn()}`;
    if (state.sel && !state.sel.pinned && state.sel.kind === t.kind && state.sel.id === t.id && state.sel.family === familyOn()) { pendKey = ''; clearTimeout(hoverTimer); return; }
    if (key === pendKey) return;
    pendKey = key;
    clearTimeout(hoverTimer);
    hoverTimer = setTimeout(() => {
      pendKey = '';
      if (state.sel && state.sel.pinned) return;
      apply({ kind: t.kind, id: t.id, pinned: false, family: familyOn() }, { silentUrl: true });
    }, HOVER_MS);
  };

  app.clickObject = (ids, shift = false) => {
    const t = target(ids, shift);
    clearTimeout(hoverTimer); clearTimeout(clearTimer); clearTimer = 0; pendKey = '';
    if (!t) { if (state.sel || state.pfp || state.slice) apply(null); return; }
    const s = state.sel;
    if (s && s.pinned && s.kind === t.kind && s.id === t.id) { apply(null); return; }
    apply({ kind: t.kind, id: t.id, pinned: true, family: familyOn() });
  };

  /** Re-pin to a related object (parent, daughter, sibling): keyboard and card links. */
  function pinTo(kind, id) { if (T && id >= 0) apply({ kind, id, pinned: true, family: familyOn() }); }

  // ----------------------------------------------- reset on a new event bundle
  // Called by main.js after app.bundle changed and BEFORE the views' setEvent.
  app.resetSelection = (bundle) => {
    T = buildTables(bundle);
    clearTimeout(hoverTimer); clearTimeout(clearTimer); clearTimer = 0; pendKey = '';
    let sel = null;
    if (firstEvent && T && !selOff()) {
      const pf = parseInt(state.pfp, 10), sl = parseInt(state.slice, 10);
      if (Number.isFinite(pf) && pf >= 0 && pf < T.nPfp) sel = { kind: 'pfp', id: pf, pinned: true, family: familyOn() };
      else if (Number.isFinite(sl) && sl >= 0 && sl < T.nSlice) sel = { kind: 'slice', id: sl, pinned: true, family: false };
    }
    firstEvent = false;
    const hadUrl = !!(state.pfp || state.slice);
    apply(sel, { silentUrl: !hadUrl || !!sel });
    syncToolbar();
  };

  // ------------------------------------------------------------------ toolbar
  function syncToolbar() {
    const by = $('sel-by'), cb = $('cb-family');
    if (by) by.value = selbyValue(state.selby);
    if (cb) { cb.checked = familyOn(); cb.disabled = (!!T && !T.hasPfp) || selOff(); }
    const dm = $('sel-dim'), dv = $('sel-dim-v');
    if (dm) { dm.value = String(Math.round(app.selDim() * 100)); dm.disabled = selOff(); }
    if (dv) dv.textContent = `${Math.round(app.selDim() * 100)}%`;
    const grp = $('sel-group');
    if (grp) grp.hidden = !T || (!T.groupsPfp && !T.groupsSlice);
  }
  function bindToolbar() {
    const by = $('sel-by'), cb = $('cb-family');
    if (by) by.addEventListener('change', () => { app.setState({ selby: selbyValue(by.value) }); app.clearSelection(); });
    const dm = $('sel-dim');
    if (dm) dm.addEventListener('input', () => app.setState({ dim: dm.value }));
    if (cb) cb.addEventListener('change', () => { app.setState({ fam: cb.checked ? '1' : '0' }); refamily(); });
    app.on('state', (s, ch) => { if (ch.has('selby') || ch.has('fam') || ch.has('dim')) syncToolbar(); if (ch.has('view')) renderCard(); });
    syncToolbar();
  }
  function refamily() {
    const s = state.sel;
    if (s && s.kind === 'pfp') apply({ ...s, family: familyOn() }, { silentUrl: true });
  }

  // ----------------------------------------------------------------- keyboard
  window.addEventListener('keydown', (e) => {
    const tag = e.target && e.target.tagName;
    if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA' || e.ctrlKey || e.metaKey || e.altKey) return;
    if (e.key === 'Shift') { if (!e.repeat && last.ids) app.hoverObject(last.ids, true); return; }
    const s = state.sel;
    if (e.key === 'Escape') { if (s || state.pfp || state.slice) { app.clearSelection(); e.preventDefault(); } return; }
    if (e.key === 'f' && T && T.hasPfp && !selOff()) { app.setState({ fam: familyOn() ? '0' : '1' }); refamily(); syncToolbar(); return; }
    if (!s || s.kind !== 'pfp' || !T || !T.hasPfp) return;
    let to = -1;
    if (e.key === '[') to = T.parent[s.id];
    else if (e.key === ']') { const d = daughters(s.id); if (d.length) to = d[0]; }
    else if (e.key === ',' || e.key === '.') {
      const par = T.parent[s.id];
      const sib = par >= 0 ? daughters(par) : Array.from({ length: T.nPfp }, (_, i) => i).filter((i) => T.parent[i] < 0);
      const i = sib.indexOf(s.id);
      if (i >= 0 && sib.length > 1) to = sib[(i + (e.key === '.' ? 1 : sib.length - 1)) % sib.length];
    } else return;
    e.preventDefault();
    if (to >= 0) pinTo('pfp', to);
  });
  window.addEventListener('keyup', (e) => { if (e.key === 'Shift' && last.ids) app.hoverObject(last.ids, false); });

  // --------------------------------------------------------------------- card
  let card = null;
  function cardSide() { try { return localStorage.getItem('evdCardSide') || 'left'; } catch (e) { return 'left'; } }
  function buildCard() {
    card = $('sel-card');
    if (!card) return;
    card.addEventListener('pointerenter', () => { cardHeld = true; clearTimeout(clearTimer); clearTimer = 0; });
    card.addEventListener('pointerleave', () => {
      cardHeld = false;
      if (state.sel && !state.sel.pinned) clearTimer = setTimeout(() => { clearTimer = 0; if (!state.sel || !state.sel.pinned) apply(null, { silentUrl: true }); }, CLEAR_MS);
    });
    card.addEventListener('click', (e) => {
      const el = e.target.closest('[data-pfp],[data-slice],[data-act]');
      if (!el) return;
      e.preventDefault();
      if (el.dataset.pfp !== undefined) pinTo('pfp', parseInt(el.dataset.pfp, 10));
      else if (el.dataset.slice !== undefined) pinTo('slice', parseInt(el.dataset.slice, 10));
      else if (el.dataset.act === 'close') app.clearSelection();
      else if (el.dataset.act === 'focus') { const v = app.views.view3d; if (v && v.focusSelection) v.focusSelection(); }
      else if (el.dataset.act === 'pin' && state.sel) { pinTo(state.sel.kind, state.sel.id); }
      else if (el.dataset.act === 'side') { try { localStorage.setItem('evdCardSide', cardSide() === 'left' ? 'right' : 'left'); } catch (x) { /* none */ } renderCard(); }
    });
  }

  const link = (p) => `<a href="#" data-pfp="${p}" title="${esc(typeOf(p))}">${T.hasPfp ? T.id[p] : p}${T.hasPfp ? ` <i>${esc(typeOf(p))}</i>` : ''}</a>`;
  const planes = (arr, i) => (arr ? PLANES.map((n, k) => `${n} ${arr[3 * i + k]}`).join(' / ') : '');
  const f1 = (v) => (Number.isFinite(v) ? v.toFixed(1) : '-');

  function trueLine(mc, pdg, tid) {
    if (tid < 0 || !pdg) return '';
    return `<div class="row"><b>true</b> ${esc(pdgSymbol(pdg))} (track id ${tid})</div>`;
  }

  function pfpCard(s) {
    const p = s.id, b = T.b;
    let h = `<div class="hd"><span>${T.hasPfp ? 'Particle' : 'Group'} ${T.hasPfp ? T.id[p] : p}${T.hasPfp ? ` <i>${esc(typeOf(p))}</i>` : ''}</span>`;
    h += hdButtons(s) + '</div>';
    if (T.hasPfp) {
      const sc = T.score[p];
      h += `<div class="row">${Number.isFinite(sc) ? `track score <b>${sc.toFixed(2)}</b> &middot; ` : ''}slice ${T.slice[p] >= 0 ? `<a href="#" data-slice="${T.slice[p]}">${T.slice[p]}</a>` : '-'}</div>`;
      const par = T.parent[p], dd = daughters(p);
      h += `<div class="row rel"><b>parent</b> ${par >= 0 ? link(par) : '<i>none (top level)</i>'}</div>`;
      h += `<div class="row rel"><b>daughters</b> ${dd.length ? dd.slice(0, 12).map(link).join(' ') + (dd.length > 12 ? ` <i>+${dd.length - 12} more</i>` : '') : '<i>none</i>'}</div>`;
    }
    const nh = T.hasPfp ? T.n_hits[p] : (T.n_hits ? T.n_hits[p] : 0);
    h += `<div class="row"><b>hits</b> ${nh}${T.hasPfp ? ` (${planes(T.n_hits_plane, p)})` : ''}</div>`;
    h += `<div class="row"><b>space points</b> ${T.n_sp ? T.n_sp[p] : 0}</div>`;
    if (T.hasPfp) {
      const ti = T.track[p], si = T.shower[p];
      if (ti >= 0) {
        const pts = b.get('tracks/points'), off = b.get('tracks/offsets');
        let L = 0;
        if (pts && off) for (let k = off[ti]; k < off[ti + 1] - 1; k++) L += Math.hypot(pts[3 * k + 3] - pts[3 * k], pts[3 * k + 4] - pts[3 * k + 1], pts[3 * k + 5] - pts[3 * k + 2]);
        h += `<div class="row"><b>track</b> length ${L.toFixed(1)} cm, ${off ? off[ti + 1] - off[ti] : 0} points</div>`;
      }
      if (si >= 0) {
        const len = b.get('showers/length'), ang = b.get('showers/angle');
        h += `<div class="row"><b>shower</b> length ${len ? len[si].toFixed(1) : '-'} cm${ang ? `, opening ${(ang[si] * 180 / Math.PI).toFixed(1)}&deg;` : ''}</div>`;
      }
      h += trueLine(T.true_mc[p], T.true_pdg[p], T.true_tid[p]);
    }
    if (s.family && T.hasPfp) h += `<div class="row note">family: ${app.selection ? app.selection.count : 1} particles from top-level ${link(T.top[p])}</div>`;
    return h;
  }

  function sliceCard(s) {
    const i = s.id;
    let h = `<div class="hd"><span>Slice ${T.hasSlice ? T.s_id[i] : i}${T.hasSlice && T.s_nu[i] ? ' <i>neutrino hypothesis</i>' : ''}</span>${hdButtons(s)}</div>`;
    const nh = T.s_n_hits ? T.s_n_hits[i] : 0;
    h += `<div class="row"><b>hits</b> ${nh}${T.hasSlice ? ` (${planes(T.s_n_hits_plane, i)})` : ''}</div>`;
    h += `<div class="row"><b>space points</b> ${T.s_n_sp ? T.s_n_sp[i] : 0}</div>`;
    if (T.slicePfps) {
      const ps = T.slicePfps[i];
      const cnt = {};
      for (const p of ps) cnt[typeOf(p)] = (cnt[typeOf(p)] || 0) + 1;
      h += `<div class="row"><b>particles</b> ${ps.length}${ps.length ? ' (' + Object.entries(cnt).map(([k, v]) => `${v} ${esc(k)}`).join(', ') + ')' : ''}</div>`;
      const top = T.s_top[i];
      h += `<div class="row"><b>${T.s_nu[i] ? 'neutrino particle' : 'top-level particle'}</b> ${top >= 0 ? link(top) : '<i>none</i>'}</div>`;
      const v = T.s_vertex;
      if (v && Number.isFinite(v[3 * i])) h += `<div class="row"><b>vertex</b> (${f1(v[3 * i])}, ${f1(v[3 * i + 1])}, ${f1(v[3 * i + 2])}) cm</div>`;
      h += `<div class="row rel"><b>members</b> ${ps.slice(0, 24).map(link).join(' ')}${ps.length > 24 ? ` <i>+${ps.length - 24} more</i>` : ''}</div>`;
      h += trueLine(T.s_true_mc[i], T.s_true_pdg[i], T.s_true_tid[i]);
    } else h += '<div class="row note">no PFParticle table for this file</div>';
    return h;
  }

  function hdButtons(s) {
    return `<span class="btns">${s.pinned ? '' : '<a href="#" data-act="pin" title="pin (click)">pin</a>'}` +
      `<a href="#" data-act="focus" title="recentre the 3-D orbit on this selection (double-click in 3-D)">focus</a>` +
      `<a href="#" data-act="side" title="move the card to the other side">&#8644;</a>` +
      `<a href="#" data-act="close" title="clear (Esc)">&times;</a></span>`;
  }

  function renderCard() {
    if (!card) return;
    const s = state.sel;
    const show = !!(s && T && CARD_VIEWS.has(state.view));
    card.hidden = !show;
    if (!show) { card.innerHTML = ''; return; }
    card.classList.toggle('pinned', !!s.pinned);
    card.classList.toggle('right', cardSide() === 'right');
    card.innerHTML = s.kind === 'pfp' ? pfpCard(s) : sliceCard(s);
  }

  buildCard();
  bindToolbar();
}
