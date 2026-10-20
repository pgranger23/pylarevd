// truth.js -- true-particle filter model and the "truth" drawer (WP-T).
//
// WHAT IT DOES
//   * Decides which true trajectories / deposits are drawn. Filtered-out truth is hidden, not dimmed.
//     Visible(trajectory) = species ON  and  origin ON  and  not individually hidden
//     (the toolbar "truth" / "deposits" checkboxes stay the master switches, handled by the views).
//     species = the PDG_CLASSES (mu e gamma p n pi K nu other); origin = "interaction"
//     (mc/from_interaction == 1) plus one entry per generator for the particles that do not descend from
//     the interaction (mc/generator, meta.mc_generators). Defaults reproduce the old look: interaction
//     visible, every other origin hidden, every species visible. Without a true vertex nothing is "the
//     interaction": the origins are then the generators (or one "all particles" entry) and all are on.
//   * Publishes the result as ONE per-trajectory byte per mc/* row, `model.flags`
//         0 = hidden, 1 = visible, 2 = visible and highlighted (selected true particle, or the true
//         particle matched to the selected PFParticle / slice)
//     in a (texW x texH) R8 layout (texW = 2048, row r at texel (r & 2047, r >> 11)). The views upload it
//     as a texture and look it up in the vertex shader, so a filter change never rebuilds geometry.
//     `model.version` changes whenever the bytes change, `model.stamp` only when visibility changes (not
//     the highlight); `model.fbVis` (18 floats, index cls + 9 * from_interaction) is the visibility of the
//     deposits that belong to no mc/* trajectory (dep/mc == -1).
//   * app.on('truth', fn(model)) fires after every change. app.truth is the API (below).
//   * Drawer (right side, opened by the "truth" button): tab "Categories" (species, origins, counts) and
//     tab "Particles" (lazy tree with tri-state checkboxes, search, reset).
//
// URL parameters (registered here): tsp = hidden species names; tog = origin switches that differ from
// the defaults ("+Ar39GenInLAr" shown, "-interaction" hidden; "+*" / "-*" = every origin on / off first); thide = individually hidden particles, first
// loaded event only (track ids are per event): "12s" = track 12 and its whole subtree, "5-120,300" = single
// particles as track-id ranges.
//
// app.truth:  model, radiologicalsOn(), rowOfTid(tid), kids(row), subtree(row), describe(row),
//             siblings(row), depositCount(row), triState(row), openDrawer(tab), closeDrawer().

import { PDG_CLASSES, pdgClass, pdgSymbol } from './colormaps.js';

const N_SP = PDG_CLASSES.length;
const TEX_W = 2048;
const PAGE = 200;
const MAX_MATCH = 100;           // matches listed per origin
const URL_MAX = 3000;
const ALIAS = {
  11: 'electron', '-11': 'positron', 13: 'muon', '-13': 'antimuon muon', 22: 'photon', 2212: 'proton', '-2212': 'antiproton',
  2112: 'neutron', '-2112': 'antineutron', 211: 'pion', '-211': 'pion', 111: 'pion pizero', 321: 'kaon', '-321': 'kaon',
  130: 'kaon', 310: 'kaon', 311: 'kaon', 12: 'neutrino', 14: 'neutrino', 16: 'neutrino', '-12': 'neutrino antineutrino',
  '-14': 'neutrino antineutrino', '-16': 'neutrino antineutrino',
};

const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const fmtN = (n) => Number(n).toLocaleString('en-US');

function parseList(s) { return String(s || '').split(',').map((x) => x.trim()).filter(Boolean); }

/** "5-120,300" <-> sorted unique ints */
function parseRanges(s) {
  const out = [];
  for (const tok of parseList(s)) {
    const m = /^(-?\d+)(?:-(\d+))?$/.exec(tok);
    if (!m) continue;
    const a = parseInt(m[1], 10), b = m[2] !== undefined ? parseInt(m[2], 10) : a;
    for (let v = a; v <= b && out.length < 5e6; v++) out.push(v);
  }
  return out;
}
function encodeRanges(sorted) {
  const parts = [];
  for (let i = 0; i < sorted.length;) {
    let j = i;
    while (j + 1 < sorted.length && sorted[j + 1] === sorted[j] + 1) j++;
    parts.push(j > i ? `${sorted[i]}-${sorted[j]}` : String(sorted[i]));
    i = j + 1;
  }
  return parts.join(',');
}

export function initTruth(app) {
  const $ = (id) => document.getElementById(id);
  const state = app.state;
  app.registerParam('tsp', '');
  app.registerParam('tog', '');
  app.registerParam('thide', '');

  let M = null;                                   // model of the current event (null: no true trajectories)
  const spHidden = new Set(parseList(state.tsp)); // species names the user switched off
  const togDiff = new Map();                      // origin key -> bool, only where it differs from the default
  let togAll = null;                              // null: per-origin defaults; true / false: every origin on / off
  for (const tok of parseList(state.tog)) {
    if (tok === '+*') togAll = true;
    else if (tok === '-*') togAll = false;
    else if (tok[0] === '+') togDiff.set(tok.slice(1), true); else togDiff.set(tok[0] === '-' ? tok.slice(1) : tok, false);
  }
  const defEff = (o) => (togAll === null ? !!o.def : togAll);
  let firstEvent = true;
  let urlTimer = 0;

  const ui = {
    open: false, tab: 'cat', expanded: new Set(), groups: new Set(), pages: new Map(), search: '', hits: null,
    capped: false,
  };

  // ------------------------------------------------------------------ model
  function buildModel(b) {
    const pdg = b.get('mc/pdg');
    if (!pdg || !pdg.length) return null;
    const n = pdg.length, fi = b.get('mc/from_interaction'), gen = b.get('mc/generator');
    const meta = b.meta || {}, gnames = (meta.mc_generators || []).map(String), nG = gnames.length;
    const split = !!(meta.truth && meta.truth.has_split && fi);
    // slot: 0 interaction, 1 unknown/other, 2+g generator g
    const slot = new Uint8Array(n), cntSlot = new Int32Array(nG + 2);
    for (let i = 0; i < n; i++) {
      const s = (split && fi[i]) ? 0 : (gen ? (gen[i] < 0 || gen[i] >= nG ? 1 : 2 + gen[i]) : 1);
      slot[i] = s; cntSlot[s]++;
    }
    const origins = [];
    for (let s = 0; s < nG + 2; s++) {
      if (!cntSlot[s]) continue;
      let key, label;
      if (s === 0) { key = 'interaction'; label = 'interaction'; }
      else if (s === 1) {
        if (gen) { key = 'unknown'; label = 'unknown generator'; } else if (split) { key = 'other'; label = 'not from the interaction'; } else { key = 'all'; label = 'all particles'; }
      } else { key = gnames[s - 2]; label = gnames[s - 2]; }
      origins.push({ slot: s, key, label, count: cntSlot[s], def: s === 0 || !split });
    }
    origins.sort((a, c) => (a.slot === 0 ? -1 : c.slot === 0 ? 1 : c.count - a.count));
    const slotToOid = new Int16Array(nG + 2).fill(-1);
    origins.forEach((o, k) => { o.id = k; slotToOid[o.slot] = k; });
    const nO = origins.length;
    const cls = new Uint8Array(n), oid = new Uint8Array(n), key = new Uint16Array(n);
    const cntKey = new Int32Array(nO * N_SP), spTotal = new Int32Array(N_SP);
    for (let i = 0; i < n; i++) {
      const c = pdgClass(pdg[i]), o = slotToOid[slot[i]];
      cls[i] = c; oid[i] = o; key[i] = o * N_SP + c;
      cntKey[key[i]]++; spTotal[c]++;
    }
    const texH = Math.max(1, Math.ceil(n / TEX_W));
    return {
      b, n, pdg, tid: b.get('mc/track_id'), mom: b.get('mc/mother'), off: b.get('mc/offsets'), pts: b.get('mc/points'),
      proc: b.get('mc/process'), procNames: meta.mc_processes || [], fi, gen, gnames, split, hasGen: !!gen,
      origins, nO, cls, oid, key, cntKey, spTotal, nSp: N_SP,
      texW: TEX_W, texH, flags: new Uint8Array(TEX_W * texH),
      spOn: new Uint8Array(N_SP).fill(1), orgOn: new Uint8Array(nO), fbVis: new Float32Array(2 * N_SP),
      hid: null, nHid: 0, hiRows: [], selActive: false,
      stamp: 0, version: 0, tree: null, tidTable: null, tidMap: null, depCount: null,
      visKey: null, nVisible: 0,
    };
  }

  function applyChoices() {
    PDG_CLASSES.forEach((c, i) => { M.spOn[i] = spHidden.has(c.name) ? 0 : 1; });
    for (const o of M.origins) M.orgOn[o.id] = togDiff.has(o.key) ? (togDiff.get(o.key) ? 1 : 0) : (defEff(o) ? 1 : 0);
  }

  function updateFbVis() {
    // deposits that are not part of any mc/* trajectory: species from dep/pdg, origin = interaction or "anything else"
    let intOn, otherOn;
    if (M.split) {
      const oi = M.origins.find((o) => o.slot === 0);
      intOn = oi ? M.orgOn[oi.id] : 0;
      otherOn = M.origins.some((o) => o.slot !== 0 && M.orgOn[o.id]) ? 1 : 0;
    } else { intOn = otherOn = M.orgOn.some((v) => v) ? 1 : 0; }
    for (let c = 0; c < N_SP; c++) {
      M.fbVis[c] = M.spOn[c] && otherOn ? 1 : 0;              // from_interaction == 0
      M.fbVis[N_SP + c] = M.spOn[c] && intOn ? 1 : 0;         // from_interaction == 1
    }
  }

  function recompute() {
    if (!M) return;
    const tab = new Uint8Array(M.nO * N_SP);
    for (let o = 0; o < M.nO; o++) for (let c = 0; c < N_SP; c++) tab[o * N_SP + c] = M.spOn[c] && M.orgOn[o] ? 1 : 0;
    const f = M.flags, key = M.key, hid = M.hid, n = M.n;
    if (hid) for (let i = 0; i < n; i++) f[i] = hid[i] ? 0 : tab[key[i]];
    else for (let i = 0; i < n; i++) f[i] = tab[key[i]];
    for (const r of M.hiRows) if (f[r] === 1) f[r] = 2;
    updateFbVis();
    M.stamp++; M.version++;
    M.tab = tab;
    M.visKey = null;
  }

  /** visible trajectories per (origin, species) key, computed on demand (drawer / footer) */
  function visCounts() {
    if (M.visKey) return M.visKey;
    const v = new Int32Array(M.nO * N_SP);
    if (!M.hid) { for (let k = 0; k < v.length; k++) v[k] = M.tab[k] ? M.cntKey[k] : 0; }
    else { const key = M.key, hid = M.hid, tab = M.tab; for (let i = 0; i < M.n; i++) if (!hid[i] && tab[key[i]]) v[key[i]]++; }
    M.visKey = v;
    let t = 0; for (let k = 0; k < v.length; k++) t += v[k];
    M.nVisible = t;
    return v;
  }

  function emitTruth() { app.emit('truth', M); }

  // -------------------------------------------------------- track-id / tree structure
  function ensureTidIndex() {
    if (M.tidTable || M.tidMap) return;
    const n = M.n, tid = M.tid;
    let mx = 0;
    for (let i = 0; i < n; i++) if (tid[i] > mx) mx = tid[i];
    if (mx <= 8 * n + 1000000) {
      const t = new Int32Array(mx + 1).fill(-1);
      for (let i = 0; i < n; i++) if (tid[i] >= 0) t[tid[i]] = i;
      M.tidTable = t;
    } else {
      const m = new Map();
      for (let i = 0; i < n; i++) m.set(tid[i], i);
      M.tidMap = m;
    }
  }
  function rowOfTid(t) {
    if (!M || !Number.isFinite(t)) return -1;
    ensureTidIndex();
    if (M.tidTable) return t >= 0 && t < M.tidTable.length ? M.tidTable[t] : -1;
    const r = M.tidMap.get(t);
    return r === undefined ? -1 : r;
  }

  /** mother rows, children (CSR), descendant counts, roots per origin: built once per event, on first use */
  function ensureTree() {
    if (M.tree) return M.tree;
    const t0 = performance.now();
    const n = M.n, mom = M.mom, oid = M.oid;
    const momRow = new Int32Array(n);
    for (let i = 0; i < n; i++) { const m = mom[i]; const r = m > 0 ? rowOfTid(m) : -1; momRow[i] = r === i ? -1 : r; }
    const start = new Int32Array(n + 1);
    for (let i = 0; i < n; i++) if (momRow[i] >= 0) start[momRow[i] + 1]++;
    for (let i = 0; i < n; i++) start[i + 1] += start[i];
    const kids = new Int32Array(start[n]), cur = start.slice(0, n);
    for (let i = 0; i < n; i++) if (momRow[i] >= 0) kids[cur[momRow[i]]++] = i;
    // descendants: BFS order from the roots, then accumulate in reverse (no assumption on id ordering)
    const order = new Int32Array(n);
    let qn = 0;
    for (let i = 0; i < n; i++) if (momRow[i] < 0) order[qn++] = i;
    for (let h = 0; h < qn; h++) { const r = order[h]; for (let k = start[r]; k < start[r + 1]; k++) order[qn++] = kids[k]; }
    const desc = new Int32Array(n);
    for (let h = qn - 1; h >= 0; h--) { const r = order[h], p = momRow[r]; if (p >= 0) desc[p] += 1 + desc[r]; }
    // roots of each origin: no mother drawn, or the mother belongs to another origin
    const rc = new Int32Array(M.nO + 1);
    for (let i = 0; i < n; i++) if (momRow[i] < 0 || oid[momRow[i]] !== oid[i]) rc[oid[i] + 1]++;
    for (let o = 0; o < M.nO; o++) rc[o + 1] += rc[o];
    const rootAll = new Int32Array(rc[M.nO]), rcur = rc.slice(0, M.nO);
    for (let i = 0; i < n; i++) if (momRow[i] < 0 || oid[momRow[i]] !== oid[i]) rootAll[rcur[oid[i]]++] = i;
    const roots = [];
    for (let o = 0; o < M.nO; o++) roots.push(rootAll.subarray(rc[o], rc[o + 1]));
    M.tree = { momRow, start, kids, desc, roots, ms: performance.now() - t0, hidBelow: null, hidDirty: true };
    return M.tree;
  }
  const kidsOf = (r) => { const t = ensureTree(); return t.kids.subarray(t.start[r], t.start[r + 1]); };
  function subtree(r) {
    const t = ensureTree();
    const out = [r];
    for (let h = 0; h < out.length; h++) { const x = out[h]; for (let k = t.start[x]; k < t.start[x + 1]; k++) out.push(t.kids[k]); }
    return Int32Array.from(out);
  }

  function lengthOf(r) {
    const off = M.off, p = M.pts;
    let L = 0;
    for (let k = off[r]; k < off[r + 1] - 1; k++) L += Math.hypot(p[3 * k + 3] - p[3 * k], p[3 * k + 4] - p[3 * k + 1], p[3 * k + 5] - p[3 * k + 2]);
    return L;
  }
  const procOf = (r) => (M.proc ? (M.procNames[M.proc[r]] || '') : '');
  function depositCount(r) {
    const dm = M.b.get('dep/mc');
    if (!dm) return null;
    if (!M.depCount) { const c = new Int32Array(M.n); for (let i = 0; i < dm.length; i++) if (dm[i] >= 0) c[dm[i]]++; M.depCount = c; }
    return M.depCount[r];
  }
  function describe(r) {
    if (!M || r < 0 || r >= M.n) return null;
    const t = ensureTree();
    const o = M.origins[M.oid[r]];
    return {
      row: r, tid: M.tid[r], pdg: M.pdg[r], sym: pdgSymbol(M.pdg[r]), motherRow: t.momRow[r], motherTid: M.mom[r],
      process: procOf(r), origin: o, generator: M.gen && M.gen[r] >= 0 ? (M.gnames[M.gen[r]] || null) : null,
      length: lengthOf(r), npts: M.off[r + 1] - M.off[r], nKids: t.start[r + 1] - t.start[r], nDesc: t.desc[r],
      cls: M.cls[r], visible: M.flags[r] > 0, catHidden: !M.spOn[M.cls[r]] || !M.orgOn[M.oid[r]], indHidden: !!(M.hid && M.hid[r]),
    };
  }

  // ----------------------------------------------------------- user actions
  function writeUrlSoon() {
    clearTimeout(urlTimer);
    urlTimer = setTimeout(() => {
      const sp = PDG_CLASSES.filter((c) => spHidden.has(c.name)).map((c) => c.name).join(',');
      const tog = [...(togAll === null ? [] : [togAll ? '+*' : '-*']), ...[...togDiff].map(([k, v]) => (v ? '+' : '-') + k)].join(',');
      let thide = '';
      if (M && M.hid && M.nHid) {
        thide = encodeHidden();
        if (thide.length > URL_MAX) { thide = ''; app.status(`${fmtN(M.nHid)} individually hidden particles are too many to store in the URL`); }
      }
      app.setState({ tsp: sp, tog, thide });
    }, 150);
  }

  /** hidden set -> "12s,40s,5-9,300": subtrees that are hidden as a whole as `<tid>s`, the rest as id ranges */
  function encodeHidden() {
    const t = ensureTree(), hb = hidBelow(), h = M.hid, mr = t.momRow, desc = t.desc;
    const full = (r) => h[r] && hb[r] === desc[r];
    const roots = [], single = [];
    for (let i = 0; i < M.n; i++) {
      if (!h[i]) continue;
      const f = hb[i] === desc[i];
      const inFull = mr[i] >= 0 && full(mr[i]);
      if (f && desc[i] > 0 && !inFull) roots.push(M.tid[i]);
      else if (!(f && inFull)) single.push(M.tid[i]);
    }
    roots.sort((a, b) => a - b); single.sort((a, b) => a - b);
    return [...roots.map((x) => `${x}s`), ...(single.length ? [encodeRanges(single)] : [])].join(',');
  }
  function decodeHidden(str) {
    for (const tok of parseList(str)) {
      const m = /^(\d+)s$/.exec(tok);
      if (m) {
        const r = rowOfTid(parseInt(m[1], 10));
        if (r >= 0) { if (!M.hid) M.hid = new Uint8Array(M.n); setHid(subtree(r), 1); }
      } else {
        for (const x of parseRanges(tok)) { const r = rowOfTid(x); if (r >= 0) { if (!M.hid) M.hid = new Uint8Array(M.n); setHid([r], 1); } }
      }
    }
    if (!M.nHid) M.hid = null;
  }

  function changed(urlToo = true) {
    recompute();
    emitTruth();
    refreshUi();
    if (urlToo) writeUrlSoon();
  }

  function setSpecies(i, on) {
    const name = PDG_CLASSES[i].name;
    if (on) spHidden.delete(name); else spHidden.add(name);
    if (M) M.spOn[i] = on ? 1 : 0;
    changed();
  }
  function setOrigin(oi, on) {
    const o = M.origins[oi];
    M.orgOn[oi] = on ? 1 : 0;
    if (!!on === defEff(o)) togDiff.delete(o.key); else togDiff.set(o.key, !!on);
    changed();
  }
  function setAll(section, on) {
    if (!M) return;
    if (section === 'sp') { PDG_CLASSES.forEach((c, i) => { if (on) spHidden.delete(c.name); else spHidden.add(c.name); M.spOn[i] = on ? 1 : 0; }); }
    else { togAll = !!on; togDiff.clear(); for (const o of M.origins) M.orgOn[o.id] = on ? 1 : 0; }
    changed();
  }

  function setHid(rows, val) {
    if (!rows.length) return;
    if (!M.hid) { if (!val) return; M.hid = new Uint8Array(M.n); }
    const h = M.hid;
    for (let k = 0; k < rows.length; k++) { const r = rows[k]; if (h[r] !== val) { h[r] = val; M.nHid += val ? 1 : -1; } }
    if (M.tree) M.tree.hidDirty = true;
  }
  function clearHid() {
    if (M && M.hid) { M.hid = null; M.nHid = 0; if (M.tree) M.tree.hidDirty = true; changed(); }
  }

  /** hidden-descendant counts: walk up from every hidden row (cost = hidden rows x depth) */
  function hidBelow() {
    const t = ensureTree();
    if (!t.hidDirty && t.hidBelow) return t.hidBelow;
    const hb = t.hidBelow || (t.hidBelow = new Int32Array(M.n));
    hb.fill(0);
    if (M.hid) {
      const h = M.hid, mr = t.momRow;
      for (let i = 0; i < M.n; i++) {
        if (!h[i]) continue;
        let a = mr[i], g = 0;
        while (a >= 0 && g++ < 100000) { hb[a]++; a = mr[a]; }
      }
    }
    t.hidDirty = false;
    return hb;
  }
  /** 0 = hidden, 1 = visible with all descendants visible, 2 = visible, some descendant hidden */
  function triState(r) {
    if (M.hid && M.hid[r]) return 0;
    if (!M.nHid) return 1;
    return hidBelow()[r] > 0 ? 2 : 1;
  }
  /** click on a tri-state box: all -> only this particle -> hidden (with subtree) -> all */
  function cycle(r) {
    const st = triState(r), t = ensureTree();
    const leaf = t.start[r + 1] === t.start[r];
    const sub = subtree(r);
    if (st === 1) {
      if (leaf) setHid(sub, 1);
      else setHid(sub.subarray(1), 1);                  // only this particle: hide every descendant
    } else if (st === 2) setHid(sub, 1);                // hide the particle and its subtree
    else setHid(sub, 0);                                // show the particle and the whole subtree
    changed();
  }

  // ----------------------------------------------------- highlight (selection)
  function highlightRows(sel) {
    if (!sel || !M) return [];
    const b = M.b;
    if (sel.kind === 'mc') return Array.from(sel.mcRows || []);
    if (sel.kind === 'pfp') {
      const tm = b.get('pfp/true_mc');
      if (!tm) return [];
      const s = new Set();
      for (let p = 0; p < sel.mask.length && p < tm.length; p++) if (sel.mask[p] && tm[p] >= 0) s.add(tm[p]);
      return Array.from(s);
    }
    if (sel.kind === 'slice') {
      const tm = b.get('slice/true_mc');
      return tm && tm[sel.id] >= 0 ? [tm[sel.id]] : [];
    }
    return [];
  }
  app.on('selection', (sel) => {
    if (!M) return;
    const f = M.flags;
    for (const r of M.hiRows) if (f[r] === 2) f[r] = 1;
    M.hiRows = highlightRows(sel);
    for (const r of M.hiRows) if (r < M.n && f[r] === 1) f[r] = 2;
    M.selActive = !!sel;
    M.version++;
    emitTruth();
    markSelectedRow();
  });

  // ---------------------------------------------------------------- public API
  app.truth = {
    get model() { return M; },
    /** any non-interaction origin visible (what the old "radiologicals" box meant; used by the PDF export) */
    radiologicalsOn() { return !!M && M.origins.some((o) => o.slot !== 0 && M.orgOn[o.id]); },
    rowOfTid, kids: (r) => (M ? kidsOf(r) : new Int32Array(0)), subtree: (r) => (M ? subtree(r) : new Int32Array(0)),
    /** the particle's siblings (daughters of its mother; for a primary the primaries of its origin), itself included */
    siblings: (r) => { if (!M) return new Int32Array(0); const t = ensureTree(); return t.momRow[r] >= 0 ? kidsOf(t.momRow[r]) : t.roots[M.oid[r]]; },
    describe, depositCount, triState,
    openDrawer, closeDrawer,
  };

  app.resetTruth = (bundle) => {
    M = bundle ? buildModel(bundle) : null;
    ui.expanded.clear(); ui.groups.clear(); ui.pages.clear(); ui.search = ''; ui.hits = null;
    if (elSearch) elSearch.value = '';
    const hadHide = !!state.thide;
    if (M) {
      if (firstEvent && app.legacyRad && !state.tog) {          // old URL: ov=...rad  ->  every origin visible
        togAll = true; togDiff.clear();
        writeUrlSoon();
      }
      applyChoices();
      if (firstEvent && state.thide) {
        decodeHidden(state.thide);
      }
      recompute();
      ui.groups.add(M.origins.length ? 0 : -1);                 // the first origin (interaction) starts open
    }
    if (hadHide && (!firstEvent || !M)) app.setState({ thide: '' });
    firstEvent = false;
    app.legacyRad = false;
    syncButton();
    if (ui.open) { if (!M) closeDrawer(); else renderAll(); }
  };

  // ------------------------------------------------------------------- drawer UI
  const drawer = $('truth-drawer');
  let elCat, elPart, elTree, elInfo, elFoot, elSearch;

  function syncButton() {
    const b = $('btn-truth');
    if (!b) return;
    b.disabled = !M;
    b.title = M ? 'filter the true particles by species, origin or one by one' : 'no true trajectories in this event';
    b.classList.toggle('on', ui.open);
  }
  function openDrawer(tab) {
    if (!M || !drawer) return;
    if (tab) ui.tab = tab;
    ui.open = true;
    drawer.hidden = false;
    $('main').classList.add('tdrawer-open');
    syncButton();
    renderAll();
    app.requestLayout();
  }
  function closeDrawer() {
    if (!drawer) return;
    ui.open = false;
    drawer.hidden = true;
    $('main').classList.remove('tdrawer-open');
    syncButton();
    app.requestLayout();
  }

  const swatch = (i) => PDG_CLASSES[i][app.theme.name];

  function renderCats() {
    if (!M) { elCat.innerHTML = '<div class="td-note">no true trajectories in this event</div>'; return; }
    const v = visCounts();
    const spVis = new Int32Array(N_SP), orVis = new Int32Array(M.nO);
    for (let o = 0; o < M.nO; o++) for (let c = 0; c < N_SP; c++) { spVis[c] += v[o * N_SP + c]; orVis[o] += v[o * N_SP + c]; }
    let h = '<div class="td-sec"><b>species</b><span class="td-all"><a href="#" data-all="sp">all</a> <a href="#" data-none="sp">none</a></span></div>';
    PDG_CLASSES.forEach((c, i) => {
      h += `<label class="td-row${M.spTotal[i] ? '' : ' empty'}" title="${esc(c.name)}: ${fmtN(spVis[i])} visible of ${fmtN(M.spTotal[i])} trajectories">` +
        `<input type="checkbox" data-sp="${i}"${spHidden.has(c.name) ? '' : ' checked'}><span class="sw" style="background:${swatch(i)}"></span>` +
        `<span class="nm">${esc(c.name)}</span><span class="ct">${fmtN(spVis[i])} / ${fmtN(M.spTotal[i])}</span></label>`;
    });
    h += '<div class="td-sec"><b>origin</b><span class="td-all"><a href="#" data-all="or">all</a> <a href="#" data-none="or">none</a></span></div>';
    for (const o of M.origins) {
      h += `<label class="td-row" title="${esc(o.label)}${o.slot === 0 ? ' (descends from the true neutrino / beam interaction)' : ''}: ${fmtN(orVis[o.id])} visible of ${fmtN(o.count)} trajectories">` +
        `<input type="checkbox" data-or="${o.id}"${M.orgOn[o.id] ? ' checked' : ''}><span class="nm ell">${esc(o.label)}</span>` +
        `<span class="ct">${fmtN(orVis[o.id])} / ${fmtN(o.count)}</span></label>`;
    }
    const meta = M.b.meta || {};
    const notes = [];
    if (!M.hasGen) notes.push('This file has no MCParticle-MCTruth association: the generator of each particle is unknown, origins are limited to interaction / other.');
    if (meta.truth && meta.truth.mc2d_scope === 'from_interaction') notes.push('2D views draw the trajectories of the interaction only for this event (too many points to project all); the other origins show up in 3D.');
    if (!M.split) notes.push('No true vertex in this event: nothing is marked as the interaction.');
    if (notes.length) h += `<div class="td-note">${notes.map(esc).join('<br>')}</div>`;
    elCat.innerHTML = h;
  }

  // ---- particles tab
  function matchRows(q) {
    q = q.trim();
    if (!q) return null;
    let m, rows = [], rowSet = null;
    const perOrigin = new Int32Array(M.nO);
    let capped = false;
    const take = (i) => { if (perOrigin[M.oid[i]] >= MAX_MATCH) { capped = true; return; } perOrigin[M.oid[i]]++; rows.push(i); };
    if ((m = /^tid:\s*(\d+)$/i.exec(q)) || (m = /^(\d+)$/.exec(q))) {
      const r = rowOfTid(parseInt(m[1], 10)); if (r >= 0) take(r);
    } else {
      let pdgSet = null;
      if ((m = /^pdg:\s*(-?\d+)$/i.exec(q))) pdgSet = new Set([parseInt(m[1], 10)]);
      else {
        const lq = q.toLowerCase();
        if (!M.distinctPdg) { M.distinctPdg = new Set(); for (let i = 0; i < M.n; i++) M.distinctPdg.add(M.pdg[i]); }
        const distinct = M.distinctPdg;
        const exact = [], part = [];
        for (const p of distinct) {
          const sym = pdgSymbol(p).toLowerCase();
          if (sym === lq || String(p) === lq) exact.push(p);
          else if (sym.includes(lq) || (ALIAS[p] || '').includes(lq)) part.push(p);
        }
        pdgSet = new Set(exact.length ? exact : part);
      }
      for (let i = 0; i < M.n; i++) if (pdgSet.has(M.pdg[i])) take(i);
    }
    rowSet = rows;
    const t = ensureTree();
    const mark = new Uint8Array(M.n);
    for (const r of rowSet) { mark[r] = 2; let a = t.momRow[r], g = 0; while (a >= 0 && !mark[a] && g++ < 100000) { mark[a] = 1; a = t.momRow[a]; } }
    ui.capped = capped;
    return { rows, mark };
  }

  const rowCls = (r) => (!M.spOn[M.cls[r]] || !M.orgOn[M.oid[r]]);
  function rowHtml(r, depth, t) {
    const nk = t.start[r + 1] - t.start[r];
    const st = triState(r);
    const open = ui.expanded.has(r) || (ui.hits && ui.hits.mark[r] === 1 && !ui.expanded.has(-1 - r));
    const sel = app.selection && app.selection.kind === 'mc' && app.selection.id === r;
    const dim = rowCls(r) || st === 0;
    const cat = rowCls(r);
    const len = lengthOf(r);
    const tip = `track id ${M.tid[r]}, ${pdgSymbol(M.pdg[r])} (pdg ${M.pdg[r]}), ${procOf(r) || 'no process'}, ${len.toFixed(1)} cm, ${M.off[r + 1] - M.off[r]} points`;
    return `<div class="tr${sel ? ' sel' : ''}${dim ? ' dim' : ''}${ui.hits && ui.hits.mark[r] === 2 ? ' hit' : ''}" data-row="${r}" style="padding-left:${4 + 13 * depth}px" title="${esc(tip)}${cat ? ' -- hidden by the species/origin filter' : ''}">` +
      `<span class="car" data-act="car">${nk ? (open ? '▾' : '▸') : ''}</span>` +
      `<i class="tri s${st}" data-act="tri" title="click: &#9745; whole subtree visible &rarr; &#9680; only this particle (descendants hidden) &rarr; &#9744; particle and subtree hidden &rarr; &#9745;"></i>` +
      `<span class="sw" style="background:${swatch(M.cls[r])}"></span>` +
      `<span class="sym">${esc(pdgSymbol(M.pdg[r]))}</span><span class="tid">#${M.tid[r]}</span>` +
      `<span class="pr">${esc(procOf(r))}</span><span class="len">${len < 10 ? len.toFixed(1) : Math.round(len)} cm</span>` +
      `${t.desc[r] ? `<span class="ds">(+${fmtN(t.desc[r])})</span>` : ''}</div>`;
  }

  function nodeHtml(r, depth, t, out, budget) {
    out.push(rowHtml(r, depth, t));
    const open = ui.expanded.has(r) || (ui.hits && ui.hits.mark[r] === 1 && !ui.expanded.has(-1 - r));
    if (!open || t.start[r + 1] === t.start[r]) return;
    let list = t.kids.subarray(t.start[r], t.start[r + 1]);
    const showAll = ui.expanded.has(r);
    if (ui.hits && !showAll) list = list.filter((k) => ui.hits.mark[k]);
    listHtml(list, depth + 1, t, out, `n${r}`);
  }
  function listHtml(list, depth, t, out, pkey) {
    const shown = ui.pages.get(pkey) || PAGE;
    const nshow = Math.min(list.length, ui.hits ? list.length : shown);
    for (let k = 0; k < nshow; k++) nodeHtml(list[k], depth, t, out);
    if (nshow < list.length) out.push(`<div class="more" data-more="${pkey}" style="padding-left:${4 + 13 * depth}px">show ${Math.min(PAGE, list.length - nshow)} more (${fmtN(list.length - nshow)} left)</div>`);
  }

  function renderTree() {
    if (!M) { elTree.innerHTML = ''; return; }
    const t0 = performance.now();
    const t = ensureTree();
    const out = [];
    for (const o of M.origins) {
      const roots = t.roots[o.id];
      let list = roots;
      if (ui.hits) list = roots.filter((r) => ui.hits.mark[r]);
      if (ui.hits && !list.length) continue;
      const open = ui.hits ? true : ui.groups.has(o.id);
      out.push(`<div class="tg" data-g="${o.id}" title="${esc(o.label)}"><span class="car" data-act="gcar">${open ? '▾' : '▸'}</span>` +
        `<input type="checkbox" data-gorg="${o.id}"${M.orgOn[o.id] ? ' checked' : ''} title="show / hide this origin (same as the Categories tab)">` +
        `<b class="ell">${esc(o.label)}</b><span class="ct" title="${fmtN(o.count)} particles, ${fmtN(roots.length)} primaries">${fmtN(o.count)} (${fmtN(roots.length)} prim.)</span></div>`);
      if (open) listHtml(list, 1, t, out, `g${o.id}`);
    }
    elTree.innerHTML = out.join('');
    ui.lastTreeMs = performance.now() - t0;
    updateInfo();
  }

  function updateInfo() {
    if (!M) { elInfo.textContent = ''; return; }
    const parts = [];
    if (ui.hits) parts.push(`${ui.hits.rows.length}${ui.capped ? '+' : ''} match${ui.hits.rows.length === 1 ? '' : 'es'}${ui.capped ? ` (first ${MAX_MATCH} per origin)` : ''}`);
    parts.push(M.nHid ? `${fmtN(M.nHid)} hidden individually` : 'none hidden individually');
    elInfo.textContent = parts.join(' · ');
    $('td-reset').disabled = !M.nHid;
  }

  function renderFoot() {
    if (!M) { elFoot.textContent = ''; return; }
    visCounts();
    const dep = M.b.get('dep/xyz');
    let s = `showing ${fmtN(M.nVisible)} / ${fmtN(M.n)} trajectories`;
    if (dep && dep.length) {
      const dm = M.b.get('dep/mc'), dp = M.b.get('dep/pdg'), fi = M.b.get('dep/from_interaction');
      let vis = 0;
      for (let i = 0; i < dep.length / 3; i++) {
        const r = dm ? dm[i] : -1;
        if (r >= 0) { if (M.flags[r]) vis++; } else if (M.fbVis[(fi && fi[i] ? N_SP : 0) + (dp ? pdgClass(dp[i]) : N_SP - 1)]) vis++;
      }
      s += ` · ${fmtN(vis)} / ${fmtN(dep.length / 3)} deposits`;
    }
    if (!app.state.ov.truth) s += ' · truth overlay is off';
    elFoot.textContent = s;
  }

  function renderAll() {
    if (!ui.open) return;
    for (const b of drawer.querySelectorAll('.td-tabs button')) b.classList.toggle('on', b.dataset.tab === ui.tab);
    elCat.hidden = ui.tab !== 'cat'; elPart.hidden = ui.tab !== 'part';
    if (ui.tab === 'cat') renderCats(); else renderTree();
    renderFoot();
  }
  /** cheap refresh after a filter change: re-render only the visible tab */
  function refreshUi() {
    if (!ui.open) return;
    if (ui.tab === 'cat') renderCats(); else {
      const top = elTree.scrollTop;
      renderTree();
      elTree.scrollTop = top;
    }
    renderFoot();
  }
  function markSelectedRow() {
    if (!ui.open || ui.tab !== 'part' || !elTree) return;
    const s = app.selection && app.selection.kind === 'mc' ? app.selection.id : -1;
    for (const el of elTree.querySelectorAll('.tr.sel')) el.classList.remove('sel');
    if (s >= 0) { const el = elTree.querySelector(`.tr[data-row="${s}"]`); if (el) el.classList.add('sel'); }
  }

  function runSearch() {
    if (!M) return;
    ui.search = elSearch.value;
    ui.hits = ui.search.trim() ? matchRows(ui.search) : null;
    if (ui.hits && !ui.hits.rows.length) { ui.hits = { rows: [], mark: new Uint8Array(M.n) }; }
    renderTree();
    if (ui.hits && ui.hits.rows.length) {
      const el = elTree.querySelector(`.tr[data-row="${ui.hits.rows[0]}"]`);
      if (el) el.scrollIntoView({ block: 'center' });
    }
  }

  function bindDrawer() {
    if (!drawer) return;
    elCat = $('td-cat'); elPart = $('td-part'); elTree = $('td-tree'); elInfo = $('td-info'); elFoot = $('td-foot'); elSearch = $('td-search');
    $('btn-truth').addEventListener('click', () => { if (ui.open) closeDrawer(); else openDrawer(); });
    $('td-close').addEventListener('click', closeDrawer);
    drawer.querySelector('.td-tabs').addEventListener('click', (e) => {
      const b = e.target.closest('button[data-tab]'); if (!b) return;
      ui.tab = b.dataset.tab; renderAll();
    });
    elCat.addEventListener('change', (e) => {
      const el = e.target;
      if (el.dataset.sp !== undefined) { el.blur(); setSpecies(parseInt(el.dataset.sp, 10), el.checked); }
      else if (el.dataset.or !== undefined) { el.blur(); setOrigin(parseInt(el.dataset.or, 10), el.checked); }
    });
    elCat.addEventListener('click', (e) => {
      const a = e.target.closest('a[data-all],a[data-none]'); if (!a) return;
      e.preventDefault();
      setAll(a.dataset.all || a.dataset.none, a.dataset.all !== undefined);
    });
    // tree: delegated clicks / hover
    elTree.addEventListener('change', (e) => {
      const el = e.target;
      if (el.dataset.gorg !== undefined) { el.blur(); setOrigin(parseInt(el.dataset.gorg, 10), el.checked); }
    });
    elTree.addEventListener('click', (e) => {
      const more = e.target.closest('[data-more]');
      if (more) { const k = more.dataset.more; ui.pages.set(k, (ui.pages.get(k) || PAGE) + PAGE); const top = elTree.scrollTop; renderTree(); elTree.scrollTop = top; return; }
      const g = e.target.closest('.tg');
      if (g) {
        if (e.target.tagName === 'INPUT') return;
        const id = parseInt(g.dataset.g, 10);
        if (ui.groups.has(id)) ui.groups.delete(id); else ui.groups.add(id);
        const top = elTree.scrollTop; renderTree(); elTree.scrollTop = top;
        return;
      }
      const row = e.target.closest('.tr'); if (!row) return;
      const r = parseInt(row.dataset.row, 10);
      const act = e.target.dataset.act;
      if (act === 'tri') { cycle(r); return; }
      if (act === 'car') {
        if (ui.hits && ui.hits.mark[r] === 1 && !ui.expanded.has(r)) { if (ui.expanded.has(-1 - r)) ui.expanded.delete(-1 - r); else ui.expanded.add(-1 - r); }
        else if (ui.expanded.has(r)) ui.expanded.delete(r); else ui.expanded.add(r);
        const top = elTree.scrollTop; renderTree(); elTree.scrollTop = top;
        return;
      }
      app.clickMc(r);
    });
    elTree.addEventListener('pointerover', (e) => {
      const row = e.target.closest('.tr'); if (!row || row === ui.hoverEl) return;
      ui.hoverEl = row; app.hoverMc(parseInt(row.dataset.row, 10));
    });
    elTree.addEventListener('pointerleave', () => { ui.hoverEl = null; app.hoverMc(null); });
    elTree.addEventListener('pointerout', (e) => { if (!e.relatedTarget || !e.relatedTarget.closest || !e.relatedTarget.closest('.tr')) { ui.hoverEl = null; app.hoverMc(null); } });
    elSearch.addEventListener('input', () => { clearTimeout(ui.searchTimer); ui.searchTimer = setTimeout(runSearch, 150); });
    elSearch.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        clearTimeout(ui.searchTimer); runSearch();
        if (ui.hits && ui.hits.rows.length) app.select({ kind: 'mc', id: ui.hits.rows[0], pinned: true });
        e.preventDefault();
      } else if (e.key === 'Escape') {
        if (elSearch.value) { elSearch.value = ''; runSearch(); } else elSearch.blur();
        e.stopPropagation(); e.preventDefault();
      }
    });
    $('td-reset').addEventListener('click', clearHid);
    window.addEventListener('keydown', (e) => {
      if (e.key !== 'Escape' || !ui.open) return;
      const t = e.target;
      if (t && t.tagName === 'INPUT' && t.type === 'text') return;
      if (state.sel || state.pfp || state.slice || state.mc) return;      // the selection handler clears that first
      closeDrawer();
    });
    app.on('state', (s, ch) => {
      if (ch.has('theme') && ui.open) renderAll();
      if (ch.has('ov') && ui.open) renderFoot();
    });
  }
  bindDrawer();
  syncButton();
}
