// util.js -- small helpers shared by the viewer modules: HTML escaping, number labels, axis ticks,
// palette ranks of group ids, file downloads. No imports, no side effects.

/** Escape text for HTML element content and double-quoted attribute values. */
export const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

/** Grouping kinds of hits / space points that "colour by" can show as categories (view2d, view3d). */
export const CAT_KINDS = ['track', 'shower', 'pfparticle', 'cluster', 'slice'];

/** Colour-bar label: 3 significant digits, exponent outside [0.01, 1000), trailing fraction zeros dropped. */
export function fmt(v) {
  if (Math.abs(v) >= 1000 || (Math.abs(v) < 0.01 && v !== 0)) return v.toExponential(1);
  const s = v.toPrecision(3);
  return s.includes('.') ? s.replace(/\.?0+$/, '') : s;   // trailing zeros of the fraction only (100 stays 100)
}

/**
 * Axis ticks with a step of 1, 2, 5 or 10 x 10^k NEAREST to (hi - lo) / target (view2d, viewtrigger).
 * -> {ticks, step, dec}: ticks within [lo, hi] (values within 1e-9 step of 0 are 0), and the number of
 * decimals that shows the step (0..6). An empty range gives {ticks: [lo], step: 1, dec: 0}.
 */
export function niceTicks(lo, hi, target) {
  const span = hi - lo; if (!(span > 0)) return { ticks: [lo], step: 1, dec: 0 };
  const raw = span / Math.max(1, target), mag = 10 ** Math.floor(Math.log10(raw)), n = raw / mag;
  const step = (n < 1.5 ? 1 : n < 3 ? 2 : n < 7 ? 5 : 10) * mag;
  const out = []; for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-6; v += step) out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  return { ticks: out, step, dec: Math.min(6, Math.max(0, -Math.floor(Math.log10(step) + 1e-9))) };
}

/**
 * Axis ticks with the SMALLEST step of 1, 2, 5 or 10 x 10^k that is >= (hi - lo) / target (viewoptical,
 * viewlight); the last tick may lie up to half a step beyond hi. -> {ticks, step}; with `snapZero`, values
 * within 1e-9 step of 0 are 0. An empty range gives {ticks: [lo], step: 1}.
 */
export function niceTicksCeil(lo, hi, target = 7, snapZero = false) {
  const span = hi - lo;
  if (!(span > 0)) return { ticks: [lo], step: 1 };
  const raw = span / target, mag = 10 ** Math.floor(Math.log10(raw));
  let step = 10 * mag;
  for (const m of [1, 2, 5, 10]) if (m * mag >= raw) { step = m * mag; break; }
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 0.5 * step; v += step) out.push(snapZero && Math.abs(v) < 1e-9 * step ? 0 : v);
  return { ticks: out, step };
}

/** Map of every group id >= 0 in `ids` to its rank among them (ascending): the palette index of an object. */
export function rankMap(ids) {
  const set = new Set();
  for (let i = 0; i < ids.length; i++) if (ids[i] >= 0) set.add(ids[i]);
  return new Map(Array.from(set).sort((a, b) => a - b).map((id, r) => [id, r]));
}

/** Save a Blob as a file named `name` (object URL revoked after `revokeMs`). */
export function downloadBlob(blob, name, revokeMs = 10000) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = name;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), revokeMs);
}
