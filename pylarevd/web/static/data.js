// data.js -- EVDB bundle parsing and the (tiny) API client.
//
// EVDB layout: "EVDB" | u32 LE header length H | H bytes UTF-8 JSON | pad to 8 |
// array data. Each array starts on an 8-byte boundary, offsets are relative to the
// start of the array data. Typed arrays returned here are views on the fetched
// ArrayBuffer (no copy). Parsing is O(number of arrays), so it runs on the main
// thread; the expensive part (fetch + gunzip) is done by the browser off-thread.

const DTYPES = {
  f32: Float32Array, f64: Float64Array, i32: Int32Array, u32: Uint32Array,
  i16: Int16Array, u16: Uint16Array, i8: Int8Array, u8: Uint8Array,
};

export class Bundle {
  constructor(header, arrays, byteLength) {
    this.kind = header.kind;
    this.version = header.version;
    this.meta = header.meta || {};
    this.arrays = arrays;       // name -> typed array (flat)
    this.shapes = {};           // name -> shape array
    this.byteLength = byteLength;
  }
  has(name) { return Object.prototype.hasOwnProperty.call(this.arrays, name); }
  /** Typed array or null. */
  get(name) { return this.arrays[name] || null; }
  /** Number of rows (first shape dimension) of an array, 0 if missing. */
  rows(name) { const s = this.shapes[name]; return s ? s[0] : 0; }
  /** Names starting with a prefix. */
  names(prefix = '') { return Object.keys(this.arrays).filter((n) => n.startsWith(prefix)); }
}

/** Parse an EVDB ArrayBuffer into a Bundle. */
export function parseEvdb(buffer) {
  const dv = new DataView(buffer);
  if (buffer.byteLength < 8 || dv.getUint32(0, false) !== 0x45564442) { // "EVDB"
    throw new Error('not an EVDB bundle (bad magic)');
  }
  const hlen = dv.getUint32(4, true);
  const header = JSON.parse(new TextDecoder('utf-8').decode(new Uint8Array(buffer, 8, hlen)));
  if (header.format !== 'evdb') throw new Error('bad EVDB header');
  const dataStart = (8 + hlen + 7) & ~7;
  const arrays = {};
  const shapes = {};
  for (const [name, spec] of Object.entries(header.arrays || {})) {
    const T = DTYPES[spec.dtype];
    if (!T) throw new Error(`unknown dtype ${spec.dtype} for ${name}`);
    const n = spec.shape.reduce((a, b) => a * b, 1);
    const off = dataStart + spec.offset;
    if (off + n * T.BYTES_PER_ELEMENT > buffer.byteLength) throw new Error(`array ${name} overruns bundle`);
    arrays[name] = n === 0 ? new T(0) : new T(buffer, off, n);
    shapes[name] = spec.shape;
  }
  const b = new Bundle(header, arrays, buffer.byteLength);
  b.shapes = shapes;
  return b;
}

/**
 * Ragged helper: iterate polylines of a flat (m,dim) array with u32 offsets.
 * Calls fn(i, start, end) where [start,end) are row indices.
 */
export function forEachRagged(offsets, fn) {
  for (let i = 0; i + 1 < offsets.length; i++) fn(i, offsets[i], offsets[i + 1]);
}

/** Number of line segments in a polyline set (sum of max(len-1, 0)). */
export function countSegments(offsets) {
  let n = 0;
  for (let i = 0; i + 1 < offsets.length; i++) {
    const len = offsets[i + 1] - offsets[i];
    if (len > 1) n += len - 1;
  }
  return n;
}

/** Build a LINES index buffer (pairs j,j+1 within each polyline). */
export function polylineIndex(offsets) {
  const idx = new Uint32Array(2 * countSegments(offsets));
  let k = 0;
  for (let i = 0; i + 1 < offsets.length; i++) {
    const e = offsets[i + 1] - 1;
    for (let j = offsets[i]; j < e; j++) { idx[k++] = j; idx[k++] = j + 1; }
  }
  return idx;
}

/** Expand a per-object array to per-row values through offsets: out[row] = f(obj). */
export function expandRagged(offsets, perObject, OutType = Float32Array) {
  const out = new OutType(offsets[offsets.length - 1]);
  for (let i = 0; i + 1 < offsets.length; i++) out.fill(perObject[i], offsets[i], offsets[i + 1]);
  return out;
}

/** Cheap percentile range from a strided sample; ignores non-finite values. */
export function percentileRange(arr, lo = 0.02, hi = 0.98, maxSamples = 20000) {
  const n = arr.length;
  if (!n) return [0, 1];
  const step = Math.max(1, Math.floor(n / maxSamples));
  const s = [];
  for (let i = 0; i < n; i += step) { const v = arr[i]; if (Number.isFinite(v)) s.push(v); }
  if (!s.length) return [0, 1];
  s.sort((a, b) => a - b);
  const a = s[Math.floor(lo * (s.length - 1))];
  const b = s[Math.floor(hi * (s.length - 1))];
  return [a, b > a ? b : a + 1e-6];
}

// ---------------------------------------------------------------- API client
async function getJson(url, signal) {
  const r = await fetch(url, { signal });
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status} ${await r.text().catch(() => '')}`.slice(0, 300));
  return r.json();
}

async function getBundle(url, signal) {
  const t0 = performance.now();
  const r = await fetch(url, { signal });
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status} ${await r.text().catch(() => '')}`.slice(0, 300));
  const buf = await r.arrayBuffer();
  const t1 = performance.now();
  const b = parseEvdb(buf);
  b.timing = { fetch: t1 - t0, parse: performance.now() - t1 };
  return b;
}

const q = (o) => new URLSearchParams(Object.entries(o).filter(([, v]) => v !== '' && v != null)).toString();

/** Server-backed API. */
export function serverApi() {
  return {
    sample: null,
    files: (signal) => getJson('/api/files', signal),
    events: (file, signal) => getJson(`/api/events?${q({ file })}`, signal),
    geometry: (file, signal) => getBundle(`/api/geometry?${q({ file })}`, signal),
    event: (file, entry, hits, flash, signal) =>
      getBundle(`/api/event?${q({ file, entry, hits, flash })}`, signal),
  };
}

/** Static-sample API: ?sample=fd|pdsp loads samples/<name>_{event,geometry}.evdb. */
export function sampleApi(name) {
  const base = `samples/${name}`;
  const path = `sample:${name}`;
  return {
    sample: name,
    files: async () => [{ path, name: `${name} sample`, n_events: 1, detector: name }],
    events: async () => [{ entry: 0, run: 0, subrun: 0, event: 0 }],
    geometry: (file, signal) => getBundle(`${base}_geometry.evdb`, signal),
    event: (file, entry, hits, flash, signal) => getBundle(`${base}_event.evdb`, signal),
  };
}
