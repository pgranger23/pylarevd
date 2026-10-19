# pylarevd web viewer (v2) — design and work plan

Status: plan, 2026-10-19. Branch `web-viewer` (worktree `~/projects/pylarevd-web`),
off `light-animation`.

## Why

The Dash/plotly browser re-renders whole figures in Python on every control
change and ships them as JSON (1-100 MB). Animations redraw the full figure every
frame, which is why the 3-D camera fought the time slider. On a ProtoDUNE-SP
event the first render takes ~10 s and anything with all truth ~100 s.

## Direction (inspired by LUCiD's viewer, github.com/CIDeR-ML/LUCiD/tree/main/viewer)

Design ideas taken (no code copied; LUCiD ships no licence file):

1. **Static browser frontend, plain ES modules, no build step.** Three.js
   (vendored r170, MIT, `pylarevd/web/static/vendor/`) for 3-D, and WebGL or
   Canvas2D for the 2-D panels.
2. **Data shipped once per event as compact columnar binary arrays.** Every
   control change after that (colour-by, scale, overlays, theme, time, camera)
   is handled in the browser without a server round trip.
3. **Few draw calls**: one point-sprite `THREE.Points` per point set (hits,
   space points, deposits) with a custom shader; trajectories as
   `LineSegments` batched into one buffer per kind.
4. **Animation on the client**: the light animation is a time variable in the
   browser. A time change updates small attribute buffers (tile colours) or a
   shader uniform, never the geometry or the camera.
5. **Tiny stdlib server** serving the frontend plus a JSON/binary API. The
   expensive Python decode happens once per event: it is cached on disk, and
   neighbouring events are prefetched in a background thread pool.

What we keep: pylario (decoding) and the physics logic already in pylarevd
(panels, wire projection, flash repair, light onset, frame clock, flash-hit
links, PD walls, truth matching). The exporter calls them; the frontend never
re-implements physics.

## Performance targets

| action | target |
|---|---|
| switch to a cached event | < 0.5 s to first paint |
| switch to an uncached event | Python decode time + < 0.5 s |
| any display control (colour, scale, overlay, theme, panel visibility) | < 50 ms, no server call |
| light animation playback | 60 fps; camera and 2-D zoom never reset |
| 3-D with 900k true-trajectory points | interactive (> 30 fps) |

## Architecture

```
pylarevd/web/
  bundle.py      # EVDB binary format: write/read (Python), shared by export + tests
  export.py      # pylario Event -> event bundle; Geometry -> geometry bundle; CLI
  cache.py       # on-disk bundle cache, keyed by file path/mtime/size + entry + options
  server.py      # stdlib ThreadingHTTPServer; API + static files; prefetch pool; CLI
  static/
    index.html   # shell: toolbar, panels, status line
    style.css
    main.js      # app state, event navigation, URL state, wiring
    data.js      # fetch + decode EVDB bundles (DataView, zero-copy typed arrays)
    colormaps.js # turbo/magma/viridis/... as 256-entry LUTs (+ categorical palette)
    view3d.js    # Three.js 3-D view (space points, tracks, showers, vertices, truth, TPC boxes, PD walls)
    view2d.js    # wire-plane panels (hits + projected overlays), pan/zoom, hover
    viewlight.js # PD walls face-on + time strip + playback; feeds view3d PD colours
    vendor/      # three.module.js, OrbitControls.js, THREE_LICENSE
```

Launch: `python -m pylarevd.web reco.root [more.root] --port 8050`, which
prints the SSH tunnel line like the current app. Old `pylarevd.app` stays
untouched.

## EVDB bundle format v1 (one file per event, one per geometry)

```
bytes 0..3   magic "EVDB"
bytes 4..7   uint32 LE  header length H
bytes 8..8+H UTF-8 JSON header
padding to a multiple of 8
array data: little-endian, each array starting on an 8-byte boundary
```

Header JSON:

```json
{"format": "evdb", "version": 1, "kind": "event" | "geometry",
 "meta": { ...free-form JSON: ids, labels, summary text, panel specs... },
 "arrays": {"<name>": {"dtype": "f32|f64|i32|u32|i16|u16|i8|u8",
                       "shape": [n, ...], "offset": <bytes from data start>}}}
```

Rules: names are `/`-separated paths (`hits/charge`). Float data are `f32`
unless precision matters (times in us use `f64` for optical, or `f32` relative
to `t0`). Ragged collections (polylines, per-flash hit lists) are a flat array
plus `.../offsets` (`u32`, length n+1). Categorical strings go in
`meta` as lists, referenced by integer index arrays. The server sends bundles
with `Content-Encoding: gzip` when the client accepts it.

### Geometry bundle (`kind: "geometry"`)

meta: `detector`, `tpcs` (list of boxes `[xmin,ymin,zmin,xmax,ymax,zmax]`),
`walls` (list of `{x, label, extent:[zmin,zmax,ymin,ymax]}`).
arrays:
- `pd/xyz` f32 (n_tiles,3), `pd/half` f32 (n_tiles,2) half sizes `[hy, hz]`,
  `pd/wall` u8, `pd/chan_offsets` u32 + `pd/channels` u32 (channels per tile)
- `pd/chan_tile` i32 (n_opchannels) tile per optical channel, -1 if none

### Event bundle (`kind: "event"`)

meta:
- `id: [run, subrun, event]`, `entry`, `file`, `detector`
- `summary`: list of text lines (the existing `summary()` outputs)
- `truth`: `{headline, vertex:[x,y,z]|null, inside: bool|null}`
- `available`: flags `{hits, spacepoints, tracks, showers, vertices, truth_mc, deposits, optical, links}`
- `hit_tag`, `hit_tags` (all hit products), `flash_tag`, `flash_tags`
- `panels`: list of `{title, hlabel, vlabel, hrange:[lo,hi], vrange:[lo,hi], anodes:[...], edges:[...]}`
  (physical space; the exporter uses `EventDisplay` panel logic)
- `groups`: `{track: n, shower: n, pfparticle: n, cluster: n, slice: n}` counts
- `optical`: `{t0, quantum, window:[lo,hi], products, pe_floor, notes:[...]}`

arrays (all optional, present when available):
- hits: `hits/panel` u8, `hits/h` f32, `hits/v` f32 (panel coords, cm),
  `hits/charge` f32, `hits/amplitude` f32, `hits/tick` f32,
  `hits/multiplicity` u8, `hits/channel` u32, `hits/xyz`? (no; hits are 2-D),
  `hits/group/<kind>` i32 (object id per hit or -1) for each available kind
- readout coords too: `hits/channel` and `hits/tick` (for a readout mode)
- space points: `sp/xyz` f32 (n,3), `sp/charge` f32, `sp/group/<kind>` i32 if cheap
- tracks: `tracks/points` f32 (m,3), `tracks/offsets` u32, `tracks/id` i32;
  projected: `tracks2d/points` f32 (k,2), `tracks2d/offsets` u32, `tracks2d/panel` u8, `tracks2d/track` i32
- showers: `showers/start` f32 (n,3), `showers/dir` f32 (n,3), `showers/length` f32,
  `showers/angle` f32; projected cones `showers2d/...` like tracks2d
- vertices: `vtx/xyz` f32 (n,3), `vtx/role` u8 (0 other, 1 interaction, 2 primary daughter, 3 secondary);
  `vtx2d/hv` f32, `vtx2d/panel` u8, `vtx2d/index` i32
- truth trajectories: `mc/points` f32 (m,3), `mc/offsets` u32, `mc/pdg` i32,
  `mc/track_id` i32, `mc/mother` i32, `mc/process` u16 (index into meta
  `mc_processes`), `mc/from_interaction` u8 (1 if descends from the true
  interaction: the radiologicals toggle becomes a client-side filter), plus
  `mc2d/...` projected like tracks2d. NO cap: the frontend draws them all.
- deposits: `dep/xyz` f32, `dep/energy` f32, `dep/track_id` i32, `dep/from_interaction` u8
- particle labels: `labels/...` from `_matched_particle_labels` (pos, text index, kind)
- optical hits: `oph/time` f64 (us), `oph/pe` f32, `oph/channel` i32,
  `oph/tile` i32, `oph/flash` i32 (flash index or -1, from validated links)
- flashes: `flash/time` f64, `flash/pe` f32, `flash/y` f32, `flash/z` f32,
  `flash/yw` f32, `flash/zw` f32 (after `repair_flash_positions`),
  `flash/wall_share` f32 (n_flash, n_walls)
- the frame clock: `light/edges` f64 (n_frames+1); the frontend computes
  arrivals/afterglow itself from `oph/*` (cheap: < 1e5 hits).

Optical arrays cover the whole readout, not a window: the client chooses the
window and clock (log or linear) and can jump between bursts. A readout holds
~75k OpHits, about 1.5 MB.

## Server API

- `GET /` and `/static/...`: frontend
- `GET /api/files` → `[{path, name, n_events, detector}]`
- `POST /api/open {path}` → open a file (loopback only, as today)
- `GET /api/events?file=` → `[{entry, run, subrun, event}]`
- `GET /api/geometry?file=` → geometry bundle (EVDB)
- `GET /api/event?file=&entry=&hits=<tag>&flash=<tag>` → event bundle (EVDB, gzip)
- `GET /api/status` → cache/prefetch state (for a status line)

Caching: `~/.cache/pylarevd/<sha1(path,mtime,size)>/<entry>_<hits>_<flash>.evdb`.
After serving entry i, prefetch i+1, i+2 and i-1 in a `ProcessPoolExecutor`
(decode is CPU-bound Python; the box has 48 cores). The CLI
`python -m pylarevd.web.export file.root -j 32` fills the cache for a whole
file.

## Frontend behaviour (parity with the Dash app, plus the light work)

- Toolbar: file, event (prev/next, arrow keys, jump), hit product, flash
  producer, view (2D | 3D | Light | Charge+Light | 2D+3D), colour-by (charge,
  amplitude, tick, multiplicity, object kinds, x/y/z), scale (auto/log/linear),
  colormap, theme (dark/light), overlays (truth, tracks, vertices, showers,
  radiologicals, particle labels), screenshot (PNG via `canvas.toBlob`), URL
  state (shareable link; must restore on load).
- 3D: OrbitControls, y up, z beam, x drift as now; TPC wireframe; PD walls as
  instanced quads at their real position and size.
- 2D: one panel per wire orientation (from meta.panels); pan/zoom (wheel,
  drag); synchronised `h` axis across panels as now (`share_w`); hover shows
  hit info.
- Light: PD walls face-on (side by side or stacked by aspect, as now), time
  strip (PE per frame, in-flash vs no-flash, flash stems, playhead), play/pause,
  slider, log/linear clock, afterglow/cumulative, flash ellipses. Charge+Light
  shows the same time state lighting the PD walls in the 3-D view. Time changes
  touch only colour buffers.
- Summary text panel (meta.summary) and a status line (decode/cache time).

## Work packages (executed by Sonnet subagents)

| WP | owner files | depends on |
|---|---|---|
| A: bundle + exporter + cache + export CLI + tests | `web/bundle.py`, `web/export.py`, `web/cache.py`, `tests/test_web_export.py` | — |
| B: server + launcher + tests | `web/server.py`, `web/__init__.py`, `web/__main__.py`, `tests/test_web_server.py` | A's API (stubbed until ready) |
| C: frontend shell + data + 3-D | `static/index.html`, `style.css`, `main.js`, `data.js`, `colormaps.js`, `view3d.js` | spec; sample bundles from A |
| D: 2-D wire panels | `static/view2d.js` (+ hooks in `main.js` agreed with C) | C's shell |
| E: light view + charge+light | `static/viewlight.js` (+ PD colour hook into `view3d.js`) | C's shell |

Order: A and C in parallel (C starts from a fixture written to this spec),
then B, D and E in parallel, then integration, browser tests
(Playwright + SwiftShader, recipe in setup-log I10) and performance
measurements against the targets above.
