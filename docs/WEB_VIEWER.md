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

## WP-A implementation notes

Implemented in `web/bundle.py`, `export.py`, `cache.py`. Additions and deviations from the contract above:

- Strict JSON header: non-finite floats in `meta` are written as `null`. Bool arrays are stored as `u8`.
- Cache file name: `<entry>_<hits|auto>_<flash|auto>.evdb`, plus `geometry.evdb`, in `~/.cache/pylarevd/<key>/` (override the root with env `PYLAREVD_CACHE`).
- `meta` extras: `timing` (seconds per section), `notes` (why a section is missing), `mc_processes`, `label_texts`, `label_kinds` (`["track","shower","mc"]`), `panels_readout` (one entry per plane for the channel-vs-tick mode, indexed by `hits/chan_view`), `truth.{has_split, n_mc, n_from_interaction, mc2d_scope, vertex2d}`. `truth.vertex2d` is per panel `[h,v]` or null. If no true vertex exists, `mc/from_interaction` and `dep/from_interaction` are 1 everywhere and `truth.has_split` is false.
- `meta.hit_tag` / `flash_tag` hold the tag actually used (also when the request left it null). `meta.optical` additionally has `n_frames, time_scale, persistence, n_flashes, n_ophits, n_walls`; `pe_floor` is `light.PE_FLOOR`; `window`/`t0` are `LightAnimation` defaults (absolute us).
- Panels: `meta.panels[i]` adds `key, view, drift_sign, n_hits` and `faces`: list of `{dir:[d0,d1], boxes:[[xmin,ymin,zmin,xmax,ymax,zmax],...]}`. This is the exact `EventDisplay.project_points` rule so the client can project any 3-D point itself: the first face with a box containing the point claims it, `h = d0*y + d1*z`, `v = x` (use no clipping for the true vertex / showers). `hrange/vrange` are the hit extents (plus anodes and drift edges for v) padded by 3%.
- `hits/panel` is 255 for hits in no panel (unresolved WireID). Extra hit arrays: `hits/view` i8 (U/V/Z of the WireID, -1 unresolved), `hits/chan_view` i8 (plane from the channel alone; the key for readout mode), `hits/drift_sign` i8. Groups: `hits/group/{track,shower,pfparticle,cluster,slice}` i32, and `meta.groups[kind]` the object count. If the requested hit tag is not the default one but has the same hit count, alignment with the default collection is assumed (noted in `meta.notes`); otherwise groups are omitted.
- Extra reco arrays: `sp/amplitude`, `sp/tick`, `tracks/chi2`, `showers/id`, `vtx/id`. Vertex roles: 1 interaction, 2 primary daughter, 3 secondary (0 never occurs in practice).
- Projected overlays: `tracks2d/{points,offsets,panel,track}`, `showers2d/{points,offsets,panel,shower}` (closed cone outline), `vtx2d/{hv,panel,index}`, `mc2d/{points,offsets,panel,index}` (`index` is an index into `mc/*`). Every row of `offsets` is one drawn run (a trajectory broken at a face gap or a step > 20 cm becomes several runs).
- `mc2d` covers all trajectories unless the event has more than 400k trajectory points; then only `from_interaction==1` particles are projected (`truth.mc2d_scope == "from_interaction"`, noted). Projecting all 1.8M points of a ProtoDUNE-SP event took 18 s.
- Deposits are NOT pre-projected (13 bytes/deposit/panel was 12 MB on the FD sample): use `panels[i].faces`. Extra: `dep/pdg`. `dep/from_interaction` matches on `abs(track_id)`.
- Labels: `labels/{pos (n,3) f32, text u32 (index into meta.label_texts), kind u8 (0 reco track, 1 reco shower, 2 true particle), index i32 (tracks / showers / mc row, -1 if unknown), track_id, pdg}`. For events with more than 100k true particles the reco objects are matched against `from_interaction` particles only (noted; this was 13 s otherwise).
- Optical: `oph/*` is the unfiltered OpHit list; `oph/flash` indexes `flash/*`, which lists ALL flashes of the product in product order (not capped, not brightest-first). `flash/yw, flash/zw` use 30 cm where the finder gave no usable width (as `LightAnimation.flashes`); `flash/x` is NaN where unset. `flash/wall_share` is the fraction of each flash's PE per wall (all ones if links are missing or there is one wall). `light/edges` is the default log clock (120 frames).
- Geometry bundle extra: `meta.projected`, `meta.notes`. Without PD positions the `pd/*` arrays are absent.
- Failures are per section: `available.*` false plus a line in `meta.notes`. ProtoDUNE-SP: no deposits, `showers` fail in pylario (TVector3 not subscriptable), and the default hit product is `gaushit` (not disambiguated); pass `hit_tag="hitpdune"`.
- CLI: `python -m pylarevd.web.export FILE [-e all|N|A-B|A,B] [-j N] [--hits TAG] [--flash TAG]`; `cache.get_or_build_event(path, entry, hit_tag, flash_tag)` reuses one `EventFile` per process.

## WP-B implementation notes

- `python -m pylarevd.web FILE... [--port --host --workers --prefetch --allow-remote-open --precompute]`; code in `web/server.py` (`make_server`, `App`, `Handler`).
- Builds run in a forkserver `ProcessPoolExecutor`; the worker writes the raw bundle and `<entry>.evdb.gz` (level 3) into the cache and the request thread reads the file. Gzip clients get the `.gz` (Content-Encoding: gzip), others the raw file; Content-Length always set.
- Default tags: hit `hitfd` > `hitpdune` > first disambiguated non-`gaushit` > first non-`gaushit`; flash `opflash` > first. Requests without `hits=`/`flash=` resolve to these before the cache lookup, so cache file names always carry explicit tags.
- Prefetch after serving i: i+1, i+2, i-1, at most `--prefetch` (default 2, capped at workers-1) concurrent, never started while an explicit request is in flight; each new explicit request replaces the queue. `--precompute` fills a lower-priority backlog.
- `root://` paths are rejected (the cache keys on os.stat). `/api/status`: workers, in_flight, queued_prefetch, counters {builds, hits, joined, errors}, recent timings, per-file cached entries.
- Errors: JSON `{error}` with 400 (bad entry/tag/body), 403 (open disabled), 404 (unknown file/route/static), 500 (decode failure).

## WP-C implementation notes

Files: `static/{index.html,style.css,main.js,data.js,colormaps.js,view3d.js}` plus no-op
stubs `view2d.js` and `viewlight.js` (to be replaced by WP-D / WP-E, which own them).
Everything is plain ES modules; no build step.

### View module interface (also in the header comment of `main.js`)

A view is a module whose default export is an object:

```js
export default {
  name: 'view2d',
  init(container, app) {},        // once at startup; container = its <section class="panel">
  setGeometry(geometry) {},       // optional; geometry Bundle changed (new file); before setEvent
  setEvent(bundle, geometry) {},  // new event Bundle; null = clear
  onState(state, changed) {},     // any state change; `changed` = Set of keys; first call has all keys
  setVisible(visible) {},         // optional; panel shown/hidden by the view selector
  resize() {},                    // optional; container size changed (called after setVisible(true))
  getCanvas() {},                 // optional; canvas the screenshot button composites
};
```

`main.js` imports the three modules statically and calls the hooks inside try/catch (a
broken view cannot take the app down). `app` (passed to `init`, also `window.__evd`):
`state`, `setState(patch)`, `registerParam(name, default)` (extra string state key kept in
the URL; call in `init`), `on('state'|'event'|'geometry', fn)`, `bundle`, `geometry`,
`theme` (colour set from `colormaps.js` `THEMES`), `views` (registry: `app.views.view3d`
etc.), `status(text, isError)`, `requestLayout()`.

State keys: `file entry hits flash view color scale cmap theme ov{truth,rad,dep,tracks,
showers,vtx,labels,pd}` plus registered params. Views must not rebuild data-derived geometry
in `onState` unless a data-defining key changed.

Layout: `#main[data-view]` is a CSS grid; `view` in `3d | 2d | light | charge+light | 2d+3d`
decides which panels are displayed (`charge+light` = 3D panel + light panel, `2d+3d` = 2D +
3D). Hidden panels get `setVisible(false)` and must not render.

### Light hook into the 3-D view

`app.views.view3d.setPdColors(Float32Array rgb|null, alpha?)`: `rgb` is `n_tiles*3`, 0..1
sRGB, tile order = geometry `pd/xyz`; `null` restores the neutral walls. It only updates an
instance-colour attribute and schedules a render. During playback call
`view3d.setAnimating(true)` (render loop on) and `setAnimating(false)` afterwards (back to
render-on-demand). `view3d.pdTileCount`, `requestRender()`, `resetView()` are also public.

### URL state

All controls live in the query string and are restored on load: `file`, `entry` (entry
number, not list index), `hits`, `flash`, `view`, `color`, `scale`, `cmap`, `theme`,
`ov=truth,tracks,...` (comma list, only written when it differs from the default; `ov=`
means none), and `sample=fd|pdsp` (static-bundle mode, no server). Registered module params
are written when they differ from their default. Controls are populated from the URL after
the option lists have been fetched, and the state follows what the server actually returned
(e.g. the resolved hit product). The camera is not in the URL.

### 3-D conventions

GPU buffers stay in detector coordinates; a root group rotates detector `(x,y,z)` into scene
`(z, y, -x)`: screen-right = z (beam), up = y, x (drift) toward the viewer; proper rotation,
no mirroring. A corner gizmo labels the axes. Space points: one `THREE.Points` + shader
(colour by value through a LUT texture, log/linear/auto, categorical by `sp/group/<kind>`,
or detector x/y/z). Tracks and truth: one indexed `LineSegments` each; radiologicals
(`mc/from_interaction == 0`) are hidden in the vertex shader from a per-vertex flag, so the
toggle never rebuilds anything. Truth colour: uniform `mctrack`, or per pdg class with colour
by "truth pdg". PD walls: `InstancedMesh` of quads with an instance colour attribute.
Colour-by values not present for space points (`amplitude`, `tick`, `multiplicity` use
`sp/<name>` when the exporter writes it, else charge).

### Testing

Headless Chromium + SwiftShader (the recipe in the README) crashes the GPU process on the
first shader draw on this machine (SEGV in JIT code; every WebGL context is lost after
about 1 s and not reliably restored). Use ANGLE on Mesa llvmpipe instead:
`--use-gl=angle --use-angle=gl-egl --ignore-gpu-blocklist` (no context loss). The viewer
also redraws after `webglcontextrestored`.

## WP-E implementation notes

File: `static/viewlight.js` only (no edits to main.js, index.html, style.css, view3d.js; its CSS is injected from the module, including a rule that hides the `#summary` aside in the pure Light view because the panel shows its own notes). Port of `light.py`: `frame_edges`/`_log_clock`, `light_onset`, `hit_time_quantum`, `frame_of`/`frame_coord`, afterglow and cumulative fill, `pe_range` (floor 0.5, off below 0.2, fixed log range for the whole window), `_flash_alpha`, `_tick_positions`, the strip with flash stems, the missing-flash warning (recomputed for the current t0), side-by-side vs stacked walls (aspect > 0.45), tile outlines from `pd/half`.

- Rendering: one instanced WebGL2 draw per wall (a frame change rewrites a 4-byte-per-tile buffer: rgb + outline flag; the outline is drawn in the fragment shader), a static Canvas2D under it (axes, TPC outlines from `meta.tpcs`, colour bar), a small dynamic Canvas2D above it (flash ellipses, PE labels of the 6 brightest, vertex star, clock, hover box). The strip is a cached bitmap plus elapsed shading and playhead. Per-frame cost is O(n_tiles) table lookups, no allocation, no geometry rebuild. All level quantisation (`lvl`, 255 log levels) is done in `rebuild()`, which runs on event/param change.
- Params (registered, in the URL when not default): `lclock` log|linear, `lfill` afterglow|cumulative, `lframes` (120), `lfps` (12), `lpers` (0.7), `lwlo`/`lwhi` (window relative to t0, us; -0.25 / 10), `lt0` (absolute us, empty = auto from `meta.optical.t0`), `lt` (frame index or `all` = whole window, default), `lplay` (0|1, autoplay on load). Burst jump buttons set `lt0` to the onset of the previous/next burst (100 ns bins above max(30 PE, 1e-3 of the peak bin), grouped within 1 us, onset by `light_onset`). The colormap/theme are the global `cmap`/`theme` state.
- Playback: rAF loop at `lfps`, holds 1.2 s on the last frame, then loops. During playback `state.lt` is written silently and committed to the URL on pause/scrub (no per-frame `replaceState`). Space toggles play when the panel is visible.
- Charge+Light: every rendered frame pushes the same tile colours with `view3d.setPdColors(rgb)` (tile order = `pd/xyz`); `setAnimating(true)` only while playing. Leaving the layout or hiding the panel pauses and restores neutral walls (`setPdColors(null)`).
- Interaction: wheel zoom / drag pan in the walls (one shared zoom/centre for all walls, kept across frames and events, reset by double click or a new geometry); strip click/drag scrubs; hover a tile shows tile, channels (`pd/chan_offsets`+`pd/channels`), PE so far, PE in the window.
- Notes line: t0/window/frames summary, hit and PE totals, flash count, true vertex, plus `meta.optical.notes` (association status, repaired flash positions) and the missing-flash warning in amber.
- `getCanvas()` composites static + GL + dynamic + strip. Test hooks: `window.__evdLight.setFrame(k)`, `.benchmark(n)`.
- Known gaps: no `meta.optical.notes`-only view when the event has no optical data (a short "No light data" message is shown); `light/edges` is not used (the clock is recomputed so the window/frames/t0 can be edited); when `oph/flash` has no links the strip shows one total curve; flash ellipses are not drawn in the 3-D view.

## WP-D implementation notes

`static/view2d.js` only; no edits to `main.js`, `index.html` or `style.css` (the view injects its own CSS and builds its own small control bar inside its panel).

- **Rendering**: one WebGL2 canvas for all panels (viewport + scissor per panel, one draw call per layer per panel) plus a transparent Canvas2D on top for axes, titles, anode (dashed) / active-volume (dotted) lines, vertices, labels, hover ring, and all pointer events. Hits and deposits are `gl.POINTS`; tracks, shower outlines and true trajectories are `GL_LINES` from the `*2d/points` buffers (widened by 1-px offset passes, trajectories dashed in the shader), shower cones also get a triangle-fan fill. Hits are depth-tested with `z` from the value so brighter hits are in front (as the matplotlib sorted scatter); unassociated hits sit behind objects. Instanced screen-space quads were tried first and were 5x slower under llvmpipe.
- **Controls never rebuild geometry**: colour-by, scale, cmap, theme and overlays set uniforms or bind another already-uploaded attribute buffer (value/category buffers are created lazily the first time a colour mode is used). Colour modes: charge / amplitude / tick / multiplicity, `x` (drift, or tick in readout), `track|shower|pfparticle|cluster|slice` (colour = rank among objects present mod 8, marker = rank mod 5 in the shape cycle o ^ s D v, grey circle for unassociated, same rule as `group_marker_batches`). `y`, `z`, `pdg` have no per-hit value in 2D and fall back to charge (the legend says so). Auto scale = log for charge/amplitude, limits p0.5..p99.5 (log span capped at 4 decades) as `_charge_scale`.
- **Overlays**: `truth` = `mc2d` dashed + true vertex X (`truth.vertex2d`); `rad` filters `mc/from_interaction` (and deposits) in the vertex shader; `dep` projects `dep/xyz` client-side with `panels[i].faces` (lazy, on first use; FD 307k deposits in ~0.1 s) and draws a faint underlay; `tracks`, `showers`, `vtx` (role from `vtx/role`: cross / diamond / circle), `labels` (`labels/*` projected with the faces, clipped to boxes). Colour `track` colours reco tracks by id; colour `pdg` colours true trajectories by class.
- **Views**: per space (`wire`, `readout`), per panel `{cx,cy,hx,hy}`. Initial framing = data extent of the hits +3%, with the shared-w rule of `figure()` (panels whose span is < 35% of the pooled span get their own range). The w axis is linked across panels in wire mode (toggle "link w"; a panel with its own range is not linked). Wheel = zoom about the cursor (shift = x only, alt = y only; wheel over the x / y tick strip zooms that axis), drag = pan, double-click or "reset zoom" = reset. Zoom/pan is kept over colour/overlay/theme changes, event steps and space toggles; it is dropped only by `setGeometry` (new file/detector) or an explicit reset. While wheeling/dragging deposits are drawn as a 12% subsample, full draw 160 ms after the interaction stops.
- **Readout mode**: own control "space" in the panel bar, URL param `space2d=readout` (registered with `registerParam`). Panels from `meta.panels_readout` indexed by `hits/chan_view`, h = `hits/channel`, v = `hits/tick`; hits only (no projected overlays).
- **Hover**: CPU grid per panel (built lazily on first hover, ~3 hits per cell), 7 px radius, tooltip with channel, tick, charge, amplitude, multiplicity, (w, x) or (channel, tick), and every reco group id of the hit.
- **Legend**: `.hud-legend` DOM element in a right-hand gutter (colour bar with 3 ticks for values, swatches + marker cycle for categorical, line/marker keys for the overlays present).
- **Screenshot**: `getCanvas()` renders synchronously and returns GL canvas + overlay composited (the legend DOM element is not included).
- **Test hooks**: `app.views.view2d.debugRender()` (sync render + `gl.finish`), `debugState()` (views, rects), `debugHitPx(panel, k)`, `resetView()`.
- Known gaps: no equal-aspect option; no vertex/label hover; legend not in screenshot; `size` slider is not in the URL; hit-point size is a CSS-px diameter growing mildly with zoom; the draw is not culled for panels outside the viewport (not needed so far).

## Parity work (round 2) — shared contract

Restores the Dash app's remaining features. Owners: F (main.js, index.html,
style.css, view2d.js, export.py), G (view3d.js, new viewoptical.js), H
(server.py, new static/exports.js). Shared names, fixed here:

- `state.view` gains `'optical'` (panel `#panel-optical`, owned by viewoptical.js)
  and `'flash3d'` (uses `#panel-3d`; view3d draws flash bars instead of / on top of
  the charge, see below). F adds them to the selector, the layout CSS and the view
  registry (`import viewoptical from './viewoptical.js'`, stub until G fills it).
- `state.ov.pvtx` (default on): the Pandora interaction vertex (`vtx/role == 1`),
  toggled separately from `state.ov.vtx` (daughter + secondary vertices, roles 2
  and 3). F adds the toggle and handles 2-D; G handles 3-D.
- colour-by `'chisq'` (space points only): exporter emits `sp/chisq` f32 (F); 3-D
  colouring (G); 2-D falls back to charge.
- `merge2d` URL param (`orientation` | `view` | `none`), registered by view2d (F).
  The exporter emits extra panel sets: `meta.panels_view`, `meta.panels_none`, and
  `hits/panel_view`, `hits/panel_none` u8. Projected overlays (`tracks2d`, ...) are
  emitted for those groupings too, under `tracks2d_view/...` and
  `tracks2d_none/...` with the same layout.
- `view3d.cameraInfo()` (G) → `{position:[x,y,z], target:[x,y,z], up:[x,y,z]}` in
  detector coordinates (cm). exports.js (H) uses it for the 3-D PDF and rotation GIF.
- exports.js (H) adds the "PDF" and "GIF" buttons to the toolbar itself (DOM
  injection); F adds only `import './exports.js'` to main.js.
- `GET /api/export?file=&entry=&hits=&flash=&kind=pdf|gif&view=<state.view>&...`
  (H) renders with the existing Python renderers (EventDisplay.figure,
  Display3D.figure / rotate_gif, OpticalDisplay, FlashDisplay3D,
  LightAnimation / ChargeLightAnimation save_animation), passing the relevant
  state (colour, scale, cmap, theme, overlays, merge2d, light params, camera).
- `POST /api/open` already exists (WP-B); F adds the "open by path" box.

## WP-G notes

Files: `static/viewoptical.js` (new), `static/view3d.js`.

- **Optical view** (`viewoptical.js`): one WebGL2 canvas (viewport+scissor per panel, one `gl.POINTS` draw each) under a Canvas2D overlay (axes, colour bars, dashed true-vertex z, dotted t0, hover ring, pointer events). Times are uploaded once per event as float32 relative to `meta.optical.t0`; pan/zoom only changes uniforms. Hits panel: z of the tile (`oph/tile` -> `pd/xyz`), channel index fallback; fixed log PE range 1..3e5, brightest drawn on top. Flash panel: size ~ log PE, colour = y over the TPC y extent. Wheel zooms time about the cursor, drag pans, double click / "reset" restores; param `owin` (half-width us or `all`, select in the panel header). Tooltips for hits and flashes; header shows counts in the window, brightest flash, `meta.optical.notes`. Test hooks: `app.views.viewoptical.benchmark(n)`, `.debugState()`. Timings (llvmpipe, 73.9k hits): setEvent ~100 ms, full redraw ~35 ms.
- **Flashes 3-D** (`view3d.js`): bars are screen-space-width quads (WebGL lines are 1 px); window t0 +- 50 us, 200 brightest, built once per event; colour = time on the cmap, width/opacity ~ log PE, square dot if `flash/x` finite. In `flash3d` charge, tracks, showers, truth, deposits, labels and reco vertices are hidden, PD tiles drawn faint (alpha 0.1), true vertex always shown. In Charge+Light a "flashes" checkbox (own DOM in the 3-D panel, param `fbars=1`) draws the same bars.
- **`view3d.setFlashTime(t_us | null)`**: show only flashes with time <= t (ABSOLUTE us, the clock of `oph/time` and `meta.optical.t0`); null = all of the window. Uniform only. viewlight can call it per frame (`t0 + frame time`); bars appear only when flash3d or the `fbars` box is active. `view3d.flashWindow` = `{t0, lo, hi, shown, total}`.
- **Vertices**: `ov.pvtx` (missing = on) -> role 1; `ov.vtx` -> roles 2/3; handled through per-vertex `aSize` (0 = clipped), no rebuild. Change: the true vertex marker now follows `ov.truth` (as in 2-D) instead of `ov.vtx`.
- **chisq**: `color=chisq` uses `sp/chisq`; if absent the legend title says it falls back to charge (same note for other missing columns).
- **`view3d.cameraInfo()`** -> `{position, target, up, fov, aspect}` in detector cm. view3d maps detector (x,y,z) to three.js scene (z, y, -x), so detector x = -scene.z, y = scene.y, z = scene.x. `up` is detector +y, `fov` is the vertical FOV in degrees.

## WP-F notes

- Exporter: `GROUPINGS` builds one extra `EventDisplay(merge="view"|"none")` each; shipped per suffix `_view`/`_none`: `meta.panels<sfx>` (same fields as `panels`, incl. `faces`), `hits/panel<sfx>`, `tracks2d<sfx>/`, `showers2d<sfx>/`, `mc2d<sfx>/` (same `mc2d_scope` rule), `vtx2d<sfx>/`, `meta.truth.vertex2d<sfx>`. Deposits/labels are projected client side with the chosen set's `faces`. Also `sp/chisq` f32.
- Cost (FD entry 1 / PDSP entry 3): +2.1 MB raw (13.4 -> 15.6 MB; gz 8.2 MB) and about +2.3 s (of 8.8 s total); PDSP +2.4 MB (45 -> 47.4 MB), about +0.2 s (mc2d is already from_interaction-only there).
- view2d: `merge2d` param (`orientation|view|none`), falls back to orientation if the bundle lacks the set; layouts/zoom/overlays are cached per grouping. 2-D `ov.pvtx` = vtx role 1, `ov.vtx` = roles 2,3. `chisq` falls back to charge with a legend note.
- main.js: `pvtx` added to OVERLAYS/OV_DEFAULT (old `ov=` URLs lacking `pvtx` turn it off); views `optical` (`#panel-optical`, viewoptical) and `flash3d` (`#panel-3d`, view3d); open-by-path box uses `/api/status.allow_open` and shows `{error}` inline (hidden in `?sample=` mode).

## WP-H notes

Exports (`pylarevd/web/render.py`, `GET /api/export` in `server.py`, `static/exports.js`).

- Query: `file entry hits flash kind=pdf|png|gif view=2d|3d|light|charge+light|2d+3d|optical|flash3d`
  plus `color scale cmap theme ov=<comma list incl. pvtx>`, `merge2d`, `space2d`, light params
  (`lclock lfill lframes lfps lpers lwlo lwhi lt0 lt`; window is relative to t0 as in the viewer, `lt`
  = frame number for a static light PDF, `all`/empty = whole window), camera (`campos=x,y,z`,
  `camtgt=x,y,z`, `camup`, detector cm, from `view3d.cameraInfo()`), `rotstep` (deg per frame, 3 default,
  the button sends 4) and `rotfps`. Bad values -> 400 JSON `{error}`; unknown file 404.
- Rendering runs in the server's ProcessPoolExecutor (`render.render_export`); response has
  `Content-Disposition: attachment; filename="pylarevd_run{R}_sub{S}_ev{E}_{view}.pdf"` (`_rotation.gif`,
  `_{flashtag}.gif` for light), `X-Export-Seconds`, and `X-Export-Note` (for 2d+3d: "exports the 2-D
  panels only"). Nothing is cached. PDF = 300 dpi, Type 42 fonts; PNG = 150 dpi.
- Camera -> matplotlib: view vector v = position - target in detector (x,y,z); matplotlib axes are
  (z,x,y), so elev = asin(v.y/|v|), azim = atan2(v.x, v.z); zoom = clip(1.15 * 1.45*D/|v|, 0.65, 1.8)
  with D the TPC bounding diagonal (1.45 D is the viewer's initial distance). Roll (`up`) is ignored.
  For charge+light the same elev/azim go to the charge panel.
- Colour fallbacks: `charge`->`integral`; 2-D falls back to integral for x/y/z/chisq/pdg; 3-D for pdg.
  `dep` and `truth` both switch the Python truth overlay on (it draws deposits and trajectories together).
- GIF buttons are enabled for 3d, light, charge+light only; static sample mode disables both buttons.

## WP-I notes (reco-object highlighting: particle / slice selection)

Files: `export.py` (`_export_reco_objects`), `static/select.js` (new: model + info card), `main.js`, `index.html`, `style.css`, `view2d.js`, `view3d.js`; tests in `tests/test_web_export.py`.

**Data** (`available.pfp`, `available.slice`, `meta.pfp`, `meta.slice`). All ids below are ROWS of the tables, i.e. the same ids as `hits/group/pfparticle` (PFParticle collection index) and `hits/group/slice` (Slice collection index).
- `pfp/{id (Pandora Self), pdg (13 track-like, 11 shower-like, 12/14/16 neutrino), parent (row or -1), top (top-level ancestor row), daughter_offsets + daughters (rows), slice (row or -1; a descendant without its own link inherits its top-level one), track (row of tracks/* or -1), shower (row of showers/* or -1), n_hits, n_hits_plane (n,3: U,V,Z from hits/chan_view), n_sp, score (TrackScore, NaN if none), vertex (row of vtx/* or -1), true_mc (row of mc/* or -1), true_pdg, true_tid}`; `tracks/pfp`, `showers/pfp` (owning PFParticle row of each exported track / shower; the best-match filter and the shower placeholder filter are replayed so rows line up with the exported `tracks/*`, `showers/*`).
- `slice/{id, n_hits, n_hits_plane, n_sp, n_pfp, top (top-level PFParticle row), nu (1 if that is a 12/14/16 neutrino), vertex (n,3; vertex of the top-level PFParticle, NaN if none), true_mc, true_pdg, true_tid}`. FD Slice centres are unset (FLT_MAX) in these files, so no slice centre is exported.
- `sp/group/<kind>` for every kind that has a hit grouping (hit -> space point association, as `Display3D._colour_values`); `pfparticle` is also filled from the PFParticle -> SpacePoint association where the hit route gives nothing, `slice` is derived from the PFParticle's slice if absent. Missing `hits/group/slice` is likewise derived from PFParticle -> Slice.
- True-particle match: no hit -> true-particle association is read. `pfp/true_*` reuse `labels/*` (reco track / shower -> nearest true trajectory, the rule of the particle labels): a shower-like PFParticle (|pdg| 11) tries its shower label first, others their track label first. Slices take the match of their PFParticle with most hits. On ProtoDUNE-SP the labels are matched to `from_interaction` particles only (see WP-A notes), so only 1 of 551 PFParticles has a match; no true energy is exported (MCParticles carries none).
- Degrades per step: if PFParticles are unreadable `available.pfp=false` with a `pfp:` note and the hit/space-point/slice groups are still exported; the viewer then builds a minimal table from the group ids ("Group n" card, no hierarchy). Slices unreadable but grouped: ids only.
- Sample cost: FD +0.25 MB, PDSP +1.1 MB raw; +0.02 s (FD) / +0.27 s (PDSP) of build time. The static samples were regenerated.

**Selection model** (`select.js`, wired from `main.js`). `app.state.sel = {kind:'pfp'|'slice', id, pinned, family}` or null; `app.selection` is the resolved form `{kind, groupKind:'pfparticle'|'slice', mask:Uint8Array (by group id), n, count, id, root, stamp}` and `app.on('selection', fn)` fires on every change. API: `app.select(sel)`, `app.clearSelection()`, `app.hoverObject({pfp, slice}|null, shift)`, `app.clickObject(ids, shift)`, `app.selectionTables()`, `app.vertexSelected(v)`, `app.emit(type, ...)`. `main.js` calls `app.resetSelection(bundle)` after `app.bundle` changes and before the views' `setEvent`.
- Hover: debounced 30 ms; clearing waits 120 ms after the pointer leaves every object (so it can reach the info card, which holds the hover). Click pins; clicking the same object, empty space, or Esc unpins. A pinned selection ignores hover. "select by" particle|slice (toolbar, `selby`, default particle); holding Shift while hovering/clicking uses the other one. `none` disables highlighting (hover, click, Shift, `f` and URL restore of `pfp=`/`slice=`). Hits/points with no PFParticle select nothing in particle mode. "others" slider (toolbar, `dim`, percent, default 30) sets the brightness of everything outside the selection (2D alpha, 3D blend to background); 100 leaves it unchanged.
- Family (toolbar checkbox or `f`, `fam=1`): from the top-level ancestor down through all daughters. Keys (pinned particle): `[` parent, `]` first daughter, `,`/`.` previous/next sibling.
- URL: only a pinned selection: `pfp=<row>` or `slice=<row>` (plus `selby`, `fam` when not default). It is restored on the first loaded event if the row exists; on later events the selection is cleared.
- Info card (`#sel-card`, bottom-left, `⇄` moves it right; visible in 2d, 3d, 2d+3d, charge+light): particle: Self id, type, track score, slice link, parent -> daughters (clickable, pin), hits per plane U/V/Z, space points, track length / shower length + opening angle, matched true particle; family count. Slice: neutrino hypothesis flag, hits per plane, space points, particle count by type, neutrino / top-level PFParticle, vertex, members (clickable -> switches to particle selection), matched truth. Buttons: pin, focus (3-D), side, close.

**2-D** (`view2d.js`): per-hit group id of the selection's kind is a lazily built attribute (`aSel`, gathered in layout order); a 1024-wide R8 flag texture (`uSelTex`, 255 = selected, indexed by group id) is rebuilt per selection change (a few hundred bytes). Selected hits: enlarged, opaque, ringed in the highlight colour, drawn in front; others 15 % alpha; true trajectories 18 %. Tracks and shower outlines/fills use the same lookup through a per-vertex id buffer built from `tracks/pfp` / `showers/pfp` (slice mode goes through `pfp/slice`); selected ones are re-drawn thicker in the highlight colour. Non-selected reco vertices are dimmed. Hover/click use the existing hit grid (7 px). No buffer is rebuilt on selection change.

**3-D** (`view3d.js`): space points, track and shower lines get `aSel` + the same flag texture (`DataTexture`, R8); selected points are 1.8x / >= 5 px with an outline, the rest is mixed 84 % into the background colour (opaque, no blending), selected lines take the highlight colour, true trajectories fade, non-selected vertex markers shrink. Highlight colour yellow (dark) / orange (light). The camera never moves on selection; the card's `focus` button or double-click recentres the orbit target on the selection (distance and direction kept; `view3d.focusSelection()`). Picking is CPU, screen space: candidate space points (those with a PFParticle or slice) are projected into an 8 px bucket grid, re-projected in the background 150 ms after the camera settles (13 ms on 110k points; 18 ms the first time), each mouse move (rAF-coalesced, not while a button is down) is a 3x3 cell lookup, frontmost point within 6 px; tracks and shower cone segments are point-segment tests within 5 px. A line wins over a point only if it is more than 1 px closer.

**Measured** (headless Chromium, ANGLE on llvmpipe, FD entry 1 and PDSP entry 3): 2-D pointermove handler (nearest hit + hoverObject) median 0.3 ms, p95 0.4 ms, max 1.1 ms; 3-D pick median 0.2 ms, max 0.4 ms (2.6 ms and 0.9 ms observed end-to-end per rAF pick); a selection change costs 0.2 ms median CPU (max 2.3 ms) before the scheduled redraw. 3-D steady-state frame 36 ms with or without a selection (the first frame after the first selection of a page costs ~110 ms extra, a one-off llvmpipe shader-variant compile). Test hooks: `view3d.debugPick(x, y)`, `view3d.pickMs`.

**Gaps**: selected 3-D track lines are 1 px (colour only, WebGL lines); the selected PFParticle's true-trajectory is not emphasised separately from the other fading truth (fixed in WP-T); no 2-D hover on the track/shower lines themselves (only on hits) and no pick of PD tiles / vertices; 3-D picks only objects whose space points belong to a PFParticle or slice; on a hit-product mismatch (`hits/group/*` absent) the 2-D hits cannot be highlighted (the PFParticle table and 3-D space points still work); the FD `hits/group/shower` index space (pylario) can differ from `showers/*` when a placeholder shower was dropped, so `showers/pfp` (not the group) is used for the line overlays; `Esc`/`f`/`[ ] , .` are ignored while a form control has focus.

## WP-T notes (true-particle filter, true-particle selection, link to the reconstruction)

Files: `export.py`, `cache.py` (`BUNDLE_VERSION` 3), new `static/truth.js`, `select.js`, `view3d.js`, `view2d.js`, `colormaps.js` (`pdgSymbol` lives here now; `select.js` re-exports it), `main.js`, `exports.js`, `index.html`, `style.css`; tests in `tests/test_web_export.py`.

**Data added** (per event bundle).
- `mc/generator` i16 (one per `mc/*` row): index into `meta.mc_generators` (module labels, e.g. FD `generator` = the GENIE neutrino, `Ar39GenInLAr`, `Kr85GenInLAr`, `CavernwallGammas...`; PDSP `generator` = beam including halo, `cosmicgenerator`, `ar39`, `kr85`, `rn222`, `ar42`), -1 unknown. Daughters carry the tag of their ancestor. Absent (with a `meta.notes` line) when the file has no MCParticle-MCTruth association; the viewer then only separates interaction / other.
- `dep/mc` i32 (one per deposit): row in `mc/*` of `abs(dep/track_id)` (EM-shower daughters carry a negated id), -1 if that particle is not in `mc/*` (it was dropped by `min_points=2`). Computed with sort + searchsorted (`_rows_of`). Not written when there are no trajectories. On the FD sample every deposit matches; PDSP has no deposits.
- Size: FD entry 1 +1.30 MB raw (15.87 -> 17.17 MB; gz 8.17 -> 8.21 MB), PDSP entry 3 +1.77 MB raw (48.6 -> 50.4 MB; gz 24.678 -> 24.684 MB). Build time unchanged (the lookup is ~10 ms).

**Filter model** (`truth.js`, `app.truth`, `app.on('truth', fn)`). Visible(trajectory) = species ON and origin ON and not individually hidden. The toolbar "truth" and "deposits" checkboxes stay the master switches (handled by the views as before).
- Species = the 9 `PDG_CLASSES`. Origin = `interaction` (`mc/from_interaction == 1`) plus one entry per generator for the particles that do NOT descend from the interaction (`unknown generator` for -1). Defaults reproduce the old look (checked pixel-for-pixel against `origin/main` in 3-D and 2-D, FD entry 1 and PDSP entry 3, 1 pixel of 723,600 differed by z-fighting): interaction on, every other origin off, all species on. Without a true vertex (`truth.has_split` false) nothing is "the interaction": the origins are the generators (or one `all particles` entry), all on. Without generators: `interaction` + `not from the interaction`.
- The model publishes one byte per `mc/*` row, `model.flags` (0 hidden, 1 visible, 2 visible and highlighted), laid out as an R8 texture (width 2048, row r at texel `(r & 2047, r >> 11)`), `model.version` (any change) / `model.stamp` (visibility change only) and `model.fbVis` (18 floats, index `cls + 9 * from_interaction`) for the deposits that belong to no trajectory (`dep/mc == -1`: species from `dep/pdg`, origin = interaction vs "any non-interaction origin on"). Deposits otherwise follow their particle through `dep/mc`.
- URL: `tsp` = hidden species names; `tog` = origin switches that differ from the defaults, `+Ar39GenInLAr` shown / `-interaction` hidden, `+*` / `-*` = every origin on / off first (what the "all" / "none" links write; this deviates from "hidden origins" because the default hides every generator, which would otherwise be in every URL); `thide` = individually hidden particles, first loaded event only (track ids are per event, cleared when you step to another event): `12s` = track 12 and its whole subtree, `5-120,300` = single particles as track-id ranges; skipped with a status note above 3000 characters. `mc=<track id>` = pinned true particle.
- **The `radiologicals` overlay is gone**: in the old viewer `ov.rad` / `uShowRad` only gated true trajectories and deposits (2-D hits were hard-wired to "shown"), so the origin filter replaces it. An old URL with `ov=...rad...` (and no `tog`) maps to `tog=+*`, all origins visible. `ov.rad` is no longer a state key; the PDF/PNG/GIF export asks `app.truth.radiologicalsOn()` (any non-interaction origin visible) and sends `rad` to the Python renderers, which know nothing else: **the matplotlib exports ignore the species / origin / per-particle filters** (and the highlight).

**Rendering** (no geometry rebuild on a filter change).
- 3-D (`view3d.js`): every vertex of the truth line buffer has `aMc` (the row); the vertex shader looks the flag up (`texelFetch`) and moves hidden vertices off-screen; highlighted ones take `uHi`, the other visible ones fade with the "others" slider (`app.selDim()`) when a selection exists (`uTruthSel`). The index buffer is sorted by (origin, species) key and `geometry.groups` leave fully switched-off keys out of the draw, so the default PDSP view still draws only the ~2k interaction trajectories (frame 34.8 ms, same as `origin/main`). Deposits: attributes `aMc` and `aFb` (`cls + 9 * from_interaction`) plus the `uFbVis` array. Highlighted trajectories are also drawn as 3-px screen-space ribbons (WebGL lines are 1 px; up to 120k highlighted segments, beyond that 1-px lines only), deposits of highlighted particles are enlarged.
- 2-D (`view2d.js`): same texture on texture unit 2; the mc2d line set gets a per-vertex row buffer (`aMc`, location 3) built from `mc2d*/index`, deposits get `aMc` / `aFlag` (= fallback code) per panel; highlighted trajectories get a second, thicker, solid pass in the highlight colour. Note `mc2d` projects only the interaction particles for events with more than 400k trajectory points (PDSP), so the other origins show up in 3-D only; the drawer says so.

**Drawer** (`#truth-drawer`, "truth" button next to the truth checkbox, `x` / Esc to close: Esc first clears a selection, then closes). Docked as an extra grid column on windows wider than 900 px (the views resize), an overlay below that; layouts use `--cols` per view so every view keeps its tiling.
- Categories: species (colour swatch, visible / total in this event), origins ("interaction" first, then generators by count, ellipsised with the full name as tooltip), `all` / `none` per section, footer `showing N / M trajectories` (+ visible deposits).
- Particles: lazy tree. Top group "interaction" = the interaction's primaries (from_interaction particles whose mother is not in that set) and descendants by mother track id; one collapsed group per other origin (its primaries: no mother drawn, or mother in another origin; the data have `mother == 0` for those), paginated 200 at a time ("show 200 more"); the group header has the origin checkbox. Row: tri-state box, species swatch, symbol (`pdgSymbol`), `#track id`, creation process, length, `(+N)` descendants. Rows hidden by a category filter are greyed with a tooltip. Hover = transient selection, click = pin (click again unpins).
- Tri-state (derived from the hidden set): checked = particle and whole subtree visible, half = particle visible and some descendant hidden, empty = particle hidden. Click: checked -> half ("only this particle": hides all descendants) -> empty (hides the particle and its subtree) -> checked (shows the particle and the whole subtree); a leaf toggles checked/empty. Descendants can be toggled individually afterwards. `reset` clears the overrides.
- Search: name (`mu-`, `proton`, `gamma`; an exact symbol wins over substrings), `tid:123` (or a plain number), `pdg:2212`. The tree shows the matches with their ancestors expanded (first 100 matches per origin), scrolls to the first; Enter pins it.
- Structures (track-id table, mother rows, children CSR, descendant counts via BFS, roots per origin) are built on first use per event: 13 ms FD, 53-72 ms PDSP entries 1-3 (886k trajectories); `resetTruth` (class / origin / key per trajectory) 1 ms FD, 14-27 ms PDSP.

**Selection** (`select.js`). `app.state.sel.kind` may now be `'mc'` (id = row of `mc/*`; `family` = the particle's subtree). It resolves to a normal `pfparticle` mask (the PFParticles with `pfp/true_mc` in the selected rows) so the 2-D / 3-D reco highlighting is unchanged, plus `selection.mcRows`. `truth.js` turns any selection into trajectory highlighting: an mc selection highlights its rows (subtree in family mode), a PFParticle (or family) its matched `pfp/true_mc` rows, a slice its `slice/true_mc`; every other visible trajectory / deposit fades with "others". API: `app.hoverMc(row)`, `app.clickMc(row)` (tree rows, work with "select by: none"), `app.hoverObject({pfp, slice, mc})` / `clickObject` (an `mc` id counts only when no reco object is under the cursor; "select by: none" disables it). 3-D picking: the visible trajectories are projected once per camera / filter change (background after the camera settles), per-trajectory screen bounding box, then point-segment distance within 5 px. 2-D picking (stretch goal, done): where no hit is under the cursor, the nearest visible `mc2d` segment within 5 px. Keys for a pinned true particle: `[` mother, `]` first daughter, `,` `.` siblings (primaries: the primaries of its origin), `f` family. Card: symbol, track id, origin / generator, process, length, mother and daughters (clickable), deposits, matched reco particles (clickable) or a clear "no reconstructed particle matched" (on PDSP only particles of the interaction have matches, noted), "hidden by the truth filter". PFParticle / slice cards link their `true` line back to the true particle.

**Measured** (headless Chromium, ANGLE on llvmpipe; JS = click handler incl. filter recompute, texture marking and drawer refresh; frame = synchronous 3-D render drained with `readPixels`; median of 5, FD entry 1 / PDSP entries 1, 2, 3):

| action | FD e1 | PDSP e1 / e2 / e3 |
|---|---|---|
| 3-D frame with no filter change (reference) | 26 ms | 32 / 36 / 38 ms |
| species toggle: JS + frame | 4.8 + 25 ms | 4.4 + 34 / 4.3 + 37 / 5.7 + 38 ms |
| small origin toggle: JS + frame | 4.7 + 26 ms | 3.2 + 33 / 2.7 + 38 / 3.6 + 38 ms |
| tri-state click in the tree: JS | 8 ms | 11 / 11 / 13 ms |
| mc selection change (tree hover), incl. ribbons: JS + frame | 0.8 + 27 ms | 0.8 + 32 / 0.8 + 36 / 0.8 + 40 ms |
| cosmic origin switched ON (584k / 681k / 868k trajectories): JS + frame | n/a | 2.8 + 99 / 2.8 + 115 / 3.4 + 137 ms |

A filter click costs the JS above plus one ordinary frame (the frame with a changed filter is the same as the frame without any change, so the filter itself adds nothing measurable to the rendering): about 30-45 ms from click to redraw, below the 50 ms target, except when hundreds of thousands of extra trajectories are switched on, where it is the software rasteriser drawing 3.6M line indices (99-137 ms). 2-D (PDSP e3): JS 3.5-6 ms, the 2-D frame is 98 ms with or without a change (FD with 307k deposits shown: ~790 ms, unchanged from `origin/main`). 3-D pick: median 0.3 ms with the interaction only; the first pick after switching the cosmics on 168 ms (the background prewarm normally absorbs it), then median 4 ms / max 8 ms with 584k extra trajectories; 2-D mc-line pick median 0.1 ms. Event load is unchanged within noise (`setEvent` PDSP e3 599 ms vs 593 ms on `origin/main`, FD 218 vs 196 ms; `view3d.setEvent` alone 117-137 vs 140-158 ms).

**Known gaps.** The matplotlib PDF / PNG / GIF exports ignore the species / origin / per-particle filters and the highlight (only the all-or-nothing `rad` is derived); 2-D lines of non-interaction particles do not exist on PDSP (mc2d scope); the highlight ribbon is 3 px in 3-D only, 2-D uses the 3-px pass; in 3-D a trajectory is only picked where no reco object is within the pick radius; the tri-state box shows the state of the particle, not of a hidden ancestor (a visible child of a hidden parent is checked, as asked); reco-truth matches follow the exporter's labels (on PDSP only particles of the interaction can be matched; e.g. PDSP entry 1 has none); the first hover of a true particle per event pays the tree build (13-72 ms); bundles written before version 3 are not served (the cache key includes the version); the drawer open state, tab and tree expansion are not in the URL; deposits without a trajectory use the interaction / any-other-origin rule rather than their generator; `Esc` / `f` / `[ ] , .` are ignored while a text field has focus.

## Export performance (2026-10-22)

Event-bundle build times, same pylario (with its bulk MCParticle decoder) before and after this round,
bundles checked identical array by array (meta minus timing):

| event | before | after | what went |
|---|---|---|---|
| ProtoDUNE-HD BSM MC, entry 5 (861k particles, 393 showers, 200 tracks) | 44.6 s | 13.8 s | `EventDisplay._faces` rescanned the panel's hits on every call (once per shower cone per panel): now remembered per panel; cones projected all at once per panel |
| FD atmospheric, entry 1 | 4.7 s | 2.5 s | true trajectories stacked with two `vstack`s each, per panel: now one vectorised stack reused by every panel |
| FD atmospheric, entry 4 | 3.4 s | 1.2 s | same |
| ProtoDUNE-SP, entry 1 | 8.9 s | 8.4 s | dominated by pylario (truth post-processing, neutrino-descendant ids) |

Particle labels: reco tracks are matched against per-particle bounding boxes, and particles whose
mean distance to the box already exceeds the 15 cm cut are skipped (an exact bound; same labels).
What remains on the heavy samples is mostly pylario: `mc_particles` post-processing and `_freeze`,
`neutrino_track_ids`, the generic `recob::Track` decode.
