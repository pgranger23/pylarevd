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
