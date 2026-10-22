# pylarevd — a LArSoft event display in pure python

Reads **art-ROOT files directly** and draws reconstructed hits in physical
coordinates. At display time there is no ROOT, no art, no LArSoft, and no
compiled extension — just `uproot` + `numpy` + `plotly`/`matplotlib`.

```
                       ┌──────────────── one-time, needs LArSoft ─────────────┐
   GDML + fcl  ──────► │ shim/libevdgeom.so  ──ctypes──►  export_geometry.py  │ ──► geom/*.npz  (in pylario)
                       └─────────────────────────────────────────────────────┘
                                                                                        │
   art-ROOT file ──uproot──► pylario (artio + event) ──► pylarevd (display) ──► PNG / PDF / HTML ◄┘
                            (pure python, no LArSoft)
```

> **Installing?** See **[SETUP.md](SETUP.md)** — install, first run, geometry,
> reading files over xrootd/EOS, running the tests, and a troubleshooting table.
>
> Already have it? `python -m pylarevd --check` reports what is installed, what
> is missing and how to install it.

## The web viewer (new)

A faster browser viewer: each event is decoded once in Python into a compact
binary bundle, and everything after that (colour, scale, overlays, theme, light
animation, camera) runs in the browser with WebGL, so controls respond
immediately and the 3-D camera is never reset. Plain ES modules, no build step.

```bash
python -m pylarevd.web reco.root [more.root ...] --port 8050 [--precompute] [--workers N]
ssh -N -L 8050:localhost:8050 <your-host>      # on your laptop, then open http://localhost:8050
```

The server binds to loopback only, like the Dash app. `--workers N` sets the
decoder processes (default min(8, cores)); the next two events are decoded in
the background while you look at one; `--precompute` also fills the cache for
the whole file in the background.

**Views** (selector in the toolbar):

| view | shows |
|---|---|
| 2D | one panel per wire plane, hits and projected overlays; `space` switch between physical (wire, x) and readout (channel, tick) |
| 3D | space points, tracks, shower cones, vertices, true trajectories, deposits, TPC boxes, photon-detector walls |
| Light | PD walls face-on, time strip, flash ellipses, play/pause, slider, log or linear clock |
| Charge+Light | the 3-D view with the PD walls lit by the same time state as the Light panel |
| 2D+3D | the two side by side |

**Controls.** File, event (prev/next buttons, left/right arrow keys, or type an
index in "go to" and press Enter), hit product, flash producer, colour by
(charge, amplitude, tick, multiplicity, x/y/z, track, shower, pfparticle,
cluster, slice, truth pdg), scale (auto/log/linear), colormap, dark/light
theme, overlay checkboxes (truth, radiologicals, deposits, tracks, showers,
vertices, labels, PD), reset view, screenshot (PNG of the visible panels) and
link. Files with the online trigger chain (`dunetrigger` TPs, TAs and TCs, e.g.
the ProtoDUNE-HD BSM MC) add **TPs / TAs / TCs** checkboxes for the 2-D views:
primitives as squares with a time-over-threshold bar (those in an activity
highlighted), activities as boxes, candidates as time bands; hover a primitive
for its details. Mouse: 3-D drag to rotate, right-drag to pan, wheel to zoom; 2-D wheel
to zoom, drag to pan, double click to reset; hover shows hit info. The summary
line under the panels expands to the full event summary on click.

**URL sharing.** Every control, including the open file and entry, is in the
query string and restored on load; "link" copies it. The camera is not saved.
`?sample=fd` or `?sample=pdsp` opens a bundled static sample without a server.

**Cache.** Decoded events are cached in `~/.cache/pylarevd` (override with the
environment variable `PYLAREVD_CACHE`), keyed on file path, mtime and size, so
a revisited event opens instantly and several people can share one cache. To
decode a whole file up front, in parallel:

```bash
python -m pylarevd.web.export reco.root -j 32          # also: -e 0-9, --hits TAG, --flash TAG
```

Design and format notes are in [docs/WEB_VIEWER.md](docs/WEB_VIEWER.md). The
original Dash app (`python -m pylarevd.app`, next section) still exists and is
unchanged.

## The Dash browser app

This is how most people use pylarevd. It is a small web app: you run it **on the
machine that has the files**, and view it **in the browser on your laptop**
through an SSH tunnel. Nothing is rendered on your laptop and no data is copied
to it.

**Step 1 — on the machine with the files:**

```bash
# on a CERN/DUNE machine everything is already available:
source /cvmfs/sft.cern.ch/lcg/views/LCG_110/x86_64-el9-gcc13-opt/setup.sh
export PYTHONPATH=/path/to/pylarevd:$PYTHONPATH
# elsewhere:  pip install -e ".[all]"

python -m pylarevd.app reco.root [more.root ...] --port 8050

# or start with nothing and open files from the browser:
python -m pylarevd.app --port 8050
```

Leave it running. It prints the tunnel command for you.

**Step 2 — on your laptop, in a second terminal:**

```bash
ssh -N -L 8050:localhost:8050 <your-host>
```

This command produces no output and does not return — that is correct, it is
holding the tunnel open. Leave it running too.

**Step 3 — open <http://localhost:8050> in your browser.**

Note `localhost`, not the remote hostname: the tunnel makes the remote port
appear as a local one. The server **binds to loopback only**, so it is not
reachable from the network and there is nothing to expose — which is also why
the tunnel is required rather than optional.

If port 8050 is taken (a colleague on a shared machine, or your own earlier
session), use `--port 8051` and change both numbers in the tunnel to match.

Once it is up: step through events with the ◀ ▶ buttons or the **left/right
arrow keys**, switch between 2-D physical, 2-D readout, 3-D, optical, 3-D
flash and animated light views, colour hits by charge or by the object that owns them, toggle truth
and reco overlays, and switch theme and colormap. Every control is mirrored into
the URL, so you can paste a link to exactly what you are looking at — see
[Browser reference](#browser-reference) for the details, including opening files
that were not on the command line.

## Command line and python API

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_110/x86_64-el9-gcc13-opt/setup.sh
export PYTHONPATH=/path/to/pylarevd:$PYTHONPATH

# what is in the file?
python -m pylarevd reco.root --list

# static image + interactive page for events 0-2, with truth overlaid
python -m pylarevd reco.root -e 0-2 --html --truth -o evd_out
```

```python
from pylarevd import EventFile

f  = EventFile("reco.root")          # geometry auto-detected from pylarevd/geom/
ev = f[0]

ev.display().save("event0.png")             # static, publication quality
ev.display(truth=True).save_html("ev0.html")  # interactive, self-contained
ev.display_3d().show()                        # 3-D space points

h = ev.hits()                     # numpy columns, physical coordinates attached
h.w, h.x, h.integral, h.view      # wire coord [cm], drift [cm], charge, plane
bright = h[h.integral > 200]      # boolean-mask selection
```

## Why this needed custom decoding

`uproot` opens art files but **cannot read the data products**:

```
ValueError: basket 0 in tree/branch recob::Hits_hitfd__Reco1.obj has the wrong
number of bytes (727141) for interpretation AsStridedObjects(Model_recob_Hit_v17)
```

art writes `std::vector<T>` **member-wise** (ROOT's `kStreamedMemberWise` bit,
version `0x4009`). uproot's automatic model assumes the object-wise layout and
computes the wrong stride. Member-wise is *columnar*:

```
byte count (4) │ version|0x4000 (2) │ value-class version (2) │ N (4)
[ member 1 × N ][ member 2 × N ][ member 3 × N ] …
```

which is actually the better layout for us — each member is a contiguous
big-endian column, so a whole event decodes as a few numpy views with no
per-object work. **6671 hits decode in 4.8 ms.**

`artio.py` derives the layout from the **streamer info in the file itself**, so
it works for any product whose members are PODs or nested POD structs, rather
than hard-coding `recob::Hit`. Three rules were established byte-exactly:

| Rule | Effect |
|---|---|
| Nested objects keep per-object headers inside their column | `geo::WireID` = 6+6+6+6 headers + bool + 4×uint = **41 B** |
| "Foreign" classes (no `ClassDef`, version 1) add a 4-byte checksum | `ROOT::Math::PositionVector3D` header is **10 B**, not 6 |
| `Double32_t` is a 32-bit float on disk, and `fArrayLength` gives array extent | `recob::SpacePoint` = **44 B**, not 28 |

Verified totals: `recob::Hit` 109 B, `sim::SimEnergyDeposit` 132 B,
`recob::SpacePoint` 44 B — each matching the on-disk basket size exactly.

`art::EventAuxiliary` also defeats uproot (bundled model is a different
version), so run/subrun/event is recovered by walking ROOT's self-describing
byte-count prefixes.

## Geometry

Wire positions come from **LArSoft's own channel map**, never from hard-coded
angles or pitches. `shim/geom_shim.cc` is a ~150-line flat-C wrapper around
`lar::standalone::SetupGeometry`, driven from python by `ctypes`.

> **Why a compiled shim rather than pure PyROOT/cppyy?**
> Driving *gallery* from PyROOT works with no compilation at all (verified).
> Driving the *geometry* does not: `larcorealg/Geometry/GeometryCore.h` pulls in
> `range/v3`, which includes `<span>`, and cling's modulemap gates that behind
> C++20 while ROOT 6.28 runs C++17 — it segfaults. Forcing `-std=c++20` breaks
> against the C++17 PCH. `ctypes` over a `.so` bypasses cling entirely.

Export once per geometry (inside the SL7 container), then never again:

```bash
./inlar.sh "bash shim/build_shim.sh"
./inlar.sh "python -m pylarevd.export_geometry \
    --fcl geom_dune10kt_1x2x6.fcl --out pylarevd/geom/dune10kt_v6_1x2x6.npz"
```

The `.npz` (0.6 MB) holds every wire endpoint, the per-plane drift
linearisation and TPC active volumes. Nothing in the display is
detector-specific — point it at a different geometry's `.npz` and it works.

## Wire wrapping

In a DUNE FD-HD APA the induction wires wrap around the frame, so one readout
channel is bonded to segments on **both** drift faces. Anything plotted against
*channel* folds the detector onto itself and a single muon breaks into ~6
disconnected clusters. This is undone geometrically, per wire:

```
u = (uy, uz)     wire direction from the geometry, canonicalised to uy ≥ 0
p = (-uz, uy)    measurement direction, perpendicular to the wire
w = p · (y, z)   continuous wire coordinate [cm]   (reduces to z for collection)
```

paired with `x = x0(plane) + slope(plane) · tick`.

Because the wires wrap, **U on one face is parallel to V on the other** (−35.71°
vs +35.71° in `dune10kt_v6_1x2x6`). So panels are grouped by *wire orientation*,
not by plane label — that is the grouping under which both drift volumes share
one consistent `w` axis and a track stays continuous through the anode. Verified
against truth: 0.03–0.11 cm in both volumes using a single common frame.

| `merge=` | panels | note |
|---|---|---|
| `orientation` (default) | 3 | one per wire direction; mixes U and V channels by design |
| `view` | 3 | one per U/V/Z label; `w` is *not* continuous at the anode for induction |
| `none` | 6 | one per (view, drift volume); mirrors how Pandora reconstructs |

`space="readout"` switches to the classic **channel vs tick** view that LArSoft's
own display uses. It needs no geometry, so it works on undisambiguated hits — and
it deliberately folds the wrapped faces together, which is what the electronics
actually sees.

A dashed line marks the anode; dotted lines mark the active-volume edge. Charge
that drifts in from another readout window (radiologicals — ~6.6% of hits in the
`atmnu_radio` sample) lands beyond those dotted lines, since the tick→x map has
no t0 correction.

## Validation

The display is checked against MC truth, not just eyeballed:

| Check | Result |
|---|---|
| `PlaneWireToChannel(WireID) == Hit::Channel()` | **6671 / 6671** hits |
| Geometry covers every readout channel | **30720 / 30720** |
| Median hit → true energy deposit distance | **0.051–0.053 cm** (U/V/Z) |
| Drift velocity implied by `slope` | 0.0803 cm/tick = 0.16 cm/µs at 2 MHz |
| Decoder vs. gallery/LArSoft C++ on the same hit | identical to all printed digits |
| Track trajectory points vs. space points | **6675 / 6675**, median 0.051 cm apart |
| Brightest optical flash vs. TPC vertex | t = 0.07 µs, 10375x background, Δz = 50 cm |
| Hits per track (associations) vs. trajectory points | identical, 11 tracks |
| Wire ROIs satisfy `last - offset == len(values)` | 10305 / 10305 |
| SimChannel IDEs vs. the true muon trajectory | 2.6 cm (trajectory sampling) |

0.05 cm is **10× below the 0.467 cm wire pitch**; a wrapping error instead puts
a whole population metres off the track.

```bash
python -m pytest tests/ -q     # 170 tests, ~3.5 min
```

## Themes, colormaps and output targets

```python
ev.display(theme="light", colormap="magma", preset="paper2").save("fig.pdf")
python -m pylarevd f.root -e 0 --theme light --colormap magma --preset paper2
python -m pylarevd --list-colormaps
```

**Themes** are `dark` (screen) and `light` (print). Every colour lives in
`theme.py`, in two complete sets. Two invariants the module exists to keep:
object-identity colours and overlay/chrome colours are **disjoint** (they shared
four exact hex values before, so a hit could be painted the same amber as the
shower cone drawn over it), and identity never rests on colour alone — the
8-colour palette is cycled against 5 markers, giving 40 distinguishable objects
and surviving greyscale and colour-vision deficiency.

**Colormaps**: `turbo`, `magma`, `viridis`, `cividis`, `inferno`, `plasma`.
`turbo` is the default because it is the familiar rainbow and reads well on
screen — but it is the one ramp whose lightness is *not* monotonic (129 of 255
steps reverse, net lightness change 0.03), so charge ordering flattens in
greyscale. `--list-colormaps` says which are monotonic; pick `magma` for print.

**Presets** set figure size and font scale together, targeting the final
physical size: `screen`, `paper1`, `paper2`, `slide`, `poster`. Authoring at the
target beats shrinking a 14-inch figure into a paper column, where an 8 pt label
lands at 0.68 mm.

## Styling

Both backends render a dark theme, defined once at the top of `display.py`:

```python
FIG_BG   = "#0f172a"   # dark navy figure    AXES_BG = "#020617"  # near-black panels
LEGEND_BG= "#1e293b"   LEGEND_EDGE = "#475569"
FG       = "#e2e8f0"   FG_MUTED = "#94a3b8"   GRID = "#334155"
SAVE_DPI = 180
```

Images are saved with `facecolor=fig.get_facecolor()`, without which matplotlib
writes a white border around the dark panels.

Charge is drawn on a **selectable ramp with a logarithmic colour axis**. Both
matter: hit integral spans decades — a minimum-ionising track sits far above the
radiological background — so on a linear ramp nearly every background hit
collapses into the bottom colour and the picture loses its texture.

`turbo` avoids `jet`'s worst false banding, but it is **not** luminance-monotonic
— see the colormap table above. Each theme trims the ramp at whichever end would
vanish against its background (the near-black bottom on dark, the near-white top
on light).

Hits are drawn at 75% opacity and sorted by charge, because 76–87% of markers in
a busy panel are overlapped: without that, draw order rather than charge decides
what you see.

Scaling is per-display: `colour_scale="auto"` (default) uses log for `integral`
and `amplitude`, linear for `tick` and `multiplicity`; force it with
`colour_scale="log"|"linear"`, or `--colour-scale` on the CLI. Non-positive
charges are clamped to the bottom of the scale rather than dropped. plotly has
no logarithmic colour axis for markers, so the interactive backend colours by
log₁₀ and relabels the ticks — the hover box still reports the real value.

## Layout
 
`pylarevd` builds on the pure-Python reader and data model provided by **`pylario`**:

| Package | Path | Role |
|---|---|---|
| `pylario` | `pylario/artio.py` | member-wise art-ROOT decoder + `art::Assns` |
| `pylario` | `pylario/streamers.py` | sequential reader for variable-length products |
| `pylario` | `pylario/geometry.py` | wire coordinates, drift conversion, orientation keys |
| `pylario` | `pylario/event.py` | `EventFile` / `Event` / `Hits` user API |
| `pylario` | `pylario/physics.py` | PDG names, interaction modes, kinematics |
| `pylarevd` | `pylarevd/display.py` | panels, 2-D/3-D/optical rendering (matplotlib + plotly) |
| `pylarevd` | `pylarevd/light.py` | photon-detector hits and flashes animated in time |
| `pylarevd` | `pylarevd/theme.py` | themes, colormaps, and marker cycles |
| `pylarevd` | `pylarevd/cli.py` | `python -m pylarevd` batch rendering |
| `pylarevd` | `pylarevd/app.py` | `python -m pylarevd.app` interactive browser (Dash) |
| `pylarevd` | `pylarevd/_deps.py` | environment and dependency checker |

## Reconstructed and true objects

Products carrying variable-length members (a trajectory of N points, a sparse
waveform, a vector of daughters) do not fit the fixed-stride decoder, so they go
through a sequential reader (`streamers.py`). That covers:

| | |
|---|---|
| `recob::Track` | trajectories as 3-D polylines (masked -999 points dropped) |
| `recob::Shower` | start, axis, length, opening angle |
| `recob::Vertex` | interaction points |
| `recob::PFParticle` | Pandora hierarchy (parent/daughters) |
| `simb::MCParticle` | true trajectories, PDG, process |
| `recob::OpFlash` / `recob::OpHit` | photon-detector activity |
| `recob::Wire` | sparse ROI waveforms |

```python
ev.tracks(); ev.showers(); ev.vertices(); ev.mc_particles(); ev.optical()
ev.display(reco=True, truth=True).save("event.png")   # tracks, vertices,
                                                      # shower cones, true paths
ev.display_optical().show()                           # PDS time view
ev.display_3d().show()                                # 3-D with track polylines
```

Three on-disk rules had to be established byte-exactly, none of them documented:

- A **variable STL member's column** carries one shared 6-byte header, then a
  4-byte count per object.
- A member-wise container of a **foreign** class (no `ClassDef` — every
  `ROOT::Math` type) writes `version 0 + checksum` and then stores its elements
  *object-wise*, each with a 10-byte prefix.
- Whether a by-value member carries a prefix is **not recorded anywhere**, and
  the obvious test fails — a coordinate like 36.85 has the byte-count bit set in
  its float encoding. So both readings are tried and the one fitting the
  enclosing byte count wins.

A fourth rule matters most: a member-wise vector of a class is stored **one
column per member, recursively**. Missing that is what made `recob::Wire`
decode 30720 objects of which only the 5120 *empty* ones parsed — every wire
carrying a waveform silently reported no signal.

The invariant that keeps this honest is that a parse must consume its byte
count **exactly**. Accepting "at most" lets a truncated read look identical to
a correct one. Where a member genuinely cannot be modelled, the byte count says
where the object ends, so the reader resynchronises and flags the shortfall
under `streamers.UNPARSED` — but it can no longer do that by accident.

Decoded products are cached per event, and streamer lookups per file; without
the latter, decoding eleven tracks cost 125k redundant streamer resolutions.

## Associations

`art::Assns<L,R,D>` streams as two parallel vectors of `art::Ptr`. An `art::Ptr`
is dictionary-emitted as `pair<art::RefCore, unsigned long>`, and because the
vector is member-wise the pair is stored **columnar** — every RefCore (22 B: a
6-byte header, the ProductID, and empty transients), then every key. Both
columns are fixed-stride, so an association decodes as two numpy views.

```python
ev.hit_group("track")        # per hit: index of its track, or -1
ev.hit_group("pfparticle")   # via clusters, which is how Pandora records it
ev.display(colour_by="track", reco=True).save("by_track.png")
```

`colour_by` accepts `track`, `shower`, `slice`, `cluster` and `pfparticle` as
well as the continuous quantities; hits belonging to nothing stay grey.

`hit_group` refuses a `tag` that selects a different hit collection from the one
reconstruction associated against — the indices would otherwise be silently
wrong (cluster members are single-plane-pure 24/30 against `hitfd`, 5/30 against
`gaushit`).

Note that several modules write the *same* association — `pandoraShower` also
emits `Hit<->Track` links for its own track collection — so the producer is
pinned per route rather than guessed. Assns carrying data (`recob::TrackHitMeta`)
have a third vector this reader does not model, so the plain `void` variants are
preferred.

### Browser reference

[Running it](#the-interactive-browser-start-here) is at the top of this file.
It steps through events without regenerating anything: file/event selection with
prev/next, hit product, colour-by (including by owning object) and scale,
2D physical / 2D readout / 3D / optical / flashes 3D / light vs time /
charge + light, panel
grouping, the flash producer and light-animation settings, and truth and
reco overlays. Controls a view ignores are greyed out — including the truth and
reco checkboxes, which only apply to the physical views. Binds to localhost only
unless you pass `--host`.

Files listed on the command line appear in the dropdown; anything else can be
opened from the **open by path** box by pasting a full path and pressing Enter:

```
/eos/user/<u>/<user>/prodgenie_atmnu_..._reco1.root
```

It joins the dropdown alongside the others, so you can flip between an ad-hoc
file and the ones you started with. `~` is expanded. Bad paths report why
(missing, a directory, unreadable, not an art file) next to the box instead of
producing an empty plot. Files that share a basename are labelled with their
parent directory so two `reco.root` from different passes stay distinguishable.

Because this reads any file the account can reach, it is enabled only when the
server is bound to loopback — the default. With `--host` set to anything else,
the box is disabled unless you also pass `--allow-remote-open`.

Every control is mirrored into the URL, so the address bar always describes what
you are looking at and can be pasted to someone else:

```
http://localhost:8050/?file=/eos/user/r/.../reco.root&entry=3&tag=hitfd&colour=track
    &scale=auto&mode=2d&merge=view&theme=light&colormap=magma
    &truth=1&reco=0&rse=20000063:0:4
```

The file is recorded as a full path, so a link opens correctly even against a
server that was started with different arguments — it opens the file on arrival
(subject to the same loopback rule above). If the file is missing, the rest of
the link is still applied, so you keep the view and lose only the file.

`rse` is the run:subrun:event of the displayed event, and it outranks `entry`
when the link is opened. Entry numbers are an artefact of how a file was
written — the same physics event sits at a different entry in a different file —
so a link carrying the event identity still lands correctly when pointed at
another reconstruction pass of the same data. If the file does not contain that
event, the link falls back to `entry` rather than failing.

The CLI covers the same ground: `--mode {2d,readout,3d,optical,flash3d,light,chargelight}`,
`--reco`, `--flash-tag`, and the full `--colour-by` set.

## True neutrino interaction

`simb::MCTruth` carries the generator record. It is decoded into a `Neutrino`,
and every view puts the headline in its title:

```python
nu = ev.neutrino()          # None for cosmic / radiological samples
nu.headline()               # 'nu_tau CC DIS   E = 18.76 GeV'
nu.is_cc, nu.mode_name      # True, 'DIS'
nu.energy, nu.vertex        # GeV, (3,) cm
nu.w, nu.x, nu.y, nu.q2     # inelasticity / Bjorken kinematics
nu.target_name              # 'Ar-40'
nu.fs_pdg, nu.fs_energy     # final state: what Geant4 was handed
print(nu.describe())
```

```
nu_tau CC DIS   E = 18.76 GeV
  target Ar-40   hit nucleon n   beam neutrino
  W = 2.17 GeV   x = 0.569   y = 0.251   Q2 = 5.03 GeV2
  vertex (-220.2, -38.9, 557.8) cm
  final state: pi- (6.82 GeV), nu_tau (6.20 GeV), p (3.66 GeV), 2x pi0 (3.00 GeV)
```

The integer codes are translated in `pylarevd.physics` (mode, interaction type,
origin, GENIE status, PDG names including nuclei such as `1000180400` ->
`Ar-40` and 2p2h clusters). An unrecognised code is reported as its number
rather than given a made-up name.

A sample can hold dozens of `MCTruth` products -- a physics generator plus one
per radiological chain -- so the one labelled `generator` is preferred, and
records with no neutrino set return `None` instead of a row of zeros.

Decoding this exposed a reader bug worth naming: a wrong framing guess made
`n = cur.u4()` read a garbage element count, and the reader faithfully parsed
the phantom elements until the buffer ran out. One `simb::MCTruth` turned into
2.35 million `TLorentzVector` reads and never finished. An element count larger
than the bytes remaining is now rejected outright, which takes the same decode
to 0.2 s.

### Containment

An event that looks empty is usually not a reconstruction failure -- the
interaction happened at or beyond the edge of the active argon and most of it
escaped. Every view says so rather than leaving you to guess:

* the **true vertex** is marked (cyan `X`) on each 2-D panel and in 3-D, drawn
  even when it falls outside the active volume, which is the case worth seeing;
* the title carries `[vertex OUTSIDE active volume]` when it does;
* the block beneath reports where the vertex sits and how much energy was
  actually deposited.

```
nu_tau CC RES   E = 7.36 GeV   [vertex OUTSIDE active volume]
  ...
  6.6 cm outside the active volume (6.6 cm above y = 600)
  visible energy 292 MeV of 7361 MeV true (4.0%)
```

`Geometry.containment(point)` tests every TPC individually, not the box that
encloses the detector -- a point can sit inside that box while being in a gap
between drift volumes. `Geometry.in_active(points)` is the vectorised form.

In the tau sample this accounts for the quiet events: hit count tracks
*deposited* energy at r = 0.95 (0.97 hits/MeV), but true neutrino energy only at
r = 0.62. Of 120 events, 8 have no hits at all and every one of those has a
vertex 39-300 cm outside the active volume.

## Pandora track/shower duplication

Pandora fits **every** PFParticle both ways and writes both, so `pandoraTrack`
and `pandoraShower` each hold one object per particle -- 11 tracks and 11
showers for 12 PFParticles in the sample here. Drawing all of each paints every
particle twice, once as a line and once as a cone.

Which one was meant is in the PFParticle metadata: `TrackScore` from
`larpandoraobj::PFParticleMetadata`, above 0.5 for track-like. `tracks()` and
`showers()` apply it by default:

```python
ev.tracks()                      # 9  - Pandora's intended interpretation
ev.showers()                     # 2
ev.tracks(best_match=False)      # 11 - the raw collection
ev.pfp_track_scores()            # TrackScore per PFParticle, NaN where absent
ev.pandora_best_match()          # {"tracks": mask, "showers": mask}
```

The threshold is checked, not assumed: on these samples it reproduces the PDG
code Pandora assigns each PFParticle (13 track-like vs 11 shower-like) exactly.
A PFParticle with no score -- the neutrino candidate itself -- is kept rather
than dropped, and a sample with no metadata keeps everything, because guessing
would be worse than a visible duplicate.

**One caveat:** `best_match` reindexes. `hit_group()` indices refer to the full
collection, which is what reconstruction associated against, so pass
`best_match=False` when comparing the two. `Tracks.id` is unaffected.

## Flashes in 3-D

```bash
python -m pylarevd reco.root -e 0 --mode flash3d
```

Flashes in the detector volume, sized by log PE and coloured by time, with the
photon detectors drawn faintly for context and the true vertex marked.

A flash localises light in (y, z) and time but **not** in drift: light alone
cannot fix x without a matched TPC object, and LArSoft records the unset value
as `DBL_MAX`. Rather than draw every flash at x = 0 as though it were measured,
each is drawn as a **bar spanning the drift** -- the bar is the measurement, the
position along it is unknown. If a sample does reconstruct x, flashes are drawn
as points instead. `OpticalActivity.has_x` says which, and the summary states it
outright.

The time window defaults to +-50 us around the brightest burst of OpHit light,
matching the 2-D optical view and the light animation; the whole readout is
milliseconds wide and dominated by radiological light. It used to centre on the
brightest *flash*, which fails exactly when the flash finder dropped the
interaction's flash (see below). Both the window and any cap on the number drawn
are stated in the title rather than silently applied. `--flash-tag opslicer`
(or `display_flashes_3d(ev, "opslicer")`) picks another flash producer.

## Light vs time (animated PDS view)

```bash
python -m pylarevd reco.root -e 0 --mode light            # GIF + static map
python -m pylarevd reco.root -e 0 --mode light --html     # + interactive page
```

```python
from pylarevd import display_light
a = display_light(ev)                  # or display_light(ev, "opslicer")
a.save_animation("light.gif")          # .mp4 too, if ffmpeg is on PATH
a.save_html("light.html")              # play button + time slider
a.save("light.png")                    # static: PE collected over the window
a.save("t.png", frame=20)              # one frame
print(a.summary())
```

Every photon detector is a tile on the wall it is mounted on (beam z across, y
up; one panel per APA plane, so three for the full FD-HD module). Tiles light
up as their OpHits arrive, and reconstructed flashes appear as their +-1 sigma
(y, z) ellipse at their own time, bright when new and fading to a ghost. A strip
underneath shows the PE arriving in each frame, split into hits that belong to a
flash and hits that belong to none, with every flash as a stem at its time and a
playhead. The true vertex is a star. In the browser pick **Light vs time
(PDS)**; the **animated GIF** button then exports this animation.

**The clock.** By default frames are log-spaced after `t0`, the onset of the
brightest OpHit light burst: a short linear pre-roll, then one 16 ns hit
quantum per frame while the prompt light spreads out from the vertex, then
geometric growth to ~0.4 us per frame by +10 us for the slow component and
later flashes (Michel electrons). On the FD-HD sample the prompt light reaches
detectors 2-4 m from the vertex ~70 ns after the onset and those beyond 8 m
after ~250 ns, so a uniform clock would show it in a single frame.
`--light-clock linear` (or `time_scale="linear"`) gives a uniform one;
`--window LO,HI` and `--t0` override the defaults.

**The tiles.** `afterglow` (default) keeps 70 % of a tile's PE from frame to
frame, so an arrival flares up and fades -- that is what makes arrivals
visible; `cumulative` accumulates instead. A tile that receives light in the
current frame is outlined. The colour range is fixed for the whole animation,
so frames compare. Tiles are cells of the detector grid, not the sensor
outline.

**Flash-hit links** come from the `OpFlash`<->`OpHit` association and are
checked, not trusted: every flash's PE must equal the summed PE of its hits.
On the FD-HD sample `opflash` passes exactly; `opslicer`'s association links
its brightest flash to 1853 hits spread over the whole 4.5 ms readout, fails,
and is not used (the strip then shows total OpHit PE).

**Charge and light together.** `--mode chargelight` (browser: **Charge + light
(3D + PDS)**, python: `display_charge_light(ev)`) puts the same animation beside
the 3-D charge: space points (grey, brightness ~ log charge -- the light owns
the colour map), tracks, vertices and truth as in the 3-D view, with the
photon-detector wall drawn where it is on the anode plane, lighting up frame by
frame, and each flash a bar across the drift at its (y, z). The charge itself is
static -- it is deposited within nanoseconds and read out over milliseconds of
drift, so there is no shared clock to animate it on. In the browser the 3-D
panel is a normal plotly scene: rotate and zoom it while it plays, or between
steps of the slider, and the view is kept. (plotly on its own put the camera
back on every frame unless the drag ended inside the 3-D canvas; a small script
in the page and in saved HTML, `KEEP_VIEW_JS`, stops that.) The GIF uses a
fixed camera (`elev`, `azim`), and draws the charge on the viewer's side of the
wall over the tiles and the rest under them.

```bash
python -m pylarevd reco.root -e 0 --mode chargelight --truth --html
```

**Missing and misplaced flashes.** Two failures this view makes visible, both on
the FD-HD atmospheric sample:

- `opflash` sometimes has **no flash at the interaction at all** -- 23 kPE of
  hits at t0 and nothing reconstructed (the `RemoveLateLight` divide-by-width
  bug in LArSoft's `OpFlashAlg`). The summary says so and names the other flash
  producers in the file; `opslicer` finds that flash (26 kPE).
- `opslicer` stores **negative widths and a centre pinned to one detector** for
  every flash. Wherever a flash's width is negative or not a number, its centre
  and width are recomputed as the PE-weighted mean and RMS of the detectors in
  `fPEperOpDet` -- which reproduces `opflash`'s stored centres to 1e-12 cm --
  and the summary counts them. All three optical views apply this.

## Hit disambiguation

In a wrapped-wire detector the physical views are only meaningful for a
**disambiguated** hit collection. Drawing a raw hit finder's output puts each
hit on an arbitrary one of the candidate wires: measured on the sample here,
`gaushit` sits a median **3.9 cm** from the true energy deposit against
**0.05 cm** for `hitfd` -- a picture that looks like broken reconstruction
rather than the wrong input.

Disambiguation is exactly what splits a wrapped channel across physical wires,
so its absence is detectable, and the display says so:

```
! recob::Hits_gaushit__Reco1 looks NOT disambiguated (no channel split across
  wires, in a detector with 19200 wrapped channels) -- physical positions will
  be wrong; use a disambiguated collection such as hitfd
```

The check is skipped in `space="readout"`, which folds the faces together by
design, and on detectors with no wrapped channels, where there is nothing to
disambiguate.

## Radiologicals

A radiological sample carries tens of thousands of decays unrelated to the
interaction -- 32.9k of the 34.6k true particles in the sample here are
radiological *primaries*. They are real, so they are drawn by default, but they
can be separated:

```bash
python -m pylarevd reco.root -e 0 --mode 3d --truth --no-radiologicals
```

```python
ev.display_3d(truth=True, radiologicals=False)
ev.display(truth=True, radiologicals=False)
ev.neutrino_track_ids()      # track ids descending from the interaction
```

and in the browser via the **radiologicals** checkbox next to the overlays.

The split is by ancestry, not by energy or position cuts: every particle is
walked to its root ancestor, and the interaction's descendants are those whose
root starts at the neutrino vertex. On the sample here that distance is 0 for
the interaction and a median 773 cm for the background, so the separation is
unambiguous. Without a neutrino (a cosmic file) there is nothing to separate
from, and the filter keeps everything rather than guessing.

Note that the 3-D view frames the *reconstructed* data by default, which
already crops most radiologicals out of shot; `focus="detector"` frames the
whole cryostat if you want to see them in context.

## What the display will tell you

Rather than fail quietly, the views state what they know:

* **`visible energy N MeV of M MeV true (P%)`** — the ionisation from the
  *interaction*, not the event total. Counting every radiological deposit
  against the neutrino's energy gave fractions over 100% on 9 of 10 events.
* **`[vertex OUTSIDE active volume]`** plus the distance and the face missed.
* **`... looks NOT disambiguated`** — judged over the whole collection, not one
  event. A cosmic track that stays on a single drift face splits no channel,
  which made a per-event test flag correct data half the time.
* **`radiologicals hidden`** when the filter is on, so a shared figure cannot
  be mistaken for an unfiltered one.
* **`outgoing lepton: tau- (5.09 GeV)`** — a tau decays and so never appears in
  the GENIE final state, which for a tau-appearance sample omitted the one
  particle the event is about.
* **`run R / subrun S / event E is not in <file>`** when following an event
  across files fails, instead of silently showing a different one.

## Working with the arrays

Product accessors are memoised and hand back the **same object** each time, so
the arrays they expose are read-only. Copy before modifying:

```python
h = ev.hits()
h.integral[:] = 0          # ValueError: assignment destination is read-only
q = h.integral.copy()      # yours to change
sub = h[h.integral > 200]  # selections own their data
```

Mutating them used to corrupt the cached product for the rest of the event's
life, silently. The same applies to `Geometry`, which is shared by every
`EventFile` opened against the same `.npz`.

`hit_group()` and `tracks()`/`showers()` agree by default: indices from the
former index the collection the latter returns. Pass `best_match=False` to
both if you want the raw Pandora collections instead.

## Known limits

- Products with variable-length members (`simb::MCParticle` trajectories,
  `art::Assns`) are **not** decodable by the fixed-stride reader; it raises a
  clear error rather than guessing. Hits, SpacePoints and SimEnergyDeposits are
  covered.
- Packed `Double32_t` (with a declared range) is refused rather than
  mis-decoded. Nothing in the DUNE products used here is packed.
- Static 3-D goes through matplotlib: plotly's 3-D needs WebGL, which a
  headless node cannot provide. Interactive 3-D is fine in a real browser.
- `art::Assns` carrying data (`recob::TrackHitMeta`) are not decoded; the plain
  `void` variants are used instead. Routes must name classes in the order art
  canonicalises them (alphabetical), which is how the branch is named.
- In the physical views only a disambiguated hit collection (`hitfd`) gives
  meaningful panels; `gaushit` WireIDs are not resolved across the wrap. The
  readout view has no such restriction — it keys off the channel.
- Hits whose WireID is unset get NaN coordinates and are excluded from the
  physical views; `EventDisplay.summary()` reports the count.
