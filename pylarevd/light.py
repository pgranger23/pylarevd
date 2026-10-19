"""Time-resolved photon-detector display: watch the light arrive.

Scintillation light reaches the photon detectors (PDs) within nanoseconds of an
interaction and keeps trickling in for microseconds (the slow component has
tau ~ 1.5 us). A static optical view flattens all of that onto one picture.
:class:`LightAnimation` plays it back instead: every PD is a tile on the wall
it is mounted on, and the tiles light up as their OpHits arrive, while the
reconstructed flashes appear at their own time and place.

Two timescales matter, and one clock cannot serve both:

* **prompt** -- light spreads out from the vertex over the first ~250 ns. On the
  FD-HD 1x2x6 sample the median first-hit time is ~70 ns at 2-4 m from the
  true vertex and ~250 ns beyond 8 m, which is resolvable at the 16 ns OpHit
  time quantum;
* **delayed** -- the slow component, and flashes microseconds later (Michel
  electrons: tau_mu = 2.2 us).

So by default the frames are **log-spaced after the light onset** ``t0``: a
short linear pre-roll, then one frame per hit quantum at first, stretching to
~0.5 us per frame by the end. ``time_scale="linear"`` gives a uniform clock.

``t0`` is found from the OpHits, not from the flashes: a flash finder can drop
the interaction's flash entirely (LArSoft ``OpFlashFinder``'s late-light
removal does, on a few percent of events), and a display keyed on flashes then
centres on a radiological flash and misses the interaction altogether.

Frame content
-------------
* **walls** -- one panel per plane of photon detectors, beam (z) across and
  up (y) vertical. Tile colour is the PE seen by that detector, either with an
  afterglow (``fill="afterglow"``: a hit flares up and fades over a few
  frames, which is what makes arrivals visible) or accumulated since the
  start of the window (``fill="cumulative"``). A tile that receives light in
  the current frame is outlined.
* **flashes** -- drawn as their +-1 sigma (y, z) ellipse from the frame their
  time falls in, bright when new and fading to a ghost.
* **time strip** -- PE arriving per frame, split into hits that belong to a
  flash and hits that belong to none, with each flash as a stem at its time,
  and a playhead. Its horizontal axis is the frame clock, labelled in time.

Tiles are cells of the PD grid, not the physical sensor outline (an HD
X-ARAPUCA supercell is ~48 x 10 cm); a tile per sensor would be too thin to
read.
"""

from __future__ import annotations

import io
import os
import shutil
import subprocess
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from .display import (DEFAULT_PRESET, PRESETS, _PLOTLY_CONFIG, _TruthInfo,
                      _fonts, _pe_width, _pyplot, _style_axes, _style_colorbar,
                      _vector_text, charge_cmap, charge_colorscale)
from .theme import DEFAULT_COLORMAP, resolve_colormap, resolve_theme

#: Window after t0. The slow scintillation component (tau ~ 1.5 us) is 99.9 %
#: gone by 10 us, and 99 % of muon decays have happened -- Michel flashes fit.
DEFAULT_AFTER_US = 10.0
#: Pre-roll before t0: long enough to show that nothing was there yet.
DEFAULT_BEFORE_US = 0.25
DEFAULT_FRAMES = 120
#: Fraction of a tile's afterglow kept from one frame to the next.
DEFAULT_PERSISTENCE = 0.7
FILLS = ("afterglow", "cumulative")
TIME_SCALES = ("log", "linear")

#: Colour range floor. A single photoelectron is ~1 PE; hits reconstruct down
#: to ~0.5 PE.
PE_FLOOR = 0.5
#: Below this a tile is drawn dark: its afterglow has faded out.
PE_OFF = 0.2

#: More walls than this means the PDs are not on planes of constant x (e.g.
#: membrane-mounted detectors), and everything is projected onto one panel.
_MAX_WALLS = 6


#: Keeps a rotated or zoomed 3-D view when the animation moves on.
#:
#: Every animation frame redraws the 3-D scene, and plotly's redraw
#: (``scene.plot`` -> ``setViewport``) puts the camera back to the one saved
#: in the figure layout. plotly saves it only on a mouse-up inside the 3-D
#: canvas: not when the release lands outside it (a drag from the 3-D panel
#: over the wall panel), not after a wheel zoom (it saves before the zoom is
#: applied), and not during a drag. So the view jumped back as soon as the
#: time changed, and could not be turned at all while playing.
#:
#: The fix wraps each scene's redraw: if the layout's camera is still the one
#: the scene last drew with -- nobody but the user has moved it -- the redraw
#: skips plotly's camera reset and leaves the live view alone, even in the
#: middle of a drag. A camera the figure itself changes (a new event, "fix
#: perspective") is still applied.
#: Mouse-ups anywhere and wheel zooms also save the view the way plotly's own
#: mouse-up does, so the browser app's relayoutData sees it. Included in the
#: browser app's page and in saved HTML; figures without 3-D are untouched.
KEEP_VIEW_JS = r"""
(function () {
  if (window.__pylarevdKeepView) { return; }
  window.__pylarevdKeepView = true;
  function key(c) {
    if (!c) { return ''; }
    var out = [];
    ['eye', 'center', 'up'].forEach(function (v) {
      var p = c[v] || {};
      ['x', 'y', 'z'].forEach(function (a) {
        out.push(p[a] === undefined ? '' : Number(p[a]).toFixed(6));
      });
    });
    return out.join(',');
  }
  function scenes(gd) {
    var fl = gd._fullLayout || {}, out = [];
    Object.keys(fl).forEach(function (k) {
      if (/^scene\d*$/.test(k) && fl[k] && fl[k]._scene) { out.push(fl[k]._scene); }
    });
    return out;
  }
  function wrap(scene) {
    if (scene.__pylarevdKeepView) { return; }
    scene.__pylarevdKeepView = true;
    var plot = scene.plot;
    scene.plot = function (sceneData, fullLayout, layout) {
      var keep = false;
      try {
        var full = fullLayout && fullLayout[this.id];
        keep = !!full && this.__drawnWith !== undefined &&
               key(full.camera) === this.__drawnWith;
      } catch (e) { keep = false; }
      if (keep) {
        // Nobody but the user has moved the camera since the last draw:
        // redraw without plotly's camera reset, so a view being dragged or
        // zoomed -- even mid-gesture -- is left exactly as it is.
        this.setViewport = function (sceneLayout) {
          this.glplot.setAspectratio(sceneLayout.aspectratio);
        };
      }
      try {
        return plot.apply(this, arguments);
      } finally {
        if (keep) { delete this.setViewport; }
        try {
          this.__drawnWith = key(fullLayout[this.id].camera);
        } catch (e) { this.__drawnWith = undefined; }
      }
    };
    try { scene.__drawnWith = key(scene.fullSceneLayout.camera); } catch (e) {}
  }
  function save(gd) {
    scenes(gd).forEach(function (s) {
      try {
        if (!gd.layout || !s.isCameraChanged(gd.layout)) { return; }
        var upd = {};
        upd[s.id + '.camera'] = s.getCamera();
        s.saveLayout(gd.layout);
        s.__drawnWith = key(upd[s.id + '.camera']);
        if (gd.emit) { gd.emit('plotly_relayout', upd); }
      } catch (e) { /* a view that cannot be saved is left as it was */ }
    });
  }
  function scan() {
    document.querySelectorAll('.js-plotly-plot').forEach(function (gd) {
      scenes(gd).forEach(wrap);
      if (!gd.__pylarevdKeepView && gd.addEventListener) {
        gd.__pylarevdKeepView = true;
        gd.addEventListener('wheel', function () {
          setTimeout(function () { save(gd); }, 150);
        }, true);
      }
    });
  }
  function saveAll() {
    setTimeout(function () {
      document.querySelectorAll('.js-plotly-plot').forEach(save);
    }, 0);
  }
  document.addEventListener('mouseup', saveAll, true);
  document.addEventListener('touchend', saveAll, true);
  scan();
  setInterval(scan, 300);
})();
"""


def _fmt_dt(dt_us: float) -> str:
    """Signed time offset, in ns below 1 us."""
    a = abs(dt_us)
    if a < 5e-4:
        return "0"
    sign = "+" if dt_us > 0 else "−"
    if a < 1.0:
        return f"{sign}{a * 1000:.0f} ns"
    if a < 100.0:
        return f"{sign}{a:.2f} µs"
    return f"{sign}{a:.0f} µs"


def _fmt_pe(pe: float) -> str:
    if pe >= 1e6:
        return f"{pe / 1e6:.1f}M"
    if pe >= 1e4:
        return f"{pe / 1e3:.0f}k"
    if pe >= 1e3:
        return f"{pe / 1e3:.1f}k"
    return f"{pe:.0f}"


def _hex(rgba) -> str:
    r, g, b = (int(round(255 * float(c))) for c in rgba[:3])
    return f"#{r:02x}{g:02x}{b:02x}"


# --------------------------------------------------------------------------
# photon-detector layout


@dataclass(frozen=True)
class PhotonDetectorWalls:
    """Photon detectors as tiles on the planes they are mounted on.

    A tile is one distinct detector position; optical channels that share a
    position (several readout channels per module) share a tile. Walls are
    groups of tiles at the same drift coordinate x -- one per APA plane in a
    horizontal-drift detector.
    """

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    half_y: np.ndarray
    half_z: np.ndarray
    wall: np.ndarray            # wall index per tile
    wall_x: tuple               # mean x per wall
    channels: tuple             # optical channels per tile
    chan_tile: np.ndarray       # tile per optical channel, -1 if none
    projected: bool = False     # True when the PDs are not on constant-x planes

    def __len__(self) -> int:
        return len(self.y)

    @property
    def n_walls(self) -> int:
        return len(self.wall_x)

    def wall_label(self, w: int) -> str:
        if self.projected:
            return "all photon detectors, projected onto (z, y)"
        return f"photon detectors at x ≈ {self.wall_x[w]:.0f} cm"

    def extent(self, w: int, pad: float = 20.0) -> tuple[float, float, float, float]:
        """(zmin, zmax, ymin, ymax) of the tiles of one wall, padded."""
        m = self.wall == w
        return (float((self.z[m] - self.half_z[m]).min()) - pad,
                float((self.z[m] + self.half_z[m]).max()) + pad,
                float((self.y[m] - self.half_y[m]).min()) - pad,
                float((self.y[m] + self.half_y[m]).max()) + pad)

    @classmethod
    def from_geometry(cls, geometry, *, wall_gap: float = 50.0
                      ) -> "PhotonDetectorWalls":
        if not getattr(geometry, "has_optical", False):
            raise ValueError(
                "this geometry was exported without photon-detector positions; "
                "re-export it with pylario.export_geometry to animate the light")
        xyz = np.asarray(geometry.opdet_xyz, float)
        chans = np.flatnonzero(np.isfinite(xyz).all(axis=1))
        if not len(chans):
            raise ValueError("no optical channel has a detector position")
        xc = np.round(xyz[chans, 0], 1)

        # Group into walls at distinct x. Detectors mounted on the two faces
        # of one APA differ by a fraction of a mm and belong together.
        xs = np.unique(xc)
        cuts = xs[:-1][np.diff(xs) > wall_gap]     # upper x of each group
        bounds = np.concatenate([cuts, [np.inf]])
        wall_c = np.searchsorted(bounds, xc, side="left")
        projected = len(bounds) > _MAX_WALLS
        if projected:
            wall_c = np.zeros(len(xc), np.int64)
            bounds = np.array([np.inf])

        # A tile is a (wall, y, z) position: channels there -- several per
        # module, or the two faces of a double-sided module -- share it
        # rather than drawing on top of one another.
        key = np.column_stack([wall_c, np.round(xyz[chans, 1], 1),
                               np.round(xyz[chans, 2], 1)])
        uniq, inv = np.unique(key, axis=0, return_inverse=True)
        inv = np.asarray(inv).ravel()
        chan_tile = np.full(len(xyz), -1, np.int64)
        chan_tile[chans] = inv
        order = np.argsort(inv, kind="stable")
        splits = np.flatnonzero(np.diff(inv[order])) + 1
        per_tile = tuple(np.split(chans[order], splits))
        wall = uniq[:, 0].astype(np.int64)
        y, z = uniq[:, 1], uniq[:, 2]
        x = np.bincount(inv, weights=xyz[chans, 0]) / np.bincount(inv)
        wall_x = tuple(float(x[wall == w].mean()) for w in range(len(bounds)))

        half_y = np.empty(len(x))
        half_z = np.empty(len(x))
        # Real detector outlines when the geometry recorded them: ProtoDUNE-SP
        # mixes 2 m light-guide bars with 10 cm ARAPUCA cells, so a size
        # inferred from spacing drew the bars as slivers. A 10 cm bar is
        # still too thin to read on a 6 m wall, so the short side is
        # thickened to a quarter of the typical row spacing at least.
        ext = getattr(geometry, "opdet_half_extent", None)
        real = None
        if ext is not None:
            ext = np.asarray(ext, float)[chans]
            if np.isfinite(ext).all():
                real = np.zeros((len(x), 3))
                np.maximum.at(real, inv, ext)
        for w in range(len(wall_x)):
            m = wall == w
            if real is None:
                half_z[m] = 0.47 * _pitch(z[m])
                half_y[m] = 0.40 * _pitch(y[m])
            else:
                floor_y = 0.25 * _pitch(y[m], typical=True)
                floor_z = 0.25 * _pitch(z[m], typical=True)
                half_y[m] = np.maximum(real[m, 1], floor_y)
                half_z[m] = np.maximum(real[m, 2], min(floor_z, floor_y))
        return cls(x=x, y=y, z=z, half_y=half_y, half_z=half_z,
                   wall=wall.astype(np.int64), wall_x=wall_x,
                   channels=per_tile, chan_tile=chan_tile, projected=projected)


def _pitch(v: np.ndarray, default: float = 20.0, *, typical: bool = False) -> float:
    """Smallest (or, with ``typical``, median) spacing between distinct
    detector coordinates."""
    u = np.unique(np.round(v, 0))
    d = np.diff(u)
    d = d[d > 2.0]
    if not len(d):
        return default
    return float(np.median(d) if typical else d.min())


# --------------------------------------------------------------------------
# flash positions


def repair_flash_positions(event, optical):
    """Recompute (y, z) where the flash finder left them unusable.

    Returns ``(optical, n_repaired)``. A flash whose y or z width is negative
    or not a number has no trustworthy centre: on the FD-HD sample every
    ``opslicer`` flash has widths of about -1e6 cm and a centre pinned to a
    single detector (y = -467, z = 268 cm for a 26 kPE interaction whose
    true vertex is at y = 305, z = 626). Such flashes get the PE-weighted
    mean and RMS of the detectors that saw them, from ``fPEperOpDet``
    (indexed by optical channel: for ``opflash`` this reproduces its stored
    centre to 1e-12 cm). Flashes with valid widths are left alone.
    """
    from dataclasses import replace
    o = optical
    fl = np.flatnonzero(np.asarray(o.is_flash))
    if not len(fl):
        return o, 0
    yw = np.asarray(o.y_width, float)[fl]
    zw = np.asarray(o.z_width, float)[fl]
    # zero is a legitimate width: a flash seen by one detector
    bad = ~(np.isfinite(yw) & (yw >= 0) & np.isfinite(zw) & (zw >= 0))
    if not bad.any():
        return o, 0
    prod = str(getattr(o, "label", "")).split(" + ")[0].strip()
    try:
        raw = event._src.art.read(prod + ".", "recob::OpFlash", event.entry)
        per = raw["fPEperOpDet"]
        xyz = np.asarray(event.geometry.opdet_xyz, float)
    except Exception:
        return o, 0
    if len(per) != len(fl) or not len(xyz):
        return o, 0
    y, z = np.array(o.y, float), np.array(o.z, float)
    ywa, zwa = np.array(o.y_width, float), np.array(o.z_width, float)
    n = 0
    for j in np.flatnonzero(bad):
        w = np.asarray(per[j], float)[:len(xyz)]
        ok = (w > 0) & np.isfinite(xyz[:len(w)]).all(axis=1)
        if not ok.any():
            continue
        w, p = w[ok], xyz[:len(ok)][ok]
        my = float(np.average(p[:, 1], weights=w))
        mz = float(np.average(p[:, 2], weights=w))
        i = fl[j]
        y[i], z[i] = my, mz
        ywa[i] = float(np.sqrt(np.average((p[:, 1] - my) ** 2, weights=w)))
        zwa[i] = float(np.sqrt(np.average((p[:, 2] - mz) ** 2, weights=w)))
        n += 1
    if not n:
        return o, 0
    return replace(o, y=y, z=z, y_width=ywa, z_width=zwa), n


# --------------------------------------------------------------------------
# light onset and frame clock


def hit_time_quantum(t: np.ndarray) -> float:
    """The OpHit time step (16 ns in DUNE FD simulation), from the data.

    Frames finer than this would repeat, so it bounds the log clock.
    """
    u = np.unique(np.round(np.asarray(t, float), 4))
    d = np.round(np.diff(u), 4)
    d = d[d > 5e-4]
    if not len(d):
        return 0.016
    # the smallest gap that is COMMON, not the smallest gap: one channel with
    # an odd timing phase must not halve the clock for everything else
    vals, counts = np.unique(d, return_counts=True)
    common = vals[counts >= 0.1 * counts.max()]
    return float(np.clip(common.min(), 0.001, 0.1))


def light_onset(t: np.ndarray, pe: np.ndarray, quantum: float = 0.016
                ) -> float | None:
    """Start time of the brightest burst of OpHit light, in us.

    The burst is the 100 ns bin with the most PE over the whole readout. Its
    onset is the earliest hit quantum within 300 ns before the peak that
    already carries 2 % of the peak quantum's PE (and at least 3 PE, so that
    a stray radiological photoelectron does not pull it early). Returned half
    a quantum before that hit time, so no frame edge sits on a hit.
    """
    t = np.asarray(t, float)
    pe = np.asarray(pe, float)
    ok = np.isfinite(t) & np.isfinite(pe)
    t, pe = t[ok], np.clip(pe[ok], 0.0, None)
    if not len(t) or pe.sum() <= 0:
        return None
    lo = float(t.min())
    coarse = np.bincount(((t - lo) / 0.1).astype(np.int64), weights=pe)
    b = int(np.argmax(coarse))
    start = lo + b * 0.1 - 0.3
    stop = lo + (b + 1) * 0.1
    sel = (t >= start) & (t < stop)
    fine = np.bincount(((t[sel] - start) / quantum).astype(np.int64),
                       weights=pe[sel])
    p = int(np.argmax(fine))
    thresh = max(0.02 * fine[p], 3.0)
    above = np.flatnonzero(fine[:p + 1] >= thresh)
    k = int(above[0]) if len(above) else p
    in_bin = sel & (t >= start + k * quantum) & (t < start + (k + 1) * quantum)
    first = float(t[in_bin].min()) if in_bin.any() else start + k * quantum
    return first - 0.5 * quantum


def frame_edges(t0: float, lo: float, hi: float, n: int, scale: str,
                quantum: float) -> np.ndarray:
    """``n + 1`` increasing frame edges covering ``[lo, hi]``.

    ``linear``: uniform. ``log``: a linear pre-roll from ``lo`` to ``t0`` (10 %
    of the frames), then a clock that ticks one hit quantum per frame until
    geometric growth overtakes it (see :func:`_log_clock`).
    """
    if n < 1:
        raise ValueError("need at least one frame")
    if hi <= lo:
        raise ValueError(f"empty time window [{lo}, {hi}]")
    if scale == "linear" or hi <= t0 + quantum:
        return np.linspace(lo, hi, n + 1)
    if scale != "log":
        raise ValueError(f"time_scale={scale!r} is not one of {TIME_SCALES}")
    pre = t0 - lo
    n_pre = 0 if pre <= 0 else max(1, int(round(0.1 * n)))
    n_post = n - n_pre
    if n_post < 1:
        return np.linspace(lo, hi, n + 1)
    post = t0 + _log_clock(quantum, hi - t0, n_post)[1:]
    if n_pre:
        head = np.linspace(lo, t0, n_pre + 1)
    else:
        head = np.array([max(lo, t0)])
        post = post[post > head[0]]
        if len(post) < n_post:          # lo was after t0: stay uniform instead
            return np.linspace(lo, hi, n + 1)
    return np.concatenate([head, post])


def _log_clock(quantum: float, span: float, n: int) -> np.ndarray:
    """``n + 1`` edges from 0 to ``span``: step ``max(quantum, e * (r - 1))``.

    A pure geometric series starting at one quantum spends dozens of frames
    below the quantum -- ~40 of 108 on the FD-HD sample -- each showing the
    same hits again. Here the first frames advance one quantum at a time (the
    prompt light spreading from the vertex, at full resolution), and the clock
    turns geometric once that is the longer step. ``r`` is solved so the last
    edge lands on ``span``.
    """
    if n * quantum >= span:
        return np.linspace(0.0, span, n + 1)

    def run(r: float) -> np.ndarray:
        e = np.empty(n + 1)
        e[0] = 0.0
        for i in range(n):
            e[i + 1] = e[i] + max(quantum, e[i] * (r - 1.0))
        return e

    lo, hi = 1.0, 2.0
    while run(hi)[-1] < span:
        hi *= 2.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if run(mid)[-1] < span:
            lo = mid
        else:
            hi = mid
    e = run(hi)
    e *= span / e[-1]
    return e


def _nice_ticks(lo: float, hi: float, target: int = 7) -> np.ndarray:
    span = hi - lo
    if span <= 0:
        return np.array([lo])
    raw = span / target
    mag = 10 ** np.floor(np.log10(raw))
    step = min((m * mag for m in (1, 2, 5, 10) if m * mag >= raw),
               default=10 * mag)
    return np.arange(np.ceil(lo / step) * step, hi + 0.5 * step, step)


# --------------------------------------------------------------------------
# the display


class LightAnimation(_TruthInfo):
    """Photon-detector hits and flashes appearing as a function of time.

    Parameters
    ----------
    event
        A pylario :class:`Event`.
    optical
        An :class:`OpticalActivity`; default ``event.optical(flash_tag)``.
        Flash-hit links are only recovered when this is the unfiltered
        product, and are checked by PE closure either way.
    flash_tag
        Flash producer module label (``opflash``, ``opslicer``, ...).
    window
        ``(lo, hi)`` absolute time window in us. Default
        ``(t0 - 0.25, t0 + 10)``.
    t0
        Time origin in us. Default: the onset of the brightest light burst
        (:func:`light_onset`).
    time_scale
        ``"log"`` (default) or ``"linear"`` frame spacing.
    n_frames
        Number of frames.
    fill
        ``"afterglow"`` (default) or ``"cumulative"``.
    persistence
        Afterglow kept per frame, in [0, 1).
    """

    def __init__(self, event, optical=None, *, flash_tag: str | None = None,
                 window=None, t0: float | None = None,
                 time_scale: str = "log", n_frames: int = DEFAULT_FRAMES,
                 fill: str = "afterglow",
                 persistence: float = DEFAULT_PERSISTENCE,
                 max_flashes: int = 50,
                 theme: str = "dark", colormap: str = DEFAULT_COLORMAP,
                 preset: str = DEFAULT_PRESET):
        if time_scale not in TIME_SCALES:
            raise ValueError(f"time_scale={time_scale!r} is not one of "
                             f"{', '.join(TIME_SCALES)}")
        if fill not in FILLS:
            raise ValueError(f"fill={fill!r} is not one of {', '.join(FILLS)}")
        if not 0.0 <= persistence < 1.0:
            raise ValueError("persistence must be in [0, 1)")
        if int(n_frames) < 2:
            raise ValueError("n_frames must be at least 2")
        self.event = event
        self.geometry = event.geometry
        self._own_optical = optical is None
        self.repaired = 0
        if optical is None:
            optical, self.repaired = repair_flash_positions(
                event, event.optical(flash_tag))
        self.op = optical
        self.flash_tag = flash_tag
        self.time_scale = time_scale
        self.n_frames = int(n_frames)
        self.fill = fill
        self.persistence = float(persistence)
        self.max_flashes = int(max_flashes)
        self.theme = resolve_theme(theme)
        resolve_colormap(colormap)
        self.colormap = colormap
        self.preset = preset
        self._window_arg = None if window is None else tuple(map(float, window))
        self._t0_arg = None if t0 is None else float(t0)

    # ---- data ----------------------------------------------------------

    @cached_property
    def walls(self) -> PhotonDetectorWalls:
        return PhotonDetectorWalls.from_geometry(self.geometry)

    @cached_property
    def _split(self):
        o = self.op
        fl = np.flatnonzero(np.asarray(o.is_flash))
        hi = np.flatnonzero(~np.asarray(o.is_flash))
        return fl, hi

    @property
    def flash_product(self) -> str:
        return str(getattr(self.op, "label", "")).split(" + ")[0].strip()

    @cached_property
    def _links(self) -> tuple[np.ndarray | None, str]:
        """(flash index per OpHit or -1, note). None when not recoverable."""
        fl, hi = self._split
        nh = len(hi)
        if not self._own_optical:
            return None, "flash-hit links not read: optical data was passed in"
        # (a repaired copy keeps the product's order, so its links still hold)
        prod = self.flash_product
        stem = "recob::OpFlashs_"
        if not prod.startswith(stem) or not len(fl) or not nh:
            return None, ""
        suffix = "art::Assns_" + prod[len(stem):].rstrip(".")
        try:
            art = self.event._src.art
            cands = [(p, False) for p in art.assns_products("recob::OpFlash",
                                                             "recob::OpHit")]
            cands += [(p, True) for p in art.assns_products("recob::OpHit",
                                                            "recob::OpFlash")]
            cands = [(p, sw) for p, sw in cands if p.rstrip(".").endswith(suffix)]
            if not cands:
                return None, "no flash-hit association in this file"
            name, swapped = cands[0]
            a = art.read_assns(name, self.event.entry)
        except Exception as exc:          # links are a bonus, never a failure
            return None, f"flash-hit association unreadable ({exc})"
        f_key = np.asarray(a.right_key if swapped else a.left_key, np.int64)
        h_key = np.asarray(a.left_key if swapped else a.right_key, np.int64)
        if len(f_key) and (f_key.max() >= len(fl) or h_key.max() >= nh):
            return None, "flash-hit association indexes a different collection"
        hit_pe = np.asarray(self.op.pe, float)[hi]
        fl_pe = np.asarray(self.op.pe, float)[fl]
        # The association must index THIS hit collection. Every flash's PE is
        # the sum of its hits' PE, so check that rather than trust the name.
        closure = np.bincount(f_key, weights=hit_pe[h_key], minlength=len(fl))
        bad = np.abs(closure - fl_pe) > 1e-3 * np.abs(fl_pe) + 0.05
        if bad.mean() > 0.01:
            return None, (f"flash-hit association fails PE closure for "
                          f"{int(bad.sum())} of {len(fl)} flashes; not used")
        out = np.full(nh, -1, np.int64)
        out[h_key] = f_key
        return out, ""

    @cached_property
    def _hits(self):
        """Hit arrays: time, pe, tile, flash (-1 = none)."""
        o = self.op
        _fl, hi = self._split
        t = np.asarray(o.time, float)[hi]
        pe = np.asarray(o.pe, float)[hi]
        ch = np.asarray(o.channel, np.int64)[hi]
        ct = self.walls.chan_tile
        tile = np.where((ch >= 0) & (ch < len(ct)), ct[np.clip(ch, 0, len(ct) - 1)], -1)
        links, _note = self._links
        flash = links if links is not None else np.full(len(t), -1, np.int64)
        return t, pe, tile, flash

    @cached_property
    def quantum(self) -> float:
        return hit_time_quantum(self._hits[0])

    @cached_property
    def t0(self) -> float:
        if self._t0_arg is not None:
            return self._t0_arg
        t, pe, _tile, _fl = self._hits
        onset = light_onset(t, pe, self.quantum)
        if onset is not None:
            return onset
        fl, _ = self._split
        if len(fl):
            pe_f = np.asarray(self.op.pe, float)[fl]
            return float(np.asarray(self.op.time, float)[fl][np.argmax(pe_f)])
        return 0.0

    @property
    def window(self) -> tuple[float, float]:
        if self._window_arg is not None:
            return self._window_arg
        return (self.t0 - DEFAULT_BEFORE_US, self.t0 + DEFAULT_AFTER_US)

    @cached_property
    def edges(self) -> np.ndarray:
        lo, hi = self.window
        return frame_edges(self.t0, lo, hi, self.n_frames, self.time_scale,
                           self.quantum)

    def frame_of(self, t) -> np.ndarray:
        """Frame index per time; -1 outside the window.

        Frame k covers ``(edges[k], edges[k+1]]``; the window's own lower edge
        belongs to frame 0.
        """
        t = np.asarray(t, float)
        e = self.edges
        k = np.searchsorted(e, t, side="left") - 1
        k = np.where(t == e[0], 0, k)
        return np.where((k >= 0) & (k < self.n_frames), k, -1)

    def frame_coord(self, t) -> np.ndarray:
        """Position on the frame clock (edge k sits at k), for time axes."""
        return np.interp(np.asarray(t, float), self.edges,
                         np.arange(len(self.edges), dtype=float))

    def frame_label(self, k: int) -> str:
        """Time at the END of frame k, relative to t0."""
        return _fmt_dt(float(self.edges[k + 1]) - self.t0)

    @cached_property
    def arrivals(self) -> np.ndarray:
        """(n_frames, n_tiles) PE arriving at each tile in each frame."""
        t, pe, tile, _fl = self._hits
        k = self.frame_of(t)
        ok = (k >= 0) & (tile >= 0)
        out = np.zeros((self.n_frames, len(self.walls)))
        np.add.at(out, (k[ok], tile[ok]), pe[ok])
        return out

    @cached_property
    def values(self) -> np.ndarray:
        """(n_frames, n_tiles) displayed PE per tile, after afterglow."""
        a = self.arrivals
        if self.fill == "cumulative":
            return np.cumsum(a, axis=0)
        out = np.empty_like(a)
        acc = np.zeros(a.shape[1])
        for k in range(len(a)):
            acc = acc * self.persistence + a[k]
            out[k] = acc
        return out

    @cached_property
    def collected(self) -> np.ndarray:
        """(n_tiles,) PE collected over the whole window."""
        return self.arrivals.sum(axis=0)

    @cached_property
    def pe_range(self) -> tuple[float, float]:
        """Fixed log colour range for every frame of this animation."""
        top = max(float(self.values.max()) if self.values.size else 0.0,
                  float(self.collected.max()) if self.collected.size else 0.0)
        return PE_FLOOR, max(top, 10 * PE_FLOOR)

    @cached_property
    def strip(self) -> tuple[np.ndarray, np.ndarray]:
        """PE per frame: (in a flash, in no flash)."""
        t, pe, _tile, fl = self._hits
        k = self.frame_of(t)
        ok = k >= 0
        inside = np.bincount(k[ok & (fl >= 0)], weights=pe[ok & (fl >= 0)],
                             minlength=self.n_frames)
        outside = np.bincount(k[ok & (fl < 0)], weights=pe[ok & (fl < 0)],
                              minlength=self.n_frames)
        return inside, outside

    @cached_property
    def flashes(self) -> dict:
        """Flashes in the window, brightest first, capped at ``max_flashes``."""
        o = self.op
        fl, _ = self._split
        t = np.asarray(o.time, float)[fl]
        k = self.frame_of(t)
        idx = np.flatnonzero(k >= 0)
        pe = np.asarray(o.pe, float)[fl]
        idx = idx[np.argsort(-pe[idx], kind="stable")][:self.max_flashes]
        get = lambda a: np.asarray(a, float)[fl][idx]
        y, z = get(o.y), get(o.z)
        yw, zw = get(o.y_width), get(o.z_width)
        yw = np.where(np.isfinite(yw) & (yw > 0), yw, 30.0)
        zw = np.where(np.isfinite(zw) & (zw > 0), zw, 30.0)
        # Which walls saw each flash: by where its own hits landed.
        nw = self.walls.n_walls
        share = np.ones((len(idx), nw))
        links, _ = self._links
        if links is not None and nw > 1:
            _t, hpe, tile, hfl = self._hits
            share = np.zeros((len(idx), nw))
            pos = {int(f): j for j, f in enumerate(idx)}
            m = (hfl >= 0) & (tile >= 0)
            for f, tl, p in zip(hfl[m], tile[m], hpe[m]):
                j = pos.get(int(f))
                if j is not None:
                    share[j, self.walls.wall[tl]] += p
            tot = share.sum(axis=1, keepdims=True)
            share = np.where(tot > 0, share / np.where(tot > 0, tot, 1), 1.0)
        return dict(index=idx, time=get(o.time), pe=pe[idx], y=y, z=z,
                    y_width=yw, z_width=zw, frame=k[idx],
                    on_wall=share >= 0.2)

    def _flash_alpha(self, k: int | None) -> np.ndarray:
        """Opacity per flash at frame k: 1 when new, fading to a ghost."""
        f = self.flashes["frame"]
        if k is None:
            return np.full(len(f), 0.85)
        age = k - f
        return np.where(age < 0, 0.0,
                        np.maximum(0.45, self.persistence ** np.maximum(age, 0)))

    # ---- text ----------------------------------------------------------

    def _title(self) -> str:
        run, subrun, ev = self.event.id
        return (f"{self.geometry.detector}   run {run} / subrun {subrun} / "
                f"event {ev}   -   photon detectors vs time")

    @cached_property
    def _missing_flash(self) -> str:
        """Warn when the light onset carries no reconstructed flash."""
        t, pe, _tile, _fl = self._hits
        near = (t >= self.t0) & (t <= self.t0 + 1.0)
        prompt = float(pe[near].sum())
        if prompt < 100.0:
            return ""
        fl, _ = self._split
        ft = np.asarray(self.op.time, float)[fl]
        if ((ft >= self.t0 - 0.5) & (ft <= self.t0 + 1.0)).any():
            return ""
        others = ""
        try:
            prods = [p.split("_")[1] for p in
                     self.event._src.art.find_product("recob::OpFlash")]
            mine = self.flash_product.split("_")[1] if "_" in self.flash_product else ""
            alt = [p for p in prods if p != mine]
            if alt:
                others = f"; compare flash producer {', '.join(alt)}"
        except Exception:
            pass
        return (f"  ! {prompt:,.0f} PE of hits arrive within 1 us of t0 but "
                f"{self.flash_product or 'the flash finder'} has no flash "
                f"there -- the interaction's flash was not reconstructed{others}")

    def summary(self) -> str:
        t, pe, tile, fl = self._hits
        lo, hi = self.window
        lines = [self._title()]
        head = self._truth_headline()
        if head:
            lines.append(head)
        span = ("log-spaced after t0" if self.time_scale == "log"
                else "uniform")
        e = self.edges
        k0 = int(np.searchsorted(e, self.t0, side="right"))
        first = e[min(k0 + 1, len(e) - 1)] - e[min(k0, len(e) - 2)]
        lines.append(
            f"  t0 = {self.t0:.3f} us "
            + ("(given)" if self._t0_arg is not None
               else "(onset of the brightest OpHit light burst)"))
        lines.append(
            f"  window t0 {_fmt_dt(lo - self.t0)} .. {_fmt_dt(hi - self.t0)}, "
            f"{self.n_frames} frames, {span} "
            f"(first frame after t0 {first * 1000:.0f} ns, last "
            f"{(e[-1] - e[-2]) * 1000:.0f} ns); tiles: {self.fill}")
        win = self.frame_of(t) >= 0
        tot = float(pe[win].sum())
        in_fl = float(pe[win & (fl >= 0)].sum())
        links, note = self._links
        frac = (f", {100 * in_fl / tot:.0f}% of it in a flash"
                if links is not None and tot > 0 else "")
        lines.append(f"  {int(win.sum()):,} hits, {tot:,.0f} PE in the window"
                     f"{frac}")
        lost = int((win & (tile < 0)).sum())
        if lost:
            lines.append(f"  {lost} hits on channels with no detector "
                         f"position are not drawn")
        f = self.flashes
        if len(f["pe"]):
            lines.append(
                f"  {len(f['pe'])} flashes ({self.flash_product}); brightest "
                f"{f['pe'][0]:,.0f} PE at t0 {_fmt_dt(f['time'][0] - self.t0)},"
                f" y = {f['y'][0]:.0f}, z = {f['z'][0]:.0f} cm")
        else:
            lines.append(f"  no flashes in the window ({self.flash_product})")
        if note:
            lines.append(f"  {note}")
        if self.repaired:
            lines.append(f"  {self.repaired} flashes had no usable (y, z) from "
                         f"the flash finder (negative widths); centre and "
                         f"width recomputed from their per-detector PE")
        if self._missing_flash:
            lines.append(self._missing_flash)
        tv = self.true_vertex
        if tv is not None and np.isfinite(tv).all():
            lines.append(f"  true vertex x = {tv[0]:.0f}, y = {tv[1]:.0f}, "
                         f"z = {tv[2]:.0f} cm (star)")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (f"<LightAnimation {self.event!r} frames={self.n_frames} "
                f"{self.time_scale}/{self.fill}>")

    # ---- shared drawing helpers ---------------------------------------

    def _cmap(self):
        return charge_cmap(self.theme, self.colormap)

    def _off_colour(self) -> str:
        return "#1e293b" if self.theme.is_dark else "#e2e8f0"

    def _tile_rgba(self, v: np.ndarray) -> np.ndarray:
        from matplotlib.colors import LogNorm, to_rgba
        lo, hi = self.pe_range
        rgba = self._cmap()(LogNorm(vmin=lo, vmax=hi, clip=True)(
            np.clip(v, lo, hi)))
        rgba = np.asarray(rgba, float)
        rgba[v < PE_OFF] = to_rgba(self._off_colour())
        return rgba

    #: Tile level of a detector that is dark (its afterglow has faded).
    _OFF_LEVEL = -0.02

    def _tile_levels(self, v: np.ndarray) -> np.ndarray:
        """0..1 position on the log colour range, or ``_OFF_LEVEL`` if dark.

        The interactive figure colours tiles by number through
        :meth:`_tile_colorscale` rather than by colour string: plotly validates
        every element of every frame, and 120 frames of colour strings took
        seconds where numeric arrays take milliseconds.
        """
        lo, hi = self.pe_range
        u = (np.log10(np.clip(v, lo, hi)) - np.log10(lo)) / (np.log10(hi) - np.log10(lo))
        return np.round(np.where(v < PE_OFF, self._OFF_LEVEL, u), 3)

    def _tile_colorscale(self) -> list:
        """The tile colour map on [_OFF_LEVEL, 1], dark colour at the bottom."""
        off = self._off_colour()
        span = 1.0 - self._OFF_LEVEL
        head = [[0.0, off], [0.5 * -self._OFF_LEVEL / span, off]]
        return head + [[(-self._OFF_LEVEL + p) / span, c]
                       for p, c in charge_colorscale(self.theme, self.colormap)]

    def _tick_positions(self) -> tuple[np.ndarray, list[str]]:
        """Frame-clock positions and labels for the time strip."""
        lo, hi = self.window
        if self.time_scale == "log":
            cand = np.array([-10, -5, -2, -1, -0.5, -0.2, -0.1, -0.05, 0,
                             0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50,
                             100, 200, 500, 1000])
            cand = cand[(self.t0 + cand >= lo - 1e-9) & (self.t0 + cand <= hi + 1e-9)]
        else:
            cand = _nice_ticks(lo - self.t0, hi - self.t0)
        pos = self.frame_coord(self.t0 + cand)
        gap = 0.06 * self.n_frames
        p0 = float(self.frame_coord(self.t0))
        has0 = bool((cand == 0).any())
        keep, last = [], -np.inf
        for p, c in zip(pos, cand):
            # t0 always gets its tick; neighbours make way for it
            if c != 0 and has0 and abs(p - p0) < gap:
                continue
            if c == 0 or p - last >= gap:
                keep.append((p, c))
                last = p
        return (np.array([p for p, _ in keep]),
                [_fmt_dt(c) if c != 0 else "t₀" for _, c in keep])

    def _wall_boxes(self, w: int) -> list[tuple[float, float, float, float]]:
        """TPC outlines (zmin, zmax, ymin, ymax) seen from this wall."""
        out = set()
        for b in self.geometry.tpcs:
            out.add((round(b.zmin, 1), round(b.zmax, 1),
                     round(b.ymin, 1), round(b.ymax, 1)))
        return sorted(out)

    # ---- static / animated (matplotlib) -------------------------------

    def _scene(self, *, dpi: int | None = None):
        return _MplScene(self, dpi=dpi)

    def figure(self, frame: int | None = None):
        """One frame as a matplotlib figure; ``None`` shows the whole window.

        The whole-window view is the static light map: every tile coloured by
        the PE it collected over the window, every flash drawn.
        """
        scene = self._scene()
        scene.update(frame)
        return scene.fig

    def save(self, path: str, *, dpi: int | None = None,
             frame: int | None = None) -> str:
        """Static PNG/PDF of one frame (default: the whole window)."""
        _vector_text()
        dpi = PRESETS[self.preset]["dpi"] if dpi is None else dpi
        fig = self.figure(frame)
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        fig.savefig(path, dpi=dpi, facecolor=fig.get_facecolor())
        _pyplot().close(fig)
        return path

    def render_frames(self, *, dpi: int = 90, frames=None) -> list[np.ndarray]:
        """RGB arrays, one per frame (default all)."""
        scene = self._scene(dpi=dpi)
        ks = range(self.n_frames) if frames is None else frames
        scene.start_blit()
        out = []
        for k in ks:
            scene.update(k)
            out.append(scene.grab())
        _pyplot().close(scene.fig)
        return out

    #: GIF/MP4 resolution when none is asked for.
    anim_dpi = 90

    def animation_bytes(self, fmt: str = "gif", *, fps: float = 12.0,
                        dpi: int | None = None, hold: float = 2.0) -> bytes:
        """Encode the animation. ``fmt`` is ``gif`` or ``mp4`` (needs ffmpeg).

        The last frame is held for ``hold`` seconds so a looping GIF does not
        snap back to the start the moment the light has all arrived.
        """
        fmt = fmt.lower().lstrip(".")
        if fmt not in ("gif", "mp4"):
            raise ValueError(f"animation format {fmt!r}: use gif or mp4")
        if fmt == "mp4" and shutil.which("ffmpeg") is None:
            raise ValueError("mp4 needs the ffmpeg executable on PATH; "
                             "write a .gif instead")
        frames = self.render_frames(dpi=dpi or self.anim_dpi)
        if fmt == "gif":
            return _encode_gif(frames, fps, hold)
        return _encode_mp4(frames, fps, hold)

    def save_animation(self, path: str, *, fps: float = 12.0,
                       dpi: int | None = None, hold: float = 2.0) -> str:
        """Write the animation; the format follows the extension (.gif/.mp4)."""
        ext = os.path.splitext(path)[1] or ".gif"
        data = self.animation_bytes(ext, fps=fps, dpi=dpi, hold=hold)
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(data)
        return path

    # ---- interactive (plotly) -----------------------------------------

    def plotly_figure(self, *, height: int | None = 860, fps: float = 12.0):
        return _plotly_animation(self, height=height, fps=fps)

    def save_html(self, path: str, *, bundle_plotlyjs: bool = True,
                  **kwargs) -> str:
        fig = self.plotly_figure(**kwargs)
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        fig.write_html(path, include_plotlyjs="inline" if bundle_plotlyjs else "cdn",
                       config=_PLOTLY_CONFIG, auto_play=False,
                       post_script=KEEP_VIEW_JS)
        return path

    def show(self, **kwargs) -> None:
        self.plotly_figure(**kwargs).show(config=_PLOTLY_CONFIG)


class ChargeLightAnimation(LightAnimation):
    """The light animation beside the 3-D charge it came from.

    Left: the reconstructed space points (grey, brightness ~ log charge),
    tracks and the true vertex, with the photon-detector wall drawn where it
    is -- on the anode plane -- lighting up frame by frame, and each flash a
    bar across the drift at its (y, z) once its time has come. Right: the
    same wall face-on, and the time strip underneath, as in
    :class:`LightAnimation`. Charge is static: it is deposited within
    nanoseconds and read out over milliseconds of drift, so there is no
    common clock on which to animate it with the light.

    ``charge`` is a :class:`pylarevd.display.Display3D`; its colour settings
    are not used, since the light owns the colour map here.
    """

    #: the figure is ~1.7x wider than the light-only one
    anim_dpi = 72

    def __init__(self, event, charge, *, elev: float = 20.0,
                 azim: float = -60.0, **kwargs):
        super().__init__(event, **kwargs)
        self.charge = charge
        self.elev = float(elev)
        self.azim = float(azim)

    def _title(self) -> str:
        run, subrun, ev = self.event.id
        return (f"{self.geometry.detector}   run {run} / subrun {subrun} / "
                f"event {ev}   -   charge (3-D) and light vs time")

    def summary(self) -> str:
        d3 = self.charge
        extra = [f"  3-D charge: {len(d3.sp):,} space points ({d3.sp.label}), "
                 f"grey by log charge"]
        if d3.tracks is not None and len(d3.tracks):
            extra[0] += f"; {len(d3.tracks)} tracks"
        lines = super().summary().splitlines()
        return "\n".join(lines[:2] + extra + lines[2:])

    def __repr__(self) -> str:
        return (f"<ChargeLightAnimation {self.event!r} frames={self.n_frames} "
                f"spacepoints={len(self.charge.sp)}>")

    @cached_property
    def _charge_ranges(self):
        """(x, y, z) ranges of the 3-D panel: full drift and height, and z
        around the charge -- the whole wall when the detector is short
        enough, so the light pattern is seen in full."""
        boxes = self.geometry.tpcs
        xr = (min(b.xmin for b in boxes), max(b.xmax for b in boxes))
        yr = (min(b.ymin for b in boxes), max(b.ymax for b in boxes))
        zr = (min(b.zmin for b in boxes), max(b.zmax for b in boxes))
        span = zr[1] - zr[0]
        sp = self.charge.sp
        if span > 2.5 * (yr[1] - yr[0]) and len(sp):
            # a long detector (the full FD module is 58 m): frame the charge
            z = np.asarray(sp.xyz)[:, 2]
            c, half = 0.5 * (z.min() + z.max()), max(600.0, 0.8 * np.ptp(z))
            zr = (max(zr[0], c - half), min(zr[1], c + half))
        return xr, yr, zr

    def _charge_levels(self) -> tuple[np.ndarray, np.ndarray]:
        """Space-point positions and a 0..1 brightness from log charge."""
        sp = self.charge.sp
        xyz = np.asarray(sp.xyz, float) if len(sp) else np.zeros((0, 3))
        q = getattr(sp, "charge", None)
        if q is None or not len(xyz):
            return xyz, np.full(len(xyz), 0.7)
        lq = np.log10(np.clip(np.asarray(q, float), 1e-3, None))
        lo, hi = np.percentile(lq, [2, 99.5]) if len(lq) > 1 else (0.0, 1.0)
        return xyz, np.clip((lq - lo) / max(hi - lo, 1e-9), 0, 1)

    def _charge_points(self) -> tuple[np.ndarray, np.ndarray]:
        """Space-point positions and their grey-scale RGBA colours."""
        from matplotlib.colors import to_rgb
        xyz, f = self._charge_levels()
        dim = np.array(to_rgb(self.theme.grid))
        bright = np.array(to_rgb(self.theme.fg))
        rgb = dim + (bright - dim) * (0.25 + 0.75 * f)[:, None]
        return xyz, np.column_stack([rgb, np.full(len(rgb), 0.9)])


# --------------------------------------------------------------------------
# matplotlib scene


class _MplScene:
    """The figure, built once; :meth:`update` redraws it for one frame."""

    def __init__(self, anim: LightAnimation, dpi: int | None = None):
        import matplotlib.colors as mcolors
        from matplotlib.collections import LineCollection, PolyCollection
        from matplotlib.patches import Ellipse, Rectangle

        plt = _pyplot()
        self.a = a = anim
        t, fonts = a.theme, _fonts(a.preset)
        self.t, self.fonts = t, fonts
        W = a.walls
        scale = PRESETS[a.preset]["fonts"]

        # Layout in inches: walls stacked, strip under them, colourbar right;
        # with charge, a 3-D panel to the left of the walls.
        charge = getattr(a, "charge", None)
        fig_w = 11.0 * max(1.0, scale ** 0.5)
        left, right_w, gap = 0.75, 1.15, 0.35
        c3d_w = 0.0
        if charge is not None:
            c3d_w = 7.4 * max(1.0, scale ** 0.5)
            fig_w += c3d_w
        wall_left = left + c3d_w + (0.35 if charge is not None else 0.0)
        avail_w = fig_w - wall_left - right_w
        spans = [W.extent(w) for w in range(W.n_walls)]
        aspects = [(s[3] - s[2]) / (s[1] - s[0]) for s in spans]
        n_w = W.n_walls
        # Walls about as tall as they are wide (ProtoDUNE: two 7 x 6 m APA
        # planes) sit side by side; long ones (the FD module) are stacked.
        self.side_by_side = side = n_w > 1 and min(aspects) > 0.45
        if side:
            each_w = (avail_w - (gap + 0.4) * (n_w - 1)) / n_w
            wall_h = [each_w * r for r in aspects]
            if max(wall_h) > 6.4:
                f = 6.4 / max(wall_h)
                wall_h = [h * f for h in wall_h]
                each_w *= f
            block_h = max(wall_h)
            wall_w = n_w * each_w + (gap + 0.4) * (n_w - 1)
        else:
            wall_w = avail_w
            wall_h = [wall_w * r for r in aspects]
            cap = 6.4 / max(1, n_w) if n_w > 1 else 6.4
            if max(wall_h) > cap:              # tall detector: shrink width
                f = cap / max(wall_h)
                wall_h = [h * f for h in wall_h]
                wall_w *= f
            block_h = sum(wall_h) + gap * (n_w - 1)
        head_h = 1.05 + 0.18 * (scale - 1)
        strip_h, bottom = 1.55 * max(1.0, scale ** 0.5), 0.55
        strip_gap = 0.8 * max(1.0, scale ** 0.5)     # room for the z label
        fig_h = head_h + block_h + strip_gap + strip_h + bottom
        self.fig = fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi or 100)
        fig.patch.set_facecolor(t.fig_bg)

        def frac(x0, y0, w, h):       # inches from bottom-left -> fractions
            return [x0 / fig_w, y0 / fig_h, w / fig_w, h / fig_h]

        # ---- walls
        self.wall_axes, self.tiles, self.ellipses = [], [], []
        top = fig_h - head_h
        cmap = a._cmap()
        lo, hi = a.pe_range
        tile_idx = []
        for w in range(W.n_walls):
            if side:
                ax = fig.add_axes(frac(wall_left + w * (each_w + gap + 0.4),
                                       top - wall_h[w], each_w, wall_h[w]))
            else:
                y0 = top - wall_h[w]
                ax = fig.add_axes(frac(wall_left, y0, wall_w, wall_h[w]))
                top = y0 - gap
            ax.set_facecolor(t.axes_bg)
            zmin, zmax, ymin, ymax = spans[w]
            for (bz0, bz1, by0, by1) in a._wall_boxes(w):
                ax.add_patch(Rectangle((bz0, by0), bz1 - bz0, by1 - by0,
                                       fill=False, edgecolor=t.grid,
                                       linewidth=0.5, alpha=0.6, zorder=1))
            m = np.flatnonzero(W.wall == w)
            tile_idx.append(m)
            verts = [((W.z[i] - W.half_z[i], W.y[i] - W.half_y[i]),
                      (W.z[i] + W.half_z[i], W.y[i] - W.half_y[i]),
                      (W.z[i] + W.half_z[i], W.y[i] + W.half_y[i]),
                      (W.z[i] - W.half_z[i], W.y[i] + W.half_y[i])) for i in m]
            pc = PolyCollection(verts, facecolors=a._off_colour(),
                                edgecolors="none", linewidths=0.0, zorder=2)
            ax.add_collection(pc)
            self.tiles.append(pc)
            ax.set_xlim(zmin, zmax)
            ax.set_ylim(ymin, ymax)
            ax.set_aspect("equal", adjustable="box")
            if w == 0 or not side:
                ax.set_ylabel("y [cm]", fontsize=fonts["label"])
            if side or w == W.n_walls - 1:
                ax.set_xlabel("z [cm]  (beam)", fontsize=fonts["label"])
            _style_axes(ax, t, fonts)
            ax.grid(False)
            if W.n_walls > 1 or W.projected:
                ax.set_title(W.wall_label(w), fontsize=fonts["label"],
                             color=t.fg_muted, loc="left", pad=3)
            self.wall_axes.append(ax)
        self.tile_idx = tile_idx

        # flashes: one ellipse (+ label) per flash and wall it was seen on
        f = a.flashes
        widths = _pe_width(f["pe"], lo=0.8, hi=3.2)
        self.flash_art = []           # (flash j, ellipse, centre, label|None)
        labelled = set(range(min(6, len(f["pe"]))))
        for j in range(len(f["pe"])):
            for w in range(W.n_walls):
                if not f["on_wall"][j, w]:
                    continue
                ax = self.wall_axes[w]
                el = Ellipse((f["z"][j], f["y"][j]), 2 * f["z_width"][j],
                             2 * f["y_width"][j], fill=False,
                             edgecolor=t.track, linewidth=widths[j], zorder=5,
                             visible=False)
                ax.add_patch(el)
                c, = ax.plot([f["z"][j]], [f["y"][j]], "+", color=t.track,
                             markersize=7, mew=1.4, zorder=6, visible=False)
                lab = None
                if j in labelled:
                    # on the ellipse at 45 degrees, clear of the centre
                    # (where the true vertex usually is)
                    lab = ax.text(f["z"][j] + 0.71 * f["z_width"][j],
                                  f["y"][j] - 0.71 * f["y_width"][j],
                                  f" {_fmt_pe(f['pe'][j])} PE",
                                  color=t.track, fontsize=fonts["legend"],
                                  ha="left", va="top", zorder=7,
                                  visible=False, clip_on=True)
                self.flash_art.append((j, el, c, lab))

        # true vertex, projected onto every wall
        tv = a.true_vertex
        self.has_vertex = tv is not None and np.isfinite(tv).all()
        self.vertex_art = []
        for ax in self.wall_axes:
            if self.has_vertex:
                v, = ax.plot([tv[2]], [tv[1]], marker="*", markersize=15,
                             color=t.mctrack, markeredgecolor=t.axes_bg,
                             markeredgewidth=1.0, linestyle="none", zorder=8)
                self.vertex_art.append(v)

        # colourbar
        cax = fig.add_axes(frac(wall_left + wall_w + 0.18, fig_h - head_h - block_h,
                                0.16, block_h))
        sm = plt.cm.ScalarMappable(norm=mcolors.LogNorm(vmin=lo, vmax=hi),
                                   cmap=cmap)
        cb = fig.colorbar(sm, cax=cax)
        _style_colorbar(cb, "PE per detector" + (" (afterglow)" if a.fill ==
                                                 "afterglow" else " (so far)"),
                        t, fonts)

        # ---- strip
        sax = fig.add_axes(frac(left, bottom, wall_left + wall_w - left, strip_h))
        self.strip_ax = sax
        sax.set_facecolor(t.axes_bg)
        n = a.n_frames
        edges = np.arange(n + 1, dtype=float)
        inside, outside = a.strip
        links, _ = a._links
        floor = 0.3
        top_pe = max(float(max(inside.max(initial=0), outside.max(initial=0))),
                     float(f["pe"].max()) if len(f["pe"]) else 0.0, 10.0)
        in_col = t.categorical[0]
        out_col = t.categorical[4]
        if links is not None:
            sax.stairs(np.where(inside > 0, inside, np.nan), edges, color=in_col,
                       linewidth=1.3, label="hits in a flash", baseline=None)
            sax.stairs(np.where(outside > 0, outside, np.nan), edges,
                       color=out_col, linewidth=1.1, label="hits in no flash",
                       baseline=None)
        else:
            tot = inside + outside
            sax.stairs(np.where(tot > 0, tot, np.nan), edges, color=in_col,
                       linewidth=1.3, label="OpHit PE", baseline=None)
        if len(f["pe"]):
            xf = a.frame_coord(f["time"])
            sax.vlines(xf, floor, f["pe"], color=t.track, linewidth=0.9,
                       alpha=0.8, zorder=4)
            sax.plot(xf, f["pe"], "o", color=t.track, markersize=4,
                     markeredgecolor=t.axes_bg, markeredgewidth=0.6,
                     label="flash (total PE)", zorder=5, linestyle="none")
        x0 = float(a.frame_coord(a.t0))
        sax.axvline(x0, color=t.fg_muted, linestyle=":", linewidth=0.8)
        sax.set_yscale("log")
        sax.set_ylim(floor, top_pe * 3)
        sax.set_xlim(0, n)
        pos, labels = a._tick_positions()
        sax.set_xticks(pos)
        sax.set_xticklabels(labels)
        sax.set_ylabel("PE per frame", fontsize=fonts["label"])
        clock = ("frames log-spaced after t₀" if a.time_scale == "log"
                 else "uniform frames")
        sax.set_xlabel(f"t − t₀   ({clock};  t₀ = "
                       f"{a.t0:.3f} µs)", fontsize=fonts["label"])
        _style_axes(sax, t, fonts)
        self.elapsed = Rectangle((0, floor), 0, top_pe * 3, facecolor=t.fg,
                                 alpha=0.08, edgecolor="none", zorder=0)
        sax.add_patch(self.elapsed)
        self.playhead = sax.axvline(0, color=t.fg, linewidth=1.4, zorder=6)
        leg = sax.legend(loc="upper right", fontsize=fonts["legend"],
                         facecolor=t.legend_bg, edgecolor=t.legend_edge,
                         labelcolor=t.fg, framealpha=0.85, ncol=3)
        leg.set_zorder(10)

        # ---- header
        head = a._truth_headline()
        fig.text(left / fig_w, 1 - 0.14 / fig_h, a._title(),
                 fontsize=fonts["suptitle"], color=t.fg, ha="left", va="top")
        if head:
            fig.text(left / fig_w, 1 - 0.40 / fig_h, head,
                     fontsize=fonts["label"], color=t.mctrack, ha="left",
                     va="top")
        self.clock = fig.text(left / fig_w, 1 - (head_h - 0.12) / fig_h, "",
                              fontsize=fonts["suptitle"] * 1.35, color=t.fg,
                              ha="left", va="bottom", family="monospace")
        self.status = fig.text((wall_left + wall_w) / fig_w,
                               1 - (head_h - 0.12) / fig_h, "",
                               fontsize=fonts["label"], color=t.fg_muted,
                               ha="right", va="bottom")
        key = []
        if self.has_vertex:
            key.append("★ true vertex")
        if len(f["pe"]):
            key.append("◯ flash ±1σ")
        key.append("□ outline: light this frame")
        fig.text((wall_left + wall_w) / fig_w, 1 - 0.40 / fig_h,
                 "     ".join(key), fontsize=fonts["legend"],
                 color=t.fg_muted, ha="right", va="top")
        self._cum = np.cumsum(inside + outside)
        self._fg_rgba = np.array(mcolors.to_rgba(t.fg))

        # ---- 3-D charge, beside the walls
        self.tiles3d = None
        self.front3d = []
        self.bars3d = []
        if charge is not None:
            block_bottom = fig_h - head_h - block_h
            ax3 = fig.add_axes(frac(left - 0.55, block_bottom - 0.45,
                                    c3d_w + 0.1, block_h + 0.35),
                               projection="3d")
            self._build_charge3d(ax3, charge)

    def _dynamic(self) -> list:
        """Artists that change between frames, in drawing (z) order."""
        out = list(self.tiles)
        for _j, el, c, lab in self.flash_art:
            out += [el, c] + ([lab] if lab is not None else [])
        out += self.vertex_art
        out += [self.elapsed, self.playhead, self.clock, self.status]
        # 3-D: the wall, then flash bars, then whatever is in front of it
        if self.tiles3d is not None:
            out.append(self.tiles3d)
        out += [bar for _j, bar in self.bars3d]
        out += self.front3d
        return out

    def start_blit(self) -> None:
        """Draw everything static once and keep it as the background.

        A full redraw costs ~0.28 s a frame, almost all of it re-rendering
        tick marks (the log strip has hundreds) that never change.
        """
        for art in self._dynamic():
            art.set_animated(True)
        self.fig.canvas.draw()
        self._bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)

    def grab(self) -> np.ndarray:
        """RGB image of the current state, blitted over the background."""
        canvas = self.fig.canvas
        canvas.restore_region(self._bg)
        for art in self._dynamic():
            if art.get_visible():
                if hasattr(art, "do_3d_projection"):
                    # the colours of a 3-D collection are applied when it is
                    # projected, not when they are set
                    art.do_3d_projection()
                art.axes.draw_artist(art) if art.axes is not None \
                    else self.fig.draw_artist(art)
        return np.asarray(canvas.buffer_rgba())[..., :3].copy()

    def update(self, k: int | None) -> None:
        a, t = self.a, self.t
        if k is None:
            vals = a.collected
            new = np.zeros(len(a.walls), bool)
            alpha = a._flash_alpha(None)
            self.clock.set_text("whole window")
            self.playhead.set_visible(False)
            self.elapsed.set_width(0)
            pe_so_far = float(self._cum[-1]) if len(self._cum) else 0.0
            nflash = len(a.flashes["pe"])
        else:
            k = int(np.clip(k, 0, a.n_frames - 1))
            vals = a.values[k]
            new = a.arrivals[k] > 0
            alpha = a._flash_alpha(k)
            self.clock.set_text(f"t − t₀ = {a.frame_label(k)}")
            self.playhead.set_visible(True)
            self.playhead.set_xdata([k + 1, k + 1])
            self.elapsed.set_width(k + 1)
            pe_so_far = float(self._cum[k])
            nflash = int((a.flashes["frame"] <= k).sum())
        rgba = a._tile_rgba(vals)
        for pc, m in zip(self.tiles, self.tile_idx):
            pc.set_facecolor(rgba[m])
            pc.set_edgecolor(np.where(new[m][:, None], self._fg_rgba,
                                      (0.0, 0.0, 0.0, 0.0)))
            pc.set_linewidth(np.where(new[m], 1.1, 0.0))
        for j, el, c, lab in self.flash_art:
            on = bool(alpha[j] > 0)
            for art in (el, c):
                art.set_visible(on)
                art.set_alpha(float(alpha[j]) if on else 0.0)
            if lab is not None:
                lab.set_visible(on)
                lab.set_alpha(max(0.75, float(alpha[j])) if on else 0.0)
        if self.tiles3d is not None:
            self.tiles3d.set_facecolor(rgba[self.tile3d_idx])
        for j, bar in self.bars3d:
            on = bool(alpha[j] > 0)
            bar.set_visible(on)
            bar.set_alpha(float(alpha[j]) if on else 0.0)
        what = "PE in the window" if k is None else "PE so far"
        self.status.set_text(f"{pe_so_far:,.0f} {what}   \u00b7   "
                             f"{nflash} flash{'es' * (nflash != 1)}")


    def _build_charge3d(self, ax, d3) -> None:
        """Space points, tracks and the true vertex, with the PD wall in place.

        Drawn in the detector frame the 3-D view uses (horizontal z, depth x,
        vertical y). Charge is grey-scale -- brightness ~ log charge -- so it
        cannot be mistaken for light, which owns the colour map.
        Matplotlib has no depth buffer across collections, so the charge is
        split by which side of the photon-detector wall it lies on: behind it
        is drawn first, in front of it after the tiles.
        """
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        a, t, fonts = self.a, self.t, self.fonts
        W = a.walls
        ax.set_facecolor(t.axes_bg)
        ax.computed_zorder = False
        xr, yr, zr = a._charge_ranges
        az = np.radians(a.azim)

        # outline of the region shown, and of each photon-detector wall
        z0, z1 = zr
        x0, x1 = xr
        y0, y1 = yr
        for (xa, xb) in ((x0, x0), (x1, x1)):
            ax.plot([z0, z1, z1, z0, z0], [xa, xa, xa, xa, xa],
                    [y0, y0, y1, y1, y0], color=t.grid, lw=0.5, alpha=0.5)
        for (zz, yy) in ((z0, y0), (z1, y0), (z1, y1), (z0, y1)):
            ax.plot([zz, zz], [x0, x1], [yy, yy], color=t.grid, lw=0.5,
                    alpha=0.5)

        sel = np.flatnonzero((W.z + W.half_z >= z0) & (W.z - W.half_z <= z1))
        self.tile3d_idx = sel
        verts = [[(max(z0, W.z[i] - W.half_z[i]), W.x[i], W.y[i] - W.half_y[i]),
                  (min(z1, W.z[i] + W.half_z[i]), W.x[i], W.y[i] - W.half_y[i]),
                  (min(z1, W.z[i] + W.half_z[i]), W.x[i], W.y[i] + W.half_y[i]),
                  (max(z0, W.z[i] - W.half_z[i]), W.x[i], W.y[i] + W.half_y[i])]
                 for i in sel]
        self.tiles3d = Poly3DCollection(verts, facecolors=a._off_colour(),
                                        edgecolors="none", zorder=3)
        ax.add_collection3d(self.tiles3d)

        pts, rgba = a._charge_points()
        wall_x = W.wall_x[0] if W.n_walls == 1 else None
        if len(pts):
            inside = ((pts[:, 2] >= z0) & (pts[:, 2] <= z1))
            pts, rgba = pts[inside], rgba[inside]
            if wall_x is not None:
                # the eye sits at +sin(azim) along the drift axis
                front = (pts[:, 0] - wall_x) * np.sin(az) > 0
            else:
                front = np.ones(len(pts), bool)
            back = ~front
            size = 4.0 if len(pts) < 3000 else (2.2 if len(pts) < 15000 else 1.2)
            if back.any():
                ax.scatter(pts[back, 2], pts[back, 0], pts[back, 1], s=size,
                           c=rgba[back], linewidths=0, depthshade=False,
                           zorder=2)
            if front.any():
                self.front3d.append(ax.scatter(
                    pts[front, 2], pts[front, 0], pts[front, 1], s=size,
                    c=rgba[front], linewidths=0, depthshade=False, zorder=5))
        if d3.tracks is not None:
            for p in d3.tracks.points:
                p = np.asarray(p)
                if len(p) < 2:
                    continue
                keep = (p[:, 2] >= z0) & (p[:, 2] <= z1)
                if keep.sum() < 2:
                    continue
                ln, = ax.plot(p[keep, 2], p[keep, 0], p[keep, 1], color=t.track,
                              lw=0.9, alpha=0.8, zorder=6)
                self.front3d.append(ln)
        tv = a.true_vertex
        if tv is not None and np.isfinite(tv).all():
            v, = ax.plot([tv[2]], [tv[0]], [tv[1]], marker="*", markersize=13,
                         color=t.mctrack, markeredgecolor=t.axes_bg,
                         markeredgewidth=0.8, linestyle="none", zorder=7)
            self.front3d.append(v)

        # flashes: a bar across the drift at their (y, z) -- light does not
        # measure x
        f = a.flashes
        widths = _pe_width(f["pe"], lo=0.8, hi=3.2)
        for j in range(len(f["pe"])):
            if not (z0 <= f["z"][j] <= z1):
                continue
            bar, = ax.plot([f["z"][j]] * 2, [x0, x1], [f["y"][j]] * 2,
                           color=t.track, lw=widths[j] * 1.3, zorder=4,
                           visible=False)
            self.bars3d.append((j, bar))

        ax.set_xlim(z0, z1)
        ax.set_ylim(x0, x1)
        ax.set_zlim(y0, y1)
        ax.set_box_aspect((z1 - z0, x1 - x0, y1 - y0), zoom=1.05)
        ax.view_init(elev=a.elev, azim=a.azim)
        ax.set_xlabel("z [cm]", fontsize=fonts["label"], labelpad=2)
        ax.set_ylabel("x [cm] (drift)", fontsize=fonts["label"], labelpad=2)
        ax.set_zlabel("y [cm]", fontsize=fonts["label"], labelpad=2)
        _style_axes(ax, t, fonts, three_d=True)
        n = len(d3.sp)
        ax.set_title(f"charge: {n:,} space points (grey: log charge)  +  "
                     "light on the anode wall", fontsize=fonts["label"],
                     color=t.fg_muted, pad=0)


def _encode_gif(frames: list[np.ndarray], fps: float, hold: float) -> bytes:
    """Animated GIF with ONE palette, so colours do not shimmer frame to frame."""
    from PIL import Image
    if not frames:
        raise ValueError("no frames to encode")
    step = max(1, len(frames) // 12)
    sample = np.concatenate([f[::2, ::2] for f in frames[::step]], axis=0)
    pal = Image.fromarray(sample).quantize(colors=255,
                                           method=Image.Quantize.MEDIANCUT)
    # Map every 5-bit-per-channel colour to its palette entry once, then
    # index frames through that table: ~50x faster than quantising each frame.
    g = (np.arange(32, dtype=np.uint8) << 3) | 4
    cube = np.stack(np.meshgrid(g, g, g, indexing="ij"), axis=-1).reshape(-1, 1, 3)
    lut = np.asarray(Image.fromarray(cube).quantize(
        palette=pal, dither=Image.Dither.NONE)).reshape(32, 32, 32)
    ims = []
    for f in frames:
        q = f >> 3
        im = Image.fromarray(lut[q[..., 0], q[..., 1], q[..., 2]], mode="P")
        im.putpalette(pal.getpalette())
        ims.append(im)
    dur = [int(round(1000 / fps))] * len(ims)
    dur[-1] = int(round(1000 * hold))
    buf = io.BytesIO()
    ims[0].save(buf, format="GIF", save_all=True, append_images=ims[1:],
                duration=dur, loop=0, disposal=1, optimize=False)
    return buf.getvalue()


def _encode_mp4(frames: list[np.ndarray], fps: float, hold: float) -> bytes:
    """H.264 via an ffmpeg pipe; the held last frame is repeated."""
    import tempfile
    h, w = frames[0].shape[:2]
    extra = [frames[-1]] * max(0, int(round(hold * fps)) - 1)
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "light.mp4")
        cmd = ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo",
               "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
               "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p",
               "-vcodec", "libx264", out]
        proc = subprocess.run(cmd, input=b"".join(
            np.ascontiguousarray(f).tobytes() for f in frames + extra),
            capture_output=True)
        if proc.returncode != 0:
            raise ValueError(f"ffmpeg failed: {proc.stderr.decode()[-400:]}")
        with open(out, "rb") as fh:
            return fh.read()


# --------------------------------------------------------------------------
# plotly animation


def _rect_path(z, y, hz, hy) -> tuple[list, list]:
    xs, ys = [], []
    for zc, yc, a, b in zip(z, y, hz, hy):
        xs += [zc - a, zc + a, zc + a, zc - a, zc - a, None]
        ys += [yc - b, yc - b, yc + b, yc + b, yc - b, None]
    return xs, ys


def _ellipse_path(z, y, zw, yw, n: int = 40) -> tuple[list, list]:
    th = np.linspace(0, 2 * np.pi, n)
    xs, ys = [], []
    for zc, yc, a, b in zip(z, y, zw, yw):
        xs += list(np.round(zc + a * np.cos(th), 1)) + [None]
        ys += list(np.round(yc + b * np.sin(th), 1)) + [None]
    return xs, ys


def _r(v, nd: int = 1) -> list:
    """Rounded list with NaN -> None: frames are JSON, and size is the cost."""
    arr = np.round(np.asarray(v, float), nd)
    return [None if not np.isfinite(x) else float(x) for x in arr]


def _plotly_charge3d(a, fig, add) -> None:
    """The 3-D panel: charge from Display3D, the PD wall and flash bars."""
    from matplotlib.colors import to_rgb
    t, W, f = a.theme, a.walls, a.flashes
    d3 = a.charge
    _xyz, level = a._charge_levels()
    dim = np.array(to_rgb(t.grid))
    bright = np.array(to_rgb(t.fg))
    grey = [[0.0, _hex(dim + (bright - dim) * 0.25)], [1.0, _hex(bright)]]
    for tr in d3.plotly_figure().data:
        tr = tr.to_plotly_json()
        tr.pop("scene", None)
        if tr.get("name") == "space points":
            # light owns the colour map here; charge is grey
            tr["marker"] = dict(size=2.4 if len(level) < 15000 else 1.4,
                                color=_r(level, 2), cmin=0, cmax=1,
                                colorscale=grey, showscale=False, opacity=0.9)
            tr["name"] = "space points (grey: log charge)"
        add(tr, 1, col=1)

    xr, yr, zr = a._charge_ranges
    ever = np.maximum(a.values.max(axis=0), a.collected) >= PE_OFF
    lighting = dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0,
                    fresnel=0.0)

    def mesh(idx):
        z0 = W.z[idx] - W.half_z[idx]
        z1 = W.z[idx] + W.half_z[idx]
        y0 = W.y[idx] - W.half_y[idx]
        y1 = W.y[idx] + W.half_y[idx]
        X = np.column_stack([z0, z1, z1, z0]).ravel()
        Y = np.repeat(W.x[idx], 4)
        Z = np.column_stack([y0, y0, y1, y1]).ravel()
        b = 4 * np.arange(len(idx))
        return dict(type="mesh3d", x=_r(X), y=_r(Y, 2), z=_r(Z),
                    i=np.column_stack([b, b]).ravel().tolist(),
                    j=np.column_stack([b + 1, b + 2]).ravel().tolist(),
                    k=np.column_stack([b + 2, b + 3]).ravel().tolist(),
                    flatshading=True, lighting=lighting, hoverinfo="skip",
                    showlegend=False, showscale=False)

    dark = np.flatnonzero(~ever)
    if len(dark):
        add({**mesh(dark), "color": a._off_colour(),
             "name": "photon detectors"}, 1, col=1)
    lit = np.flatnonzero(ever)
    if len(lit):
        def wall3d(k):
            v = (a.collected if k is None else a.values[k])[lit]
            return dict(intensity=np.repeat(a._tile_levels(v), 2)
                        .astype(np.float32))
        add({**mesh(lit), "name": "photon detectors (lit)",
             "intensitymode": "cell", "colorscale": a._tile_colorscale(),
             "cmin": a._OFF_LEVEL, "cmax": 1.0}, 1, wall3d, col=1)

    def bars(k, fresh):
        on = np.ones(len(f["pe"]), bool) if k is None else (
            (f["frame"] == k) if fresh else (f["frame"] < k))
        if k is None and fresh:
            on[:] = False
        sel = np.flatnonzero(on)
        xs, ys, zs = [], [], []
        for j in sel:
            xs += [float(f["z"][j])] * 2 + [None]
            ys += [xr[0], xr[1], None]
            zs += [float(f["y"][j])] * 2 + [None]
        return dict(x=xs or [None], y=ys or [None], z=zs or [None])

    add(dict(type="scatter3d", mode="lines", hoverinfo="skip",
             line=dict(color=t.track, width=3), opacity=0.4,
             name="earlier flashes (bar: x unknown)", showlegend=False), 1,
        lambda k: bars(k, False), col=1)
    add(dict(type="scatter3d", mode="lines", hoverinfo="skip",
             line=dict(color=t.track, width=8),
             name="new flash (bar: x unknown)", showlegend=False), 1,
        lambda k: bars(k, True), col=1)

    pane = dict(backgroundcolor=t.axes_bg, gridcolor=t.grid,
                zerolinecolor=t.grid, color=t.fg_muted)
    el, az = np.radians(a.elev), np.radians(a.azim)
    r = 1.9
    fig.update_scenes(
        xaxis=dict(title="z [cm] (beam)", range=list(zr), **pane),
        yaxis=dict(title="x [cm] (drift)", range=list(xr), **pane),
        zaxis=dict(title="y [cm] (up)", range=list(yr), **pane),
        aspectmode="data",
        camera=dict(eye=dict(x=r * np.cos(el) * np.cos(az),
                             y=r * np.cos(el) * np.sin(az),
                             z=r * np.sin(el))))


def _plotly_animation(a: LightAnimation, *, height: int | None, fps: float):
    """Animated plotly figure: one frame per clock tick, slider and play.

    Frames carry only what changes -- tile colours and PE, arrival outlines,
    flash outlines and markers, the playhead -- and plotly merges them into
    the traces. Sending whole traces each frame made the figure 5.4 MB and
    took 5 s to build for 120 frames of a 480-detector wall.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    t = a.theme
    W = a.walls
    nw = W.n_walls
    spans = [W.extent(w) for w in range(nw)]
    ratio = [(s[3] - s[2]) / (s[1] - s[0]) for s in spans]
    strip_share = 0.24
    wall_share = [(1 - strip_share) * r / sum(ratio) for r in ratio]
    titles = [W.wall_label(w) if (nw > 1 or W.projected) else ""
              for w in range(nw)]
    charge = getattr(a, "charge", None)
    srow = nw + 1
    # Same rule as the static figure: walls about as tall as they are wide
    # (ProtoDUNE's two APA planes) go side by side, long ones are stacked.
    side = nw > 1 and min(ratio) > 0.45
    wcol = 1 if charge is None else 2
    cells = [(w + 1, wcol) for w in range(nw)]
    if side:
        c0 = 1 if charge is None else 2
        ncol = nw + c0 - 1
        row1 = ([{"type": "scene"}] if charge is not None else []) \
            + [{"type": "xy"} for _ in range(nw)]
        row1_titles = ([f"charge: {len(charge.sp):,} space points (grey: log "
                        "charge) + light on the photon detectors"]
                       if charge is not None else []) + titles
        fig = make_subplots(
            rows=2, cols=ncol,
            specs=[row1, [{"type": "xy", "colspan": ncol}] + [None] * (ncol - 1)],
            column_widths=([0.4] + [0.6 / nw] * nw) if charge is not None else None,
            row_heights=[1 - strip_share, strip_share],
            horizontal_spacing=0.05, vertical_spacing=0.1,
            subplot_titles=row1_titles + [""])
        srow = 2
        cells = [(1, c0 + w) for w in range(nw)]
    elif charge is None:
        fig = make_subplots(rows=nw + 1, cols=1,
                            row_heights=wall_share + [strip_share],
                            vertical_spacing=0.08 if nw == 1 else 0.05,
                            subplot_titles=titles + [""])
    else:
        # the 3-D charge spans the wall rows on the left; the strip spans all
        specs = ([[{"type": "scene", "rowspan": nw}, {"type": "xy"}]]
                 + [[None, {"type": "xy"}] for _ in range(nw - 1)]
                 + [[{"type": "xy", "colspan": 2}, None]])
        order = []                    # make_subplots titles run row by row
        for w in range(nw):
            order += ([f"charge: {len(charge.sp):,} space points "
                       "(grey: log charge) + light on the anode wall"]
                      if w == 0 else []) + [titles[w]]
        fig = make_subplots(rows=nw + 1, cols=2, specs=specs,
                            column_widths=[0.5, 0.5],
                            row_heights=wall_share + [strip_share],
                            horizontal_spacing=0.05,
                            vertical_spacing=0.08 if nw == 1 else 0.05,
                            subplot_titles=order + [""])
    f = a.flashes
    lo, hi = a.pe_range
    n = a.n_frames
    dyn: list[tuple[int, callable]] = []     # (trace index, frame -> dict)
    cur = {"col": cells[0][1]}               # the wall column being filled

    def add(static: dict, row: int, fn=None, k0: int | None = 0,
            col: int | None = None):
        """Add a trace; ``fn(k)`` gives its per-frame properties."""
        props = dict(static)
        if fn is not None:
            for key, val in fn(k0).items():
                if isinstance(val, dict) and isinstance(props.get(key), dict):
                    props[key] = {**props[key], **val}
                else:
                    props[key] = val
        if col is None:
            col = 1 if row == srow else cur["col"]
        fig.add_trace(props, row=row, col=col)
        if fn is not None:
            # Frame entries MUST repeat the trace type: plotly.py validates a
            # typeless entry as a scatter, and the animation then turns the
            # detector bars into one scatter line on the first frame.
            kind = static["type"]
            dyn.append((len(fig.data) - 1,
                        lambda k, fn=fn, kind=kind: {"type": kind, **fn(k)}))

    for w in range(nw):
        row, wc = cells[w]
        cur["col"] = wc
        m = np.flatnonzero(W.wall == w)
        bx, by = [], []
        for (z0, z1, y0, y1) in a._wall_boxes(w):
            bx += [z0, z1, z1, z0, z0, None]
            by += [y0, y0, y1, y1, y0, None]
        add(dict(type="scatter", x=bx, y=by, mode="lines", hoverinfo="skip",
                 line=dict(color=t.grid, width=1), showlegend=False), row)

        hover_txt = [f"y {W.y[i]:.0f} cm · ch "
                     + ", ".join(str(c) for c in W.channels[i][:4])
                     + ("..." if len(W.channels[i]) > 4 else "") for i in m]

        def tiles(k, m=m):
            v = (a.collected if k is None else a.values[k])[m]
            # float32: plotly ships numpy arrays base64-encoded, so
            # precision is size
            return dict(marker=dict(color=a._tile_levels(v).astype(np.float32)),
                        customdata=np.round(v, 1).astype(np.float32))

        add(dict(type="bar", marker=dict(colorscale=a._tile_colorscale(),
                                         cmin=a._OFF_LEVEL, cmax=1.0,
                                         line=dict(width=0)),
                 x=_r(W.z[m]), y=_r(2 * W.half_y[m]),
                 base=_r(W.y[m] - W.half_y[m]), width=_r(2 * W.half_z[m]),
                 text=hover_txt, textposition="none",
                 hovertemplate=("z %{x:.0f} cm · %{text}<br>"
                                "%{customdata:.1f} PE<extra></extra>"),
                 showlegend=False, name="photon detectors"), row, tiles)

        def arrivals(k, m=m):
            sel = m[a.arrivals[k][m] > 0] if k is not None else m[:0]
            xs, ys = _rect_path(W.z[sel], W.y[sel], W.half_z[sel], W.half_y[sel])
            return dict(x=_r(xs) if xs else [None], y=_r(ys) if ys else [None])

        add(dict(type="scatter", mode="lines", line=dict(color=t.fg, width=1.2),
                 hoverinfo="skip", name="light this frame", legendgroup="new",
                 showlegend=(w == 0)), row, arrivals)

        def ellipses(k, w=w, fresh=True):
            on = f["on_wall"][:, w]
            if k is None:
                sel = np.flatnonzero(on) if not fresh else on[:0]
            else:
                sel = np.flatnonzero(on & ((f["frame"] == k) if fresh
                                           else (f["frame"] < k)))
            xs, ys = _ellipse_path(f["z"][sel], f["y"][sel],
                                   f["z_width"][sel], f["y_width"][sel])
            return dict(x=xs or [None], y=ys or [None])

        add(dict(type="scatter", mode="lines", opacity=0.5, hoverinfo="skip",
                 line=dict(color=t.track, width=1.2), name="earlier flashes",
                 legendgroup="old", showlegend=(w == 0)), row,
            lambda k, w=w: ellipses(k, w, fresh=False))
        add(dict(type="scatter", mode="lines", hoverinfo="skip",
                 line=dict(color=t.track, width=2.8),
                 name="new flash (±1σ)", legendgroup="fresh",
                 showlegend=(w == 0)), row,
            lambda k, w=w: ellipses(k, w, fresh=True))

        def centres(k, w=w):
            on = f["on_wall"][:, w]
            sel = np.flatnonzero(on if k is None else on & (f["frame"] <= k))
            if not len(sel):
                return dict(x=[None], y=[None], text=[""], customdata=[[0, 0, 0, 0]])
            return dict(
                x=_r(f["z"][sel]), y=_r(f["y"][sel]),
                text=[f"{_fmt_pe(p)} PE" if r < 6 else ""
                      for r, p in zip(sel, f["pe"][sel])],
                customdata=np.round(np.column_stack(
                    [f["pe"][sel], f["time"][sel] - a.t0, f["y_width"][sel],
                     f["z_width"][sel]]), 3).tolist())

        add(dict(type="scatter", mode="markers+text", name="flashes",
                 marker=dict(symbol="cross-thin", size=11,
                             line=dict(color=t.track, width=2)),
                 textposition="bottom right",
                 textfont=dict(color=t.track, size=11),
                 hovertemplate=("flash %{customdata[0]:,.0f} PE<br>"
                                "t − t₀ = %{customdata[1]:.3f} µs<br>"
                                "y %{y:.0f} ± %{customdata[2]:.0f} cm<br>"
                                "z %{x:.0f} ± %{customdata[3]:.0f} cm"
                                "<extra></extra>"),
                 showlegend=False), row, centres)
        tv = a.true_vertex
        if tv is not None and np.isfinite(tv).all():
            add(dict(type="scatter", x=[tv[2]], y=[tv[1]], mode="markers",
                     marker=dict(symbol="star", size=16, color=t.mctrack,
                                 line=dict(color=t.axes_bg, width=1)),
                     name="true vertex", showlegend=(w == 0),
                     hovertemplate=(f"true vertex<br>x {tv[0]:.0f}, "
                                    f"y {tv[1]:.0f}, z {tv[2]:.0f} cm"
                                    "<extra></extra>")), row)
        zmin, zmax, ymin, ymax = spans[w]
        xa = fig.get_subplot(row, wc).xaxis.plotly_name.replace("axis", "")
        # Equal aspect, and shrink the panel rather than pad the axis: the
        # default filled the width with z = -1000..2300 cm of nothing.
        fig.update_xaxes(range=[zmin, zmax], row=row, col=wc,
                         title_text="z [cm] (beam)" if (side or row == nw) else None,
                         showgrid=False, zeroline=False, constrain="domain")
        fig.update_yaxes(range=[ymin, ymax], row=row, col=wc,
                         title_text="y [cm]" if (w == 0 or not side) else None,
                         scaleanchor=xa, scaleratio=1, showgrid=False,
                         zeroline=False, constrain="domain")

    if charge is not None:
        _plotly_charge3d(a, fig, add)

    # colourbar carrier: bars take explicit colours, so the scale lives here
    decades = np.arange(np.floor(np.log10(lo)), np.ceil(np.log10(hi)) + 1)
    add(dict(type="scatter", x=[None], y=[None], mode="markers",
             hoverinfo="skip", showlegend=False,
             marker=dict(colorscale=charge_colorscale(t, a.colormap),
                         cmin=float(np.log10(lo)), cmax=float(np.log10(hi)),
                         color=[float(np.log10(lo))], size=0.1,
                         colorbar=dict(
                             title=dict(text=("PE (afterglow)"
                                              if a.fill == "afterglow"
                                              else "PE (so far)"), side="right"),
                             thickness=12, len=0.6, y=0.62,
                             tickvals=decades.tolist(),
                             ticktext=[_fmt_pe(10.0 ** d) if d >= 0
                                       else f"{10.0 ** d:g}" for d in decades]))),
        1)

    # ---- strip
    inside, outside = a.strip
    links, _ = a._links
    xs = np.repeat(np.arange(n + 1, dtype=float), 2)[1:-1]
    k_of = np.repeat(np.arange(n), 2)
    series = ([("hits in a flash", inside, t.categorical[0]),
               ("hits in no flash", outside, t.categorical[4])]
              if links is not None else
              [("OpHit PE", inside + outside, t.categorical[0])])
    for name, v, col in series:
        ys = np.repeat(np.where(v > 0, v, np.nan), 2)
        add(dict(type="scatter", x=_r(xs, 0), y=_r(ys), mode="lines", name=name,
                 line=dict(color=col, width=1.6),
                 customdata=[a.frame_label(k) for k in k_of],
                 hovertemplate=(f"{name}<br>%{{y:,.1f}} PE in the frame ending "
                                "at %{customdata}<extra></extra>")), srow)
    floor = 0.3
    top_pe = max(float(np.max(np.concatenate([inside, outside, [10.0]]))),
                 float(f["pe"].max()) if len(f["pe"]) else 0.0)
    ytop = top_pe * 3
    if len(f["pe"]):
        xf = a.frame_coord(f["time"])
        sx, sy = [], []
        for x, p in zip(xf, f["pe"]):
            sx += [float(x), float(x), None]
            sy += [floor, float(p), None]
        add(dict(type="scatter", x=sx, y=sy, mode="lines", hoverinfo="skip",
                 line=dict(color=t.track, width=1), showlegend=False), srow)
        add(dict(type="scatter", x=_r(xf, 2), y=_r(f["pe"]), mode="markers",
                 name="flash (total PE)",
                 marker=dict(color=t.track, size=7,
                             line=dict(color=t.axes_bg, width=1)),
                 customdata=_r(f["time"] - a.t0, 3),
                 hovertemplate=("flash %{y:,.0f} PE<br>t − t₀ = "
                                "%{customdata:.3f} µs<extra></extra>")), srow)
    x0 = float(a.frame_coord(a.t0))
    add(dict(type="scatter", x=[x0, x0], y=[floor, ytop], mode="lines",
             hoverinfo="skip", line=dict(color=t.fg_muted, width=1, dash="dot"),
             showlegend=False), srow)

    def elapsed(k):
        e = 0 if k is None else k + 1
        return dict(x=[0, e, e, 0, 0])

    add(dict(type="scatter", y=[floor, floor, ytop, ytop, floor], fill="toself",
             mode="lines", line=dict(width=0),
             fillcolor="rgba(148,163,184,0.15)", hoverinfo="skip",
             showlegend=False), srow, elapsed)

    def playhead(k):
        e = n if k is None else k + 1
        return dict(x=[e, e])

    add(dict(type="scatter", y=[floor, ytop], mode="lines",
             line=dict(color=t.fg, width=2), hoverinfo="skip",
             showlegend=False), srow, playhead)
    pos, labels = a._tick_positions()
    clock = ("frames log-spaced after t₀" if a.time_scale == "log"
             else "uniform frames")
    fig.update_xaxes(range=[0, n], tickvals=pos.tolist(), ticktext=labels,
                     title_text=(f"t − t₀   ({clock};  t₀ = "
                                 f"{a.t0:.3f} µs)"),
                     row=srow, col=1, showgrid=False, zeroline=False)
    fig.update_yaxes(type="log", range=[np.log10(floor), np.log10(ytop)],
                     title_text="PE per frame", row=srow, col=1,
                     gridcolor=t.grid, zeroline=False)

    # ---- frames, slider, buttons
    idx = [i for i, _ in dyn]
    fig.frames = [dict(name=str(k), traces=idx, data=[fn(k) for _, fn in dyn])
                  for k in range(n)]
    dur = int(round(1000 / fps))
    steps = [dict(method="animate", label=a.frame_label(k),
                  args=[[str(k)], dict(mode="immediate",
                                       frame=dict(duration=0, redraw=True),
                                       transition=dict(duration=0))])
             for k in range(n)]
    head = a._truth_headline()
    title = a._title() + (f"<br><span style='font-size:11px;color:{t.mctrack}'>"
                          f"{head}</span>" if head else "")
    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color=t.fg)),
        height=height, barmode="overlay", bargap=0,
        template="plotly_dark" if t.is_dark else "plotly_white",
        paper_bgcolor=t.fig_bg, plot_bgcolor=t.axes_bg,
        font=dict(color=t.fg_muted),
        legend=dict(orientation="h", x=1.0, xanchor="right",
                    y=1.06 if side else 1.0,
                    yanchor="bottom", bgcolor="rgba(0,0,0,0)",
                    font=dict(color=t.fg, size=11)),
        margin=dict(l=60, r=30, t=90, b=40),
        uirevision="light",
        updatemenus=[dict(
            type="buttons", direction="left", x=0.0, xanchor="left", y=-0.02,
            yanchor="top", pad=dict(t=50, r=10), showactive=False,
            bgcolor=t.legend_bg, bordercolor=t.legend_edge,
            font=dict(color=t.fg),
            buttons=[dict(label="▶ play", method="animate",
                          args=[None, dict(frame=dict(duration=dur, redraw=True),
                                           transition=dict(duration=0),
                                           fromcurrent=True, mode="immediate")]),
                     dict(label="❚❚ pause", method="animate",
                          args=[[None], dict(frame=dict(duration=0, redraw=False),
                                             transition=dict(duration=0),
                                             mode="immediate")])])],
        sliders=[dict(active=0, steps=steps, x=0.16, len=0.84, y=-0.02,
                      yanchor="top", pad=dict(t=40, b=10),
                      bgcolor=t.legend_bg, bordercolor=t.legend_edge,
                      activebgcolor=t.fg_muted, tickcolor=t.fg_muted,
                      # 120 step labels cannot be read; the current value
                      # above the slider says the time instead
                      font=dict(color="rgba(0,0,0,0)", size=8), ticklen=3,
                      currentvalue=dict(prefix="t − t₀ = ",
                                        font=dict(color=t.fg, size=13),
                                        xanchor="left"),
                      transition=dict(duration=0))])
    for ann in fig.layout.annotations:
        ann.font.size = 11
        ann.font.color = t.fg_muted
    return fig
