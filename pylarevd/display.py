"""Event display rendering: static (matplotlib) and interactive (plotly).

Both backends draw the same thing, so a figure you tune interactively saves to
the same picture.

Panels
------
By default hits are grouped by *wire orientation*: one panel per direction the
wires actually measure.  Because induction wires wrap around the APA frame, U
on one face is parallel to V on the other, so this grouping -- not the U/V
label -- is what lets both drift volumes share a single consistent ``w`` axis.
The horizontal axis is the continuous wire coordinate ``w``, the vertical one
the drift coordinate ``x``; a dashed line marks the anode and dotted lines the
active-volume edge.  ``merge="view"`` and ``merge="none"`` give the other two
groupings, and ``space="readout"`` switches to the classic channel-vs-tick view
that needs no geometry at all.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from .geometry import VIEW_NAMES

from .theme import (COLORMAPS, DEFAULT_COLORMAP, MARKER_CYCLE,
                    PLOTLY_MARKER_CYCLE, THEMES, Theme, resolve_colormap,
                    resolve_theme)

SAVE_DPI = 180

#: Interaction settings, shared so a shown figure behaves like a saved one.
_PLOTLY_CONFIG = {"scrollZoom": True, "displaylogo": False}

#: A trajectory step longer than this is a gap, not a segment.
MAX_STEP_CM = 20.0

#: Font sizes per output target, scaled with the figure. Authoring at the final
#: physical size beats shrinking a 14-inch figure into a paper column, where an
#: 8 pt label lands at 0.68 mm.
PRESETS: dict[str, dict] = {
    "screen": dict(panel_width=7.0, panel_height=2.5, fonts=1.0, dpi=180),
    "paper1": dict(panel_width=3.4, panel_height=1.5, fonts=0.80, dpi=600),
    "paper2": dict(panel_width=7.0, panel_height=2.6, fonts=1.15, dpi=600),
    "slide": dict(panel_width=6.6, panel_height=2.6, fonts=2.0, dpi=200),
    "poster": dict(panel_width=9.0, panel_height=3.4, fonts=2.6, dpi=300),
}
DEFAULT_PRESET = "screen"

_BASE_FONTS = dict(tick=7.0, label=8.0, title=9.0, suptitle=10.0, legend=7.0,
                   note=8.0)


#: Chrome that is read up close rather than from across the room. Scaling the
#: truth block and legend by the full preset multiplier put a 20.8 pt note on a
#: poster, where the text bands then claimed more of the canvas than the panels
#: and constrained_layout gave up. A poster viewer reads the title from 2 m and
#: the kinematics from arm's length, so these grow with the square root.
_REFERENCE_FONTS = ("note", "legend")


def _fonts(preset: str) -> dict:
    scale = PRESETS[preset]["fonts"]
    return {k: v * (scale ** 0.5 if k in _REFERENCE_FONTS else scale)
            for k, v in _BASE_FONTS.items()}


def charge_cmap(theme: Theme, name: str = DEFAULT_COLORMAP):
    """The charge ramp, trimmed so both ends stay visible on this theme."""
    import matplotlib as mpl
    import matplotlib.colors as mcolors
    base = mpl.colormaps[resolve_colormap(name).name]
    lo, hi = theme.cmap_floor, theme.cmap_ceiling
    return mcolors.LinearSegmentedColormap.from_list(
        f"{name}_{theme.name}", base(np.linspace(lo, hi, 256)))


def _sample_colorscale(scale, value, lo, hi) -> str:
    """One rgb colour from a plotly colorscale at a normalised position.

    Plotly cannot vary line width within a trace, so flashes are drawn one
    trace each -- which means their colour has to be resolved here rather than
    handed to a shared colouraxis.
    """
    span = float(hi) - float(lo)
    f = 0.0 if span <= 0 else (float(value) - float(lo)) / span
    f = min(max(f, 0.0), 1.0)
    stops = [(float(p), c) for p, c in scale]
    for (p0, c0), (p1, c1) in zip(stops, stops[1:]):
        if f <= p1:
            return c1 if p1 == p0 else c0 if f < 0.5 * (p0 + p1) else c1
    return stops[-1][1]


def charge_colorscale(theme: Theme, name: str = DEFAULT_COLORMAP) -> list[list]:
    """The same ramp as a plotly colorscale, so both backends match."""
    import matplotlib as mpl
    base = mpl.colormaps[resolve_colormap(name).name]
    lo, hi = theme.cmap_floor, theme.cmap_ceiling
    stops = np.linspace(0.0, 1.0, 32)
    return [[float(s), mpl.colors.to_hex(base(lo + s * (hi - lo)))]
            for s in stops]


#: Hits overlap heavily (~85% occluded in a busy panel), so a little
#: transparency lets density read through instead of hiding it.
_ALPHA = 0.75


def _join(polylines):
    """Concatenate polylines into one NaN-separated pair of arrays."""
    if not polylines:
        return np.empty(0), np.empty(0)
    ws, xs = [], []
    for w, x in polylines:
        ws.append(np.append(w, np.nan))
        xs.append(np.append(x, np.nan))
    return np.concatenate(ws), np.concatenate(xs)


def _join_segments(segments) -> np.ndarray:
    """NaN-separate polylines WITHOUT gap-breaking inside them.

    For lines that are straight by construction (a shower axis), the gap test
    is not just unnecessary -- it splits the segment's own two endpoints and
    nothing is drawn.
    """
    usable = [np.asarray(p) for p in segments if len(p) >= 2]
    if not usable:
        return np.zeros((0, 3))
    sep = np.full((1, 3), np.nan)
    return np.vstack([np.vstack([p, sep]) for p in usable])


def _join3d(polylines, max_step: float = MAX_STEP_CM) -> np.ndarray:
    """Many 3-D polylines as one NaN-separated (N, 3) array.

    One ``ax.plot`` per particle is 34k artists on a radiological sample and
    takes ~50 s; batched into a single call it is milliseconds. The 2-D path
    learned this already (:func:`_join`) -- this is the same trick in 3-D, and
    it keeps both backends drawing from one array.
    """
    usable = [np.asarray(p) for p in polylines if len(p) >= 2]
    if not usable:
        return np.zeros((0, 3))
    sep = np.full((1, 3), np.nan)
    return np.vstack([np.vstack([_split_polyline(p, max_step), sep])
                      for p in usable])


def _hover_template(readout: bool, xlabel: str, ylabel: str) -> str:
    """Per-hit hover text.

    In readout space the axes already *are* channel and tick, so repeating them
    from customdata would show each value twice at differing precision.
    """
    head = f"{xlabel} %{{x:.2f}}<br>{ylabel} %{{y:.2f}}<br>"
    rows = ["integral %{customdata[2]:.1f} ADC",
            "amplitude %{customdata[3]:.1f} ADC",
            "TPC %{customdata[4]:d}, wire %{customdata[5]:d}",
            "multiplicity %{customdata[6]:d}"]
    if not readout:
        rows = ["channel %{customdata[0]:d}",
                "tick %{customdata[1]:.1f}"] + rows
    return head + "<br>".join(rows) + "<extra></extra>"


#: A trajectory step longer than this is a gap, not a segment. Real steps are
#: sub-centimetre; reconstruction occasionally leaves metre-scale holes, and
#: joining across one draws a line through detector the particle never crossed.
MAX_STEP_CM = 20.0


def _gap_breaks(xyz: np.ndarray, max_step: float = MAX_STEP_CM) -> np.ndarray:
    """Indices at which a polyline should be broken."""
    if len(xyz) < 2:
        return np.empty(0, int)
    step = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    return np.flatnonzero(step > max_step) + 1


def _wrap_for(text: str, width_in: float, fontsize: float,
              char_factor: float = 0.5) -> str:
    """Wrap *text* to a figure of *width_in* inches at *fontsize* points.

    An unwrapped title is what actually blew the print presets: at paper1 the
    one-line header rendered 5.59 in wide on a 3.40 in canvas, and
    ``bbox_inches="tight"`` then grew the saved figure to fit it -- so a figure
    authored for a journal column arrived 84% too wide and every font landed
    ~55% too small once the journal scaled it back.

    ``char_factor`` is the mean glyph width as a fraction of the point size:
    0.5 for the proportional default face, ~0.6 for monospace.
    """
    import textwrap
    per_line = max(12, int(width_in * 72.0 / max(char_factor * fontsize, 1e-6)))
    return "\n".join(textwrap.fill(line, per_line, replace_whitespace=False)
                     for line in text.splitlines())


def _vector_text() -> None:
    """Embed real (Type 42) fonts in PDF/PS rather than matplotlib's Type 3.

    Type 3 extracts as text but is rejected or flagged by many journal
    production pipelines, and some RIPs render it poorly at small sizes.
    """
    import matplotlib
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42


def _data_ylim(values, pad: float = 0.08):
    """(lo, hi) framing the data with a margin, or (None, None) if degenerate."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return None, None
    lo, hi = float(v.min()), float(v.max())
    if hi <= lo:
        return lo - 1.0, hi + 1.0
    m = pad * (hi - lo)
    return lo - m, hi + m


def _split_polyline(xyz: np.ndarray, max_step: float = MAX_STEP_CM) -> np.ndarray:
    """Insert NaN rows at gaps so a drawn line breaks instead of leaping.

    Both backends treat NaN as a break, in 2-D and 3-D alike.
    """
    gaps = _gap_breaks(xyz, max_step)
    if not len(gaps):
        return xyz
    return np.insert(xyz, gaps, np.nan, axis=0)


#: Fixed PE range for flash marker size. Normalising to whatever is in the
#: current window made a 4 PE flash render larger than a 56 PE one.
PE_SIZE_RANGE = (1.0, 3e5)


def _pe_width(pe, lo: float = 0.4, hi: float = 3.0) -> np.ndarray:
    """Line width from PE on the FIXED :data:`PE_SIZE_RANGE`, not the window.

    Normalising within the displayed window is degenerate for a single flash --
    ``max == min`` collapsed the brightest flash in the event to the thinnest
    possible line -- and makes width incomparable between two renders. Same
    reasoning as :func:`_pe_size`, which the 2-D view already uses.
    """
    p0, p1 = np.log10(PE_SIZE_RANGE[0]), np.log10(PE_SIZE_RANGE[1])
    f = (np.log10(np.clip(np.asarray(pe, float), PE_SIZE_RANGE[0],
                          PE_SIZE_RANGE[1])) - p0) / (p1 - p0)
    return lo + (hi - lo) * f


def _pe_size(pe: np.ndarray) -> np.ndarray:
    """Marker area from photoelectrons, log-scaled over a FIXED range."""
    lo, hi = PE_SIZE_RANGE
    scaled = np.log10(np.clip(pe, lo, None) / lo) / np.log10(hi / lo)
    return 6.0 + 70.0 * np.clip(scaled, 0.0, 1.0)


def _si(v: float) -> str:
    """Compact number for a colourbar tick."""
    return f"{v:.0f}" if v >= 10 else f"{v:.2g}"


def _pyplot():
    """Import pyplot, selecting a headless backend only if safe to do so.

    Choosing a backend is process-global state.  If the caller has already
    imported pyplot they have made that choice -- possibly deliberately, in a
    notebook or their own script -- so we must not override it.
    """
    import sys
    import matplotlib
    if "matplotlib.pyplot" not in sys.modules and not os.environ.get("DISPLAY"):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _style_axes(ax, t: Theme, fonts: dict, *, three_d: bool = False) -> None:
    """Apply the dark theme to one axes.

    Matplotlib defaults to black text, which is invisible on ``AXES_BG``, so
    every text and line element has to be recoloured explicitly.
    """
    import matplotlib.colors as mcolors
    ax.set_facecolor(t.axes_bg)
    ax.tick_params(colors=t.fg_muted, labelsize=fonts["tick"], which="both")
    ax.xaxis.label.set_color(t.fg_muted)
    ax.yaxis.label.set_color(t.fg_muted)
    ax.xaxis.label.set_fontsize(fonts["label"])
    ax.yaxis.label.set_fontsize(fonts["label"])
    ax.title.set_color(t.fg)
    ax.title.set_fontsize(fonts["title"])
    if three_d:
        # 3-D panes are separate artists from the 2-D facecolor
        pane = mcolors.to_rgba(t.axes_bg)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_pane_color(pane)
            axis.line.set_color(t.grid)
            axis._axinfo["grid"]["color"] = t.grid
        ax.zaxis.label.set_color(t.fg_muted)
        ax.zaxis.label.set_fontsize(fonts["label"])
    else:
        for spine in ax.spines.values():
            spine.set_color(t.grid)
        ax.grid(color=t.grid, alpha=0.45, linewidth=0.5)


#: The 3-D legend scales markers up because space points are drawn 1-2 px.
_LEGEND_MARKERSCALE_3D = 8

#: Smallest share of the canvas height the panels are allowed to keep.
_MIN_AXES_FRACTION = 0.45


def _fit_bands(bottom: float, top: float) -> tuple[float, float]:
    """Shrink the reserved text bands so the panels keep enough room."""
    total = bottom + top
    budget = 1.0 - _MIN_AXES_FRACTION
    if total <= budget or total <= 0:
        return bottom, top
    k = budget / total
    return bottom * k, top * k


def _preset_figsize(preset: str, w: float, h: float) -> tuple[float, float]:
    """Scale a figure's nominal size by the preset, keeping its aspect.

    The 3-D, optical and flash figures hardcoded their canvas, so --preset only
    changed their fonts -- large type on a canvas that never grew.
    """
    spec = PRESETS[preset]
    scale = spec["panel_width"] / PRESETS["screen"]["panel_width"]
    return (w * scale, h * scale)


def _casing(t: Theme, width: float = 2.0):
    """Path effect drawing a contrasting outline under an overlay.

    Hue cannot separate the overlays from the charge ramps -- six colormaps
    cover most of colour space, and a palette satisfying that plus contrast
    plus colour-vision constraints comes out neon. A casing separates them
    structurally instead, whatever colormap is in use.
    """
    from matplotlib import patheffects
    return [patheffects.withStroke(linewidth=width, foreground=t.axes_bg)]


def _style_legend(ax, t: Theme, **kwargs):
    return ax.legend(facecolor=t.legend_bg, edgecolor=t.legend_edge,
                     labelcolor=t.fg, framealpha=0.9, **kwargs)


def _style_colorbar(cb, label: str, t: Theme, fonts: dict) -> None:
    cb.set_label(label, color=t.fg_muted, fontsize=fonts["label"])
    cb.ax.yaxis.set_tick_params(color=t.fg_muted, labelsize=fonts["tick"])
    cb.outline.set_edgecolor(t.grid)
    for lbl in cb.ax.get_yticklabels():
        lbl.set_color(t.fg_muted)


@dataclass(frozen=True)
class PanelSpec:
    key: int
    view: int
    drift_sign: int      # 0 when both drift volumes share the panel
    title: str
    mask: np.ndarray


def _composition(hits, mask) -> str:
    """Describe which (view, drift side) combinations feed a panel."""
    seen = sorted({(int(v), int(s)) for v, s in
                   zip(hits.view[mask], hits.drift_sign[mask])})
    return ", ".join(f"{VIEW_NAMES.get(v, v)}{'+x' if s > 0 else '-x'}"
                     for v, s in seen)


def _panels(hits, geometry, *, min_hits: int = 1, merge: str = "orientation"
            ) -> list[PanelSpec]:
    """Group hits into panels.

    ``merge`` selects the grouping:

    ``"orientation"`` (default)
        One panel per *wire direction*.  Because the induction wires wrap around
        the APA frame, U on one face is parallel to V on the other, so this is
        the grouping under which both drift volumes share a single consistent
        ``w`` axis and a track stays continuous through the anode.  A panel
        therefore mixes U-channel and V-channel hits by design.

    ``"view"``
        One panel per U/V/Z label.  Compact, but for induction it joins two
        different measurement frames, so ``w`` is not continuous at the anode.

    ``"none"``
        One panel per (view, drift volume).  Nothing is merged; every axis is
        unambiguous.  This mirrors how Pandora treats each drift volume as an
        independent LArTPC.
    """
    out: list[PanelSpec] = []
    finite = np.isfinite(hits.w) & np.isfinite(hits.x)

    if merge == "none":
        for key in np.unique(hits.panel):
            mask = (hits.panel == key) & finite
            if np.count_nonzero(mask) < min_hits:
                continue
            view = int(hits.view[mask][0])
            sign = int(hits.drift_sign[mask][0])
            side = "+x drift" if sign > 0 else "-x drift"
            out.append(PanelSpec(int(key), view, sign,
                                 f"{geometry.view_name(view)} plane, {side}", mask))
        out.sort(key=lambda p: (p.view, -p.drift_sign))
        return out

    if merge == "view":
        for view in np.unique(hits.view[finite]):
            mask = (hits.view == view) & finite
            if np.count_nonzero(mask) < min_hits:
                continue
            name = geometry.view_name(int(view))
            both = len(np.unique(hits.drift_sign[mask])) > 1
            suffix = "" if both else (
                ", +x drift" if hits.drift_sign[mask][0] > 0 else ", -x drift")
            out.append(PanelSpec(int(view), int(view), 0, f"{name} plane{suffix}", mask))
        return out

    # orientation
    for key in np.unique(hits.orientation[finite]):
        mask = (hits.orientation == key) & finite
        if np.count_nonzero(mask) < min_hits:
            continue
        angle = key / 10.0
        kind = "Collection" if abs(angle) < 1.0 else "Induction"
        title = (f"{kind} {angle:+.1f}°  ({_composition(hits, mask)})"
                 if kind == "Induction"
                 else f"{kind}  ({_composition(hits, mask)})")
        out.append(PanelSpec(int(key), int(hits.view[mask][0]), 0, title, mask))
    out.sort(key=lambda p: p.key)
    return out


def _readout_panels(hits, geometry, *, min_hits: int = 1) -> list[PanelSpec]:
    """Panels for the classic readout view: one per plane, channel vs tick.

    This is the convention LArSoft's own event display uses.  It needs no
    geometry beyond mapping a channel to its plane, so unlike the physical
    views it works on hit collections whose WireIDs were never resolved.
    """
    out: list[PanelSpec] = []
    for view in np.unique(hits.chan_view):
        if view < 0:
            continue
        mask = hits.chan_view == view
        if np.count_nonzero(mask) < min_hits:
            continue
        name = geometry.view_name(int(view))
        out.append(PanelSpec(int(view), int(view), 0, f"{name} plane (readout)", mask))
    return out


def _drift_bounds(hits, geometry, panel: PanelSpec) -> tuple[float, float]:
    """Active-volume drift limits of the TPCs feeding a panel.

    Ticks convert to x by a plain linear map with no t0 correction, so
    out-of-time charge (radiologicals, other drift windows) lands beyond the
    cathode.  Those hits are real, but they are not inside the detector, and
    without a marker the picture gives no way to tell.
    """
    tpcs = {(int(c), int(t)) for c, t in
            zip(hits.cryo[panel.mask], hits.tpc[panel.mask])}
    lo, hi = np.inf, -np.inf
    for box in geometry.tpcs:
        if (box.cryo, box.tpc) in tpcs:
            lo, hi = min(lo, box.xmin), max(hi, box.xmax)
    return (lo, hi) if np.isfinite(lo) else (np.nan, np.nan)


def _anode_positions(hits, geometry, panel: PanelSpec) -> list[float]:
    """Drift coordinates of the readout planes feeding a panel.

    Marking these makes the seam between the two drift volumes explicit, so a
    merged induction panel cannot be misread as one continuous picture.
    """
    rows = geometry.plane_rows(hits.cryo[panel.mask], hits.tpc[panel.mask],
                               hits.plane[panel.mask])
    rows = rows[rows >= 0]
    if rows.size == 0:
        return []
    return sorted({round(float(v), 3) for v in geometry.p_x0[np.unique(rows)]})


def _charge_scale(values: np.ndarray, *, log: bool, pct: float = 99.5
                  ) -> tuple[float, float]:
    """Robust colour limits: a few saturated hits must not wash out the rest.

    Hit charge spans orders of magnitude -- a minimum-ionising track sits far
    above the radiological background -- so on a linear ramp nearly everything
    collapses into the bottom colour.  On the log scale the limits are taken
    over strictly positive values only.
    """
    v = values[np.isfinite(values)]
    if log:
        v = v[v > 0]
    if v.size == 0:
        return (1.0, 10.0) if log else (0.0, 1.0)
    lo = float(np.percentile(v, 0.5))
    hi = float(np.percentile(v, pct))
    if log:
        lo = max(lo, hi / 1e4 if hi > 0 else 1.0)   # cap the span at four decades
        if hi <= lo:
            hi = lo * 10.0
    else:
        lo = max(0.0, lo)
        if hi <= lo:
            hi = lo + 1.0
    return lo, hi


class _TruthInfo:
    """Shared access to the event's true interaction.

    All three displays want the same headline, and truth must never be able to
    break a display: an unreadable or absent MCTruth degrades to "no neutrino",
    not to an exception.
    """

    @property
    def neutrino(self):
        try:
            return self.event.neutrino()
        except Exception:
            return None

    @property
    def disambiguation_warning(self) -> str:
        """Set when the hit collection has not been disambiguated.

        In a wrapped-wire detector the physical views are only meaningful for a
        disambiguated collection; drawn from a raw hit finder they put charge on
        an arbitrary one of the candidate wires. Measured on the sample here,
        that is a median 3.9 cm from truth against 0.05 cm -- a picture that
        looks like broken reconstruction rather than the wrong input.
        """
        if self.space == "readout":
            return ""       # channel/tick folds the faces together by design
        if not self.geometry.wrapped_channels or len(self.hits) < 100:
            return ""
        if self.hits.channels_split_across_wires:
            return ""       # this event settles it on its own
        # Otherwise ask the whole collection: one event that happens to stay on
        # a single drift face proves nothing.
        try:
            tag = self.hits.product.split("_")[1]
            if self.event._src.is_disambiguated(tag) is not False:
                return ""
        except Exception:
            return ""
        return (f"{self.hits.product.rstrip('.')} looks NOT disambiguated "
                f"(no channel split across wires, in a detector with "
                f"{self.geometry.wrapped_channels} wrapped channels) -- "
                f"physical positions will be wrong; use a disambiguated "
                f"collection such as hitfd")

    @property
    def containment(self):
        """Where the true vertex sits relative to the active volume, or None."""
        nu = self.neutrino
        if nu is None:
            return None
        try:
            return self.geometry.containment(nu.vertex)
        except Exception:
            return None

    def _truth_block(self) -> str:
        nu = self.neutrino
        if nu is None:
            return ""
        lines = [self._truth_headline()] + nu.describe().splitlines()[1:]
        con = self.containment
        if con is not None:
            lines.append(f"  {con.description}")
        # Visible energy explains a sparse event far better than the hit count
        # does: an interaction at the wall can bring in 10 GeV and leave 300 MeV.
        try:
            vis = self.event.neutrino_visible_energy()
        except Exception:
            vis = None
        if vis is not None and nu.energy > 0:
            lines.append(f"  visible energy {vis:.0f} MeV of {nu.energy * 1000:.0f} "
                         f"MeV true ({100 * vis / (nu.energy * 1000):.1f}%)")
        # A filtered picture must say so: a shared PNG was indistinguishable
        # from an unfiltered one.
        if getattr(self, "radiologicals", True) is False:
            lines.append("  radiologicals hidden: showing only truth "
                         "descending from the interaction")
        return "\n".join(lines)

    def _truth_headline(self) -> str:
        nu = self.neutrino
        if nu is None:
            return ""
        con = self.containment
        flag = "" if con is None or con.inside else "   [vertex OUTSIDE active volume]"
        return nu.headline() + flag


class EventDisplay(_TruthInfo):
    """Renders one event.

    ``colour_by`` selects the hit quantity mapped to colour: ``"integral"``
    (charge, the default), ``"amplitude"``, ``"tick"`` or ``"multiplicity"``.

    ``colour_scale`` is ``"log"``, ``"linear"``, or ``"auto"`` (the default),
    which uses log for the charge-like quantities and linear for the rest.
    """

    MERGES = ("orientation", "view", "none")
    SPACES = ("physical", "readout")
    #: Continuous quantities, plus the categorical "which object owns this hit"
    #: modes that need associations.
    COLOUR_BY = ("integral", "amplitude", "tick", "multiplicity",
                 "track", "shower", "slice", "cluster", "pfparticle")
    CATEGORICAL_BY = ("track", "shower", "slice", "cluster", "pfparticle")
    SCALES = ("auto", "log", "linear")

    def __init__(self, event, hits=None, *, truth=None, tracks=None, vertices=None,
                 showers=None, mc=None, colour_by: str = "integral",
                 colour_scale: str = "auto", min_hits_per_panel: int = 1,
                 merge: str = "orientation", space: str = "physical",
                 radiologicals: bool = True,
                 theme: str = "dark", colormap: str = DEFAULT_COLORMAP,
                 preset: str = DEFAULT_PRESET):
        for name, value, allowed in (("merge", merge, self.MERGES),
                                     ("space", space, self.SPACES),
                                     ("colour_by", colour_by, self.COLOUR_BY),
                                     ("colour_scale", colour_scale, self.SCALES)):
            if value not in allowed:
                raise ValueError(
                    f"{name}={value!r} is not one of {', '.join(allowed)}")
        self.event = event
        self.radiologicals = radiologicals
        self.hits = hits if hits is not None else event.hits()
        self.truth = truth
        self.tracks = tracks
        self.vertices = vertices
        self.showers = showers
        self.mc = mc
        self.geometry = event.geometry
        self.colour_by = colour_by
        self.colour_scale = colour_scale
        self.merge = merge
        self.space = space
        self.theme = resolve_theme(theme)
        resolve_colormap(colormap)
        self.colormap = colormap
        if preset not in PRESETS:
            raise ValueError(
                f"preset={preset!r} is not one of {', '.join(PRESETS)}")
        self.preset = preset
        self.panels = (_readout_panels(self.hits, self.geometry,
                                       min_hits=min_hits_per_panel)
                       if space == "readout"
                       else _panels(self.hits, self.geometry,
                                    min_hits=min_hits_per_panel, merge=merge))

    @property
    def _axes(self) -> tuple[np.ndarray, np.ndarray, str, str]:
        """(horizontal, vertical, x label, y label) for the current space."""
        if self.space == "readout":
            return (self.hits.channel.astype(float), self.hits.tick,
                    "channel", "peak time [ticks]")
        return self.hits.w, self.hits.x, "wire coordinate w [cm]", "drift x [cm]"

    @property
    def is_categorical(self) -> bool:
        return self.colour_by in self.CATEGORICAL_BY

    @property
    def use_log(self) -> bool:
        if self.is_categorical:
            return False
        if self.colour_scale == "auto":
            return self.colour_by in ("integral", "amplitude")
        return self.colour_scale == "log"

    @cached_property
    def groups(self) -> np.ndarray:
        """Owning-object index per hit (-1 where unassociated)."""
        return self.event.hit_group(self.colour_by)

    def group_colours(self) -> np.ndarray:
        """One colour per hit, cycling the palette and greying the orphans."""
        t = self.theme
        g = self.groups
        cols = np.array([t.unassociated] * len(g), dtype=object)
        known = g >= 0
        cols[known] = [t.categorical[i % len(t.categorical)] for i in g[known]]
        return cols

    def group_marker_batches(self, plotly: bool = False):
        """Yield (mask, colour, marker) so identity survives without colour.

        Eight colours cycled against five markers give 40 distinguishable
        objects; colour alone repeats after eight, and disappears entirely in
        greyscale or for a colour-blind reader.
        """
        t = self.theme
        markers = PLOTLY_MARKER_CYCLE if plotly else MARKER_CYCLE
        g = self.groups
        n_colour = len(t.categorical)
        batches = []
        if (g < 0).any():
            batches.append((g < 0, t.unassociated, markers[0], None))
        # Style by rank among the objects present, not by the raw index: object
        # ids are sparse (cluster ids run into the hundreds), so keying on them
        # lets two objects 40 apart land on the same colour AND marker.
        for rank, obj in enumerate(np.unique(g[g >= 0])):
            # Colour and marker advance INDEPENDENTLY. 8 and 5 are coprime, so
            # the pair still cycles with period 40 and uses every combination --
            # but consecutive objects now differ in both, instead of sharing a
            # marker for the first eight (colour-only identity, which fails in
            # greyscale and under colour-vision deficiency) or sharing a colour
            # for the first five (the mirror of the same mistake).
            batches.append((g == obj,
                            t.categorical[rank % n_colour],
                            markers[rank % len(markers)],
                            int(obj)))
        return batches

    def _faces(self, panel: PanelSpec):
        """Yield (measurement direction, TPC boxes) for each face in a panel.

        A merged panel spans both APA faces, whose wires are mirror images and
        so have different measurement directions. Anything projected into the
        panel has to use the frame of the face it physically sits in.
        """
        g, h = self.geometry, self.hits
        for sign in np.unique(h.drift_sign[panel.mask]):
            sub = panel.mask & (h.drift_sign == sign)
            rows = g.plane_rows(h.cryo[sub], h.tpc[sub], h.plane[sub])
            rows = rows[rows >= 0]
            if rows.size == 0:
                continue
            p = g.plane_measure_dir[rows[0]]
            if not np.all(np.isfinite(p)):
                continue
            tpcs = {(int(c), int(t)) for c, t in zip(h.cryo[sub], h.tpc[sub])}
            boxes = [b for b in g.tpcs if (b.cryo, b.tpc) in tpcs]
            if boxes:
                yield p, boxes

    @staticmethod
    def _inside(boxes, x, y, z) -> np.ndarray:
        keep = np.zeros(np.shape(x), bool)
        for b in boxes:
            keep |= ((x >= b.xmin) & (x <= b.xmax) & (y >= b.ymin) & (y <= b.ymax) &
                     (z >= b.zmin) & (z <= b.zmax))
        return keep

    def project_points(self, panel: PanelSpec, x, y, z, *, clip: bool = True):
        """Project 3-D points into a panel's (w, x) plane, preserving order.

        Each point is projected with the measurement direction of the face it
        physically sits in; points in no face come back as NaN.  Order is
        preserved and gaps are left as NaN deliberately: a polyline drawn
        through these must break at a gap, not leap across it.  Concatenating
        per-face instead would reorder a track that crosses between faces and
        draw a metres-long line joining the two halves.
        """
        x = np.asarray(x, np.float64)
        y = np.asarray(y, np.float64)
        z = np.asarray(z, np.float64)
        w_out = np.full(x.shape, np.nan)
        x_out = np.full(x.shape, np.nan)
        for p, boxes in self._faces(panel):
            inside = self._inside(boxes, x, y, z) if clip else np.ones(x.shape, bool)
            take = inside & np.isnan(w_out)          # first face to claim a point wins
            if not take.any():
                continue
            w_out[take] = p[0] * y[take] + p[1] * z[take]
            x_out[take] = x[take]
        if not np.isfinite(w_out).any():
            return None
        return w_out, x_out

    def _truth_wx(self, panel: PanelSpec) -> tuple[np.ndarray, np.ndarray] | None:
        """Project true deposits into one panel's (w, x) plane.

        Only deposits lying inside the active volume of the TPCs that actually
        feed this panel are kept -- charge from the opposite drift volume is
        read out elsewhere and would just be a misleading smear here.
        """
        if self.truth is None or len(self.truth) == 0:
            return None
        return self.project_points(panel, self.truth.x, self.truth.y, self.truth.z)

    def _polylines(self, panel: PanelSpec, points) -> list[tuple[np.ndarray, np.ndarray]]:
        """Project a set of 3-D polylines, breaking them at gaps.

        Everything is concatenated into one array and projected in a single
        vectorised call, then split back apart by NaN. Projecting per polyline
        costs 40 s for one panel of a radiological sample's 34k true
        trajectories; this costs milliseconds.
        """
        usable = [xyz for xyz in points if len(xyz) >= 2]
        if not usable:
            return []
        sep = np.full((1, 3), np.nan)
        stacked = np.vstack([np.vstack([xyz, sep]) for xyz in usable])
        pr = self.project_points(panel, stacked[:, 0], stacked[:, 1], stacked[:, 2])
        if pr is None:
            return []
        w, x = pr
        # Break where a step is implausibly long; the NaN separators already
        # break between polylines, and NaN steps compare False here.
        with np.errstate(invalid="ignore"):
            step = np.linalg.norm(np.diff(stacked, axis=0), axis=1)
            gaps = np.flatnonzero(step > MAX_STEP_CM) + 1
        if len(gaps):
            w = np.insert(w, gaps, np.nan)
            x = np.insert(x, gaps, np.nan)
        if np.isfinite(w).sum() < 2:
            return []
        return [(w, x)]

    def _track_wx(self, panel: PanelSpec) -> list[tuple[np.ndarray, np.ndarray]]:
        """Each track's polyline projected into this panel."""
        if not self.tracks:
            return []
        return self._polylines(panel, self.tracks.points)

    def _mc_wx(self, panel: PanelSpec) -> list[tuple[np.ndarray, np.ndarray]]:
        """Each true trajectory projected into this panel."""
        if self.mc is None or not len(self.mc):
            return []
        return self._polylines(panel, self.mc.points)

    def _shower_wx(self, panel: PanelSpec, *, n_azimuth: int = 24
                   ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Shower cones as a projected silhouette.

        The cone's base is a disc perpendicular to the shower axis. Its
        appearance in a given panel depends on where that disc points relative
        to the panel's measurement direction, so the whole rim is sampled and
        projected. Taking a single arbitrary perpendicular instead makes the
        drawn width an artefact of that choice -- and for collection wires,
        where the measurement direction is z, a perpendicular built from the z
        axis projects to zero width, drawing a cone with no opening at all.
        """
        if self.showers is None or not len(self.showers):
            return []
        out = []
        for start, direction, length, angle in zip(
                self.showers.start, self.showers.direction,
                self.showers.length, self.showers.open_angle):
            if not (np.isfinite(length) and np.isfinite(angle)) or length <= 0:
                continue
            norm = np.linalg.norm(direction)
            if not np.isfinite(norm) or norm <= 0:
                continue
            d = direction / norm
            # any two unit vectors spanning the plane perpendicular to the axis
            helper = np.array([0.0, 0.0, 1.0])
            if abs(np.dot(helper, d)) > 0.9:
                helper = np.array([1.0, 0.0, 0.0])
            u1 = np.cross(d, helper)
            u1 /= np.linalg.norm(u1)
            u2 = np.cross(d, u1)

            r = length * np.tan(np.clip(angle, 0.0, 1.4))
            tip = start + d * length
            phi = np.linspace(0.0, 2.0 * np.pi, n_azimuth, endpoint=False)
            rim = tip + r * (np.cos(phi)[:, None] * u1 + np.sin(phi)[:, None] * u2)
            outline = np.vstack([start, rim, start])
            pr = self.project_points(panel, outline[:, 0], outline[:, 1],
                                     outline[:, 2], clip=False)
            if pr is None:
                continue
            w, x = pr
            good = np.isfinite(w) & np.isfinite(x)
            if good.sum() > 2:
                out.append((w[good], x[good]))
        return out

    def _vertex_wx(self, panel: PanelSpec):
        if self.vertices is None or len(self.vertices) == 0:
            return None
        v = self.vertices.xyz
        return self.project_points(panel, v[:, 0], v[:, 1], v[:, 2])

    def _true_vertex_wx(self, panel: PanelSpec):
        """The true interaction vertex projected into this panel, or None.

        Projected with ``clip=False`` on purpose: a vertex just outside the
        active volume is precisely the case worth looking at, and clipping to
        the drift faces would silently drop it.
        """
        nu = self.neutrino
        if nu is None or self.space == "readout":
            return None            # readout axes are channel/tick, not position
        v = np.asarray(nu.vertex, float)
        if not np.isfinite(v).all():
            return None
        pr = self.project_points(panel, v[0:1], v[1:2], v[2:3], clip=False)
        if pr is None:
            return None
        w, x = pr
        if not (np.isfinite(w[0]) and np.isfinite(x[0])):
            return None
        return float(w[0]), float(x[0])

    def __repr__(self) -> str:
        return f"<EventDisplay {self.event!r} panels={len(self.panels)} hits={len(self.hits)}>"

    # ---- shared helpers ------------------------------------------------

    def _colour_values(self) -> tuple[np.ndarray, str]:
        h = self.hits
        if self.is_categorical:
            return self.groups.astype(float), f"{self.colour_by} index"
        return {
            "integral": (h.integral, "hit integral [ADC]"),
            "amplitude": (h.amplitude, "peak amplitude [ADC]"),
            "tick": (h.tick, "peak time [ticks]"),
            "multiplicity": (h.multiplicity.astype(float), "multiplicity"),
        }[self.colour_by]

    def _title(self) -> str:
        run, subrun, ev = self.event.id
        return (f"{self.geometry.detector}   run {run} / subrun {subrun} / event {ev}"
                f"   -   {len(self.hits)} hits   "
                f"({self.hits.label.split('[')[0].strip()})")

    @property
    def neutrino(self):
        """The true interaction, or None. Cheap: the event memoises it."""
        try:
            return self.event.neutrino()
        except Exception:
            return None          # truth is a bonus; never break a display for it

    def _plotly_title(self) -> str:
        """Title plus the interaction, as HTML for plotly's title slot."""
        nu = self.neutrino
        if nu is None:
            return self._title()
        head, *rest = self._truth_block().splitlines()
        detail = "<br>".join(line.strip() for line in rest)
        return (f"{self._title()}"
                f"<br><span style='font-size:12px'>{head}</span>"
                f"<br><span style='font-size:10px;color:{self.theme.fg_muted}'>"
                f"{detail}</span>")

    def summary(self) -> str:
        """One-line-per-panel text summary, handy without a display."""
        lines = [self._title()]
        if self.neutrino is not None:
            lines.extend(self._truth_block().splitlines())
        bad = self.hits.n_bad_geometry
        if bad:
            lines.append(f"  ! {bad} hits had a WireID absent from the geometry")
        warn = self.disambiguation_warning
        if warn:
            lines.append(f"  ! {warn}")
        if self.tracks is not None or self.showers is not None:
            note = self.event.pandora_selection_note()
            if note:
                lines.append(f"  {note}")
        hx, hy, xlabel, ylabel = self._axes
        unit = "" if self.space == "readout" else " cm"
        xs, ys = xlabel.split(" [")[0], ylabel.split(" [")[0]
        for p in self.panels:
            n = int(np.count_nonzero(p.mask))
            a, b = hx[p.mask], hy[p.mask]
            lines.append(f"  {p.title:38s} {n:6d} hits   "
                         f"{xs} [{a.min():8.1f},{a.max():8.1f}]{unit}   "
                         f"{ys} [{b.min():7.1f},{b.max():7.1f}]{unit}")
        return "\n".join(lines)

    # ---- static (matplotlib) -------------------------------------------

    def figure(self, *, ncols: int | None = None, size: float | None = None,
               marker_size: float = 4.0, share_w: bool = True):
        """Build a matplotlib figure."""
        plt = _pyplot()
        t, fonts = self.theme, _fonts(self.preset)
        spec = PRESETS[self.preset]

        if not self.panels:
            raise ValueError("no hits with usable geometry to display")

        n = len(self.panels)
        # Panels are wide and short, so stacking a few in one column both fills
        # the canvas (a 2-column grid leaves a quarter of it empty at n=3) and
        # lets a track be followed across projections down a shared w axis.
        ncols = ncols or (1 if n <= 3 else 2)
        nrows = int(np.ceil(n / ncols))
        size = spec["panel_height"] if size is None else size
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(spec["panel_width"] * ncols, size * nrows),
                                 squeeze=False, constrained_layout=True)
        fig.patch.set_facecolor(t.fig_bg)
        cvals, clabel = self._colour_values()
        log = self.use_log
        vmin, vmax = _charge_scale(cvals, log=log)
        cmap = charge_cmap(t, self.colormap)
        if log:
            from matplotlib.colors import LogNorm
            norm = LogNorm(vmin=vmin, vmax=vmax, clip=True)
            # LogNorm cannot represent non-positive values; clamp them to the
            # bottom of the scale rather than dropping those hits entirely.
            cvals = np.where(cvals > 0, cvals, vmin)
        else:
            from matplotlib.colors import Normalize
            norm = Normalize(vmin=vmin, vmax=vmax)

        hx, hy, xlabel, ylabel = self._axes
        readout = self.space == "readout"
        wall = hx[np.isfinite(hx)]
        # Shared only across panels that actually overlap in w: pooling every
        # panel's hits gave a sparse panel a range 10x its own content, so 90%
        # of it was empty. Panels whose own span is a small fraction of the
        # pooled one get framed on their own data instead.
        wlim = (wall.min(), wall.max()) if share_w and wall.size and not readout else None

        # Everything measured in points has to follow the canvas, or a preset
        # authored at 3.4 in draws the same dots as one authored at 9 in.
        mscale = float(np.clip(spec["panel_width"] / 7.0, 0.45, 1.6))
        marker_size = marker_size * mscale ** 1.5
        sc = None
        _legend_rows = 0
        drew_truth = False
        for i, (ax, p) in enumerate(zip(axes.flat, self.panels)):
            m = p.mask
            tw = None if readout else self._truth_wx(p)
            if tw is not None:
                # A faint underlay: context for the hits, not the subject.
                # Marker size scales with the canvas -- points are absolute, so
                # the same 0.5 pt dot that reads as a stipple at screen width
                # merges into a solid grey block on a journal column.
                ax.plot(tw[0], tw[1], ".", color=t.truth,
                        markersize=0.5 * mscale, alpha=0.35 * mscale,
                        label="true deposits", rasterized=True, zorder=1)
                drew_truth = True
            if self.is_categorical:
                sc = None
                for sel, colour, marker, _obj in self.group_marker_batches():
                    sel = sel & m
                    if sel.any():
                        ax.scatter(hx[sel], hy[sel], c=colour, marker=marker,
                                   s=marker_size, linewidths=0, alpha=_ALPHA,
                                   rasterized=True, zorder=3)
            else:
                # Draw faint hits first so a bright one is never hidden behind
                # a dim one: with ~85% of markers overlapping, draw order would
                # otherwise decide what you see.
                order = np.argsort(cvals[m])
                sc = ax.scatter(hx[m][order], hy[m][order], c=cvals[m][order],
                                s=marker_size, cmap=cmap, norm=norm,
                                linewidths=0, alpha=_ALPHA, rasterized=True,
                                zorder=3)
            if not readout:
                mc_w, mc_x = _join(self._mc_wx(p))
                if len(mc_w):
                    # One artist for every trajectory, joined by NaN. One
                    # ax.plot per particle is 34k artists on a radiological
                    # sample and takes minutes.
                    ax.plot(mc_w, mc_x, "-", color=t.mctrack, linewidth=0.7,
                            alpha=0.7, zorder=3.5, label="true trajectories",
                            dashes=(4, 2), rasterized=True)
                for j, (sw_, sx_) in enumerate(self._shower_wx(p)):
                    ax.fill(sw_, sx_, color=t.shower, alpha=0.22, zorder=2,
                            label="shower cones" if j == 0 else None)
                    ax.plot(sw_, sx_, "-", color=t.shower, linewidth=1.2,
                            path_effects=_casing(t, 2.6),
                            alpha=0.8, zorder=3.6)
                trk_w, trk_x = _join(self._track_wx(p))
                if len(trk_w):
                    ax.plot(trk_w, trk_x, "-", color=t.track, linewidth=0.9,
                            alpha=0.85, zorder=4, label="reco tracks",
                            rasterized=True)
                vx = self._vertex_wx(p)
                if vx is not None:
                    ax.plot(vx[0], vx[1], "*", color=t.vertex, markersize=7,
                            markeredgecolor=t.axes_bg, markeredgewidth=0.5,
                            zorder=5, label="vertices", linestyle="none")
            tv = self._true_vertex_wx(p)
            if tv is not None:
                # Distinct marker as well as colour: on a busy panel the true
                # and reconstructed vertices otherwise merge into one blob.
                ax.plot([tv[0]], [tv[1]], "X", color=t.mctrack, markersize=7,
                        markeredgecolor=t.axes_bg, markeredgewidth=0.8,
                        zorder=7, label="true vertex", linestyle="none")
            if not readout:
                # Frame the DATA, then draw the reference lines without letting
                # them reopen the range. axhline participates in autoscale, so
                # an anode at x = 0 and an edge at x = -746 forced every panel
                # to the full drift span -- on a sparse event that left the
                # hits in 26% of the panel height with the rest empty.
                ylo, yhi = _data_ylim(hy[p.mask])
                if ylo is not None:
                    ax.set_ylim(ylo, yhi)
                for k, xa in enumerate(_anode_positions(self.hits, self.geometry, p)):
                    ax.axhline(xa, color=t.fg_muted, linewidth=0.8, alpha=0.8,
                               linestyle="--", zorder=2,
                               label="anode" if (k == 0 and i == 0) else None)
                for k, xc in enumerate(_drift_bounds(self.hits, self.geometry, p)):
                    if np.isfinite(xc):
                        ax.axhline(xc, color=t.fg_muted, linewidth=0.8,
                                   alpha=0.6, linestyle=":", zorder=2,
                                   label=("active volume edge"
                                          if (k == 0 and i == 0) else None))
                if ylo is not None:
                    ax.set_ylim(ylo, yhi)      # axhline reopened it; pin again
            ax.set_title(p.title, fontsize=9)
            ax.set_xlabel(xlabel, fontsize=8)
            ax.set_ylabel(ylabel, fontsize=8)
            _style_axes(ax, t, fonts)
            if wlim:
                own = hx[p.mask]
                own = own[np.isfinite(own)]
                pooled = wlim[1] - wlim[0]
                if len(own) and pooled > 0 and np.ptp(own) < 0.35 * pooled:
                    lo, hi = _data_ylim(own)          # same padding rule as y
                    ax.set_xlim(lo, hi)
                else:
                    ax.set_xlim(*wlim)
        for ax in axes.flat[n:]:
            ax.set_visible(False)
        if (drew_truth or self.tracks or self.vertices is not None
                or self.mc is not None or self.showers is not None
                or self.neutrino is not None):
            # one figure-level entry: a per-panel legend would appear only on
            # the panels that happen to contain truth, which reads as an error
            handles, labels = axes.flat[0].get_legend_handles_labels()
            if handles:
                # markerscale has to stay modest: the legend now mixes lines,
                # a filled patch and a star marker, and a large scale blows the
                # star up over the panels.
                # Below the title, not beside it: with truth and reco both on
                # the legend is wide enough to sit on top of the title text.
                # Wrap the legend to the figure's actual width. Forcing one
                # row spilled it past both edges, and bbox_inches="tight" then
                # grew the canvas to fit -- paper1 saved 70% wider than the
                # column it was authored for, so every font landed ~55% too
                # small once a journal scaled it back down.
                per_entry = 1.6 * (fonts["legend"] / _BASE_FONTS["legend"])
                fig_w = spec["panel_width"] * ncols
                max_cols = max(1, int(fig_w / max(per_entry, 0.1)))
                n_cols = max(1, min(len(handles), max_cols))
                _legend_rows = int(np.ceil(len(handles) / n_cols))
                # Inside the first panel, not a figure-level legend: an
                # "outside" location competes with the suptitle for the top
                # band and the truth block for the bottom one, and every
                # arrangement collided on some preset. Axes legends are laid
                # out by matplotlib with no figure-level interaction at all.
                leg = axes.flat[0].legend(
                                 handles, labels, loc="upper left",
                                 ncols=n_cols,
                                 fontsize=fonts["legend"], markerscale=1.6,
                                 framealpha=0.9, facecolor=t.legend_bg,
                                 edgecolor=t.legend_edge, labelcolor=t.fg,
                                 handlelength=1.8, columnspacing=1.2,
                                 borderpad=0.4, labelspacing=0.3)
                leg.set_zorder(10)
        if sc is not None:
            cb = fig.colorbar(sc, ax=axes.ravel().tolist(), pad=0.01, fraction=0.03)
            _style_colorbar(cb, clabel, t, fonts)
        elif self.is_categorical:
            n_obj = int(self.groups.max()) + 1 if (self.groups >= 0).any() else 0
            n_orphan = int((self.groups < 0).sum())
            # Appended to the truth block below rather than drawn at the
            # opposite corner: anchored independently, the two texts printed
            # over each other whenever both were present.
            self._extra_note = (f"coloured by {self.colour_by}: {n_obj} objects "
                                f"(colour x marker), {n_orphan} unassociated hits")
        fig_w = spec["panel_width"] * ncols
        truth = self._truth_block()
        note = getattr(self, "_extra_note", "")
        head = truth.splitlines()[0] if truth else ""
        rest = truth.splitlines()[1:] if truth else []
        if note:
            rest = rest + ["  " + note]

        # Two engine-managed slots only: the suptitle at the top and a
        # supxlabel underneath the panels. constrained_layout reserves room for
        # both, so neither can land on the data. Every scheme that positioned a
        # block itself eventually overlapped something on some preset.
        fig.suptitle(_wrap_for(self._title() + (f"\n{head}" if head else ""),
                               fig_w, fonts["suptitle"]),
                     fontsize=fonts["suptitle"], color=t.fg)
        if rest:
            fig.supxlabel(_wrap_for("\n".join(rest), fig_w, fonts["note"],
                                    char_factor=0.62),
                          color=t.fg_muted, fontsize=fonts["note"],
                          ha="left", x=0.01, family="monospace",
                          linespacing=1.4)
        return fig

    def save(self, path: str, *, dpi: int | None = None, **kwargs) -> str:
        """Save a static image (png, pdf, svg - anything matplotlib supports)."""
        _vector_text()
        fig = self.figure(**kwargs)
        dpi = PRESETS[self.preset]["dpi"] if dpi is None else dpi
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        # without facecolor= the saved file gets a white border round the panels
        fig.savefig(path, dpi=dpi, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        import matplotlib.pyplot as plt
        plt.close(fig)
        return path

    # ---- interactive (plotly) ------------------------------------------

    def plotly_figure(self, *, ncols: int | None = None, marker_size: float = 3.0,
                      height_per_row: int = 300):
        """Build an interactive plotly figure with hover details per hit."""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        t = self.theme

        if not self.panels:
            raise ValueError("no hits with usable geometry to display")

        n = len(self.panels)
        ncols = ncols or (1 if n <= 3 else 2)
        nrows = int(np.ceil(n / ncols))
        fig = make_subplots(rows=nrows, cols=ncols,
                            subplot_titles=[p.title for p in self.panels],
                            horizontal_spacing=0.07,
                            vertical_spacing=0.12 / max(nrows, 1))

        cvals, clabel = self._colour_values()
        log = self.use_log
        vmin, vmax = _charge_scale(cvals, log=log)
        cmap = charge_cmap(t, self.colormap)
        raw = cvals
        cbar = dict(title=clabel, thickness=12, len=0.9)
        if log:
            # plotly has no logarithmic colour axis for markers, so colour by
            # log10 and relabel the ticks with the original values.
            cvals = np.log10(np.where(raw > 0, raw, vmin))
            vmin, vmax = np.log10(vmin), np.log10(vmax)
            decades = np.arange(np.ceil(vmin), np.floor(vmax) + 1)
            if decades.size >= 2:
                cbar.update(tickvals=decades.tolist(),
                            ticktext=[f"10<sup>{int(t)}</sup>" for t in decades])
            else:
                # Less than two full decades: whole-decade labels would leave
                # the bar with one tick or none, so label actual values instead.
                vals = np.linspace(vmin, vmax, 5)
                cbar.update(tickvals=vals.tolist(),
                            ticktext=[_si(10 ** v) for v in vals])
        h = self.hits
        hx, hy, xlabel, ylabel = self._axes
        readout = self.space == "readout"

        for i, p in enumerate(self.panels):
            r, c = divmod(i, ncols)
            m = p.mask
            tw = None if readout else self._truth_wx(p)
            if tw is not None:
                fig.add_trace(
                    go.Scattergl(x=tw[0], y=tw[1], mode="markers",
                                 marker=dict(size=1.5, color=t.truth, opacity=0.30),
                                 name="true deposits", showlegend=(i == 0),
                                 legendgroup="truth", hoverinfo="skip"),
                    row=r + 1, col=c + 1)
            if not readout:
                mc_w, mc_x = _join(self._mc_wx(p))
                if len(mc_w):
                    fig.add_trace(go.Scattergl(
                        x=mc_w, y=mc_x, mode="lines",
                        line=dict(color=t.mctrack, width=0.9, dash="dash"),
                        opacity=0.75, name="true trajectories", legendgroup="mc",
                        showlegend=(i == 0), hoverinfo="skip"),
                        row=r + 1, col=c + 1)
                for j, (sw_, sx_) in enumerate(self._shower_wx(p)):
                    fig.add_trace(go.Scattergl(
                        x=sw_, y=sx_, mode="lines", fill="toself",
                        line=dict(color=t.shower, width=0.8), opacity=0.35,
                        name="shower cones", legendgroup="shw",
                        showlegend=(i == 0 and j == 0), hoverinfo="skip"),
                        row=r + 1, col=c + 1)
                trk_w, trk_x = _join(self._track_wx(p))
                if len(trk_w):
                    fig.add_trace(go.Scattergl(
                        x=trk_w, y=trk_x, mode="lines",
                        line=dict(color=t.track, width=1.2),
                        name="reco tracks", legendgroup="trk",
                        showlegend=(i == 0), hoverinfo="skip"),
                        row=r + 1, col=c + 1)
                vx = self._vertex_wx(p)
                if vx is not None:
                    fig.add_trace(go.Scattergl(
                        x=vx[0], y=vx[1], mode="markers",
                        marker=dict(symbol="star", size=11, color=t.vertex,
                                    line=dict(color=t.axes_bg, width=0.6)),
                        name="vertices", legendgroup="vtx",
                        showlegend=(i == 0), hoverinfo="skip"),
                        row=r + 1, col=c + 1)
            tv = self._true_vertex_wx(p)
            if tv is not None:
                nu = self.neutrino
                fig.add_trace(go.Scattergl(
                    x=[tv[0]], y=[tv[1]], mode="markers",
                    marker=dict(symbol="x", size=13, color=t.mctrack,
                                line=dict(color=t.axes_bg, width=1)),
                    name="true vertex", legendgroup="tvtx",
                    showlegend=(i == 0),
                    hovertemplate=(f"true vertex<br>{nu.headline()}"
                                   f"<br>({nu.vertex[0]:.1f}, {nu.vertex[1]:.1f}, "
                                   f"{nu.vertex[2]:.1f}) cm<extra></extra>")),
                    row=r + 1, col=c + 1)
            custom = np.column_stack([
                h.channel[m], h.tick[m], h.integral[m], h.amplitude[m],
                h.tpc[m], h.wire[m], h.multiplicity[m],
            ])
            if self.is_categorical:
                # One trace per colour x marker batch, as the static backend
                # does. Colouring a single trace instead left identity resting
                # on hue alone -- in the browser, which is where events are
                # actually scanned -- and that is exactly what pairing colour
                # with a marker exists to prevent.
                for sel, colour, marker, obj in self.group_marker_batches(plotly=True):
                    sel = sel & m
                    if not sel.any():
                        continue
                    label = "unassociated" if obj is None else f"{self.colour_by} {obj}"
                    fig.add_trace(go.Scattergl(
                        x=hx[sel], y=hy[sel], mode="markers",
                        marker=dict(size=marker_size, color=colour,
                                    symbol=marker, opacity=_ALPHA),
                        customdata=np.column_stack([
                            h.channel[sel], h.tick[sel], h.integral[sel],
                            h.amplitude[sel], h.tpc[sel], h.wire[sel],
                            h.multiplicity[sel]]),
                        hovertemplate=_hover_template(readout, xlabel, ylabel),
                        name=label, legendgroup=label,
                        showlegend=(i == 0)),
                        row=r + 1, col=c + 1)
            else:
                fig.add_trace(
                    go.Scattergl(
                        x=hx[m], y=hy[m], mode="markers",
                        marker=dict(size=marker_size, color=cvals[m],
                                    colorscale=charge_colorscale(t, self.colormap),
                                    cmin=vmin, cmax=vmax, opacity=_ALPHA,
                                    showscale=(i == 0), colorbar=cbar),
                        customdata=custom,
                        hovertemplate=_hover_template(readout, xlabel, ylabel),
                        name=p.title, showlegend=False),
                    row=r + 1, col=c + 1)
            if not readout:
                for xa in _anode_positions(h, self.geometry, p):
                    fig.add_hline(y=xa,
                                  line=dict(color=t.fg_muted, width=0.8, dash="dash"),
                                  opacity=0.45, row=r + 1, col=c + 1)
                for xc in _drift_bounds(h, self.geometry, p):
                    if np.isfinite(xc):
                        fig.add_hline(y=xc,
                                      line=dict(color=t.fg_muted, width=0.8,
                                                dash="dot"),
                                      opacity=0.55, row=r + 1, col=c + 1)
            fig.update_xaxes(title_text=xlabel, row=r + 1, col=c + 1,
                             title_font_size=10, tickfont_size=9)
            fig.update_yaxes(title_text=ylabel, row=r + 1, col=c + 1,
                             title_font_size=10, tickfont_size=9)

        for slot in range(n + 1, nrows * ncols + 1):
            # subplot slots with no panel: hide them rather than leave an empty
            # framed box (the matplotlib path already does this)
            suffix = "" if slot == 1 else str(slot)
            fig.layout[f"xaxis{suffix}"].visible = False
            fig.layout[f"yaxis{suffix}"].visible = False

        fig.update_layout(
            title=dict(text=self._plotly_title(), font=dict(size=13, color=t.fg)),
            height=height_per_row * nrows + 90,
            template=("plotly_dark" if t.is_dark else "plotly_white"),
            paper_bgcolor=t.fig_bg, plot_bgcolor=t.axes_bg,
            font=dict(color=t.fg_muted),
            legend=dict(bgcolor=t.legend_bg, bordercolor=t.legend_edge, borderwidth=1,
                        font=dict(color=t.fg)),
            # The truth block adds four lines to the title; without more room
            # for it the top panel gets overdrawn.
            margin=dict(l=60, r=20, t=80 if self.neutrino is None else 150, b=50),
            # Keep the user's zoom/pan when they step to the next event: the
            # panel layout is what the view depends on, not which event is in
            # it. Without this, scanning a sample re-zooms on every keypress.
            # File identity belongs in the key: a camera set on one file must
            # not persist onto another with a different extent. Event id stays
            # OUT, so stepping events still keeps the view.
            uirevision=f"{self.geometry.detector}|{self.space}|"
                       f"{self.merge}|{len(self.panels)}",
            dragmode="pan",
        )
        fig.update_xaxes(gridcolor=t.grid, zerolinecolor=t.grid)
        fig.update_yaxes(gridcolor=t.grid, zerolinecolor=t.grid)
        for a in fig.layout.annotations:
            a.font.size = 11
            a.font.color = t.fg
        return fig

    def save_html(self, path: str, *, bundle_plotlyjs: bool = True, **kwargs) -> str:
        """Write an interactive HTML page.

        ``bundle_plotlyjs=True`` (the default) inlines plotly.js so the page
        works offline -- larger, but it renders anywhere. ``False`` loads it
        from a CDN instead: much smaller, but blank without internet access.
        """
        fig = self.plotly_figure(**kwargs)
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        fig.write_html(path, include_plotlyjs=("inline" if bundle_plotlyjs else "cdn"),
                       config=_PLOTLY_CONFIG)
        return path

    def show(self, **kwargs) -> None:
        """Open the interactive figure (browser, or inline in a notebook)."""
        self.plotly_figure(**kwargs).show(config=_PLOTLY_CONFIG)


def _tpc_wireframe(box) -> tuple[list, list, list]:
    """Edges of one TPC active volume as a single polyline.

    Segments are separated by NaN, which both plotly and matplotlib treat as a
    break, so all twelve edges draw as one cheap trace.
    """
    xs, ys, zs = [], [], []
    c = [(box.xmin, box.ymin, box.zmin), (box.xmax, box.ymin, box.zmin),
         (box.xmax, box.ymax, box.zmin), (box.xmin, box.ymax, box.zmin),
         (box.xmin, box.ymin, box.zmax), (box.xmax, box.ymin, box.zmax),
         (box.xmax, box.ymax, box.zmax), (box.xmin, box.ymax, box.zmax)]
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    for a, b in edges:
        xs += [c[a][0], c[b][0], np.nan]
        ys += [c[a][1], c[b][1], np.nan]
        zs += [c[a][2], c[b][2], np.nan]
    return xs, ys, zs


class Display3D(_TruthInfo):
    """Interactive 3-D view of reconstructed space points.

    Hits are inherently two-dimensional (one projection each), so the 3-D
    picture is built from ``recob::SpacePoint``, which is what reconstruction
    produces by matching hits across planes.
    """

    #: What a space point can be coloured by. Drift-x is deliberately not the
    #: default: it is the depth axis, so colouring by it duplicates position.
    COLOUR_BY = ("x", "y", "z", "charge")

    FOCUS = ("data", "detector")

    def __init__(self, event, spacepoints=None, *, truth=None, tracks=None,
                 vertices=None, showers=None, mc=None,
                 colour_by: str = "y", draw_tpcs: bool = True,
                 focus: str = "data", theme: str = "dark",
                 radiologicals: bool = True,
                 colormap: str = DEFAULT_COLORMAP, preset: str = DEFAULT_PRESET):
        self.radiologicals = radiologicals
        if focus not in self.FOCUS:
            raise ValueError(f"focus={focus!r} is not one of {', '.join(self.FOCUS)}")
        if colour_by not in self.COLOUR_BY:
            raise ValueError(
                f"colour_by={colour_by!r} is not one of {', '.join(self.COLOUR_BY)}. "
                "Space points carry no charge, so the 2-D hit quantities "
                "(integral, amplitude, ...) are not available here.")
        self.event = event
        self.sp = spacepoints if spacepoints is not None else event.spacepoints()
        self.truth = truth
        self.tracks = tracks
        self.colour_by = colour_by
        self.vertices = vertices
        self.showers = showers
        self.mc = mc
        self.geometry = event.geometry
        self.draw_tpcs = draw_tpcs
        self.focus = focus
        self.theme = resolve_theme(theme)
        resolve_colormap(colormap)
        self.colormap = colormap
        self.preset = preset

    def _limits(self, pad: float = 0.08):
        """(x, y, z) ranges to show, in detector coordinates.

        ``focus="detector"`` frames the whole cryostat, which is honest but
        renders a localised interaction as a handful of pixels. ``"data"``
        frames what is actually in the event; the TPC wireframe still draws and
        simply extends past the edges.
        """
        boxes = self.geometry.tpcs
        if self.focus == "detector" and boxes:
            return ((min(b.xmin for b in boxes), max(b.xmax for b in boxes)),
                    (min(b.ymin for b in boxes), max(b.ymax for b in boxes)),
                    (min(b.zmin for b in boxes), max(b.zmax for b in boxes)))
        pts = [self.sp.xyz] if len(self.sp) else []
        if self.tracks is not None:
            pts += [p for p in self.tracks.points if len(p)]
        if not pts:
            return None
        allp = np.vstack(pts)
        out = []
        for axis in range(3):
            lo, hi = float(allp[:, axis].min()), float(allp[:, axis].max())
            margin = max((hi - lo) * pad, 5.0)
            out.append((lo - margin, hi + margin))
        return tuple(out)

    def _colour_values(self) -> tuple[np.ndarray, str]:
        if self.colour_by == "charge":
            return self.sp.charge, "summed hit charge [ADC]"
        idx = {"x": 0, "y": 1, "z": 2}[self.colour_by]
        return self.sp.xyz[:, idx], f"{self.colour_by} [cm]"

    def summary(self) -> str:
        """One-line-per-item text summary, matching the other display classes."""
        run, subrun, ev = self.event.id
        lines = [f"{self.geometry.detector}   run {run} / subrun {subrun} / "
                 f"event {ev}   -   {len(self.sp)} space points "
                 f"({self.sp.label})"]
        if self.neutrino is not None:
            lines.extend(self._truth_block().splitlines())
        if self.tracks is not None and len(self.tracks):
            pts = sum(len(p) for p in self.tracks.points)
            lines.append(f"  {len(self.tracks)} tracks ({pts} trajectory points)")
        if self.vertices is not None and len(self.vertices):
            lines.append(f"  {len(self.vertices)} vertices")
        if self.showers is not None and len(self.showers):
            lines.append(f"  {len(self.showers)} showers")
        if self.tracks is not None or self.showers is not None:
            note = self.event.pandora_selection_note()
            if note:
                lines.append(f"  {note}")
        if self.mc is not None and len(self.mc):
            lines.append(f"  {len(self.mc)} true trajectories")
        lines.append(f"  coloured by {self.colour_by}; framing: {self.focus}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"<Display3D {self.event!r} spacepoints={len(self.sp)}>"

    def plotly_figure(self, *, marker_size: float = 1.6, height: int = 800):
        import plotly.graph_objects as go
        t = self.theme

        fig = go.Figure()
        if self.draw_tpcs:
            xs, ys, zs = [], [], []
            for box in self.geometry.tpcs:
                bx, by, bz = _tpc_wireframe(box)
                xs += bx
                ys += by
                zs += bz
            fig.add_trace(go.Scatter3d(
                x=zs, y=xs, z=ys, mode="lines",
                line=dict(color=t.grid, width=1),
                name="TPC volumes", hoverinfo="skip"))

        if self.truth is not None and len(self.truth):
            fig.add_trace(go.Scatter3d(
                x=self.truth.z, y=self.truth.x, z=self.truth.y, mode="markers",
                marker=dict(size=1.0, color=t.fg_muted, opacity=0.25),
                name="true deposits", hoverinfo="skip"))

        # Batched into one trace each, and drawn in BOTH backends: the
        # interactive view used to omit vertices, showers and true
        # trajectories entirely, which made the browser -- the plotly path --
        # the one missing data.
        if self.mc is not None:
            q = _join3d(self.mc.points)
            if len(q):
                fig.add_trace(go.Scatter3d(
                    x=q[:, 2], y=q[:, 0], z=q[:, 1], mode="lines",
                    line=dict(color=t.mctrack, width=1, dash="dash"),
                    opacity=0.6, name="true trajectories", hoverinfo="skip"))

        if self.tracks is not None:
            poly = _join3d(self.tracks.points)
            if len(poly):
                fig.add_trace(go.Scatter3d(
                    x=poly[:, 2], y=poly[:, 0], z=poly[:, 1], mode="lines",
                    line=dict(color=t.track, width=3), name="reco tracks",
                    hoverinfo="skip"))

        if self.showers is not None and len(self.showers):
            # NOT _join3d: a shower axis is one straight segment whose length
            # exceeds MAX_STEP_CM by construction, so the gap-break inserted a
            # NaN between its own two endpoints and the line drew nothing.
            axes = _join_segments([np.vstack([st, st + di * ln]) for st, di, ln in
                                   zip(self.showers.start, self.showers.direction,
                                       self.showers.length)])
            if len(axes):
                fig.add_trace(go.Scatter3d(
                    x=axes[:, 2], y=axes[:, 0], z=axes[:, 1], mode="lines",
                    line=dict(color=t.shower, width=4), opacity=0.85,
                    name="shower axes", hoverinfo="skip"))

        if self.vertices is not None and len(self.vertices):
            v = self.vertices.xyz
            fig.add_trace(go.Scatter3d(
                x=v[:, 2], y=v[:, 0], z=v[:, 1], mode="markers",
                marker=dict(symbol="diamond", size=5, color=t.vertex),
                name="vertices",
                hovertemplate="reco vertex<br>z %{x:.1f}<br>x %{y:.1f}"
                              "<br>y %{z:.1f} cm<extra></extra>"))

        xyz = self.sp.xyz
        cvals, clabel = self._colour_values()
        fig.add_trace(go.Scatter3d(
            x=xyz[:, 2], y=xyz[:, 0], z=xyz[:, 1], mode="markers",
            marker=dict(size=marker_size, color=cvals, colorscale=charge_colorscale(t, self.colormap),
                        showscale=True,
                        colorbar=dict(title=clabel, thickness=12, len=0.75)),
            name="space points",
            hovertemplate="z %{x:.1f}<br>x %{y:.1f}<br>y %{z:.1f} cm<extra></extra>"))

        tnu = self.neutrino
        if tnu is not None and np.isfinite(tnu.vertex).all():
            tv = tnu.vertex
            fig.add_trace(go.Scatter3d(
                x=[tv[2]], y=[tv[0]], z=[tv[1]], mode="markers",
                marker=dict(symbol="x", size=6, color=t.mctrack,
                            line=dict(color=t.axes_bg, width=1)),
                name="true vertex",
                hovertemplate=(f"true vertex<br>{tnu.headline()}"
                               f"<br>({tv[0]:.1f}, {tv[1]:.1f}, {tv[2]:.1f}) cm"
                               "<extra></extra>")))

        run, subrun, ev = self.event.id
        limits = self._limits()
        rng = limits if limits else (None, None, None)
        pane = dict(backgroundcolor=t.axes_bg, gridcolor=t.grid, zerolinecolor=t.grid,
                    color=t.fg_muted)
        fig.update_layout(
            title=dict(text=(f"{self.geometry.detector}   run {run} / subrun {subrun} / "
                             f"event {ev}   -   {len(self.sp)} space points"),
                       font=dict(color=t.fg)),
            height=height, template=("plotly_dark" if t.is_dark else "plotly_white"),
            paper_bgcolor=t.fig_bg, font=dict(color=t.fg_muted),
            legend=dict(bgcolor=t.legend_bg, bordercolor=t.legend_edge, borderwidth=1,
                        font=dict(color=t.fg)),
            scene=dict(xaxis=dict(title="z [cm] (beam)", range=rng[2], **pane),
                       yaxis=dict(title="x [cm] (drift)", range=rng[0], **pane),
                       zaxis=dict(title="y [cm] (up)", range=rng[1], **pane),
                       aspectmode="data"),
            # Framing is what the camera depends on; stepping events must not
            # throw away a viewpoint the user set up.
            uirevision=f"3d|{self.geometry.detector}|{self.focus}",
            margin=dict(l=0, r=0, t=50, b=0))
        return fig

    def save_html(self, path: str, *, bundle_plotlyjs: bool = True, **kwargs) -> str:
        """Write an interactive HTML page (see :meth:`EventDisplay.save_html`)."""
        fig = self.plotly_figure(**kwargs)
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        fig.write_html(path, include_plotlyjs=("inline" if bundle_plotlyjs else "cdn"),
                       config=_PLOTLY_CONFIG)
        return path

    def show(self, **kwargs) -> None:
        self.plotly_figure(**kwargs).show(config=_PLOTLY_CONFIG)

    # ---- static (matplotlib) -------------------------------------------
    #
    # plotly's 3-D traces need WebGL, which a headless machine cannot provide,
    # so static 3-D images go through matplotlib instead.

    def figure(self, *, marker_size: float = 0.6, elev: float = 18.0,
               azim: float = -60.0, figsize: tuple[float, float] | None = None):
        plt = _pyplot()

        t, fonts = self.theme, _fonts(self.preset)
        fig = plt.figure(figsize=figsize or _preset_figsize(self.preset, 11.0, 7.0))
        fig.patch.set_facecolor(t.fig_bg)
        ax = fig.add_subplot(111, projection="3d")

        if self.draw_tpcs:
            for box in self.geometry.tpcs:
                bx, by, bz = _tpc_wireframe(box)
                ax.plot(bz, bx, by, color=t.grid, linewidth=0.4, alpha=0.8)

        if self.truth is not None and len(self.truth):
            ax.scatter(self.truth.z, self.truth.x, self.truth.y, s=0.2,
                       color=t.truth, alpha=0.3, linewidths=0, label="true deposits")

        if self.tracks is not None:
            poly = _join3d(self.tracks.points)
            if len(poly):
                ax.plot(poly[:, 2], poly[:, 0], poly[:, 1], "-", color=t.track,
                        linewidth=1.0, alpha=0.8, label="reco tracks", zorder=5)

        if self.mc is not None:
            q = _join3d(self.mc.points)
            if len(q):
                ax.plot(q[:, 2], q[:, 0], q[:, 1], "--", color=t.mctrack,
                        linewidth=0.6, alpha=0.6, label="true trajectories")
        if self.vertices is not None and len(self.vertices):
            v = self.vertices.xyz
            ax.scatter(v[:, 2], v[:, 0], v[:, 1], marker="*", s=90,
                       color=t.vertex, edgecolors=t.axes_bg, linewidths=0.5,
                       label="vertices", zorder=6)
        tnu = self.neutrino
        if tnu is not None and np.isfinite(tnu.vertex).all():
            tv = tnu.vertex
            ax.scatter([tv[2]], [tv[0]], [tv[1]], marker="X", s=110,
                       color=t.mctrack, edgecolors=t.axes_bg, linewidths=0.8,
                       label="true vertex", zorder=7)
        if self.showers is not None and len(self.showers):
            for j, (st, di, ln) in enumerate(zip(self.showers.start,
                                                 self.showers.direction,
                                                 self.showers.length)):
                tip = st + di * ln
                ax.plot([st[2], tip[2]], [st[0], tip[0]], [st[1], tip[1]],
                        "-", color=t.shower, linewidth=1.4, alpha=0.85,
                        label="shower axes" if j == 0 else None)

        xyz = self.sp.xyz
        cvals, clabel = self._colour_values()
        sc = ax.scatter(xyz[:, 2], xyz[:, 0], xyz[:, 1], c=cvals, s=marker_size,
                        cmap=charge_cmap(t, self.colormap), linewidths=0,
                        alpha=_ALPHA, label="space points")
        cb = fig.colorbar(sc, ax=ax, shrink=0.55, pad=0.10)
        _style_colorbar(cb, clabel, t, fonts)

        ax.set_xlabel("z [cm]  (beam)", fontsize=8)
        ax.set_ylabel("x [cm]  (drift)", fontsize=8)
        ax.set_zlabel("y [cm]  (up)", fontsize=8)
        _style_axes(ax, t, fonts, three_d=True)
        ax.view_init(elev=elev, azim=azim)
        limits = self._limits()
        if limits:
            (xl, yl, zl) = limits            # detector x, y, z
            # Screen-vertical is detector y, the gravity axis -- putting drift-x
            # there defeats a LArTPC reader's spatial intuition.
            ax.set_xlim(*zl)                 # plot axes are (z, x, y)
            ax.set_ylim(*xl)
            ax.set_zlim(*yl)
        ax.set_box_aspect((np.ptp(ax.get_xlim()), np.ptp(ax.get_ylim()),
                           np.ptp(ax.get_zlim())))
        run, subrun, ev = self.event.id
        head = self._truth_headline()
        ax.set_title(f"{self.geometry.detector}   run {run} / subrun {subrun} / "
                     f"event {ev}   -   {len(self.sp)} space points"
                     + (f"\n{head}" if head else ""),
                     fontsize=fonts["suptitle"], color=t.fg)
        # markerscale exists for the space points, which are drawn 1-2 px; it
        # would blow the vertex marker up until it covered the legend text, so
        # that one entry gets a proxy sized to survive the same multiplier.
        handles, labels = ax.get_legend_handles_labels()
        from matplotlib.lines import Line2D
        # Every point marker drawn large on the plot needs a proxy here: the
        # legend multiplies marker size by _LEGEND_MARKERSCALE_3D (which exists
        # for the 1-2 px space points), so a full-size star or X comes out
        # bigger than the legend box and covers the text beneath it.
        proxies = {"true vertex": ("X", t.mctrack), "vertices": ("*", t.vertex)}
        handles = [
            Line2D([], [], marker=proxies[lab][0], linestyle="none",
                   color=proxies[lab][1],
                   markersize=7 / _LEGEND_MARKERSCALE_3D,
                   markeredgecolor=t.axes_bg, markeredgewidth=0.1)
            if lab in proxies else h
            for h, lab in zip(handles, labels)]
        if handles:
            leg = ax.legend(handles, labels, facecolor=t.legend_bg,
                            edgecolor=t.legend_edge, labelcolor=t.fg,
                            framealpha=0.9, fontsize=fonts["legend"],
                            markerscale=_LEGEND_MARKERSCALE_3D,
                            loc="upper left", bbox_to_anchor=(0.0, 0.92))
            leg.set_zorder(10)
        return fig

    def save(self, path: str, *, dpi: int | None = None, **kwargs) -> str:
        _vector_text()
        # None must fall back to the preset, not to matplotlib's 100 dpi: the
        # CLI always passes dpi=a.dpi, which defaults to None.
        dpi = PRESETS[self.preset]["dpi"] if dpi is None else dpi
        fig = self.figure(**kwargs)
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        import matplotlib.pyplot as plt
        plt.close(fig)
        return path


class FlashDisplay3D(_TruthInfo):
    """Reconstructed optical flashes in the detector volume.

    A flash localises light in (y, z) and in time. Most flash finders leave the
    drift coordinate unset -- light alone cannot fix it without a matched TPC
    object, and LArSoft stores that as DBL_MAX. So a flash is drawn here as a
    bar spanning the drift, not as a point at x = 0: the bar is the measurement,
    the position along it is unknown.

    Photon detectors are drawn faintly for context, and the true vertex when
    the event has one -- which is how you see whether the light lines up with
    where the interaction actually was.
    """

    def __init__(self, event, optical=None, *, time_range=None, max_flashes=200,
                 theme: str = "dark", colormap: str = DEFAULT_COLORMAP,
                 preset: str = DEFAULT_PRESET, show_opdets: bool = True):
        self.event = event
        self.op = optical if optical is not None else event.optical()
        self.geometry = event.geometry
        self.time_range = time_range
        self.max_flashes = max_flashes
        self.show_opdets = show_opdets
        self.theme = resolve_theme(theme)
        resolve_colormap(colormap)
        self.colormap = colormap
        self.preset = preset

    @property
    def window(self):
        """(lo, hi) time window, defaulting around the brightest flash.

        The readout is milliseconds wide and dominated by radiological light;
        without a default window the brightest 200 flashes are drawn from the
        whole of it and the physics flash is just one bar among them. Same rule
        as the 2-D optical view, so the two agree on what they are showing.
        """
        if self.time_range is not None:
            return tuple(self.time_range)
        o = self.op
        if not len(o) or not o.is_flash.any():
            return None
        brightest = int(np.argmax(np.where(o.is_flash, o.pe, -np.inf)))
        centre = float(o.time[brightest])
        return (centre - 50.0, centre + 50.0)

    @cached_property
    def _selected(self):
        """(flashes to draw, how many were in the window before capping)."""
        o = self.op
        sel = o.is_flash & np.isfinite(o.y) & np.isfinite(o.z)
        win = self.window
        if win is not None:
            sel &= (o.time >= win[0]) & (o.time <= win[1])
        idx = np.flatnonzero(sel)
        if not len(idx):
            return o[idx], 0
        order = idx[np.argsort(-o.pe[idx])]
        return o[order[:self.max_flashes]], len(idx)

    @property
    def flashes(self):
        """Flashes in the time window, brightest first, capped at max_flashes."""
        return self._selected[0]

    def _colour_time_range(self) -> tuple[float, float]:
        """Span for the flash-time colour axis, never degenerate."""
        win = self.window
        f = self.flashes
        if win is not None and win[1] > win[0]:
            return float(win[0]), float(win[1])
        if len(f) and f.time.max() > f.time.min():
            return float(f.time.min()), float(f.time.max())
        centre = float(f.time[0]) if len(f) else 0.0
        return centre - 1.0, centre + 1.0

    @property
    def _drift_range(self) -> tuple[float, float]:
        boxes = self.geometry.tpcs
        if not boxes:
            return (-1.0, 1.0)
        return (min(b.xmin for b in boxes), max(b.xmax for b in boxes))

    def _title(self) -> str:
        run, subrun, ev = self.event.id
        shown, total = len(self.flashes), self._selected[1]
        more = "" if shown >= total else f", brightest {shown} shown"
        win = self.window
        span = "" if win is None else f"   t in [{win[0]:.1f}, {win[1]:.1f}] us"
        return (f"{self.geometry.detector}   run {run} / subrun {subrun} / "
                f"event {ev}   -   {total} flashes{more}{span}")

    def summary(self) -> str:
        f = self.flashes
        lines = [self._title()]
        head = self._truth_headline()
        if head:
            lines.append(head)
        if not len(f):
            lines.append("  no flashes in the time window")
            return "\n".join(lines)
        lines.append(f"  drift position: "
                     + ("reconstructed" if self.op.has_x else
                        "NOT reconstructed by this flash finder - drawn as a "
                        "bar spanning the drift"))
        lines.append(f"  brightest {f.pe[0]:.0f} PE at t = {f.time[0]:.2f} us, "
                     f"y = {f.y[0]:.1f} cm, z = {f.z[0]:.1f} cm")
        nu = self.neutrino
        if nu is not None:
            dz = f.z[0] - nu.vertex[2]
            dy = f.y[0] - nu.vertex[1]
            lines.append(f"  brightest flash vs true vertex: "
                         f"dy = {dy:+.1f} cm, dz = {dz:+.1f} cm")
        return "\n".join(lines)

    # ---- static (matplotlib) -------------------------------------------

    def figure(self, *, elev: float = 22.0, azim: float = -60.0):
        plt = _pyplot()
        t, fonts = self.theme, _fonts(self.preset)
        fig = plt.figure(figsize=_preset_figsize(self.preset, 11.0, 8.0))
        ax = fig.add_subplot(111, projection="3d")
        fig.patch.set_facecolor(t.fig_bg)
        ax.set_facecolor(t.axes_bg)

        for box in self.geometry.tpcs:
            xs, ys, zs = _tpc_wireframe(box)
            ax.plot(zs, xs, ys, color=t.grid, linewidth=0.4, alpha=0.35)

        if self.show_opdets and self.geometry.has_optical:
            od = self.geometry.opdet_xyz
            od = od[np.isfinite(od).all(axis=1)]
            ax.scatter(od[:, 2], od[:, 0], od[:, 1], s=2, color=t.fg_muted,
                       alpha=0.35, linewidths=0, label="photon detectors")

        f = self.flashes
        if len(f):
            lo, hi = self._drift_range
            from matplotlib.colors import Normalize
            # Colour spans the requested window, not just the flashes that
            # happen to be in it: with one flash in view vmin == vmax, which
            # Normalize maps to 0.0 -- the bottom of the colormap, i.e. the
            # brightest flash in the event rendered near-black.
            tlo, thi = self._colour_time_range()
            norm = Normalize(vmin=tlo, vmax=thi)
            cmap = charge_cmap(t, self.colormap)
            widths = _pe_width(f.pe)
            for j in range(len(f)):
                colour = cmap(norm(f.time[j]))
                if np.isfinite(f.x[j]):
                    ax.scatter([f.z[j]], [f.x[j]], [f.y[j]], s=40 * widths[j],
                               color=colour, linewidths=0)
                else:
                    # x unmeasured: the flash is a line across the drift
                    ax.plot([f.z[j], f.z[j]], [lo, hi], [f.y[j], f.y[j]],
                            "-", color=colour, linewidth=widths[j], alpha=0.8,
                            label="flashes" if j == 0 else None)
            sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            cb = fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.10)
            _style_colorbar(cb, "flash time [us]", t, fonts)

        nu = self.neutrino
        if nu is not None and np.isfinite(nu.vertex).all():
            v = nu.vertex
            ax.scatter([v[2]], [v[0]], [v[1]], marker="X", s=110,
                       color=t.mctrack, edgecolors=t.axes_bg, linewidths=0.8,
                       label="true vertex")

        ax.set_xlabel("z [cm]  (beam)", fontsize=fonts["label"])
        ax.set_ylabel("x [cm]  (drift)", fontsize=fonts["label"])
        ax.set_zlabel("y [cm]  (up)", fontsize=fonts["label"])
        _style_axes(ax, t, fonts, three_d=True)
        ax.set_box_aspect((np.ptp(ax.get_xlim()), np.ptp(ax.get_ylim()),
                           np.ptp(ax.get_zlim())))
        ax.view_init(elev=elev, azim=azim)
        head = self._truth_headline()
        ax.set_title(self._title() + (f"\n{head}" if head else ""),
                     fontsize=fonts["suptitle"], color=t.fg)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            leg = ax.legend(handles, labels, facecolor=t.legend_bg,
                            edgecolor=t.legend_edge, labelcolor=t.fg,
                            framealpha=0.9, fontsize=fonts["legend"],
                            loc="upper left", bbox_to_anchor=(0.0, 0.92))
            leg.set_zorder(10)
        return fig

    def save(self, path: str, *, dpi: int | None = None, **kwargs) -> str:
        _vector_text()
        # None must fall back to the preset, not to matplotlib's 100 dpi: the
        # CLI always passes dpi=a.dpi, which defaults to None.
        dpi = PRESETS[self.preset]["dpi"] if dpi is None else dpi
        fig = self.figure(**kwargs)
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        _pyplot().close(fig)
        return path

    # ---- interactive (plotly) ------------------------------------------

    def plotly_figure(self):
        import plotly.graph_objects as go
        t = self.theme
        fig = go.Figure()

        for box in self.geometry.tpcs:
            xs, ys, zs = _tpc_wireframe(box)
            fig.add_trace(go.Scatter3d(x=zs, y=xs, z=ys, mode="lines",
                                       line=dict(color=t.grid, width=1),
                                       hoverinfo="skip", showlegend=False))

        if self.show_opdets and self.geometry.has_optical:
            od = self.geometry.opdet_xyz
            od = od[np.isfinite(od).all(axis=1)]
            fig.add_trace(go.Scatter3d(
                x=od[:, 2], y=od[:, 0], z=od[:, 1], mode="markers",
                marker=dict(size=1.5, color=t.fg_muted, opacity=0.45),
                name="photon detectors", hoverinfo="skip"))

        f = self.flashes
        if len(f):
            lo, hi = self._drift_range
            scale = charge_colorscale(t, self.colormap)
            width = _pe_width(f.pe, lo=1.0, hi=8.0)
            tlo, thi = self._colour_time_range()
            # One trace per flash: each needs its own width, which plotly does
            # not vary within a line trace.
            for j in range(len(f)):
                hover = (f"flash<br>{f.pe[j]:.0f} PE<br>t = {f.time[j]:.3f} us"
                         f"<br>y = {f.y[j]:.1f} +- {f.y_width[j]:.1f} cm"
                         f"<br>z = {f.z[j]:.1f} +- {f.z_width[j]:.1f} cm"
                         "<extra></extra>")
                if np.isfinite(f.x[j]):
                    xs, ys, zs = [f.x[j]], [f.y[j]], [f.z[j]]
                    fig.add_trace(go.Scatter3d(
                        x=zs, y=xs, z=ys, mode="markers",
                        marker=dict(size=width[j], color=[f.time[j]],
                                    colorscale=scale, cmin=tlo, cmax=thi),
                        name="flashes", legendgroup="fl",
                        showlegend=(j == 0), hovertemplate=hover))
                else:
                    fig.add_trace(go.Scatter3d(
                        x=[f.z[j], f.z[j]], y=[lo, hi], z=[f.y[j], f.y[j]],
                        mode="lines",
                        line=dict(color=_sample_colorscale(scale, f.time[j],
                                                           tlo, thi),
                                  width=width[j]),
                        name="flashes", legendgroup="fl",
                        showlegend=(j == 0), hovertemplate=hover))
            # a dummy trace carries the colourbar the line traces cannot
            fig.add_trace(go.Scatter3d(
                x=[None], y=[None], z=[None], mode="markers",
                marker=dict(colorscale=scale, cmin=tlo, cmax=thi,
                            color=[tlo],
                            colorbar=dict(title="flash time [us]", thickness=12,
                                          len=0.7), size=0.1),
                showlegend=False, hoverinfo="skip"))

        nu = self.neutrino
        if nu is not None and np.isfinite(nu.vertex).all():
            v = nu.vertex
            fig.add_trace(go.Scatter3d(
                x=[v[2]], y=[v[0]], z=[v[1]], mode="markers",
                marker=dict(symbol="x", size=6, color=t.mctrack),
                name="true vertex",
                hovertemplate=(f"true vertex<br>{nu.headline()}<extra></extra>")))

        pane = dict(backgroundcolor=t.axes_bg, gridcolor=t.grid,
                    zerolinecolor=t.grid, color=t.fg_muted)
        fig.update_layout(
            title=dict(text=self._title(), font=dict(size=13, color=t.fg)),
            paper_bgcolor=t.fig_bg, font_color=t.fg_muted,
            scene=dict(xaxis=dict(title="z [cm] (beam)", **pane),
                       yaxis=dict(title="x [cm] (drift)", **pane),
                       zaxis=dict(title="y [cm] (up)", **pane),
                       aspectmode="data"),
            legend=dict(bgcolor=t.legend_bg, bordercolor=t.legend_edge,
                        borderwidth=1, font=dict(color=t.fg)),
            uirevision="flash3d",      # keep the camera across event steps
            margin=dict(l=0, r=0, t=50, b=0))
        return fig

    def __repr__(self) -> str:
        return f"<FlashDisplay3D {self.event!r} flashes={len(self.flashes)}>"


class OpticalDisplay(_TruthInfo):
    """Photon-detector activity: the time dimension the TPC views lack.

    Two panels sharing a time axis -- individual OpHits against their channel,
    and reconstructed flashes against z. Charge in a LArTPC takes milliseconds
    to drift, while scintillation light arrives in nanoseconds, so this is what
    actually timestamps an interaction.
    """

    def __init__(self, event, optical=None, *, time_range=None,
                 theme: str = "dark", colormap: str = DEFAULT_COLORMAP,
                 preset: str = DEFAULT_PRESET):
        self.event = event
        self.op = optical if optical is not None else event.optical()
        self.geometry = event.geometry
        self.time_range = time_range
        self.theme = resolve_theme(theme)
        resolve_colormap(colormap)
        self.colormap = colormap
        self.preset = preset

    @property
    def _window(self):
        """(lo, hi) time window, defaulting to a span around the brightest flash.

        The full readout is milliseconds wide and dominated by radiological
        activity; the physics flash is nanoseconds wide. Showing everything by
        default hides the one thing worth looking at.
        """
        if self.time_range is not None:
            return tuple(self.time_range)
        o = self.op
        if not len(o):
            return None
        brightest = int(np.argmax(np.where(o.is_flash, o.pe, -np.inf))) \
            if o.is_flash.any() else int(np.argmax(o.pe))
        centre = float(o.time[brightest])
        return (centre - 50.0, centre + 50.0)

    def _flash_y_range(self) -> tuple[float, float]:
        """Symmetric y span from the geometry, not a hardcoded detector size.

        This was ``vmin=-600, vmax=600`` -- correct for the samples to hand and
        wrong for any other detector, in a module whose premise is that nothing
        is detector-specific.
        """
        boxes = self.geometry.tpcs
        if not boxes:
            return (-1.0, 1.0)
        return (min(b.ymin for b in boxes), max(b.ymax for b in boxes))

    def _mask(self, extra=None):
        lo, hi = self._window or (-np.inf, np.inf)
        m = (self.op.time >= lo) & (self.op.time <= hi)
        return m if extra is None else (m & extra)

    def __repr__(self) -> str:
        return f"<OpticalDisplay {self.event!r} {self.op!r}>"

    def _title(self) -> str:
        run, subrun, ev = self.event.id
        o = self.op
        return (f"{self.geometry.detector}   run {run} / subrun {subrun} / event {ev}"
                f"   -   {int(o.is_flash.sum())} flashes, "
                f"{int((~o.is_flash).sum())} optical hits")

    def summary(self) -> str:
        o = self.op
        lines = [self._title()]
        if self.neutrino is not None:
            lines.append(self._truth_headline())
        for label, mask in (("flashes", o.is_flash), ("op hits", ~o.is_flash)):
            if not mask.any():
                continue
            t, pe = o.time[mask], o.pe[mask]
            lines.append(f"  {label:9s} {mask.sum():6d}   t [{t.min():9.2f},"
                         f"{t.max():9.2f}] us   PE [{pe.min():.1f},{pe.max():.1f}]")
        if o.is_flash.any():
            i = int(np.argmax(np.where(o.is_flash, o.pe, -np.inf)))
            median = float(np.median(o.pe[o.is_flash]))
            lines.append(f"  brightest flash  t = {o.time[i]:.2f} us, "
                         f"{o.pe[i]:.0f} PE ({o.pe[i] / max(median, 1e-9):.0f}x the "
                         f"median flash), y = {o.y[i]:.0f}, z = {o.z[i]:.0f} cm")
        nonpos = int((o.pe <= 0).sum())
        if nonpos:
            lines.append(f"  {nonpos} entries have PE <= 0 (baseline undershoot); "
                         "they sit at the bottom of the log colour scale")
        window = self._window
        if window:
            lines.append(f"  showing t in [{window[0]:.1f}, {window[1]:.1f}] us "
                         "- pass time_range=(lo, hi) to change, or (-inf, inf) for all")
        return "\n".join(lines)

    def figure(self, *, figsize: tuple[float, float] | None = None):
        plt = _pyplot()
        t, fonts = self.theme, _fonts(self.preset)
        o = self.op
        fig, axes = plt.subplots(
            2, 1, figsize=figsize or _preset_figsize(self.preset, 12.0, 6.0),
            constrained_layout=True,
                                 sharex=True)
        fig.patch.set_facecolor(t.fig_bg)
        hits, fl = self._mask(~o.is_flash), self._mask(o.is_flash)

        # With photon-detector positions the hits panel can share the flash
        # panel's z axis -- and therefore the TPC views' z axis -- instead of
        # plotting a detector index that carries no spatial meaning.
        positioned = self.geometry.has_optical and np.isfinite(o.z[hits]).any()
        hit_y = o.z if positioned else o.channel.astype(float)
        hit_label = ("photon detector z [cm]" if positioned
                     else "optical channel (index, not position)")

        ax = axes[0]
        if hits.any():
            pe = np.clip(o.pe[hits], 1e-3, None)
            # Fixed range, not the window's own min/max: the same OpHit changed
            # colour by 7.6x purely from zooming, and no two renders could be
            # compared. Same reasoning as PE_SIZE_RANGE for flash size.
            plo, phi = PE_SIZE_RANGE
            sc = ax.scatter(o.time[hits], hit_y[hits], c=pe, s=3,
                            cmap=charge_cmap(t, self.colormap),
                            norm=__import__("matplotlib.colors", fromlist=["LogNorm"])
                            .LogNorm(vmin=plo, vmax=phi, clip=True),
                            linewidths=0, rasterized=True)
            cb = fig.colorbar(sc, ax=ax, pad=0.01, fraction=0.03)
            _style_colorbar(cb, "photoelectrons", t, fonts)
        ax.set_ylabel(hit_label, fontsize=fonts["label"])
        ax.set_title("optical hits", fontsize=fonts["title"])
        _style_axes(ax, t, fonts)

        ax = axes[1]
        if fl.any():
            # Size from PE on a fixed log range; colour from y. Flashes carry a
            # y as well as a z, and plotting z alone threw away half of their
            # reconstructed position.
            ylo, yhi = self._flash_y_range()
            fsc = ax.scatter(o.time[fl], o.z[fl], s=_pe_size(o.pe[fl]),
                             c=o.y[fl], cmap="coolwarm", vmin=ylo, vmax=yhi,
                             alpha=0.85, linewidths=0,
                             label="flashes (size ~ PE)")
            cb2 = fig.colorbar(fsc, ax=ax, pad=0.01, fraction=0.03)
            _style_colorbar(cb2, "flash y [cm]", t, fonts)
            _style_legend(ax, t, fontsize=fonts["legend"], loc="upper right",
                          markerscale=1)
        ax.set_xlabel("time [us]", fontsize=fonts["label"])
        ax.set_ylabel("flash z [cm]", fontsize=fonts["label"])
        ax.set_title("reconstructed flashes", fontsize=fonts["title"])
        _style_axes(ax, t, fonts)

        head = self._truth_headline()
        fig.suptitle(self._title() + (f"\n{head}" if head else ""),
                     fontsize=fonts["suptitle"], color=t.fg)
        return fig

    def save(self, path: str, *, dpi: int | None = None, **kwargs) -> str:
        _vector_text()
        # None must fall back to the preset, not to matplotlib's 100 dpi: the
        # CLI always passes dpi=a.dpi, which defaults to None.
        dpi = PRESETS[self.preset]["dpi"] if dpi is None else dpi
        fig = self.figure(**kwargs)
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
        _pyplot().close(fig)
        return path

    def plotly_figure(self, *, height: int = 700):
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        o = self.op
        t = self.theme
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            subplot_titles=["optical hits",
                                            "reconstructed flashes"],
                            vertical_spacing=0.09)
        hits, fl = self._mask(~o.is_flash), self._mask(o.is_flash)
        positioned = self.geometry.has_optical and np.isfinite(o.z[hits]).any()
        hit_y = o.z if positioned else o.channel.astype(float)
        hit_label = ("photon detector z [cm]" if positioned
                     else "optical channel (index)")
        if hits.any():
            raw_pe = o.pe[hits]
            pe = np.clip(raw_pe, 1e-3, None)
            log_pe = np.log10(pe)
            decades = np.arange(np.floor(log_pe.min()), np.ceil(log_pe.max()) + 1)
            fig.add_trace(go.Scattergl(
                x=o.time[hits], y=hit_y[hits], mode="markers",
                marker=dict(size=3, color=log_pe,
                            colorscale=charge_colorscale(t, self.colormap), showscale=True,
                            colorbar=dict(
                                title="PE", thickness=12, len=0.45, y=0.78,
                                # colour is log10(PE); label the ticks with the
                                # real values so the axis means what it says
                                tickvals=decades.tolist(),
                                ticktext=[_si(10.0 ** v) for v in decades])),
                name="op hits",
                customdata=np.column_stack([raw_pe, o.channel[hits]]),
                hovertemplate="t %{x:.2f} us<br>z %{y:.1f} cm<br>"
                              "%{customdata[0]:.2f} PE  (ch %{customdata[1]:d})"
                              "<extra></extra>"),
                row=1, col=1)
        if fl.any():
            ylo, yhi = self._flash_y_range()
            fig.add_trace(go.Scattergl(
                x=o.time[fl], y=o.z[fl], mode="markers",
                # Colour by y, as the static figure does: the two backends are
                # meant to draw the same picture, and a flat colour threw away
                # half of the reconstructed flash position.
                marker=dict(size=np.sqrt(_pe_size(o.pe[fl])), color=o.y[fl],
                            colorscale="RdBu_r", cmin=ylo, cmax=yhi,
                            opacity=0.8, showscale=True,
                            colorbar=dict(title="flash y [cm]", thickness=12,
                                          len=0.4, y=0.2)),
                name="flashes", customdata=np.column_stack([o.pe[fl], o.y[fl]]),
                hovertemplate="t %{x:.2f} us<br>z %{y:.1f} cm<br>"
                              "y %{customdata[1]:.1f} cm<br>"
                              "%{customdata[0]:.0f} PE<extra></extra>"),
                row=2, col=1)
        fig.update_xaxes(title_text="time [us]", row=2, col=1, gridcolor=t.grid)
        fig.update_xaxes(gridcolor=t.grid, row=1, col=1)
        fig.update_yaxes(title_text=hit_label, row=1, col=1, gridcolor=t.grid)
        fig.update_yaxes(title_text="flash z [cm]", row=2, col=1, gridcolor=t.grid)
        fig.update_layout(title=dict(text=self._title(), font=dict(size=13, color=t.fg)),
                          height=height, template=("plotly_dark" if t.is_dark else "plotly_white"),
                          paper_bgcolor=t.fig_bg, plot_bgcolor=t.axes_bg,
                          font=dict(color=t.fg_muted), dragmode="pan",
                          uirevision="optical",   # keep zoom across event steps
                          margin=dict(l=60, r=20, t=80, b=50))
        for a in fig.layout.annotations:
            a.font.size = 11
            a.font.color = t.fg
        return fig

    def save_html(self, path: str, *, bundle_plotlyjs: bool = True, **kwargs) -> str:
        fig = self.plotly_figure(**kwargs)
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        fig.write_html(path, include_plotlyjs=("inline" if bundle_plotlyjs else "cdn"),
                       config=_PLOTLY_CONFIG)
        return path

    def show(self, **kwargs) -> None:
        self.plotly_figure(**kwargs).show(config=_PLOTLY_CONFIG)
