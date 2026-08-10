"""Colour themes and charge colormaps.

Every colour the display uses lives here, in one of two themes, so a figure can
be retargeted from screen to paper without touching drawing code.

Two rules this module exists to enforce:

* **No colour means two things.** Object-identity colours (``categorical``) and
  overlay/chrome colours (``track``, ``vertex``, ``shower``, ``mctrack``,
  ``truth``, ``grid``) are drawn from disjoint sets. They previously shared
  four exact hex values, so a hit could be painted the same amber as the shower
  cone drawn over it.
* **Identity does not rest on colour alone.** ``categorical`` is paired with a
  marker cycle, giving 8 x 5 = 40 distinguishable combinations before anything
  repeats. Colour and marker advance independently, so consecutive objects
  differ in both; ranks 5 apart share a marker and are therefore kept apart in
  greyscale luminance as well.

What this module does NOT claim: that the overlay colours are separable from
the charge colormaps by hue. Six ramps between them cover most of colour space,
and a joint solve that also satisfies contrast and colour-vision constraints
only returns neon. Overlays are told apart from hits by *shape and casing*
instead -- dashed vs solid, star vs cross, and a contrasting outline drawn
under every overlay marker and line (see ``_CASING`` in :mod:`pylarevd.display`).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Theme:
    """A complete colour scheme for one output target."""

    name: str
    fig_bg: str            # figure surround
    axes_bg: str           # panel interior
    legend_bg: str
    legend_edge: str
    fg: str                # primary text
    fg_muted: str          # ticks, axis labels
    grid: str
    track: str             # reconstructed tracks
    vertex: str
    shower: str
    mctrack: str           # true trajectories
    truth: str             # true energy deposits
    unassociated: str      # hits belonging to no reconstructed object
    categorical: tuple[str, ...]
    cmap_floor: float      # how far into the colormap to start (see below)
    cmap_ceiling: float = 1.0

    @property
    def is_dark(self) -> bool:
        return self.name == "dark"


#: Markers cycled alongside ``categorical`` so identity survives without colour.
MARKER_CYCLE = ("o", "^", "s", "D", "v")
PLOTLY_MARKER_CYCLE = ("circle", "triangle-up", "square", "diamond",
                       "triangle-down")

DARK = Theme(
    name="dark",
    fig_bg="#0f172a", axes_bg="#020617",
    legend_bg="#1e293b", legend_edge="#475569",
    fg="#e2e8f0", fg_muted="#94a3b8", grid="#4f5e75",
    track="#ffffff", vertex="#ff2d78",
    # Violet, not amber: the warm colormaps (turbo, inferno, plasma) reach
    # #ffb020 at ordinary charge values, so a shower cone drawn over a bright
    # hit cluster became metameric with it.
    shower="#c084fc",
    mctrack="#00d5ff", truth="#8a94a6", unassociated="#4e5d76",
    categorical=("#5b9bd5", "#e15759", "#59a14f", "#b07aa1",
                 "#f28e2b", "#76b7b2", "#edc948", "#ff9da7"),
    # The bottom of most colormaps is near-black and vanishes on a dark panel.
    cmap_floor=0.08,
)

LIGHT = Theme(
    name="light",
    fig_bg="#ffffff", axes_bg="#ffffff",
    legend_bg="#f8fafc", legend_edge="#94a3b8",
    fg="#0f172a", fg_muted="#475569", grid="#8394ac",
    track="#111827", vertex="#be185d", shower="#7c3aed",
    mctrack="#0369a1", truth="#6b7280", unassociated="#8993a3",
    # Ranks 5 apart share a marker (the cycle is 5 long), so those pairs must
    # also separate in greyscale: #00838f and #b03a5b were within 1.5 and 4.2
    # luma of their partners, i.e. the same ink in print.
    categorical=("#1f6fb2", "#c0392b", "#2e7d32", "#7b4397",
                 "#d2691e", "#006670", "#9a7d0a", "#c34a6c"),
    # On white the *pale* end of a colormap disappears instead, so trim the top.
    cmap_floor=0.0, cmap_ceiling=0.92,
)

THEMES: dict[str, Theme] = {"dark": DARK, "light": LIGHT}


@dataclass(frozen=True)
class Colormap:
    """A charge colormap plus a note on what it is good for."""

    name: str
    monotonic: bool        # lightness increases steadily -> survives greyscale
    note: str


#: Charge ramps offered to the user. ``monotonic`` records whether lightness
#: rises steadily across the ramp, which is what makes charge ordering readable
#: in greyscale and under colour-vision deficiency. ``turbo`` does not: it is a
#: much-improved rainbow, but its lightness doubles back repeatedly.
COLORMAPS: dict[str, Colormap] = {
    "turbo": Colormap("turbo", False,
                      "rainbow; familiar and high-contrast on screen, but "
                      "lightness is not monotonic so it flattens in greyscale"),
    "magma": Colormap("magma", True,
                      "black-red-yellow; keeps the bright-core reading and is "
                      "monotonic, so it prints"),
    "viridis": Colormap("viridis", True, "blue-green-yellow; monotonic"),
    "cividis": Colormap("cividis", True,
                        "blue-yellow; designed to look the same under the "
                        "common colour-vision deficiencies"),
    "inferno": Colormap("inferno", True, "black-red-white; monotonic"),
    "plasma": Colormap("plasma", True, "purple-orange-yellow; monotonic"),
}

DEFAULT_COLORMAP = "turbo"


def resolve_theme(theme) -> Theme:
    if isinstance(theme, Theme):
        return theme
    try:
        return THEMES[theme]
    except KeyError:
        raise ValueError(
            f"theme={theme!r} is not one of {', '.join(THEMES)}") from None


def resolve_colormap(name: str) -> Colormap:
    try:
        return COLORMAPS[name]
    except KeyError:
        raise ValueError(
            f"colormap={name!r} is not one of {', '.join(COLORMAPS)}") from None
