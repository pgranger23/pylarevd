"""Regression tests for the display.py / theme.py fixes of the phase-2 review."""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

import pylario
from pylario import EventFile

ATMNU = os.environ.get("PYLAREVD_ATMNU", "")
ROCKMU = os.environ.get("PYLAREVD_ROCKMU", "")
GEOM = os.path.join(os.path.dirname(os.path.abspath(pylario.__file__)),
                    "geom", "dune10kt_v6_1x2x6.npz")
needs_atmnu = pytest.mark.skipif(not os.path.exists(ATMNU) or not os.path.exists(GEOM),
                                 reason="atmospheric sample or geometry absent")
needs_rockmu = pytest.mark.skipif(not os.path.exists(ROCKMU), reason="rock-muon sample absent")


def _close(fig):
    import matplotlib.pyplot as plt
    plt.close(fig)


# 1. the 2-D legend is gathered from every panel, not the first one only

def test_legend_entries_cover_every_axes_once():
    import matplotlib.pyplot as plt
    from pylarevd.display import _legend_entries
    fig, (a, b) = plt.subplots(2)
    a.plot([0, 1], label="true deposits")
    b.plot([0, 1], label="true deposits")
    b.plot([0, 1], label="reco tracks")           # only in the second panel
    handles, labels = _legend_entries([a, b])
    assert labels == ["true deposits", "reco tracks"]
    assert len(handles) == 2
    _close(fig)


@needs_atmnu
def test_2d_legend_lists_overlays_of_every_panel():
    from pylarevd.display import _legend_entries
    d = EventFile(ATMNU, geometry=GEOM)[1].display(reco=True, truth=True)
    fig = d.figure()
    panels = fig.axes[:len(d.panels)]
    _, every = _legend_entries(panels[:len(d.panels)])
    leg = panels[0].get_legend()
    assert leg is not None
    assert [t.get_text() for t in leg.get_texts()] == every
    _close(fig)


# 2. --preset sizes the 3-D figure (screen keeps its size)

@needs_atmnu
@pytest.mark.parametrize("preset", ["screen", "paper1", "poster"])
def test_3d_figure_follows_the_preset(preset):
    from pylarevd.display import PRESETS
    ev = EventFile(ATMNU, geometry=GEOM)[0]
    fig = ev.display_3d(preset=preset).figure()
    k = PRESETS[preset]["panel_width"] / PRESETS["screen"]["panel_width"]
    assert tuple(fig.get_size_inches()) == pytest.approx((9.8 * k, 8.2 * k))
    _close(fig)
    explicit = ev.display_3d(preset=preset).figure(figsize=(5.0, 4.0))
    assert tuple(explicit.get_size_inches()) == pytest.approx((5.0, 4.0))
    _close(explicit)


# 3 + 8. the categorical note counts objects, and is not left on the instance

@needs_atmnu
def test_categorical_note_counts_objects_not_the_largest_id():
    d = EventFile(ATMNU, geometry=GEOM)[1].display(colour_by="cluster")
    g = d.groups
    n_obj = len(np.unique(g[g >= 0]))
    assert n_obj < int(g.max()) + 1, "cluster ids are expected to be sparse here"
    fig = d.figure()
    note = fig._supxlabel.get_text().replace("\n", " ")
    assert f"{n_obj} objects" in note
    _close(fig)
    assert not hasattr(d, "_extra_note")
    d.colour_by = "integral"                      # no longer categorical
    fig = d.figure()
    text = fig._supxlabel.get_text() if fig._supxlabel is not None else ""
    assert "objects (colour x marker)" not in text
    _close(fig)


# 4. FlashDisplay3D.save_html forwards kwargs and the shared plotly config

@needs_atmnu
def test_flash3d_html_uses_the_shared_config_and_kwargs(tmp_path):
    from pylarevd.display import FlashDisplay3D
    d = EventFile(ATMNU, geometry=GEOM)[1].display_flashes_3d()
    path = d.save_html(str(tmp_path / "f.html"), bundle_plotlyjs=False, height=432)
    html = open(path).read()
    assert '"scrollZoom": true' in html and '"displaylogo": false' in html
    assert '"height":432' in html.replace(" ", "")
    assert callable(FlashDisplay3D.show)


# 5. Display3D validates colour_scale and preset up front

@needs_atmnu
@pytest.mark.parametrize("kwargs", [dict(colour_scale="loglog"),
                                    dict(preset="billboard")])
def test_display3d_rejects_bad_options(kwargs):
    ev = EventFile(ATMNU, geometry=GEOM)[0]
    with pytest.raises(ValueError, match=next(iter(kwargs))):
        ev.display_3d(**kwargs)


@needs_atmnu
@pytest.mark.parametrize("maker", ["display_flashes_3d", "display_optical"])
def test_light_displays_reject_a_bad_preset(maker):
    ev = EventFile(ATMNU, geometry=GEOM)[0]
    with pytest.raises(ValueError, match="preset"):
        getattr(ev, maker)(preset="billboard")


# 6. min_hits_per_panel=0 never yields an empty panel

def test_panels_never_empty_even_with_min_hits_zero():
    from pylarevd.display import _panels
    hits = SimpleNamespace(
        w=np.array([1.0, 2.0, np.nan]), x=np.array([1.0, 2.0, 3.0]),
        panel=np.array([0, 0, 7]),            # panel 7: no hit with geometry
        view=np.array([2, 2, 2]), drift_sign=np.array([1, 1, 1]),
        orientation=np.array([0, 0, 0]))
    geometry = SimpleNamespace(view_name=lambda v: "Z")
    for merge in ("none", "view", "orientation"):
        panels = _panels(hits, geometry, min_hits=0, merge=merge)
        assert panels and all(p.mask.any() for p in panels), merge


@needs_rockmu
def test_summary_with_min_hits_zero():
    for merge in ("none", "view", "orientation"):
        d = EventFile(ROCKMU)[0].display(merge=merge, min_hits_per_panel=0)
        assert len(d.summary().splitlines()) > len(d.panels)


# 7. the theme docstring no longer claims every overlay is cased

def test_theme_doc_names_what_is_cased():
    import pylarevd.theme as theme
    assert "under every overlay marker and line" not in theme.__doc__
    assert "``_casing``" in theme.__doc__
