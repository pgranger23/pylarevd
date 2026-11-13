"""Regression tests for the pylarevd.light fixes (phase-2 review).

The data tests use ``PYLAREVD_ATMNU`` (as tests/test_light.py does) and, for
the side-by-side layout, ``PYLAREVD_PDSP`` (a ProtoDUNE-SP reco file with two
photon-detector walls); each skips without its file.
"""

from __future__ import annotations

import os
import types

import numpy as np
import pytest

from pylarevd.light import LightAnimation, _encode_gif, _encode_mp4, frame_edges

_DATA = os.environ.get("PYLAREVD_TEST_DATA", "")
ATMNU = os.environ.get(
    "PYLAREVD_ATMNU", os.path.join(_DATA, "atmnu_radio_reco.root") if _DATA else "")
PDSP = os.environ.get("PYLAREVD_PDSP", "")
needs_atmnu = pytest.mark.skipif(not os.path.exists(ATMNU),
                                 reason="neutrino sample absent")
needs_pdsp = pytest.mark.skipif(not os.path.exists(PDSP),
                                reason="ProtoDUNE-SP sample absent")


# ---- frame_edges validates the scale before any early return ----------------

@pytest.mark.parametrize("hi", [5.01, 15.0])
def test_frame_edges_rejects_an_unknown_scale_even_for_a_short_window(hi):
    # hi <= t0 + quantum used to return uniform edges before the check
    with pytest.raises(ValueError, match="time_scale"):
        frame_edges(5.0, 4.0, hi, 4, "bogus", 0.016)


# ---- _encode_mp4 with no frames ----------------------------------------------

def test_encoders_reject_no_frames_alike():
    for enc in (_encode_gif, _encode_mp4):
        with pytest.raises(ValueError, match="no frames"):
            enc([], 12.0, 1.0)


# ---- TPC outlines --------------------------------------------------------------

def test_tpc_outlines_are_distinct_rounded_and_sorted():
    box = lambda z0, z1, y0, y1: types.SimpleNamespace(  # noqa: E731
        zmin=z0, zmax=z1, ymin=y0, ymax=y1)
    fake = types.SimpleNamespace(geometry=types.SimpleNamespace(tpcs=[
        box(100.04, 200.0, -10.0, 10.0), box(0.0, 100.0, -10.0, 10.0),
        box(100.0, 200.01, -10.0, 10.0), box(0.0, 100.0, -10.0, 10.0)]))
    out = LightAnimation._tpc_outlines.func(fake)
    assert out == [(0.0, 100.0, -10.0, 10.0), (100.0, 200.0, -10.0, 10.0)]


# ---- summary: "first frame after t0" -----------------------------------------

@pytest.fixture(scope="module")
def ev():
    from pylario import EventFile
    return EventFile(ATMNU)[0]


def _first_quoted_ns(summary: str) -> int:
    tail = summary.split("first frame after t0 ", 1)[1]
    return int(tail.split(" ns", 1)[0])


@needs_atmnu
def test_summary_first_frame_is_the_one_starting_at_t0_on_the_log_clock(ev):
    from pylarevd import display_light
    # 6 frames: the log clock turns geometric at once (16, 64, 320 ns, ...),
    # so the second frame after t0 differs from the first
    a = display_light(ev, n_frames=6)
    e = a.edges
    k = int(np.flatnonzero(e == a.t0)[0])          # t0 is an edge
    want = round((e[k + 1] - e[k]) * 1000)
    assert want != round((e[k + 2] - e[k + 1]) * 1000)
    assert _first_quoted_ns(a.summary()) == want


@needs_atmnu
def test_summary_first_frame_is_the_one_holding_t0_on_the_linear_clock(ev):
    from pylarevd import display_light
    a = display_light(ev, n_frames=7, time_scale="linear")
    e = a.edges
    k = int(np.searchsorted(e, a.t0, side="right")) - 1
    assert e[k] <= a.t0 < e[k + 1]
    assert _first_quoted_ns(a.summary()) == round((e[k + 1] - e[k]) * 1000)


@needs_atmnu
def test_summary_first_frame_for_a_t0_outside_the_window(ev):
    from pylarevd import display_light
    a = display_light(ev, n_frames=5, time_scale="linear", t0=0.0,
                      window=(10.0, 20.0))
    assert _first_quoted_ns(a.summary()) == 2000


# ---- plotly charge-panel title -----------------------------------------------

def _charge_titles(a):
    fig = a.plotly_figure()
    return [ann.text for ann in fig.layout.annotations
            if ann.text and ann.text.startswith("charge:")]


@needs_atmnu
def test_plotly_charge_title_stacked_layout(ev):
    from pylarevd import display_charge_light
    a = display_charge_light(ev, n_frames=4)
    titles = _charge_titles(a)
    assert len(titles) == 1 and titles[0].endswith("+ light on the anode wall")


@needs_pdsp
def test_plotly_charge_title_side_by_side_layout():
    from pylario import EventFile

    from pylarevd import display_charge_light
    a = display_charge_light(EventFile(PDSP)[0], n_frames=4)
    assert a.walls.n_walls == 2
    titles = _charge_titles(a)
    assert len(titles) == 1 and titles[0].endswith("+ light on the anode wall")
