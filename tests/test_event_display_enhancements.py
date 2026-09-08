"""Tests for the 8 event display enhancements."""

import io
import os
import numpy as np
import pytest

from pylarevd import EventFile
from pylarevd.app import build_app, render
from pylarevd.display import Display3D

SAMPLE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "samples")
ATMNU_PATH = os.path.join(SAMPLE_DIR, "atmnu_max_weighted_HD_w19433s1_job000100_20260829T125152Z_reco2.root")
NDK_PATH = os.path.join(SAMPLE_DIR, "ndk_n_to_eminus_Kplus_HD_w19409s1_job000001_20260828T150556Z_reco2.root")
MUSUN_PATH = os.path.join(SAMPLE_DIR, "musun_HD_w19435s1_job003240_20260829T170934Z_reco2.root")

needs_ndk = pytest.mark.skipif(not os.path.exists(NDK_PATH), reason="NDK sample file absent")
needs_atmnu = pytest.mark.skipif(not os.path.exists(ATMNU_PATH), reason="Atmospheric neutrino sample file absent")
needs_musun = pytest.mark.skipif(not os.path.exists(MUSUN_PATH), reason="Cosmic sample file absent")
needs_all = pytest.mark.skipif(
    not (os.path.exists(NDK_PATH) and os.path.exists(ATMNU_PATH) and os.path.exists(MUSUN_PATH)),
    reason="All sample files required"
)


@needs_all
def test_true_vertex_universal():
    """True vertex should be available for neutrino, NDK, and cosmic events."""
    ev_atmnu = EventFile(ATMNU_PATH)[0]
    tv_atmnu = ev_atmnu.true_vertex()
    assert tv_atmnu is not None
    assert len(tv_atmnu) == 3
    assert np.isfinite(tv_atmnu).all()

    ev_ndk = EventFile(NDK_PATH)[0]
    tv_ndk = ev_ndk.true_vertex()
    assert tv_ndk is not None
    assert len(tv_ndk) == 3
    assert np.isfinite(tv_ndk).all()

    ti_ndk = ev_ndk.true_interaction()
    assert ti_ndk is not None
    assert "n -> e⁻ + K⁺" in ti_ndk.headline() or "n -> e- + K+" in ti_ndk.headline()
    assert ti_ndk.latex in ("$n \\to e^- + K^+$", "$n \\to e^- K^+$")

    ev_musun = EventFile(MUSUN_PATH)[0]
    tv_musun = ev_musun.true_vertex()
    assert tv_musun is not None
    assert len(tv_musun) == 3
    assert np.isfinite(tv_musun).all()


@needs_ndk
def test_spacepoints_physics_quantities():
    """SpacePoints dataclass should provide hit physics quantities."""
    ev = EventFile(NDK_PATH)[0]
    sp = ev.spacepoints()
    assert hasattr(sp, "amplitude")
    assert hasattr(sp, "tick")
    assert hasattr(sp, "multiplicity")
    assert len(sp.amplitude) == len(sp)
    assert len(sp.tick) == len(sp)
    assert len(sp.multiplicity) == len(sp)


@needs_ndk
def test_display3d_colour_options_and_vertex():
    """Display3D must support all physics colour quantities and show true vertex with LaTeX decay."""
    ev = EventFile(NDK_PATH)[0]
    for cb in ("integral", "amplitude", "tick", "multiplicity", "track", "x", "y", "z"):
        d3 = ev.display_3d(colour_by=cb, colour_scale="auto")
        fig = d3.plotly_figure()
        trace_names = [tr.name for tr in fig.data]
        assert "space points" in trace_names
        assert "true vertex" in trace_names
        assert fig.layout.uirevision is not None
        tv_trace = [tr for tr in fig.data if tr.name == "true vertex"][0]
        assert "n" in tv_trace.hovertemplate and "K" in tv_trace.hovertemplate

    # Check vertex roles according to user's approach
    # Event 10 (entry 9)
    ev10 = EventFile(NDK_PATH)[9]
    d3_ev10 = ev10.display_3d(tracks=True)
    fig_ev10 = d3_ev10.plotly_figure()
    ev10_names = [tr.name for tr in fig_ev10.data]
    assert "pandora interaction vertex" in ev10_names
    assert "primary daughter vertices" in ev10_names
    assert "secondary vertices" in ev10_names

    pan_tr = [tr for tr in fig_ev10.data if tr.name == "pandora interaction vertex"][0]
    dau_tr = [tr for tr in fig_ev10.data if tr.name == "primary daughter vertices"][0]
    sec_tr = [tr for tr in fig_ev10.data if tr.name == "secondary vertices"][0]

    assert pan_tr.marker.symbol == "x"
    assert dau_tr.marker.symbol == "diamond"
    assert sec_tr.marker.symbol == "circle"

    # Test rotating GIF generation
    gif_bytes = d3_ev10.rotate_gif(elev=20.0, start_azim=0.0, step=60, fps=10)
    assert len(gif_bytes) > 1000
    assert gif_bytes[:4] == b"GIF8"


@needs_ndk
def test_display3d_particle_symbols(tmp_path):
    """Display3D should display particle symbols for matched tracks and showers."""
    ev = EventFile(NDK_PATH)[0]
    d3 = ev.display_3d(truth=True, tracks=True, particle_symbols=True)
    fig = d3.plotly_figure()
    trace_names = [tr.name for tr in fig.data]
    assert "particle symbols" in trace_names
    sym_trace = [tr for tr in fig.data if tr.name == "particle symbols"][0]
    assert any("K" in txt for txt in sym_trace.text)
    assert any("e" in txt for txt in sym_trace.text)

    # Matplotlib PDF save with particle symbols
    pdf_file = tmp_path / "test_syms.pdf"
    d3.save(str(pdf_file), elev=25.0, azim=-45.0, particle_symbols=True)
    assert pdf_file.exists()
    assert pdf_file.stat().st_size > 5000


@needs_ndk
def test_display3d_figure_and_pdf_save(tmp_path):
    """Display3D.save should export vector PDF matching custom elev/azim perspective."""
    ev = EventFile(NDK_PATH)[0]
    d3 = ev.display_3d(colour_by="integral")
    pdf_file = tmp_path / "test_snapshot.pdf"
    d3.save(str(pdf_file), elev=30.0, azim=45.0)
    assert pdf_file.exists()
    assert pdf_file.stat().st_size > 5000


@needs_ndk
def test_app_callbacks_and_snapshot():
    """Dash app should support fixing perspective, particle symbols, and PDF snapshots."""
    app = build_app([NDK_PATH])
    assert app is not None

    # Test PDF snapshot callback
    snap_fn = app.callback_map["download-snapshot.data"]["callback"].__wrapped__
    relayout = {"scene.camera": {"up": {"x": 0, "y": 0, "z": 1}, "center": {"x": 0, "y": 0, "z": 0}, "eye": {"x": 1.5, "y": -1.2, "z": 0.8}}}
    res = snap_fn(1, NDK_PATH, 0, None, "integral", "auto", "3d", ["on"], "orientation", ["on"], "dark", "cividis", ["on"], True, relayout, None, None)
    assert res is not None
    assert res.get("type") == "application/pdf"
    assert res.get("filename").endswith(".pdf")
    assert len(res.get("content")) > 1000

    # Test particle symbols toggle
    toggle_fn = app.callback_map["..show-symbols.data...particle-symbols.children...particle-symbols.style.."]["callback"].__wrapped__
    active, label, style = toggle_fn(1, False)
    assert active is True
    assert "particle symbols" in label

    # Check index_string CSS contains dropdown contrast styling
    assert ".Select-control" in app.index_string
    assert "#1e293b" in app.index_string
    assert "[data-theme=\"light\"]" in app.index_string


@needs_ndk
def test_hiding_elements_and_pandora_vertex():
    """Verify granular element visibility and hiding pandora interaction vertex."""
    from pylarevd.app import render, _parse_reco
    ev = EventFile(NDK_PATH)[9]
    files = {NDK_PATH: EventFile(NDK_PATH)}

    # Parse reco helper
    assert _parse_reco(["tracks", "vertices"]) == (True, True, False)
    assert _parse_reco(["tracks", "vertices", "pandora_vtx"]) == (True, True, True)
    assert _parse_reco(["on"]) == (True, True, True)
    assert _parse_reco([]) == (False, False, False)

    # 1. 3D with pandora_vertex=False
    d3_no_pan = ev.display_3d(tracks=True, pandora_vertex=False)
    fig3_no_pan = d3_no_pan.plotly_figure()
    names3_no_pan = [tr.name for tr in fig3_no_pan.data]
    assert "pandora interaction vertex" not in names3_no_pan
    assert "primary daughter vertices" in names3_no_pan
    assert "reco tracks" in names3_no_pan

    # 2. 3D with pandora_vertex=True
    d3_pan = ev.display_3d(tracks=True, pandora_vertex=True)
    fig3_pan = d3_pan.plotly_figure()
    names3_pan = [tr.name for tr in fig3_pan.data]
    assert "pandora interaction vertex" in names3_pan

    # 3. Matplotlib 3D figure
    fig_mpl_no_pan = d3_no_pan.figure()
    leg_texts = [t.get_text() for t in fig_mpl_no_pan.axes[0].get_legend().get_texts()]
    assert "pandora interaction vertex" not in leg_texts
    assert "primary daughter vertices" in leg_texts

    # 4. App render with reco=["tracks", "vertices"]
    fig_render_no_pan, _ = render(files, NDK_PATH, 9, None, "integral", "auto", "3d", truth=[], reco=["tracks", "vertices"])
    render_names = [tr.name for tr in fig_render_no_pan.data]
    assert "pandora interaction vertex" not in render_names
    assert "primary daughter vertices" in render_names
    assert "reco tracks" in render_names

    # 5. 2D display with pandora_vertex=False
    d2_no_pan = ev.display(tag=None, reco=True, pandora_vertex=False)
    fig2_no_pan = d2_no_pan.plotly_figure()
    names2_no_pan = [tr.name for tr in fig2_no_pan.data]
    assert "pandora interaction vertex" not in names2_no_pan


@needs_ndk
def test_event_display_2d_particle_symbols(tmp_path):
    """EventDisplay (2D) should project and render particle symbols in Plotly and Matplotlib."""
    ev = EventFile(NDK_PATH)[9]
    d2 = ev.display(reco=True, particle_symbols=True)

    # 1. Plotly 2D
    fig_plotly = d2.plotly_figure()
    sym_traces = [tr for tr in fig_plotly.data if tr.name == "particle symbols"]
    assert len(sym_traces) > 0
    all_texts = [txt for tr in sym_traces for txt in (tr.text if hasattr(tr, "text") and tr.text is not None else [])]
    assert any("K" in t for t in all_texts)
    assert any("μ" in t or "mu" in t or "e" in t for t in all_texts)

    # 2. Matplotlib 2D
    fig_mpl = d2.figure(particle_symbols=True)
    texts_mpl = [t.get_text() for ax in fig_mpl.axes for t in ax.texts]
    assert any("K" in t for t in texts_mpl)

    # 3. App render callback in 2D mode
    files = {NDK_PATH: EventFile(NDK_PATH)}
    fig_render, _ = render(
        files, NDK_PATH, 9, None, "integral", "auto", "2d",
        truth=[], reco=["tracks", "vertices"], particle_symbols=True
    )
    assert any(tr.name == "particle symbols" for tr in fig_render.data)



