"""Tests for the time-resolved photon-detector display (pylarevd.light).

The synthetic tests need nothing but numpy. The data tests use the same
``PYLAREVD_ATMNU`` sample as the rest of the suite (a neutrino file with OpHits,
OpFlashes and their association) and skip without it.
"""

from __future__ import annotations

import os
import types

import numpy as np
import pytest

from pylarevd.light import (PhotonDetectorWalls, _log_clock, frame_edges,
                            hit_time_quantum, light_onset)

_DATA = os.environ.get("PYLAREVD_TEST_DATA", "")
ATMNU = os.environ.get(
    "PYLAREVD_ATMNU", os.path.join(_DATA, "atmnu_radio_reco.root") if _DATA else "")
needs_atmnu = pytest.mark.skipif(not os.path.exists(ATMNU),
                                 reason="neutrino sample absent")


# ---- frame clock -----------------------------------------------------------

def test_log_clock_starts_at_the_quantum_and_ends_on_the_span():
    q, span, n = 0.016, 10.0, 108
    e = _log_clock(q, span, n)
    assert len(e) == n + 1
    assert e[0] == 0.0 and e[-1] == pytest.approx(span)
    steps = np.diff(e)
    assert (steps > 0).all()
    # never finer than the data: a sub-quantum frame repeats the previous one
    assert steps.min() >= q * (1 - 1e-6) * (span / e[-1])
    # one quantum per frame at first, then geometric growth
    assert steps[:5] == pytest.approx(q, rel=1e-3)
    assert steps[-1] > 10 * q
    assert (np.diff(steps) >= -1e-9).all()


def test_log_clock_falls_back_to_uniform_when_quanta_fill_the_span():
    e = _log_clock(0.1, 1.0, 20)
    assert np.allclose(np.diff(e), 0.05)


def test_frame_edges_log_puts_t0_on_an_edge_after_the_pre_roll():
    e = frame_edges(t0=5.0, lo=4.75, hi=15.0, n=120, scale="log", quantum=0.016)
    assert len(e) == 121
    assert (np.diff(e) > 0).all()
    assert e[0] == pytest.approx(4.75) and e[-1] == pytest.approx(15.0)
    k0 = int(np.argmin(np.abs(e - 5.0)))
    assert e[k0] == pytest.approx(5.0)
    assert k0 == 12                                   # 10 % of the frames
    assert e[k0 + 1] - e[k0] == pytest.approx(0.016, rel=1e-3)


def test_frame_edges_linear_is_uniform_and_rejects_bad_input():
    e = frame_edges(0.0, -1.0, 9.0, 10, "linear", 0.016)
    assert np.allclose(np.diff(e), 1.0)
    with pytest.raises(ValueError):
        frame_edges(0.0, 1.0, 1.0, 10, "linear", 0.016)
    with pytest.raises(ValueError):
        frame_edges(0.0, -1.0, 9.0, 10, "cubic", 0.016)


# ---- onset -----------------------------------------------------------------

def _synthetic_hits(seed=1, onset=5.008, q=0.016):
    rng = np.random.default_rng(seed)
    # radiological background: 1-2 PE hits over a 4.5 ms readout, on the grid
    tb = np.round(rng.uniform(-2250, 2250, 20000) / q) * q
    pb = rng.uniform(0.8, 2.0, len(tb))
    # an interaction: a prompt burst plus a slow tail, on the grid
    tp = onset + q * np.floor(rng.exponential(0.02, 3000) / q)
    tl = onset + q * np.floor(rng.exponential(1.5, 1000) / q)
    t = np.concatenate([tb, tp, tl]) + 0.0035          # grid phase, as in data
    pe = np.concatenate([pb, rng.uniform(1, 30, len(tp)), rng.uniform(0.8, 2, len(tl))])
    return t, pe


def test_hit_time_quantum_reads_the_grid():
    t, _ = _synthetic_hits()
    assert hit_time_quantum(t) == pytest.approx(0.016, abs=1e-6)


def test_light_onset_finds_the_burst_not_the_background():
    t, pe = _synthetic_hits(onset=5.008)            # on the 16 ns grid
    t0 = light_onset(t, pe, 0.016)
    # half a quantum before the first hit of the burst
    assert t0 == pytest.approx(5.008 + 0.0035 - 0.008, abs=1e-6)


def test_hit_time_quantum_ignores_a_stray_phase():
    t, _ = _synthetic_hits()
    odd = np.array([100.0115, 200.0115, 300.0115])  # half a quantum off-grid
    assert hit_time_quantum(np.concatenate([t, odd])) == pytest.approx(0.016, abs=1e-6)


def test_light_onset_handles_nothing():
    assert light_onset(np.array([]), np.array([])) is None
    assert light_onset(np.array([1.0]), np.array([0.0])) is None


# ---- walls -----------------------------------------------------------------

def _fake_geometry(xs, ny=4, nz=6, dup=1):
    rows = []
    for x in xs:
        for iy in range(ny):
            for iz in range(nz):
                for _ in range(dup):              # several channels per module
                    rows.append((x, -300 + 200 * iy, 50 + 100 * iz))
    return types.SimpleNamespace(has_optical=True, opdet_xyz=np.array(rows, float))


def test_walls_group_by_drift_position_and_merge_shared_channels():
    g = _fake_geometry([-727.0, -0.05, 0.05, 727.0], dup=2)
    w = PhotonDetectorWalls.from_geometry(g)
    # the two faces of the central APA (+-0.05 cm) are one wall
    assert w.n_walls == 3
    assert not w.projected
    # one tile per (wall, y, z): the central wall's two faces coincide here,
    # so its tiles carry both faces' channels instead of overlapping
    assert len(w) == 3 * 4 * 6
    n_ch = sorted({len(c) for c in w.channels})
    assert n_ch == [2, 4]
    assert np.bincount([len(c) for c in w.channels])[4] == 4 * 6
    assert (w.chan_tile >= 0).all()
    # a tile per module, tiles narrower than their pitch
    assert np.allclose(w.half_z, 0.47 * 100)
    assert np.allclose(w.half_y, 0.40 * 200)


def test_walls_project_when_detectors_are_not_on_planes():
    g = _fake_geometry(np.arange(10) * 100.0)       # a membrane-like spread
    w = PhotonDetectorWalls.from_geometry(g)
    assert w.projected and w.n_walls == 1


def test_walls_refuse_a_geometry_without_detectors():
    with pytest.raises(ValueError, match="photon-detector"):
        PhotonDetectorWalls.from_geometry(types.SimpleNamespace(has_optical=False))


# ---- data ------------------------------------------------------------------

@pytest.fixture(scope="module")
def ev():
    from pylario import EventFile
    return EventFile(ATMNU)[0]


@needs_atmnu
def test_animation_accounts_for_every_pe(ev):
    from pylarevd import display_light
    a = display_light(ev, n_frames=40)
    t, pe, tile, _fl = a._hits
    inside = (a.frame_of(t) >= 0) & (tile >= 0)
    assert a.arrivals.sum() == pytest.approx(pe[inside].sum())
    assert len(a.edges) == 41 and (np.diff(a.edges) > 0).all()
    # cumulative fill ends where the whole-window map is
    c = display_light(ev, n_frames=40, fill="cumulative")
    assert np.allclose(c.values[-1], c.collected)
    # afterglow never exceeds the cumulative total
    assert (a.values <= c.values + 1e-9).all()
    lo, hi = a.pe_range
    assert 0 < lo < hi
    assert "t0 =" in a.summary()


@needs_atmnu
def test_flash_hit_links_close_on_pe(ev):
    from pylarevd import display_light
    a = display_light(ev, "opflash", n_frames=10)
    links, note = a._links
    if links is None:
        pytest.skip(f"no usable association: {note}")
    fl = np.flatnonzero(a.op.is_flash)
    _t, pe, _tile, hfl = a._hits
    sums = np.bincount(hfl[hfl >= 0], weights=pe[hfl >= 0], minlength=len(fl))
    assert np.allclose(sums, np.asarray(a.op.pe)[fl], rtol=1e-3, atol=0.05)


@needs_atmnu
def test_repair_reproduces_stored_flash_centres(ev):
    """Recomputing (y, z) from fPEperOpDet must give back what opflash stored.

    That pins the channel indexing of fPEperOpDet and the weighting, which is
    what the repair relies on for producers (opslicer) that store garbage.
    """
    from dataclasses import replace
    from pylarevd.light import repair_flash_positions
    o = ev.optical("opflash")
    fl = np.asarray(o.is_flash)
    good = fl & (np.asarray(o.y_width) > 0) & (np.asarray(o.z_width) > 0)
    broken = replace(o, y_width=np.where(fl, -1.0, o.y_width))
    fixed, n = repair_flash_positions(ev, broken)
    assert n == int(fl.sum())
    assert np.allclose(np.asarray(fixed.y)[good], np.asarray(o.y)[good], atol=1e-6)
    assert np.allclose(np.asarray(fixed.z)[good], np.asarray(o.z)[good], atol=1e-6)
    # untouched when nothing is broken
    same, n0 = repair_flash_positions(ev, o)
    assert n0 == 0 and same is o


@needs_atmnu
def test_plotly_frames_keep_their_trace_types(ev):
    """A typeless frame entry becomes a scatter and collapses the bars."""
    from pylarevd import display_light
    fig = display_light(ev, n_frames=8).plotly_figure()
    assert len(fig.frames) == 8
    for fr in fig.frames:
        for i, d in zip(fr.traces, fr.data):
            assert d.type == fig.data[i].type
    assert fig.layout.sliders and fig.layout.updatemenus


@needs_atmnu
def test_gif_has_one_frame_per_tick_and_holds_the_last(ev, tmp_path):
    from PIL import Image
    from pylarevd import display_light
    path = display_light(ev, n_frames=6).save_animation(
        str(tmp_path / "l.gif"), fps=10, dpi=50, hold=1.5)
    im = Image.open(path)
    assert im.n_frames == 6
    im.seek(5)
    assert im.info["duration"] == 1500
    with pytest.raises(ValueError):
        display_light(ev, n_frames=6).save_animation(str(tmp_path / "l.webm"))


@needs_atmnu
def test_static_frames_render(ev, tmp_path):
    from pylarevd import display_light
    a = display_light(ev, n_frames=10)
    assert os.path.getsize(a.save(str(tmp_path / "all.png"), dpi=60)) > 10_000
    assert os.path.getsize(a.save(str(tmp_path / "f3.png"), dpi=60, frame=3)) > 10_000


@needs_atmnu
def test_flash3d_plotly_takes_height(ev):
    """The browser passes height=None to every view; this used to TypeError."""
    from pylarevd import display_flashes_3d
    assert display_flashes_3d(ev).plotly_figure(height=None) is not None


@needs_atmnu
def test_app_renders_the_light_view():
    pytest.importorskip("dash")
    from pylario import EventFile
    from pylarevd.app import render
    files = {ATMNU: EventFile(ATMNU)}
    fig, summary = render(files, ATMNU, 0, None, "integral", "auto", "light",
                          [], light_scale="linear", light_fill="cumulative")
    assert not summary.startswith("ERROR"), summary
    assert len(fig.frames) > 0
    fig, summary = render(files, ATMNU, 0, None, "integral", "auto", "flash3d", [])
    assert not summary.startswith("ERROR"), summary


@needs_atmnu
def test_cli_light_mode_writes_an_animation(tmp_path):
    from pylarevd.cli import main
    out = tmp_path / "out"
    assert main([ATMNU, "-e", "0", "--mode", "light", "--frames", "5",
                 "-o", str(out)]) == 0
    names = os.listdir(out)
    assert any(n.endswith("_light.gif") for n in names)
    assert any(n.endswith(".png") for n in names)
    assert "_light.gif" in (out / "manifest.csv").read_text()


# ---- charge + light ----------------------------------------------------------

@needs_atmnu
def test_charge_light_figure_has_a_3d_panel_and_typed_frames(ev):
    from pylarevd import display_charge_light
    a = display_charge_light(ev, n_frames=6)
    assert "space points" in a.summary()
    fig = a.plotly_figure()
    kinds = {d.type for d in fig.data}
    assert {"scatter3d", "mesh3d", "bar"} <= kinds
    for fr in fig.frames:
        for i, d in zip(fr.traces, fr.data):
            assert d.type == fig.data[i].type
    # the lit wall is coloured by number on the tile colour scale
    mesh = next(d for d in fig.data if d.type == "mesh3d" and d.intensity is not None)
    assert mesh.intensitymode == "cell"
    assert len(mesh.intensity) == len(mesh.i)


@needs_atmnu
def test_charge_light_frames_render_and_blit_like_a_full_draw(ev, tmp_path):
    from pylarevd import display_charge_light
    a = display_charge_light(ev, n_frames=6)
    assert os.path.getsize(a.save(str(tmp_path / "cl.png"), dpi=50, frame=3)) > 10_000
    blit = a.render_frames(dpi=50, frames=[3])[0]
    assert blit.ndim == 3 and blit.shape[2] == 3 and blit.std() > 5


@needs_atmnu
def test_app_renders_charge_and_light():
    pytest.importorskip("dash")
    from pylario import EventFile
    from pylarevd.app import render
    files = {ATMNU: EventFile(ATMNU)}
    fig, summary = render(files, ATMNU, 0, None, "integral", "auto",
                          "chargelight", [], reco=["tracks"])
    assert not summary.startswith("ERROR"), summary
    assert any(d.type == "mesh3d" for d in fig.data) and len(fig.frames) > 0
