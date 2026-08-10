"""Regression tests for pylarevd.

These pin the facts that were established byte-by-byte against dunesw output.
If a future art/LArSoft version changes a layout, these fail loudly rather than
silently producing a plausible but wrong picture.

Run:  python -m pytest tests/ -v      (needs the LCG view sourced)
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from pylarevd import EventFile
from pylarevd.artio import ArtFile, ArtReadError
from pylarevd.geometry import Geometry, GeometryError

# Sample files are not distributed with the code. Point the suite at your own
# with PYLAREVD_TEST_DATA (a directory holding the two files below), or set
# either path explicitly. Tests needing them skip when they are absent.
_DATA = os.environ.get("PYLAREVD_TEST_DATA", "")
ROCKMU = os.environ.get(
    "PYLAREVD_ROCKMU", os.path.join(_DATA, "rock_muons_reco1.root") if _DATA else "")
ATMNU = os.environ.get(
    "PYLAREVD_ATMNU", os.path.join(_DATA, "atmnu_radio_reco.root") if _DATA else "")
GEOM = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "pylarevd", "geom", "dune10kt_v6_1x2x6.npz")

needs_data = pytest.mark.skipif(not os.path.exists(ROCKMU), reason="sample file absent")
needs_geom = pytest.mark.skipif(not os.path.exists(GEOM), reason="geometry absent")


# ---- layout: derived from streamer info, verified against real bytes -------

@needs_data
@pytest.mark.parametrize("classname,itemsize", [
    ("recob::Hit", 109),           # 68 flat + 41 for the nested geo::WireID
    ("sim::SimEnergyDeposit", 132),  # two ROOT::Math positions at 44 B each
    ("recob::SpacePoint", 44),     # Double32_t arrays: 4 + 12 + 24 + 4
])
def test_record_itemsize(classname, itemsize):
    assert ArtFile(ROCKMU).layout(classname).itemsize == itemsize


@needs_data
def test_wireid_nested_headers():
    """geo::WireID is 41 B: four 6-byte class headers + bool + four uints."""
    lay = ArtFile(ROCKMU).layout("recob::Hit")
    blk = next(b for b in lay.blocks if b.name == "fWireID")
    assert blk.itemsize == 41
    assert not blk.is_flat
    assert [c.name for c in blk.columns] == [
        "fWireID.isValid", "fWireID.Cryostat", "fWireID.TPC",
        "fWireID.Plane", "fWireID.Wire"]


# ---- decoding --------------------------------------------------------------

@needs_data
def test_hits_decode_known_values():
    """First hit of event 0, cross-checked against gallery/LArSoft."""
    h = ArtFile(ROCKMU).read("recob::Hits_hitfd__Reco1.", "recob::Hit", 0)
    assert len(h["fChannel"]) == 6671
    assert h["fChannel"][0] == 20548
    assert h["fPeakTime"][0] == pytest.approx(2627.44, abs=0.01)
    assert h["fWireID.Wire"][0] == 868
    assert h["fWireID.Plane"][0] == 0
    assert h["fWireID.TPC"][0] == 16


@needs_data
def test_event_ids():
    ids = ArtFile(ROCKMU).event_ids()
    assert len(ids) == 10
    assert tuple(ids[0]) == (1, 0, 1)
    assert tuple(ids[9]) == (1, 0, 10)


# ---- geometry --------------------------------------------------------------

@needs_geom
def test_geometry_covers_all_channels():
    g = Geometry(GEOM)
    assert g.detector == "dune10kt_v6_1x2x6"
    assert g.nchannels == 30720
    assert len(np.unique(g.w_chan)) == 30720
    assert g.has_drift


@needs_data
@needs_geom
def test_channel_map_matches_reconstruction():
    """The decisive check: geometry's channel for each hit's WireID must equal
    the channel reconstruction stored on the hit."""
    f = EventFile(ROCKMU, geometry=GEOM)
    h = f[0].hits()
    rows = f.geometry.wire_rows(h.cryo, h.tpc, h.plane, h.wire)
    assert (rows >= 0).all()
    assert np.array_equal(f.geometry.w_chan[rows], h.channel)


@needs_data
@needs_geom
def test_hits_lie_on_truth():
    """Reconstructed hits must sit on the true energy depositions.

    A wire-wrapping or drift-conversion error puts a whole population metres
    away, so the tolerance here is deliberately far below the 0.467 cm pitch.
    """
    f = EventFile(ROCKMU, geometry=GEOM)
    ev = f[0]
    h, t = ev.hits(), ev.truth_deposits()
    g = f.geometry
    for key in np.unique(h.panel):
        m = (h.panel == key) & np.isfinite(h.w) & np.isfinite(h.x)
        if m.sum() < 10:
            continue
        row = g.plane_rows(h.cryo[m], h.tpc[m], h.plane[m])[0]
        p = g.plane_measure_dir[row]
        wt = p[0] * t.y + p[1] * t.z
        hw, hx = h.w[m], h.x[m]
        dmin = np.empty(len(hw))
        for i in range(0, len(hw), 2000):
            sl = slice(i, min(i + 2000, len(hw)))
            d = np.hypot(hw[sl, None] - wt[None, :], hx[sl, None] - t.x[None, :])
            dmin[sl] = d.min(axis=1)
        assert np.median(dmin) < 0.1, f"panel {key}: median {np.median(dmin):.3f} cm"


@needs_data
@needs_geom
def test_drift_and_panels():
    f = EventFile(ROCKMU, geometry=GEOM)
    g = f.geometry
    # 0.16 cm/us at 2 MHz sampling
    assert np.allclose(np.abs(np.unique(g.p_slope)), 0.0803, atol=1e-3)
    # both drift sides exist, and hits never mix them within one panel
    h = f[0].hits()
    for key in np.unique(h.panel):
        m = h.panel == key
        assert len(np.unique(h.drift_sign[m])) == 1
        assert len(np.unique(h.view[m])) == 1


# ---- end to end ------------------------------------------------------------

@needs_data
@needs_geom
@pytest.mark.parametrize("path", [ROCKMU, ATMNU])
def test_render(tmp_path, path):
    if not os.path.exists(path):
        pytest.skip("sample absent")
    f = EventFile(path, geometry=GEOM)
    d = f[0].display()
    assert d.panels
    png = d.save(str(tmp_path / "e.png"))
    html = d.save_html(str(tmp_path / "e.html"))
    assert os.path.getsize(png) > 10_000
    assert os.path.getsize(html) > 100_000


@needs_data
@needs_geom
def test_panel_groupings():
    """Each grouping yields the expected panels and loses no hits."""
    f = EventFile(ATMNU, geometry=GEOM)
    ev = f[0]
    by_orient = ev.display(merge="orientation")
    by_view = ev.display(merge="view")
    split = ev.display(merge="none")
    assert len(by_orient.panels) == 3      # two induction angles + collection
    assert len(by_view.panels) == 3        # U, V, Z
    assert len(split.panels) == 6          # every (view, drift volume)
    assert {p.title for p in by_view.panels} == {"U plane", "V plane", "Z plane"}
    counts = {sum(int(p.mask.sum()) for p in d.panels)
              for d in (by_orient, by_view, split)}
    assert len(counts) == 1, f"grouping changed the hit count: {counts}"


@needs_data
@needs_geom
def test_orientation_panels_contain_only_parallel_wires():
    """The property that makes orientation merging correct.

    Every hit in an orientation panel must share one measurement direction --
    that is what lets both drift volumes sit on a single w axis.  Merging by
    view label does *not* have this property for induction planes, which is
    exactly why the two give different pictures.
    """
    f = EventFile(ATMNU, geometry=GEOM)
    ev = f[0]
    g = f.geometry
    h = ev.hits()
    rows = g.plane_rows(h.cryo, h.tpc, h.plane)

    for p in ev.display(merge="orientation").panels:
        dirs = np.unique(np.round(g.plane_measure_dir[rows[p.mask]], 6), axis=0)
        assert len(dirs) == 1, f"{p.title} mixes {len(dirs)} wire directions"
        # and it really does span both drift volumes for this event
        assert len(np.unique(h.drift_sign[p.mask])) == 2

    # the contrast: merging by view mixes directions on the induction panels
    mixed = 0
    for p in ev.display(merge="view").panels:
        if len(np.unique(np.round(g.plane_measure_dir[rows[p.mask]], 6), axis=0)) > 1:
            mixed += 1
    assert mixed == 2, "expected both induction views to mix wire directions"


@needs_data
@needs_geom
def test_readout_view_needs_no_wireid():
    """The readout view keys off the channel, so it works without a WireID."""
    f = EventFile(ATMNU, geometry=GEOM)
    ev = f[0]
    d = ev.display(space="readout")
    assert len(d.panels) == 3
    assert all("readout" in p.title for p in d.panels)
    h = ev.hits()
    # every hit is placed, including any whose geometry lookup failed
    assert sum(int(p.mask.sum()) for p in d.panels) == int((h.chan_view >= 0).sum())


@needs_data
@needs_geom
def test_merged_truth_uses_per_face_frames():
    """Truth in a merged panel is projected per drift face, and must still land
    on the hits rather than smearing across the seam."""
    f = EventFile(ROCKMU, geometry=GEOM)
    d = f[0].display(truth=True)
    for p in d.panels:
        tw = d._truth_wx(p)
        assert tw is not None
        w, x = tw
        assert len(w) == len(x) and len(w) > 0
        assert np.isfinite(w).all() and np.isfinite(x).all()


@needs_data
@needs_geom
def test_hit_selection_roundtrip():
    f = EventFile(ROCKMU, geometry=GEOM)
    h = f[0].hits()
    sel = h[h.integral > 100]
    assert len(sel) < len(h)
    assert (sel.integral > 100).all()
    assert len(sel.w) == len(sel)


# ---- regressions from the interface audit ----------------------------------

@needs_geom
def test_invalid_wireid_is_rejected_not_wrapped():
    """LArSoft's unset-WireID sentinel must not resolve to a real wire.

    0xFFFFFFFF narrows to -1, which numpy would happily index from the end of
    the table, silently giving a plausible wire for a hit that has none.
    """
    g = Geometry(GEOM)
    one = lambda v: np.array([v])           # noqa: E731
    for args in [(one(-1), one(0), one(0), one(0)),
                 (one(0), one(-1), one(0), one(0)),
                 (one(0), one(0), one(-1), one(0)),
                 (one(0), one(0), one(0), one(-1))]:
        assert g.wire_rows(*args)[0] == -1
    assert g.plane_rows(one(-1), one(0), one(0))[0] == -1
    assert g.plane_rows(one(0), one(-1), one(0))[0] == -1
    # and a huge index must not raise
    assert g.wire_rows(one(0), one(0), one(0), one(10 ** 9))[0] == -1



def test_unreadable_file_raises_artreaderror():
    """Callers contract on ArtReadError; a raw OSError must not escape."""
    with pytest.raises(ArtReadError):
        ArtFile("/nonexistent/definitely-not-here.root")


@needs_data
@needs_geom
def test_hits_label_tracks_selection_and_rejects_scalars():
    f = EventFile(ROCKMU, geometry=GEOM)
    h = f[0].hits()
    sub = h[h.plane == 0]
    assert f"{len(sub)} hits" in sub.label
    assert f"{len(h)} hits" not in sub.label
    for bad in (0, np.array(True)):
        with pytest.raises(IndexError):
            h[bad]


@needs_data
@needs_geom
@pytest.mark.parametrize("kwargs", [
    {"merge": "bogus"}, {"space": "bogus"},
    {"colour_by": "bogus"}, {"colour_scale": "bogus"},
])
def test_bad_display_options_raise(kwargs):
    """A typo must not silently render a different view."""
    f = EventFile(ROCKMU, geometry=GEOM)
    with pytest.raises(ValueError):
        f[0].display(**kwargs)


@needs_data
@needs_geom
def test_log_colorbar_always_labelled():
    """Sub-decade ranges must still get readable tick labels."""
    f = EventFile(ROCKMU, geometry=GEOM)
    fig = f[0].display().plotly_figure()
    bar = [t for t in fig.data if getattr(t.marker, "showscale", False)][0].marker.colorbar
    assert len(bar.tickvals) >= 2
    assert len(bar.ticktext) == len(bar.tickvals)


@needs_data
@needs_geom
def test_app_survives_bad_input():
    """A stale entry or an unreadable file must not take the browser down."""
    from pylarevd.app import build_app, render
    app = build_app(["/nonexistent/file.root", ROCKMU])   # bad one skipped
    files = app._pylarevd_files
    assert len(files) == 1
    _, summary = render(files, ROCKMU, 999, "hitfd", "integral", "auto", "2d",
                        [], "orientation")
    assert summary.startswith("ERROR")


def test_cli_event_spec_rejects_nonsense():
    from pylarevd.cli import _parse_events
    assert _parse_events("2-5", 10) == [2, 3, 4, 5]
    assert _parse_events("all", 3) == [0, 1, 2]
    for bad in ("5-2", "-1", ""):
        with pytest.raises(ValueError):
            _parse_events(bad, 10)


def test_cli_never_overwrites(tmp_path):
    """Different files share run/subrun/event numbers; renders must not collide."""
    from pylarevd.cli import _unique_path
    taken: set[str] = set()
    a = _unique_path(str(tmp_path / "r1_s0_e1.png"), taken)
    b = _unique_path(str(tmp_path / "r1_s0_e1.png"), taken)
    assert a != b


@needs_data
@needs_geom
def test_display3d_options():
    f = EventFile(ATMNU, geometry=GEOM)
    d = f[0].display_3d(colour_by="charge")
    assert d.plotly_figure() is not None
    assert "space points" in d.summary()
    with pytest.raises(ValueError):
        f[0].display_3d(colour_by="integral")   # not a space-point quantity


@needs_data
@needs_geom
def test_figures_are_closed_by_save(tmp_path):
    import matplotlib.pyplot as plt
    f = EventFile(ROCKMU, geometry=GEOM)
    d = f[0].display()
    before = set(plt.get_fignums())
    for i in range(3):
        d.save(str(tmp_path / f"e{i}.png"))
    assert set(plt.get_fignums()) == before


# ---- variable-length products ----------------------------------------------

@needs_data
@pytest.mark.parametrize("classname", [
    "recob::PFParticle", "recob::OpFlash", "recob::Shower",
    "recob::Track", "recob::Wire", "recob::Vertex", "recob::OpHit",
])
def test_variable_length_products_decode(classname):
    """Products carrying STL containers go through the sequential reader."""
    if not os.path.exists(ATMNU):
        pytest.skip("sample absent")
    f = ArtFile(ATMNU)
    product = next((p for p in f.products() if p.startswith(classname + "s_")), None)
    if product is None:
        pytest.skip(f"{classname} absent")
    decoded = f.read(product, classname, 0)
    assert decoded and len(decoded[next(iter(decoded))]) > 0


@needs_data
@needs_geom
def test_track_trajectories_match_spacepoints():
    """The decisive check on trajectory decoding.

    Pandora builds one trajectory point per space point, so the valid points
    must both match in number and sit on top of them. A byte-level mistake in
    the nested-container parse would break one or the other.
    """
    f = EventFile(ATMNU, geometry=GEOM)
    ev = f[0]
    # best_match=False: this compares against the raw collection Pandora wrote,
    # one trajectory point per space point, before the duplicate fits are cut.
    tracks = ev.tracks(best_match=False)
    sp = ev.spacepoints().xyz
    total = sum(len(p) for p in tracks.points)
    assert total == len(sp), f"{total} trajectory points vs {len(sp)} space points"

    medians = []
    for xyz in tracks.points:
        if len(xyz) < 3:
            continue
        step = max(1, len(xyz) // 50)
        medians.append(np.median([np.linalg.norm(sp - q, axis=1).min()
                                  for q in xyz[::step]]))
    assert np.median(medians) < 0.5, f"median {np.median(medians):.3f} cm"


@needs_data
@needs_geom
def test_track_points_are_inside_the_detector():
    """Masked points are stored as -999 and must be dropped, not drawn."""
    f = EventFile(ATMNU, geometry=GEOM)
    g = f.geometry
    xlim = max(max(abs(b.xmin), abs(b.xmax)) for b in g.tpcs)
    zlim = max(b.zmax for b in g.tpcs)
    for xyz in f[0].tracks().points:
        if not len(xyz):
            continue
        assert np.abs(xyz[:, 0]).max() <= xlim + 1
        assert -1 <= xyz[:, 2].min() and xyz[:, 2].max() <= zlim + 1


@needs_data
@needs_geom
def test_vertex_matches_track_start():
    f = EventFile(ATMNU, geometry=GEOM)
    ev = f[0]
    v = ev.vertices()
    starts = np.array([p[0] for p in ev.tracks(best_match=False).points if len(p)])
    assert len(v) > 0
    assert np.linalg.norm(starts - v.xyz[0], axis=1).min() < 2.0


@needs_data
def test_optical_activity():
    if not os.path.exists(ATMNU):
        pytest.skip("sample absent")
    f = EventFile(ATMNU, geometry=GEOM)
    o = f[0].optical()
    assert o.is_flash.sum() > 0
    assert (~o.is_flash).sum() > 0          # hits too
    assert np.isfinite(o.time).all()
    assert np.isfinite(o.y[o.is_flash]).all()


@needs_data
@needs_geom
def test_reco_overlay_renders(tmp_path):
    f = EventFile(ATMNU, geometry=GEOM)
    d = f[0].display(reco=True)
    assert d.tracks is not None and len(d.tracks) > 0
    for panel in d.panels:
        assert d._track_wx(panel)            # every panel gets projected tracks
    assert os.path.getsize(d.save(str(tmp_path / "reco.png"))) > 10_000


# ---- reco and truth objects -------------------------------------------------

@needs_data
@needs_geom
def test_mc_trajectories():
    """True trajectories must decode and start where the particle entered."""
    f = EventFile(ROCKMU, geometry=GEOM)
    mc = f[0].mc_particles()
    assert len(mc) > 0
    primaries = mc.primaries()
    assert len(primaries) == 1                      # one rock muon
    assert primaries.pdg[0] == 13                   # mu-
    track = primaries.points[0]
    assert len(track) > 50
    # a rock muon enters through the detector boundary
    g = f.geometry
    ylim = max(b.ymax for b in g.tpcs)
    assert abs(abs(track[0][1]) - ylim) < 5.0


@needs_data
@needs_geom
def test_mc_trajectory_follows_the_hits():
    """The true trajectory must lie on the reconstructed hits."""
    f = EventFile(ROCKMU, geometry=GEOM)
    ev = f[0]
    d = ev.display(truth=True)
    assert d.mc is not None and len(d.mc) > 0
    for panel in d.panels:
        projected = d._mc_wx(panel)
        assert projected, f"no true trajectory projected into {panel.title}"
        w, x = max(projected, key=lambda p: np.isfinite(p[0]).sum())
        # polylines carry NaN where a trajectory gap was deliberately broken
        finite = np.isfinite(w) & np.isfinite(x)
        w, x = w[finite], x[finite]
        hw, hx = d.hits.w[panel.mask], d.hits.x[panel.mask]
        step = max(1, len(w) // 40)
        dmin = [np.hypot(hw - a, hx - b).min() for a, b in zip(w[::step], x[::step])]
        assert np.median(dmin) < 3.0, f"{panel.title}: median {np.median(dmin):.2f} cm"


@needs_data
@needs_geom
def test_showers_decode():
    f = EventFile(ATMNU, geometry=GEOM)
    sh = f[0].showers()
    assert len(sh) > 0
    assert sh.start.shape[1] == 3 and sh.direction.shape[1] == 3
    assert np.isfinite(sh.length).all()
    # shower axes are unit vectors
    assert np.allclose(np.linalg.norm(sh.direction, axis=1), 1.0, atol=1e-3)


@needs_data
@needs_geom
def test_optical_flash_agrees_with_tpc_vertex():
    """The brightest flash should coincide with the interaction, in time and z.

    Light arrives in nanoseconds while charge drifts for milliseconds, so this
    is an independent handle on the same event -- and a strong check that the
    optical products decoded correctly.
    """
    f = EventFile(ATMNU, geometry=GEOM)
    ev = f[0]
    o = ev.optical()
    flash = o.is_flash
    brightest = int(np.argmax(np.where(flash, o.pe, -1)))
    assert abs(o.time[brightest]) < 5.0                    # at trigger time
    assert o.pe[brightest] > 100 * np.median(o.pe[flash])  # well above background
    assert abs(o.z[brightest] - ev.vertices().xyz[0][2]) < 150.0


@needs_data
@needs_geom
def test_optical_and_3d_render(tmp_path):
    f = EventFile(ATMNU, geometry=GEOM)
    ev = f[0]
    assert os.path.getsize(ev.display_optical().save(str(tmp_path / "o.png"))) > 10_000
    d3 = ev.display_3d()
    assert d3.tracks is not None and len(d3.tracks) > 0
    assert os.path.getsize(d3.save(str(tmp_path / "3d.png"))) > 10_000


# ---- associations ----------------------------------------------------------

@needs_data
def test_assns_decode_and_index_correctly():
    """Association keys must be valid indices into both collections."""
    if not os.path.exists(ATMNU):
        pytest.skip("sample absent")
    f = ArtFile(ATMNU)
    product = next(p for p in f.assns_products("recob::Hit", "recob::Track")
                   if "pandoraTrack" in p)
    a = f.read_assns(product, 0)
    n_hits = len(f.read("recob::Hits_hitfd__Reco1.", "recob::Hit", 0)["fChannel"])
    assert len(a) > 0
    assert a.left_key.min() >= 0 and a.left_key.max() < n_hits
    assert a.right_key.min() >= 0
    # one ProductID per side: a single source collection each
    assert len(np.unique(a.left_id)) == 1
    assert len(np.unique(a.right_id)) == 1


@needs_data
@needs_geom
def test_assns_agree_with_track_trajectories():
    """Pandora emits one trajectory point per associated hit.

    Comparing the two is a genuine cross-check: they come from different
    products, decoded by different code paths (numpy strides vs the sequential
    reader), so agreement is strong evidence both are right.
    """
    f = EventFile(ATMNU, geometry=GEOM)
    ev = f[0]
    groups = ev.hit_group("track")
    per_track = np.bincount(groups[groups >= 0])
    # hit_group indices refer to the full collection, so compare against it
    traj_total = [len(t) for t in ev.tracks(best_match=False).points]
    # trajectories drop masked (-999) points, so they are a subset
    assert len(per_track) == len(traj_total)
    assert (per_track >= np.array(traj_total)).all()
    assert per_track.sum() == len(
        ArtFile(ATMNU).read_assns(
            next(p for p in ArtFile(ATMNU).assns_products("recob::Hit", "recob::Track")
                 if "pandoraTrack" in p), 0))


@needs_data
@needs_geom
@pytest.mark.parametrize("kind", ["track", "shower", "slice", "cluster", "pfparticle"])
def test_hit_group_routes(kind):
    f = EventFile(ATMNU, geometry=GEOM)
    ev = f[0]
    g = ev.hit_group(kind)
    assert len(g) == len(ev.hits())
    assert (g >= -1).all()
    assert (g >= 0).any(), f"no hit associated to any {kind}"


@needs_data
@needs_geom
def test_hit_group_rejects_unknown_kind():
    f = EventFile(ATMNU, geometry=GEOM)
    with pytest.raises(ValueError):
        f[0].hit_group("nonsense")


@needs_data
@needs_geom
def test_categorical_colouring(tmp_path):
    f = EventFile(ATMNU, geometry=GEOM)
    d = f[0].display(colour_by="track")
    assert d.is_categorical and not d.use_log
    colours = d.group_colours()
    assert len(colours) == len(d.hits)
    assert (colours == d.theme.unassociated).any()   # radiologicals own nothing
    assert len(set(colours)) > 2                     # several objects coloured
    assert os.path.getsize(d.save(str(tmp_path / "c.png"))) > 10_000


# ---- regressions from the second review -------------------------------------

@needs_data
def test_wire_waveforms_are_actually_recovered():
    """The count of decoded objects is not evidence the data came out.

    Every wire used to "decode" while silently reporting no signal: only the
    empty ones parsed. Assert on content, and on an invariant the file itself
    must satisfy.
    """
    from pylarevd.streamers import UNPARSED
    f = ArtFile(ROCKMU)
    wires = f.read("recob::Wires_tpcrawdecoder_gauss_DetSim.", "recob::Wire", 0)
    roi = wires["fSignalROI"]
    assert not any(UNPARSED in r for r in roi), "some wires failed to decode"
    rois = [x for r in roi for x in (r.get("ranges") or [])]
    assert len(rois) > 1000, f"only {len(rois)} regions of interest recovered"
    for x in rois:
        assert x["last"] - x["offset"] == len(x["values"])


@needs_data
@pytest.mark.parametrize("classname", [
    "sim::SimChannel", "sim::SimPhotonsLite", "sim::OpDetBacktrackerRecord",
])
def test_container_heavy_products_decode(classname):
    f = ArtFile(ROCKMU)
    product = next((p for p in f.products() if p.startswith(classname + "s_")), None)
    if product is None:
        pytest.skip(f"{classname} absent")
    decoded = f.read(product, classname, 0)
    assert len(decoded[next(iter(decoded))]) > 0


@needs_data
@needs_geom
def test_polylines_never_leap_across_the_detector():
    """A projected track must not draw a segment longer than a real step."""
    f = EventFile(ATMNU, geometry=GEOM)
    d = f[0].display(reco=True)
    for panel in d.panels:
        for w, x in d._track_wx(panel) + d._mc_wx(panel):
            step = np.hypot(np.diff(w), np.diff(x))
            step = step[np.isfinite(step)]
            if len(step):
                assert step.max() < 50.0, f"{panel.title}: {step.max():.0f} cm segment"


@needs_data
@needs_geom
def test_shower_cones_have_width_in_every_view():
    """A cone built from one arbitrary perpendicular collapses to zero width in
    the collection view, because that perpendicular has no z component. The
    real silhouette does not.

    Judged on the median, since a shower with a genuinely tiny opening angle
    (0.0008 rad over 11 cm here) legitimately projects to almost nothing.
    """
    f = EventFile(ATMNU, geometry=GEOM)
    d = f[0].display(reco=True)
    for panel in d.panels:
        widths = [w.max() - w.min() for w, x in d._shower_wx(panel)]
        assert widths, f"{panel.title}: no cones drawn"
        assert np.median(widths) > 1.0, f"{panel.title}: cones are flat"


@needs_data
@needs_geom
def test_shower_directions_are_unit_vectors_in_every_event():
    """Failed Pandora fits are -999 placeholders and must be dropped."""
    f = EventFile(ATMNU, geometry=GEOM)
    for i in range(len(f)):
        sh = f[i].showers()
        if len(sh):
            assert np.allclose(np.linalg.norm(sh.direction, axis=1), 1.0, atol=1e-3)


@needs_geom
def test_malformed_geometry_raises_geometryerror(tmp_path):
    bad = tmp_path / "bad.npz"
    np.savez(bad, foo=np.arange(3))
    with pytest.raises(GeometryError):
        Geometry(str(bad))


@needs_data
@needs_geom
def test_hit_group_refuses_a_mismatched_tag():
    """Associations index one specific hit collection; a tag cannot change it."""
    f = EventFile(ATMNU, geometry=GEOM)
    with pytest.raises(ValueError):
        f[0].hit_group("track", tag="gaushit")


@needs_data
def test_streamer_caches_are_per_file():
    """A partial-parse decision for one file must not bind another."""
    from pylarevd import streamers as st
    st.reset_caches()
    a, b = ArtFile(ROCKMU), ArtFile(ROCKMU)
    a.read("recob::Wires_tpcrawdecoder_gauss_DetSim.", "recob::Wire", 0)
    assert st._caches(a._file.file) is not st._caches(b._file.file)


@needs_data
@needs_geom
def test_3d_polylines_are_broken_at_gaps():
    """The 3-D view must break trajectory gaps too.

    The gap-breaking originally lived only in the 2-D path, so 3-D drew
    straight lines up to 181 cm long through detector the particle never
    crossed.
    """
    from pylarevd.display import _split_polyline
    f = EventFile(ATMNU, geometry=GEOM)
    ev = f[0]
    for xyz in ev.tracks().points:
        if len(xyz) < 2:
            continue
        broken = _split_polyline(xyz)
        step = np.linalg.norm(np.diff(broken, axis=0), axis=1)
        step = step[np.isfinite(step)]
        if len(step):
            assert step.max() < 50.0, f"{step.max():.0f} cm segment in 3-D"

    fig = ev.display_3d().plotly_figure()
    for trace in [t for t in fig.data if t.name == "reco tracks"]:
        xs = np.asarray(trace.x, float)
        ys = np.asarray(trace.y, float)
        zs = np.asarray(trace.z, float)
        step = np.linalg.norm(np.diff(np.column_stack([xs, ys, zs]), axis=0), axis=1)
        step = step[np.isfinite(step)]
        if len(step):
            assert step.max() < 50.0


@needs_data
@needs_geom
def test_3d_focus_frames_the_event():
    """Framing the whole cryostat renders a localised event as a few pixels."""
    f = EventFile(ATMNU, geometry=GEOM)
    ev = f[0]
    data = ev.display_3d(focus="data")._limits()
    detector = ev.display_3d(focus="detector")._limits()
    for axis in range(3):
        span_data = data[axis][1] - data[axis][0]
        span_det = detector[axis][1] - detector[axis][0]
        assert span_data <= span_det
    assert (data[2][1] - data[2][0]) < 0.5 * (detector[2][1] - detector[2][0])
    with pytest.raises(ValueError):
        ev.display_3d(focus="bogus")


# ---- theme, colormap and presentation ---------------------------------------

@needs_data
@needs_geom
@pytest.mark.parametrize("theme", ["dark", "light"])
def test_both_themes_render_every_view(tmp_path, theme):
    f = EventFile(ATMNU, geometry=GEOM)
    ev = f[0]
    for name, disp in (("2d", ev.display(reco=True, theme=theme)),
                       ("3d", ev.display_3d(theme=theme)),
                       ("opt", ev.display_optical(theme=theme))):
        assert os.path.getsize(disp.save(str(tmp_path / f"{name}.png"))) > 5_000
        assert disp.plotly_figure() is not None


@pytest.mark.parametrize("theme_name", ["dark", "light"])
def test_object_colours_never_collide_with_overlays(theme_name):
    """A hit's identity colour must not be the same paint as an overlay.

    Four exact hex values were shared before, so a hit could be drawn in the
    same amber as the shower cone on top of it, and orphan hits matched the
    gridlines exactly.
    """
    from pylarevd.theme import THEMES
    t = THEMES[theme_name]
    reserved = {t.track, t.vertex, t.shower, t.mctrack, t.truth, t.grid,
                t.unassociated, t.axes_bg, t.fig_bg}
    clashes = [c for c in t.categorical if c in reserved]
    assert not clashes, f"{theme_name}: {clashes} used for both"
    assert len(set(t.categorical)) == len(t.categorical)


@needs_data
@needs_geom
def test_identity_survives_without_colour():
    """More objects than colours must still be distinguishable by marker."""
    f = EventFile(ATMNU, geometry=GEOM)
    d = f[0].display(colour_by="cluster")
    batches = d.group_marker_batches()
    pairs = {(colour, marker) for _m, colour, marker, _o in batches}
    assert len(pairs) == len(batches), "two objects share colour AND marker"
    assert len(batches) > len(d.theme.categorical), "test needs colour wraparound"


@pytest.mark.parametrize("name", ["turbo", "magma", "viridis", "cividis"])
def test_colormaps_are_selectable_and_described(name):
    from pylarevd.theme import COLORMAPS, resolve_colormap
    cm = resolve_colormap(name)
    assert cm.note
    # turbo is the one non-monotonic ramp and must be recorded as such, since
    # that is what decides whether charge ordering survives greyscale
    assert COLORMAPS["turbo"].monotonic is False
    assert COLORMAPS["magma"].monotonic is True


@needs_data
@needs_geom
def test_truth_overlay_is_fast_enough_to_use(tmp_path):
    """One ax.plot per particle took over three minutes on this sample."""
    import time
    f = EventFile(ATMNU, geometry=GEOM)
    d = f[0].display(truth=True, reco=True)
    start = time.time()
    d.save(str(tmp_path / "truth.png"))
    elapsed = time.time() - start
    assert elapsed < 30.0, f"truth overlay took {elapsed:.0f}s"


@needs_data
@needs_geom
def test_presets_change_figure_size_and_fonts():
    from pylarevd.display import PRESETS
    f = EventFile(ATMNU, geometry=GEOM)
    sizes = {}
    for preset in ("screen", "paper1", "slide"):
        fig = f[0].display(preset=preset).figure()
        sizes[preset] = fig.get_size_inches()[0]
        import matplotlib.pyplot as plt
        plt.close(fig)
    assert sizes["paper1"] < sizes["screen"]
    assert PRESETS["slide"]["fonts"] > PRESETS["screen"]["fonts"]
    with pytest.raises(ValueError):
        f[0].display(preset="bogus")


@needs_data
@needs_geom
def test_three_panels_fill_the_canvas():
    """A 2-column grid leaves a quarter of the figure empty at n=3."""
    f = EventFile(ATMNU, geometry=GEOM)
    fig = f[0].display().figure()
    used = sum(1 for ax in fig.axes if ax.get_visible() and ax.has_data())
    grid = [ax for ax in fig.axes if ax.get_subplotspec() is not None]
    assert used >= 3
    assert len(grid) <= used + 1, "empty panel slots in the grid"
    import matplotlib.pyplot as plt
    plt.close(fig)


@needs_data
@needs_geom
def test_reference_lines_are_in_the_legend():
    f = EventFile(ATMNU, geometry=GEOM)
    fig = f[0].display(reco=True).figure()
    # The legend lives inside the first panel now, not at figure level: an
    # "outside" figure legend competed with the title and the truth block.
    legends = list(fig.legends) + [ax.get_legend() for ax in fig.axes
                                   if ax.get_legend() is not None]
    labels = {t.get_text() for leg in legends for t in leg.get_texts()}
    assert legends, "no legend was drawn at all"
    assert "anode" in labels and "active volume edge" in labels
    import matplotlib.pyplot as plt
    plt.close(fig)


@needs_data
@needs_geom
def test_flash_marker_size_is_absolute():
    """Window-relative sizing let a 4 PE flash outdraw a 56 PE one."""
    from pylarevd.display import _pe_size
    small, large = np.array([4.0]), np.array([56.0])
    assert _pe_size(small)[0] < _pe_size(large)[0]
    # and the mapping must not depend on what else is in the array
    together = _pe_size(np.array([4.0, 56.0]))
    assert np.isclose(together[0], _pe_size(small)[0])
    assert np.isclose(together[1], _pe_size(large)[0])


@needs_data
@needs_geom
def test_capabilities_match_the_file():
    f_full = EventFile(ATMNU, geometry=GEOM)[0].capabilities()
    f_hits = EventFile(ROCKMU, geometry=GEOM)[0].capabilities()
    assert f_full["tracks"] and f_full["optical"] and f_full["group:track"]
    assert not f_hits["tracks"] and not f_hits["optical"]
    assert not f_hits["group:track"]
    assert f_hits["spacepoints"] and f_hits["truth"]


@needs_data
@needs_geom
def test_3d_puts_gravity_up():
    """Screen-vertical must be detector y, not the drift axis."""
    f = EventFile(ATMNU, geometry=GEOM)
    fig = f[0].display_3d().plotly_figure()
    assert "y" in fig.layout.scene.zaxis.title.text
    assert "up" in fig.layout.scene.zaxis.title.text
    assert "drift" in fig.layout.scene.yaxis.title.text


# ---------------------------------------------------------------------------
# URL state
# ---------------------------------------------------------------------------

_URL_VALS = ["/data/a.root", 7, "hitfd", "track", "auto", "2d", "view",
             "light", "magma", ["on"], [], ["on"]]
_BY_NAME = {"a.root": "/data/a.root", "b.root": "/data/b.root"}


def test_url_round_trip_is_lossless():
    from pylarevd.app import url_params, url_values, _URL_CONTROLS
    assert len(_URL_VALS) == len(_URL_CONTROLS)
    back = url_values(url_params(_URL_VALS), _BY_NAME)
    assert back == _URL_VALS


def test_url_omits_unset_controls():
    """A control the URL does not mention comes back as None, not a default.

    Restoring a default over a value the user picked would silently undo it.
    """
    from pylarevd.app import url_values
    got = url_values("?entry=5", _BY_NAME)
    assert got[1] == 5
    assert got[0] is None and got[2] is None


def test_url_ignores_unknown_file_and_junk_entry():
    from pylarevd.app import url_values
    assert url_values("?file=nope.root", _BY_NAME)[0] is None   # bare name, unknown
    assert url_values("?entry=abc", _BY_NAME)[1] is None


def test_url_carries_full_paths():
    """A link must resolve against a server started with other arguments."""
    from pylarevd.app import url_params, url_values
    q = url_params(_URL_VALS)
    assert "file=%2Fdata%2Fa.root" in q            # full path, not the basename
    # a path this server has never seen still comes back, for the caller to open
    ad_hoc = "/scratch/someone/prodgenie_atmnu_dune10kt_reco1_687.root"
    assert url_values(f"?file={ad_hoc}", _BY_NAME)[0] == ad_hoc


def test_file_from_url_accepts_names_paths_and_tilde():
    from pylarevd.app import file_from_url
    assert file_from_url("a.root", _BY_NAME) == "/data/a.root"      # known basename
    assert file_from_url("/data/b.root", _BY_NAME) == "/data/b.root"  # known path
    assert file_from_url("/scratch/x/y.root", _BY_NAME) == "/scratch/x/y.root"
    assert file_from_url("~/y.root", _BY_NAME) == os.path.expanduser("~/y.root")
    assert file_from_url("stray.root", _BY_NAME) is None            # not a path


def test_bare_eos_paths_become_xrootd_urls():
    """/eos is not mounted here, so a pasted EOS path must be redirected.

    Otherwise the only symptom is "no such file" for a file that exists.
    """
    from pylarevd.app import normalise_path, is_remote, file_from_url
    user = "/eos/user/r/someone/reco.root"
    assert normalise_path(user) == f"root://eosuser.cern.ch/{user}"
    assert normalise_path("/eos/experiment/dune/x.root") == \
        "root://eospublic.cern.ch//eos/experiment/dune/x.root"
    assert normalise_path("/eos/project-nu/x.root") == \
        "root://eosproject.cern.ch//eos/project-nu/x.root"
    # already a URL, or not EOS at all: untouched
    assert normalise_path("root://a.cern.ch//x.root") == "root://a.cern.ch//x.root"
    assert normalise_path("/scratch/x.root") == "/scratch/x.root"
    assert is_remote(normalise_path(user)) and not is_remote("/scratch/x.root")
    # and a link carrying a bare EOS path resolves the same way
    assert file_from_url(user, {}) == f"root://eosuser.cern.ch/{user}"


def test_open_file_reports_actionable_errors(tmp_path):
    from pylarevd.app import open_file
    files = {}
    with pytest.raises(ValueError, match="no such file"):
        open_file(files, str(tmp_path / "absent.root"))
    with pytest.raises(ValueError, match="directory, not a file"):
        open_file(files, str(tmp_path))
    with pytest.raises(ValueError, match="no path given"):
        open_file(files, "   ")
    assert files == {}          # nothing half-registered on failure


def test_file_options_disambiguate_shared_basenames():
    """Two reco.root from different passes must not look identical in the list."""
    from pylarevd.app import file_options
    labels = [o["label"] for o in file_options(
        {"/pass1/reco.root": None, "/pass2/reco.root": None, "/x/solo.root": None})]
    assert labels == ["pass1/reco.root", "pass2/reco.root", "solo.root"]
    assert all(o["title"].startswith("/") for o in file_options({"/a/b.root": None}))


@needs_data
def test_open_file_is_idempotent_and_shares_geometry():
    from pylarevd.app import open_file
    files = {}
    first = open_file(files, ROCKMU)
    assert open_file(files, ROCKMU) is first          # opened once, not twice
    assert len(files) == 1


def test_url_checklists_survive_both_states():
    from pylarevd.app import url_params, url_values, _URL_CONTROLS
    checklists = [i for i, (_cid, is_list) in enumerate(_URL_CONTROLS) if is_list]
    on = list(_URL_VALS); off = list(_URL_VALS)
    for i in checklists:
        on[i], off[i] = ["on"], []
    assert [url_values(url_params(on), _BY_NAME)[i] for i in checklists] == \
        [["on"]] * len(checklists)
    assert [url_values(url_params(off), _BY_NAME)[i] for i in checklists] == \
        [[]] * len(checklists)


def test_url_event_identity_outranks_stale_entry():
    """rse= must win: entry numbers shift between files, event ids do not."""
    from pylarevd.app import url_params, url_values

    class FakeFile:
        def index_of(self, run, subrun, event):
            return 3 if (run, subrun, event) == (10, 1, 42) else None

    q = url_params(_URL_VALS, event_id=(10, 1, 42))
    assert "rse=10%3A1%3A42" in q
    resolve = {"/data/a.root": FakeFile()}.get
    assert url_values(q, _BY_NAME, resolve)[1] == 3          # not the 7 in entry=
    assert url_values(q, _BY_NAME)[1] == 7                   # no resolver: entry wins


def test_url_survives_an_event_the_file_does_not_have():
    from pylarevd.app import url_params, url_values

    class Empty:
        def index_of(self, *a):
            return None

    q = url_params(_URL_VALS, event_id=(10, 1, 42))
    resolve = {"/data/a.root": Empty()}.get
    assert url_values(q, _BY_NAME, resolve)[1] == 7           # falls back
    assert url_values(q.replace("10%3A1%3A42", "junk"), _BY_NAME, resolve)[1] == 7


@needs_data
def test_index_of_inverts_event_id():
    f = EventFile(ROCKMU)
    assert all(f.index_of(*f[i].id) == i for i in range(len(f)))
    assert f.index_of(999999, 999, 999) is None


# ---------------------------------------------------------------------------
# Geometry selection
# ---------------------------------------------------------------------------

@needs_data
def test_geometry_config_comes_from_the_file():
    """art embeds the job's fhicl config; the detector must be read, not guessed.

    Guessing wrong is silent: channels fall outside the map and every
    coordinate becomes NaN, which looks like an empty event rather than an
    error.
    """
    cfg = ArtFile(ROCKMU).geometry_config()
    assert cfg.get("Name")                     # e.g. "dune10kt_v6_1x2x6"
    assert cfg.get("GDML", "").endswith(".gdml")


@needs_data
@needs_geom
def test_file_selects_the_geometry_it_names():
    f = EventFile(ROCKMU)
    assert f.geometry.detector == f.art.geometry_config()["Name"]


@needs_data
def test_unexported_geometry_fails_loudly(monkeypatch):
    """Better a clear error than a display full of NaN."""
    import pylarevd.event as ev_mod
    monkeypatch.setattr(ArtFile, "geometry_config",
                        lambda self: {"Name": "protodune_sp_v9"})
    with pytest.raises(GeometryError, match="protodune_sp_v9"):
        EventFile(ROCKMU)


@needs_geom
def test_load_geometry_is_cached_by_path():
    from pylarevd.event import load_geometry
    assert load_geometry(GEOM) is load_geometry(GEOM)


@needs_data
def test_basket_cache_avoids_refetching(monkeypatch):
    """Reading many entries of one branch must not re-read its basket.

    Over xrootd each re-read is a network round-trip: walking 120 entries of
    EventAuxiliary cost ~6 s before this cache, versus ~0.5 s after.
    """
    f = ArtFile(ROCKMU)
    branch = f.events["EventAuxiliary"]
    calls = []
    original = type(branch).basket

    def counted(self, i):
        calls.append(i)
        return original(self, i)

    monkeypatch.setattr(type(branch), "basket", counted)
    f.event_ids()
    n_entries = branch.num_entries
    assert len(calls) < n_entries          # not one fetch per entry
    assert len(calls) == len(set(calls))   # each basket fetched exactly once


# ---------------------------------------------------------------------------
# Truth: neutrino interaction
# ---------------------------------------------------------------------------

def test_particle_names_cover_pdg_nuclei_and_genie_codes():
    from pylarevd import physics
    assert physics.particle_name(16) == "nu_tau"
    assert physics.particle_name(-211) == "pi-"
    assert physics.particle_name(1000180400) == "Ar-40"      # nuclear code
    assert physics.particle_name(2000000201) == "np cluster"  # MEC pair
    assert physics.particle_name(99999999) == "99999999"      # unknown: as-is
    assert physics.nucleus_name(211) is None


def test_interaction_code_names():
    from pylarevd import physics
    assert physics.mode_name(2) == "DIS"
    assert physics.interaction_name(1091) == "CC DIS"
    assert physics.current_name(0) == "CC" and physics.current_name(1) == "NC"
    assert physics.origin_name(1) == "beam neutrino"
    # unknown codes are reported, not invented
    assert "77" in physics.mode_name(77)
    assert "4242" in physics.interaction_name(4242)


def test_sequence_rejects_impossible_element_counts():
    """A count bigger than the bytes left is a misread, not data.

    Without this the reader parses the phantom elements until the buffer runs
    out: one simb::MCTruth became 2.35 million TLorentzVector reads and never
    finished.
    """
    import pylarevd.streamers as st
    # 8 bytes: a count of 0x7FFFFFFF followed by nothing that could hold it
    buf = b"\x7f\xff\xff\xff" + b"\x00" * 4
    cur = st.Cursor(buf, 0)
    with pytest.raises(st.StreamerError, match="exceeds"):
        st.read_sequence(None, "TLorentzVector", cur, headered=False)


@needs_data
def test_neutrino_truth_is_read_and_self_consistent():
    ev = EventFile(ATMNU)[0] if os.path.exists(ATMNU) else None
    if ev is None:
        pytest.skip("atmnu sample absent")
    nu = ev.neutrino()
    assert nu is not None
    assert abs(nu.pdg) in (12, 14, 16)
    assert nu.energy > 0
    assert nu.ccnc in (0, 1)
    assert nu.target_name.startswith("Ar")           # argon target
    # the outgoing lepton cannot carry more than the neutrino brought
    if nu.is_cc and np.isfinite(nu.lepton_energy):
        assert nu.lepton_energy <= nu.energy * 1.001
    # final state is a subset of the full part list, and every entry has status 1
    assert len(nu.fs_pdg) <= len(nu.all_pdg)
    assert set(nu.all_status[np.isin(nu.all_pdg, nu.fs_pdg)]) >= {1}
    assert nu.headline() and nu.describe().count("\n") >= 3


@needs_data
def test_events_without_a_neutrino_return_none():
    """Cosmic and radiological samples carry MCTruth but set no neutrino."""
    assert EventFile(ROCKMU)[0].neutrino() is None


@needs_data
def test_neutrino_vertex_agrees_with_reconstruction():
    """The true vertex should land near the reconstructed one."""
    if not os.path.exists(ATMNU):
        pytest.skip("atmnu sample absent")
    ev = EventFile(ATMNU)[0]
    nu = ev.neutrino()
    vtx = ev.vertices()
    if vtx is None or not len(vtx):
        pytest.skip("no reconstructed vertex")
    d = np.linalg.norm(vtx.xyz - nu.vertex, axis=1).min()
    assert d < 50.0, f"true vertex {nu.vertex} is {d:.1f} cm from any reco vertex"


# ---------------------------------------------------------------------------
# Containment / visible energy / true vertex
# ---------------------------------------------------------------------------

@needs_geom
def test_containment_is_per_tpc_not_a_bounding_box():
    """A point can sit in the box enclosing the detector yet in no TPC.

    Event 687/0/95 is exactly that case: 6.6 cm above the top face.
    """
    from pylarevd.geometry import Geometry
    g = Geometry(GEOM)
    box = g.active_boxes
    centre = 0.5 * (box[0, :3] + box[0, 3:])
    con = g.containment(centre)
    assert con.inside and bool(con) and con.distance > 0
    assert "inside" in con.description

    # a metre above the topmost face is outside, and the report says which face
    top = float(box[:, 4].max())
    out = g.containment([centre[0], top + 100.0, centre[2]])
    assert not out.inside and not bool(out)
    assert 99.0 < out.distance < 101.0
    assert "above y" in out.description


@needs_geom
def test_in_active_is_vectorised():
    from pylarevd.geometry import Geometry
    g = Geometry(GEOM)
    box = g.active_boxes
    centre = 0.5 * (box[0, :3] + box[0, 3:])
    far = np.array([1e5, 1e5, 1e5])
    got = g.in_active(np.vstack([centre, far, centre]))
    assert got.tolist() == [True, False, True]


@needs_data
@needs_geom
def test_true_vertex_projects_onto_the_collection_axis():
    """For vertical collection wires the wire coordinate IS z.

    That makes the projection checkable without trusting the projection code.
    """
    ev = EventFile(ATMNU)[0] if os.path.exists(ATMNU) else None
    if ev is None:
        pytest.skip("atmnu sample absent")
    nu = ev.neutrino()
    d = ev.display()
    coll = [p for p in d.panels if "Collection" in p.title]
    if not coll:
        pytest.skip("no collection panel")
    wx = d._true_vertex_wx(coll[0])
    assert wx is not None
    assert abs(wx[0] - nu.vertex[2]) < 1.0      # w == z
    assert abs(wx[1] - nu.vertex[0]) < 1e-6     # drift == x


@needs_data
def test_readout_view_has_no_true_vertex():
    """Readout axes are channel/tick; a position would be meaningless there."""
    if not os.path.exists(ATMNU):
        pytest.skip("atmnu sample absent")
    d = EventFile(ATMNU)[0].display(space="readout")
    assert all(d._true_vertex_wx(p) is None for p in d.panels)


@needs_data
def test_uncontained_events_are_flagged_and_explained():
    if not os.path.exists(ATMNU):
        pytest.skip("atmnu sample absent")
    d = EventFile(ATMNU)[0].display()
    block = d._truth_block()
    assert "visible energy" in block
    assert d.containment is not None
    # a contained event must NOT carry the warning
    if d.containment.inside:
        assert "OUTSIDE" not in d._truth_headline()
    assert "visible energy" in d.summary()


@needs_data
def test_deposited_energy_matches_the_deposits():
    if not os.path.exists(ATMNU):
        pytest.skip("atmnu sample absent")
    ev = EventFile(ATMNU)[0]
    assert ev.deposited_energy() == pytest.approx(float(ev.truth_deposits().edep.sum()))


@needs_data
def test_no_neutrino_means_no_containment_or_flag():
    d = EventFile(ROCKMU)[0].display()
    assert d.neutrino is None
    assert d.containment is None
    assert d._truth_headline() == "" and d._truth_block() == ""
    assert all(d._true_vertex_wx(p) is None for p in d.panels)


# ---------------------------------------------------------------------------
# Pandora track/shower best match
# ---------------------------------------------------------------------------

@needs_data
def test_pandora_writes_both_interpretations_of_every_pfparticle():
    """The premise of the best-match filter, checked rather than assumed."""
    if not os.path.exists(ATMNU):
        pytest.skip("atmnu sample absent")
    ev = EventFile(ATMNU, geometry=GEOM)[0]
    n_track = len(ev.tracks(best_match=False))
    n_shower = len(ev.showers(best_match=False))
    assert n_track > 0 and n_shower > 0
    # both collections are built from the same PFParticles
    assert abs(n_track - n_shower) <= 1


@needs_data
def test_best_match_keeps_each_pfparticle_exactly_once():
    if not os.path.exists(ATMNU):
        pytest.skip("atmnu sample absent")
    ev = EventFile(ATMNU, geometry=GEOM)[0]
    sel = ev.pandora_best_match()
    assert sel, "no Pandora metadata to decide with"
    # a given object is never kept as both a track and a shower
    assert not (sel["tracks"] & sel["showers"]).any()
    kept = int(sel["tracks"].sum() + sel["showers"].sum())
    scores = ev.pfp_track_scores()
    assert kept == int(np.isfinite(scores).sum())


@needs_data
def test_track_score_agrees_with_pandoras_own_pdg_code():
    """TrackScore > 0.5 must reproduce Pandora's 13-vs-11 assignment.

    Two independent statements of the same decision; if they disagreed, the
    threshold would be wrong.
    """
    if not os.path.exists(ATMNU):
        pytest.skip("atmnu sample absent")
    ev = EventFile(ATMNU, geometry=GEOM)[0]
    scores = ev.pfp_track_scores()
    raw = ev._src.art.read("recob::PFParticles_pandora__Reco2.",
                           "recob::PFParticle", ev.entry)
    pdg = np.asarray(raw["fPdgCode"], int)
    have = np.isfinite(scores[:len(pdg)])
    is_track_by_score = scores[:len(pdg)][have] >= 0.5
    is_track_by_pdg = pdg[have] == 13
    assert (is_track_by_score == is_track_by_pdg).all()


@needs_data
def test_best_match_is_the_default_and_reduces_the_drawing():
    if not os.path.exists(ATMNU):
        pytest.skip("atmnu sample absent")
    ev = EventFile(ATMNU, geometry=GEOM)[0]
    assert len(ev.tracks()) + len(ev.showers()) < \
        len(ev.tracks(best_match=False)) + len(ev.showers(best_match=False))
    assert "Pandora best match" in ev.pandora_selection_note()


@needs_data
def test_no_metadata_means_draw_everything():
    """Without Pandora metadata nothing may be dropped -- guessing is worse."""
    ev = EventFile(ROCKMU)[0]
    if ev.pfp_track_scores() is not None:
        pytest.skip("this sample does have metadata")
    assert ev.pandora_best_match() == {}
    assert ev.pandora_selection_note() is None


# ---------------------------------------------------------------------------
# 3-D flash display
# ---------------------------------------------------------------------------

@needs_data
def test_flash_x_sentinel_becomes_nan():
    """LArSoft leaves an unreconstructed flash x at DBL_MAX.

    Taken literally that is 1.8e308, which would place every flash at infinity
    and silently blow up any axis range computed from it.
    """
    if not os.path.exists(ATMNU):
        pytest.skip("atmnu sample absent")
    o = EventFile(ATMNU, geometry=GEOM)[0].optical()
    assert np.nanmax(np.abs(o.x)) < 1e30
    # has_x asks about flashes; op hits get x from their detector and must not
    # make an unreconstructed flash collection look reconstructed
    assert o.has_x == bool(np.isfinite(o.x[o.is_flash]).any())


@needs_data
@needs_geom
def test_flash3d_defaults_to_the_brightest_flash_window():
    if not os.path.exists(ATMNU):
        pytest.skip("atmnu sample absent")
    ev = EventFile(ATMNU, geometry=GEOM)[0]
    d = ev.display_flashes_3d()
    lo, hi = d.window
    o = ev.optical()
    brightest_t = float(o.time[np.argmax(np.where(o.is_flash, o.pe, -np.inf))])
    assert lo <= brightest_t <= hi
    f = d.flashes
    assert len(f) and ((f.time >= lo) & (f.time <= hi)).all()
    assert (np.diff(f.pe) <= 0).all()          # brightest first


@needs_data
@needs_geom
def test_flash3d_caps_and_reports_what_it_dropped():
    if not os.path.exists(ATMNU):
        pytest.skip("atmnu sample absent")
    ev = EventFile(ATMNU, geometry=GEOM)[0]
    d = ev.display_flashes_3d(time_range=(-3000, 3000), max_flashes=5)
    assert len(d.flashes) == 5
    assert "brightest 5 shown" in d._title()       # no silent truncation
    assert "NOT reconstructed" in d.summary()      # x is unmeasured here


@needs_data
@needs_geom
def test_flash3d_renders_in_both_backends():
    if not os.path.exists(ATMNU):
        pytest.skip("atmnu sample absent")
    d = EventFile(ATMNU, geometry=GEOM)[0].display_flashes_3d(max_flashes=8)
    fig = d.plotly_figure()
    names = {tr.name for tr in fig.data}
    assert "flashes" in names and "photon detectors" in names
    assert "drift" in fig.layout.scene.yaxis.title.text
    assert "up" in fig.layout.scene.zaxis.title.text
    mpl = d.figure()
    assert mpl is not None
    import matplotlib.pyplot as plt
    plt.close(mpl)


@needs_data
def test_partial_parse_memo_does_not_poison_later_objects():
    """A class that fails to parse in one context must still parse in another.

    simb::MCParticle reads whole inside MCTruth::fPartList but not inside
    MCNeutrino::fNu. Caching the resulting truncation per class made decoding
    order-dependent: reading one event taught the reader to truncate every
    later object of that class, so every subsequent event silently lost its
    neutrino truth. Reading events out of order must not change what they say.
    """
    f = EventFile(ATMNU) if os.path.exists(ATMNU) else None
    if f is None or len(f) < 3:
        pytest.skip("need a multi-event truth sample")
    fresh = [EventFile(ATMNU)[i].neutrino() for i in range(min(3, len(f)))]
    if not any(n is not None for n in fresh):
        pytest.skip("sample has no neutrino truth")

    shared = EventFile(ATMNU)
    shared[len(shared) - 1].neutrino()          # visit a later event first
    after = [shared[i].neutrino() for i in range(min(3, len(f)))]
    for i, (a, b) in enumerate(zip(fresh, after)):
        assert (a is None) == (b is None), \
            f"entry {i} truth appeared/vanished depending on browse order"
        if a is not None:
            assert a.energy == b.energy and a.pdg == b.pdg


# ---------------------------------------------------------------------------
# UI-review fixes
# ---------------------------------------------------------------------------

@needs_data
@needs_geom
def test_undisambiguated_collection_is_flagged():
    """A raw hit finder puts charge metres from truth; say so.

    Disambiguation is what splits a wrapped channel across physical wires, so a
    collection that never does it has not been disambiguated.
    """
    if not os.path.exists(ATMNU):
        pytest.skip("atmnu sample absent")
    ev = EventFile(ATMNU, geometry=GEOM)[0]
    assert ev.geometry.wrapped_channels > 0
    assert ev.hits("hitfd").channels_split_across_wires > 0
    assert ev.hits("gaushit").channels_split_across_wires == 0
    assert "NOT disambiguated" in ev.display("gaushit").disambiguation_warning
    assert ev.display("hitfd").disambiguation_warning == ""
    # readout space folds the faces together on purpose
    assert ev.display("gaushit", space="readout").disambiguation_warning == ""


def test_colour_and_marker_advance_independently():
    """Neither colour nor marker may carry identity alone.

    Marker-per-block-of-8 gave the first eight objects one marker; the mirror
    fix gave the first five one colour. 8 and 5 are coprime, so advancing both
    keeps all 40 combinations while separating consecutive objects twice over.
    """
    from pylarevd.theme import DARK, MARKER_CYCLE
    n = len(DARK.categorical)
    pairs = [(r % n, MARKER_CYCLE[r % len(MARKER_CYCLE)]) for r in range(41)]
    assert len(set(pairs[:40])) == 40
    assert pairs[40] == pairs[0]
    assert all(pairs[i][0] != pairs[i + 1][0] for i in range(40))
    assert all(pairs[i][1] != pairs[i + 1][1] for i in range(40))


def test_flash_width_is_absolute_not_window_relative():
    """A lone flash must not collapse to the thinnest line.

    Normalising within the window made max == min, so the brightest flash in
    the event rendered as a hairline at the bottom of the colormap.
    """
    from pylarevd.display import _pe_width
    assert _pe_width(279123.0) > _pe_width(1000.0) > _pe_width(4.0)
    # one flash, alone, still lands near the top of the range
    assert _pe_width(np.array([279123.0]))[0] > 2.5


@needs_data
@needs_geom
def test_flash3d_colour_range_is_never_degenerate():
    if not os.path.exists(ATMNU):
        pytest.skip("atmnu sample absent")
    ev = EventFile(ATMNU, geometry=GEOM)[0]
    for window in ((-2, 2), (-50, 50), (-3000, 3000)):
        lo, hi = ev.display_flashes_3d(time_range=window)._colour_time_range()
        assert hi > lo, f"degenerate colour range for window {window}"


@needs_data
@needs_geom
def test_panels_frame_their_data_not_the_whole_drift():
    """Reference lines must not drag the axis open around a sparse event."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ev = EventFile(ROCKMU)[0]
    d = ev.display()
    fig = d.figure()
    try:
        h, panel = ev.hits(), d.panels[0]
        span = float(np.ptp(h.x[panel.mask]))
        shown = float(np.ptp(fig.axes[0].get_ylim()))
        assert span / shown > 0.6, f"data fills only {100 * span / shown:.0f}% of the panel"
    finally:
        plt.close(fig)


@needs_data
@needs_geom
def test_presets_save_at_their_declared_width():
    """A paper1 figure must arrive at a journal column width, not 84% over.

    Long unwrapped text plus bbox_inches="tight" used to grow the canvas, so
    every font landed too small once the journal scaled it back.
    """
    from PIL import Image
    from pylarevd.display import PRESETS
    if not os.path.exists(ATMNU):
        pytest.skip("atmnu sample absent")
    ev = EventFile(ATMNU, geometry=GEOM)[0]
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        for preset in ("paper1", "slide", "poster"):
            out = os.path.join(tmp, f"{preset}.png")
            ev.display("hitfd", truth=True, preset=preset).save(out)
            got = Image.open(out).size[0] / PRESETS[preset]["dpi"]
            nominal = PRESETS[preset]["panel_width"]
            assert got < nominal * 1.15, \
                f"{preset} saved {got:.2f} in against a nominal {nominal:.2f} in"


def test_title_wrapping_respects_the_canvas():
    from pylarevd.display import _wrap_for
    long = "dune10kt_v6_1x2x6   run 20000063 / subrun 0 / event 1   -   9312 hits"
    assert "\n" in _wrap_for(long, 3.4, 8.0)        # a journal column wraps it
    assert "\n" not in _wrap_for(long, 20.0, 8.0)   # a poster does not


def test_checklists_are_disabled_per_option_not_per_component():
    """dcc.Checklist has no component-level `disabled` prop.

    Writing one put the attribute on the DOM node, where the browser's default
    disabled styling (dark grey) made the labels unreadable on the dark panel.
    Disabling belongs on each option, and the reason belongs in its label.
    """
    from dash import dcc
    assert "disabled" not in getattr(dcc.Checklist, "_prop_names", ())


@needs_data
def test_overlay_labels_say_why_they_are_unavailable():
    from pylarevd.app import build_app
    import json
    app = build_app([ROCKMU])
    client = app.server.test_client()
    deps = json.loads(client.get("/_dash-dependencies").data)
    # no callback may target the nonexistent property
    assert not any("truth.disabled" in d["output"] or "reco.disabled" in d["output"]
                   for d in deps)
    dep = next(d for d in deps if "truth.options" in d["output"])
    assert [i["id"] for i in dep["inputs"]] == ["file", "entry", "mode", "truth"]
    # and the page states both label colours rather than inheriting them
    html = client.get("/").data.decode()
    assert ".evd-check" in html and "input:disabled" in html


@needs_data
@needs_geom
def test_radiologicals_can_be_separated_from_the_interaction():
    """A radiological sample buries the interaction in unrelated decays.

    Everything from the neutrino traces back to one vertex; radiological
    generators emit tens of thousands of primaries scattered through the
    detector, so the root ancestor's start point separates them.
    """
    if not os.path.exists(ATMNU):
        pytest.skip("atmnu sample absent")
    ev = EventFile(ATMNU, geometry=GEOM)[0]
    ids = ev.neutrino_track_ids()
    assert ids is not None and len(ids)

    full = ev.display_3d(truth=True, radiologicals=True)
    cut = ev.display_3d(truth=True, radiologicals=False)
    assert len(cut.mc) < len(full.mc)
    assert len(cut.truth) < len(full.truth)
    # what survives really is the interaction: every kept deposit belongs to a
    # particle descending from the vertex
    assert np.isin(np.abs(cut.truth.track_id), ids).all()
    # and the 2-D view agrees with the 3-D one
    assert len(ev.display(truth=True, radiologicals=False).mc) == len(cut.mc)


@needs_data
def test_no_neutrino_means_nothing_to_separate():
    """Without a vertex to trace back to, the filter must keep everything."""
    ev = EventFile(ROCKMU)[0]
    assert ev.neutrino_track_ids() is None
    kept = ev.display_3d(truth=True, radiologicals=False)
    both = ev.display_3d(truth=True, radiologicals=True)
    assert len(kept.mc or []) == len(both.mc or [])


# ---------------------------------------------------------------------------
# Dependency reporting
# ---------------------------------------------------------------------------

def test_check_report_names_what_is_missing_and_how_to_fix_it():
    from pylarevd import _deps
    text = _deps.report()
    assert "python" in text and "bundled geometries" in text
    for dep, _ok, _v in _deps.status():
        assert dep.module in text
        assert dep.purpose in text


def test_require_raises_something_actionable():
    """A missing optional package must name itself and the fix, not traceback."""
    from pylarevd import _deps
    with pytest.raises(SystemExit) as exc:
        _deps.require("a_package_that_does_not_exist", "some feature")
    message = str(exc.value)
    assert "some feature" in message
    assert "pip install" in message and "--check" in message
    # and it must not fire for something that is present
    _deps.require("numpy", "everything")


def test_declared_extras_exist_in_pyproject():
    """_deps.py and pyproject.toml must not drift apart.

    An install hint pointing at an extra that does not exist is worse than no
    hint at all: it fails with a pip error instead of installing anything.
    """
    import tomllib
    from pylarevd import _deps
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "pyproject.toml"), "rb") as fh:
        pyproject = tomllib.load(fh)
    declared = set(pyproject["project"]["optional-dependencies"])
    used = {d.extra for d in _deps.DEPENDENCIES if d.extra}
    assert used <= declared, f"extras used but not declared: {used - declared}"
    # every optional dependency must be reachable from the 'all' extra
    every = " ".join(pyproject["project"]["optional-dependencies"]["all"])
    for dep in _deps.DEPENDENCIES:
        if dep.extra and dep.pip != "xrootd":     # xrootd needs a compiler
            assert dep.pip in every, f"{dep.pip} missing from the 'all' extra"


def test_requirements_txt_covers_the_dependencies():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "requirements.txt")
    assert os.path.exists(path)
    text = open(path).read()
    from pylarevd import _deps
    for dep in _deps.DEPENDENCIES:
        if dep.pip != "xrootd":
            assert dep.pip in text, f"{dep.pip} missing from requirements.txt"


# ---------------------------------------------------------------------------
# Starting with no files, and opening one by path
# ---------------------------------------------------------------------------

def test_app_starts_with_no_files():
    """Files must be optional: the browser can start empty and open by path."""
    from pylarevd.app import build_app, _files_store, render
    files = _files_store([], None)
    assert files == {}
    app = build_app([])
    assert app.server.test_client().get("/").status_code == 200
    # and it says what to do rather than showing a blank panel
    _fig, summary = render(files, None, None, "hitfd", "integral", "auto",
                           "2d", False)
    assert "paste a path" in summary


def test_unreadable_files_given_explicitly_still_fail():
    """An empty start is deliberate; files that were given and failed are not."""
    from pylarevd.app import _files_store
    with pytest.raises(SystemExit):
        _files_store(["/nope/definitely-not-here.root"], None)


@needs_data
def test_open_by_path_files_under_the_normalised_key():
    """The box must look a file up the way open_file stored it.

    A bare /eos/... path is stored as its root:// URL, so looking it up by the
    typed string crashed with a KeyError -- on exactly the paths the rewriting
    exists to make convenient.
    """
    from pylarevd.app import open_file, normalise_path
    files = {}
    opened = open_file(files, ROCKMU)
    assert normalise_path(ROCKMU) in files
    assert files[normalise_path(ROCKMU)] is opened

    # the EOS form resolves to the same key the opener would use
    eos = "/eos/user/x/xyz/sample.root"
    assert normalise_path(eos).startswith("root://")
    assert normalise_path(normalise_path(eos)) == normalise_path(eos)


# ---------------------------------------------------------------------------
# Round-2 review fixes
# ---------------------------------------------------------------------------

@needs_data
def test_opening_a_file_does_not_inherit_another_files_geometry():
    """Each file names its own detector; forcing one on another is silent.

    Reusing the first file's resolved geometry put 99% of a full-10kt file's
    hits outside the channel map with no error at all.
    """
    from pylarevd.app import _files_store, open_file
    files = _files_store([ATMNU], None)
    first = next(iter(files.values())).geometry
    opened = open_file(files, ROCKMU, None)
    assert opened.geometry.detector == opened.art.geometry_config()["Name"]
    # and nothing was forced: the object is chosen per file, not inherited
    assert opened.geometry.detector == first.detector or \
        opened.geometry is not first


@needs_data
def test_visible_energy_never_exceeds_the_neutrino_energy():
    """Radiological deposits are not the interaction's visible energy."""
    f = EventFile(ATMNU)
    seen = 0
    for i in range(len(f)):
        nu = f[i].neutrino()
        if nu is None or nu.energy <= 0:
            continue
        vis = f[i].neutrino_visible_energy()
        assert vis is not None
        seen += 1
        assert vis <= nu.energy * 1000 * 1.05, \
            f"entry {i}: {100 * vis / (nu.energy * 1000):.0f}% of the beam energy"
    assert seen, "no neutrino events to check"


@needs_data
@needs_geom
def test_disambiguation_verdict_is_per_collection_not_per_event():
    """One event that stays on a single drift face proves nothing."""
    f = EventFile(ROCKMU)
    assert f.is_disambiguated("hitfd") is True
    # so no event of a disambiguated collection may be flagged
    assert not any(f[i].display("hitfd").disambiguation_warning
                   for i in range(len(f)))
    if os.path.exists(ATMNU):                     # and a raw one still is
        g = EventFile(ATMNU)
        assert g.is_disambiguated("gaushit") is False
        assert g[0].display("gaushit").disambiguation_warning


@needs_data
def test_expensive_accessors_are_memoised():
    """Every control change re-decoded 33k MCParticles without this."""
    ev = EventFile(ATMNU)[0] if os.path.exists(ATMNU) else EventFile(ROCKMU)[0]
    for name in ("mc_particles", "truth_deposits", "tracks", "hits",
                 "spacepoints", "optical"):
        first = getattr(ev, name)()
        assert getattr(ev, name)() is first, f"{name} is not memoised"


@needs_data
@needs_geom
def test_shower_axes_actually_draw_in_the_interactive_view():
    """A trace being present is not the same as it drawing anything."""
    from pylarevd.display import _join_segments
    ev = EventFile(ATMNU, geometry=GEOM)[0]
    showers = ev.showers()
    if not len(showers):
        pytest.skip("no showers in this event")
    fig = ev.display_3d(truth=True).plotly_figure()
    trace = next((t for t in fig.data if t.name == "shower axes"), None)
    assert trace is not None
    x = np.asarray(trace.x, float)
    finite = np.isfinite(x)
    runs = max((len(r) for r in "".join("1" if v else "0" for v in finite).split("0")),
               default=0)
    assert runs >= 2, "every point is isolated between NaNs; mode='lines' draws nothing"
    # a straight segment must survive the join untouched
    seg = np.array([[0.0, 0, 0], [500.0, 0, 0]])
    assert np.isfinite(_join_segments([seg])[:2]).all()


@needs_data
@needs_geom
def test_categorical_identity_uses_markers_in_both_backends():
    from pylarevd.theme import PLOTLY_MARKER_CYCLE
    ev = EventFile(ATMNU, geometry=GEOM)[0]
    d = ev.display("hitfd", colour_by="track")
    if len(d.group_marker_batches()) < 3:
        pytest.skip("too few objects to exercise the cycle")
    symbols = {getattr(t.marker, "symbol", None) for t in d.plotly_figure().data
               if getattr(t, "marker", None) is not None}
    assert set(PLOTLY_MARKER_CYCLE[:3]) <= symbols, \
        "the browser distinguishes objects by colour alone"


def test_ophit_colour_scale_is_window_independent():
    """The same hit must not change colour because you zoomed."""
    import inspect
    from pylarevd import display
    src = inspect.getsource(display.OpticalDisplay.figure)
    assert "PE_SIZE_RANGE" in src
    assert "vmax=pe.max()" not in src


@needs_data
@needs_geom
def test_true_vertex_only_on_its_own_drift_side():
    """A panel showing one drift volume must not display the other's vertex."""
    ev = EventFile(ATMNU, geometry=GEOM)[0]
    nu = ev.neutrino()
    if nu is None:
        pytest.skip("no neutrino")
    d = ev.display("hitfd", merge="none")
    sign = np.sign(nu.vertex[0])
    for panel in d.panels:
        drawn = d._true_vertex_wx(panel) is not None
        if panel.drift_sign:
            assert drawn == (np.sign(panel.drift_sign) == sign), \
                f"{panel.title}: marker drawn on the wrong drift side"


@needs_data
def test_final_state_energies_are_kinetic_not_total():
    """Rest mass is not something the interaction supplied.

    Listing total energies made the final state sum to 18.66 GeV against a
    7.36 GeV neutrino -- a reader checking the arithmetic concludes the display
    is broken. The tolerance here is deliberately loose: on a bound nucleon the
    initial state carries Fermi motion, so a MEC or QE final state legitimately
    exceeds the beam energy by a few percent (entry 2 of this sample does, by
    9%). What it must not do is exceed it several times over.
    """
    f = EventFile(ATMNU)
    checked = 0
    for i in range(len(f)):
        nu = f[i].neutrino()
        if nu is None:
            continue
        checked += 1
        total = sum(e for _name, _n, e in nu.final_state_counts())
        assert total <= nu.energy * 1.25, \
            f"entry {i}: final state {total:.2f} GeV vs beam {nu.energy:.2f} GeV"
    assert checked


def test_data_ylim_ignores_a_few_outliers():
    from pylarevd.display import _data_ylim
    bulk = np.random.default_rng(0).normal(0, 1, 500)
    with_outliers = np.concatenate([bulk, [500.0, -500.0]])
    lo, hi = _data_ylim(with_outliers)
    assert hi - lo < 20, "a couple of stray hits still dictate the frame"


def test_batch_writes_a_manifest(tmp_path):
    from pylarevd.cli import _write_manifest
    rows = [{"entry": 0, "run": 1, "subrun": 0, "event": 4, "hits": 12,
             "headline": "nu_mu CC QE", "png": "a.png", "html": "a.html",
             "status": "ok", "error": ""},
            {"entry": 1, "status": "failed", "error": "boom"}]
    _write_manifest(str(tmp_path), rows)
    csv_text = (tmp_path / "manifest.csv").read_text()
    assert "nu_mu CC QE" in csv_text and "failed" in csv_text
    index = (tmp_path / "index.html").read_text()
    assert 'href="a.png"' in index and "boom" in index


def test_html_cdn_flag_exists():
    """--html embedded ~17.8 MB per event with no way to opt out."""
    import inspect
    from pylarevd import cli
    src = inspect.getsource(cli.main)
    assert "--html-cdn" in src
    assert "bundle_plotlyjs=not a.html_cdn" in inspect.getsource(cli.main)


def test_package_imports_without_numpy():
    """--check must run in the situation it exists to diagnose."""
    import subprocess
    import sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    blocker = os.path.join(root, "tests", "_noblock")
    os.makedirs(blocker, exist_ok=True)
    for mod in ("numpy", "uproot"):
        with open(os.path.join(blocker, f"{mod}.py"), "w") as fh:
            fh.write("raise ImportError('absent')\n")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([blocker, root]))
    out = subprocess.run([sys.executable, "-m", "pylarevd", "--check"],
                         capture_output=True, text=True, env=env, timeout=120)
    assert out.returncode == 0, out.stderr[-500:]
    assert "REQUIRED packages are missing" in out.stdout
    assert "numpy" in out.stdout
