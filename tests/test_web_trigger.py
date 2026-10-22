"""Trigger primitives / activities / candidates in the web viewer's 2-D panels."""
import os

import numpy as np
import pytest

# ProtoDUNE-HD MC with the dunetrigger chain, e.g.
# prod_protodunehd_beamneutrino_1_342725_42_1777946041_gen_g4_detsim_trigger_reco_stage1_reco_stage2.root
PDHD = os.environ.get("PYLAREVD_PDHD", "")
needs_pdhd = pytest.mark.skipif(not os.path.exists(PDHD), reason="set PYLAREVD_PDHD")


@pytest.fixture(scope="module")
def exported():
    """Event 3 (two activities, two candidates), all three panel groupings."""
    from pylario import EventFile
    from pylarevd.display import EventDisplay
    from pylarevd.web.trigger import export_trigger
    ev = EventFile(PDHD)[3]
    hits = ev.hits()
    ds = [(sf, EventDisplay(ev, hits=hits, merge=m))
          for sf, m in (("", "orientation"), ("_view", "view"), ("_none", "none"))]
    arrays, meta = {}, {}
    lines = export_trigger(ev, hits, ds, arrays, meta)
    return ev, ds, arrays, meta, lines


@needs_pdhd
def test_counts_and_summary(exported):
    ev, ds, a, meta, lines = exported
    t = meta["trigger"]
    assert (t["n_tp"], t["n_ta"], t["n_tc"]) == (31360, 2, 2)
    assert t["tc_type_names"] == ["ADCSimpleWindow"] * 2
    assert lines[0].startswith("trigger: 31,360 TPs, 2 TAs, 2 TCs")
    assert len(a["tp/channel"]) == 31360 and a["ta/n_tp"].tolist() == [627, 1094]
    assert a["ta/tc"].tolist() == [0, 1] and a["tc/n_ta"].tolist() == [1, 1]


@needs_pdhd
def test_offsets_per_plane(exported):
    off = exported[3]["trigger"]["hit_tp_offset"]
    assert 8 < off["U"] < 12 and 3 < off["V"] < 7 and -4 < off["Z"] < 0


@needs_pdhd
def test_placement(exported):
    from pylarevd.web.trigger import PLACE_AMBIGUOUS, PLACE_HIT, PLACE_UNIQUE
    ev, ds, a, meta, _ = exported
    g = ev.geometry
    place, h = a["tp/place"], a["tp/h"]
    chv = a["tp/chan_view"]
    assert np.all(place[chv == 2] == PLACE_UNIQUE)            # collection: one wire each
    assert (place == PLACE_HIT).sum() > 5000
    assert (place == PLACE_AMBIGUOUS).mean() < 0.05
    placed = place <= PLACE_HIT
    assert np.all(np.isfinite(h[placed])) and np.all(np.isnan(h[~placed]))
    # a primitive placed through a hit sits on a wire of its own channel
    sel = np.flatnonzero(place == PLACE_HIT)[:500]
    for i in sel:
        rows = np.flatnonzero(g.w_chan == a["tp/channel"][i])
        assert np.any(np.isclose(g.wire_coord[rows], h[i], atol=1e-3))
    # every grouping places every placed primitive in one of its panels
    for sf, d in ds:
        pan = a[f"tp/panel{sf}"]
        assert np.all(pan[placed] < len(d.panels)) and np.all(pan[~placed] == 255)


@needs_pdhd
def test_boxes_and_bands(exported):
    ev, ds, a, meta, _ = exported
    g = ev.geometry
    xs = [(b.xmin, b.xmax) for b in g.tpcs]
    for sf, d in ds:
        # both activities are on collection channels: boxes only in a Z panel
        assert sorted(a[f"ta2d{sf}/index"].tolist()) == [0, 1]
        assert {d.panels[k].view for k in a[f"ta2d{sf}/panel"]} == {2}
        box = a[f"tc2d{sf}/box"]
        assert len(box) and set(a[f"tc2d{sf}/index"].tolist()) == {0, 1}
        for h0, h1, v0, v1 in box:          # inside some active drift volume
            assert h0 < h1 and v0 < v1
            assert any(lo - 1e-3 <= v0 and v1 <= hi + 1e-3 for lo, hi in xs)
    ro = a["ta_ro/box"]
    assert a["ta_ro/panel"].tolist() == [2, 2]
    np.testing.assert_allclose(ro[:, 2], a["ta/tick0"])
    np.testing.assert_allclose(ro[:, 3], a["ta/tick1"])


@needs_pdhd
def test_event_without_candidates():
    from pylario import EventFile
    from pylarevd.display import EventDisplay
    from pylarevd.web.trigger import export_trigger
    ev = EventFile(PDHD)[0]
    hits = ev.hits()
    arrays, meta = {}, {}
    export_trigger(ev, hits, [("", EventDisplay(ev, hits=hits))], arrays, meta)
    assert meta["trigger"]["n_ta"] == 0 and meta["trigger"]["n_tc"] == 0
    assert len(arrays["tp/channel"]) == 55206
    assert len(arrays["ta2d/box"]) == 0 and len(arrays["tc2d/box"]) == 0
    assert np.all(arrays["tp/ta"] == -1)


@needs_pdhd
def test_readout_units_pdhd():
    from pylario import EventFile
    from pylarevd.web.trigger import readout_units
    cv, cu, units = readout_units(EventFile(PDHD).geometry)
    assert [u["name"] for u in units] == ["APA 1", "APA 2", "APA 3", "APA 4"]
    assert [u["tpcs"] for u in units] == [[0, 1], [2, 3], [4, 5], [6, 7]]
    for k in range(4):
        assert [int(((cu == k) & (cv == v)).sum()) for v in (0, 1, 2)] == [800, 800, 960]
