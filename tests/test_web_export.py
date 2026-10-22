"""EVDB bundle format, exporter and cache tests."""
import io
import os
import struct
import threading

import numpy as np
import pytest

from pylarevd.web.bundle import (DTYPES, bundle_bytes, read_bundle,
                                 write_bundle)

FD = os.environ.get("PYLAREVD_ATMNU",
                    "/home/pigrange/projects/data/reco/atmnu_radio_18643_1_20260727T121224Z_reco.root")
PD = ("/home/pigrange/projects/data/pdsp/PDSPProd4a_protoDUNE_sp_reco_stage1_"
      "p1GeV_35ms_sce_datadriven_18803601_67_20210414T032355Z.root")


# ---- format ------------------------------------------------------------

def test_roundtrip_all_dtypes_and_alignment():
    rng = np.random.default_rng(1)
    arrays = {}
    for code, dt in DTYPES.items():
        if dt.kind == "f":
            a = rng.normal(size=(7, 3)).astype(dt)
        else:
            info = np.iinfo(dt)
            a = rng.integers(info.min, info.max, size=(5, 2), endpoint=True).astype(dt)
        arrays[f"x/{code}"] = a
    arrays["odd/u8"] = np.arange(5, dtype=np.uint8)       # 5 bytes: forces padding
    arrays["empty"] = np.zeros((0, 3), np.float32)
    arrays["scalar_like"] = np.array([3.5], np.float64)
    meta = {"id": [1, 2, 3], "text": "héllo", "nan": float("nan")}
    blob = bundle_bytes("event", meta, arrays)
    assert blob[:4] == b"EVDB"
    (hlen,) = struct.unpack_from("<I", blob, 4)
    header, out = read_bundle(blob)
    assert header["kind"] == "event" and header["version"] == 1
    assert header["meta"]["text"] == "héllo" and header["meta"]["nan"] is None
    start = 8 + hlen
    start += (-start) % 8
    for name, a in arrays.items():
        e = header["arrays"][name]
        assert e["offset"] % 8 == 0
        assert (start + e["offset"]) % 8 == 0
        assert e["dtype"] in DTYPES and e["shape"] == list(a.shape)
        assert out[name].dtype == DTYPES[e["dtype"]]
        np.testing.assert_array_equal(out[name], a)
    last = header['arrays']['scalar_like']
    assert len(blob) == start + last['offset'] + 8   # padded to 8


def test_dtype_codes_exact():
    blob = bundle_bytes("event", {}, {
        "a": np.zeros(1, np.float32), "b": np.zeros(1, np.float64),
        "c": np.zeros(1, np.int32), "d": np.zeros(1, np.uint32),
        "e": np.zeros(1, np.int16), "f": np.zeros(1, np.uint16),
        "g": np.zeros(1, np.int8), "h": np.zeros(1, np.uint8),
        "i": np.zeros(1, bool)})
    h, _ = read_bundle(blob)
    assert [h["arrays"][k]["dtype"] for k in "abcdefghi"] == [
        "f32", "f64", "i32", "u32", "i16", "u16", "i8", "u8", "u8"]
    with pytest.raises(TypeError):
        bundle_bytes("event", {}, {"x": np.zeros(1, np.complex64)})


def test_ragged_offsets_roundtrip(tmp_path):
    lens = [3, 0, 5, 1]
    off = np.zeros(len(lens) + 1, np.uint32)
    off[1:] = np.cumsum(lens)
    pts = np.arange(off[-1] * 3, dtype=np.float32).reshape(-1, 3)
    p = tmp_path / "r.evdb"
    write_bundle(p, "event", {}, {"t/points": pts, "t/offsets": off})
    _, a = read_bundle(str(p))
    for i, n in enumerate(lens):
        np.testing.assert_array_equal(a["t/points"][a["t/offsets"][i]:a["t/offsets"][i + 1]],
                                      pts[off[i]:off[i + 1]])
    buf = io.BytesIO()
    write_bundle(buf, "event", {}, {"t/points": pts})
    assert read_bundle(buf.getvalue())[1]["t/points"].shape == pts.shape


def test_bad_magic():
    with pytest.raises(ValueError):
        read_bundle(b"NOPE" + b"\0" * 20)


# ---- polyline projection ------------------------------------------------

def test_project_polylines_ragged(tmp_path):
    from pylarevd.web.export import project_polylines

    class Fake:
        def project_points(self, panel, x, y, z, clip=True):
            return y.copy(), x.copy()
    polys = [np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2.]]), np.zeros((1, 3)),
             np.array([[0, 0, 0], [100, 0, 0], [101, 0, 0.]])]  # 100 cm step: break
    pts, off, own = project_polylines(Fake(), None, polys)
    assert list(own) == [0, 2]       # 2nd too short; 3rd keeps only its 2-point run
    assert list(off) == [0, 3, 5]
    assert len(off) == len(own) + 1 and off[-1] == len(pts)


def test_stack_polylines_matches_the_per_polyline_stack():
    """The vectorised stack is row for row what stacking each polyline with a NaN
    separator gave, and reusing it across panels changes nothing."""
    from pylarevd.web.export import _stack_polylines, project_polylines
    rng = np.random.default_rng(3)
    polys = [rng.normal(0, 50, (n, 3)) for n in (5, 0, 1, 2, 40, 1, 3)] + \
            [[[0, 0, 0], [1, 2, 3]]]                     # plain lists are accepted too
    stacked, owner, gap = _stack_polylines(polys)
    idx = [i for i, p in enumerate(polys) if len(p) >= 2]
    sep = np.full((1, 3), np.nan)
    ref = np.vstack([np.vstack([np.asarray(polys[i], float), sep]) for i in idx])
    np.testing.assert_array_equal(stacked, ref)
    assert owner.tolist() == sum(([i] * (len(polys[i]) + 1) for i in idx), [])
    assert gap.dtype == bool and len(gap) == len(stacked)
    assert _stack_polylines([np.zeros((1, 3)), []]) == (None, None, None)

    class Fake:
        def project_points(self, panel, x, y, z, clip=True):
            return y.copy() + panel, x.copy()
    for panel in (0, 1):
        a = project_polylines(Fake(), panel, polys)
        b = project_polylines(Fake(), panel, polys, (stacked, owner, gap))
        for u, v in zip(a, b):
            np.testing.assert_array_equal(u, v)


def test_shower_cones_with_index_match_the_plain_list():
    """with_index only adds the shower each outline belongs to."""
    from pylarevd.display import EventDisplay

    class Showers:
        start = np.array([[0., 0, 0], [10, 10, 10], [5, 5, 5]])
        direction = np.array([[0., 0, 1], [1, 0, 0], [0, 0, 0]])       # third: no axis
        length = np.array([50., 30, 20])
        open_angle = np.array([0.2, 0.1, 0.3])

        def __len__(self):
            return 3

    d = EventDisplay.__new__(EventDisplay)
    d.showers = Showers()
    d.project_points = lambda panel, x, y, z, clip=True: (y + z, x)
    plain = d._shower_wx(None)
    indexed = d._shower_wx(None, with_index=True)
    assert [j for j, _, _ in indexed] == [0, 1]
    for (w, x), (_, w2, x2) in zip(plain, indexed):
        np.testing.assert_array_equal(w, w2)
        np.testing.assert_array_equal(x, x2)


# ---- cache -----------------------------------------------------------------

def test_cache_atomic_and_keyed(tmp_path, monkeypatch):
    monkeypatch.setenv("PYLAREVD_CACHE", str(tmp_path / "c"))
    from pylarevd.web import cache
    f = tmp_path / "x.root"
    f.write_bytes(b"abc")
    d1 = cache.cache_dir(f)
    f.write_bytes(b"abcd")
    assert cache.cache_dir(f) != d1       # size changed -> new key
    dest = os.path.join(d1, "t.evdb")

    def w(i):
        cache._atomic_write(dest, b"EVDB" + bytes([i]) * 1000)
    ts = [threading.Thread(target=w, args=(i,)) for i in range(8)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    data = open(dest, "rb").read()
    assert len(data) == 1004 and len(set(data[4:])) == 1
    assert [n for n in os.listdir(d1) if n.startswith(".tmp")] == []


# ---- real files --------------------------------------------------------------

needs_fd = pytest.mark.skipif(not os.path.exists(FD), reason="FD sample missing")
needs_pd = pytest.mark.skipif(not os.path.exists(PD), reason="PDSP sample missing")


@pytest.fixture(scope="module")
def fd_event():
    from pylario import EventFile
    from pylarevd.web.export import event_bundle
    f = EventFile(FD)
    return f, read_bundle(event_bundle(f[1], hit_tag="hitfd"))


def _check_offsets(a, prefix, n_items=None):
    off = a[f"{prefix}/offsets"]
    assert off[0] == 0 and np.all(np.diff(off.astype(np.int64)) >= 0)
    assert off[-1] == len(a[f"{prefix}/points"])
    if n_items is not None:
        assert len(off) == n_items + 1


@needs_fd
def test_fd_event_content(fd_event):
    f, (h, a) = fd_event
    m = h["meta"]
    ev = f[1]
    assert m["id"] == list(ev.id) and m["entry"] == 1 and m["hit_tag"] == "hitfd"
    assert all(m["available"][k] for k in
               ("hits", "spacepoints", "tracks", "showers", "vertices",
                "truth_mc", "deposits", "optical", "links"))
    n = len(ev.hits("hitfd"))
    for k in ("panel", "h", "v", "charge", "amplitude", "tick", "multiplicity", "channel"):
        assert a[f"hits/{k}"].shape == (n,)
    assert a["hits/panel"].max() < len(m["panels"]) or (a["hits/panel"] == 255).any()
    assert set(("title", "hlabel", "vlabel", "hrange", "vrange", "anodes", "edges")) <= set(m["panels"][0])
    for kind in m["groups"]:
        g = a[f"hits/group/{kind}"]
        assert g.shape == (n,) and g.max() < m["groups"][kind]
    _check_offsets(a, "tracks", len(ev.tracks()))
    _check_offsets(a, "mc", len(a["mc/pdg"]))
    for pre, key in (("tracks2d", "track"), ("mc2d", "index"), ("showers2d", "shower")):
        _check_offsets(a, pre, len(a[f"{pre}/panel"]))
        assert len(a[f"{pre}/{key}"]) == len(a[f"{pre}/panel"])
    assert a["mc/from_interaction"].sum() == m["truth"]["n_from_interaction"] > 0
    assert len(a["mc/process"]) == len(a["mc/pdg"]) and a["mc/process"].max() < len(m["mc_processes"])
    assert len(a["dep/xyz"]) == len(a["dep/from_interaction"]) == len(a["dep/energy"])
    # optical
    assert np.all(a["oph/flash"] < len(a["flash/time"]))
    assert a["flash/wall_share"].shape[0] == len(a["flash/time"])
    assert np.all(np.diff(a["light/edges"]) > 0)
    assert m["optical"]["t0"] == pytest.approx(
        0.5 * (m["optical"]["window"][0] + m["optical"]["window"][1]), abs=6)
    assert any("hits" in s for s in m["summary"]) and m["truth"]["headline"]
    assert len(m["truth"]["vertex"]) == 3


@needs_fd
def test_fd_panel_groupings(fd_event):
    f, (h, a) = fd_event
    from pylarevd.display import EventDisplay
    m = h["meta"]
    n = len(f[1].hits("hitfd"))
    for sf, mg in (("_view", "view"), ("_none", "none")):
        d = EventDisplay(f[1], hits=f[1].hits("hitfd"), merge=mg)
        assert [p["title"] for p in m[f"panels{sf}"]] == [p.title for p in d.panels]
        assert all(p["faces"] for p in m[f"panels{sf}"])
        assert a[f"hits/panel{sf}"].shape == (n,)
        for k, p in enumerate(d.panels):
            assert np.array_equal(a[f"hits/panel{sf}"] == k, p.mask)
        for pre, key in (("tracks2d", "track"), ("mc2d", "index"), ("showers2d", "shower"),
                         ("vtx2d", None)):
            if pre == "vtx2d":
                assert len(a[f"vtx2d{sf}/hv"]) == len(a[f"vtx2d{sf}/panel"]) == len(a[f"vtx2d{sf}/index"])
                assert a[f"vtx2d{sf}/panel"].max() < len(m[f"panels{sf}"])
                continue
            _check_offsets(a, f"{pre}{sf}", len(a[f"{pre}{sf}/panel"]))
            assert len(a[f"{pre}{sf}/{key}"]) == len(a[f"{pre}{sf}/panel"])
            assert a[f"{pre}{sf}/panel"].max() < len(m[f"panels{sf}"])
        assert len(m["truth"][f"vertex2d{sf}"]) == len(d.panels)
    # the finest grouping never has fewer panels than the coarsest
    assert len(m["panels_none"]) >= len(m["panels_view"])


@needs_fd
def test_fd_spacepoint_chisq(fd_event):
    f, (h, a) = fd_event
    assert a["sp/chisq"].shape == (len(a["sp/xyz"]),) and a["sp/chisq"].dtype == np.float32


@needs_fd
def test_fd_panel_matches_display(fd_event):
    f, (h, a) = fd_event
    from pylarevd.display import EventDisplay
    d = EventDisplay(f[1], hits=f[1].hits("hitfd"))
    assert [p["title"] for p in h["meta"]["panels"]] == [p.title for p in d.panels]
    for k, p in enumerate(d.panels):
        assert np.array_equal(a["hits/panel"] == k, p.mask)


@needs_fd
def test_geometry_bundle():
    from pylario import EventFile
    from pylarevd.web.export import geometry_bundle
    g = EventFile(FD).geometry
    h, a = read_bundle(geometry_bundle(g))
    assert h["kind"] == "geometry" and h["meta"]["detector"] == g.detector
    assert len(h["meta"]["tpcs"]) == len(g.tpcs) and len(h["meta"]["tpcs"][0]) == 6
    n = len(a["pd/xyz"])
    assert a["pd/half"].shape == (n, 2) and a["pd/wall"].shape == (n,)
    assert len(a["pd/chan_offsets"]) == n + 1 and a["pd/chan_offsets"][-1] == len(a["pd/channels"])
    ct = a["pd/chan_tile"]
    assert ct.max() == n - 1 and len(h["meta"]["walls"]) >= 1


@needs_fd
def test_cache_roundtrip_and_cli(tmp_path, monkeypatch):
    monkeypatch.setenv("PYLAREVD_CACHE", str(tmp_path / "c"))
    from pylarevd.web import cache, export
    d = cache.get_or_build_event(FD, 0, "hitfd", None)
    assert cache.get_or_build_event(FD, 0, "hitfd", None) == d
    assert os.path.exists(cache.entry_path(FD, 0, "hitfd", None))
    assert cache.get_or_build_geometry(FD)[:4] == b"EVDB"
    assert export.main([FD, "-e", "1-2", "-j", "2", "--hits", "hitfd"]) == 0
    for e in (1, 2):
        assert os.path.exists(cache.entry_path(FD, e, "hitfd", None))


@needs_pd
def test_pdsp_event_degrades_gracefully():
    from pylario import EventFile
    from pylarevd.web.export import event_bundle
    f = EventFile(PD)
    h, a = read_bundle(event_bundle(f[3], hit_tag="hitpdune"))
    m = h["meta"]
    assert m["available"]["hits"] and m["available"]["truth_mc"]
    assert not m["available"]["deposits"]
    assert any("SimEnergyDeposit" in n for n in m["notes"])
    assert "dep/xyz" not in a
    assert a["mc/from_interaction"].sum() == m["truth"]["n_from_interaction"]
    assert len(a["mc/pdg"]) > 100_000


# ---- reconstructed-particle / slice tables (selection and highlighting) --------

def _check_pfp_tables(h, a, n_hits, n_sp):
    m = h["meta"]
    assert m["available"]["pfp"] and m["available"]["slice"]
    n = len(a["pfp/id"])
    for k in ("pdg", "parent", "top", "slice", "track", "shower", "n_hits", "n_sp", "vertex",
              "true_mc", "true_pdg", "true_tid", "score"):
        assert len(a[f"pfp/{k}"]) == n, k
    assert a["pfp/n_hits_plane"].shape == (n, 3)
    g = a["hits/group/pfparticle"]
    assert len(g) == n_hits and g.max() < n
    assert a["pfp/n_hits"].sum() == (g >= 0).sum()
    assert a["pfp/n_hits_plane"].sum() <= a["pfp/n_hits"].sum()
    # parent <-> daughter symmetry
    off, dau, par = a["pfp/daughter_offsets"].astype(int), a["pfp/daughters"], a["pfp/parent"]
    assert len(off) == n + 1 and off[-1] == len(dau)
    for p in range(n):
        for d in dau[off[p]:off[p + 1]]:
            assert par[d] == p
    for c in range(n):
        if par[c] >= 0:
            assert c in dau[off[par[c]]:off[par[c] + 1]]
        assert par[a["pfp/top"][c]] == -1
    # tracks / showers map to PFParticles that point back at them
    for what in ("track", "shower"):
        key = f"{what}s/pfp"
        if key in a:
            for i, p in enumerate(a[key]):
                assert p < n
                if p >= 0:
                    assert a[f"pfp/{what}"][p] >= 0
        row = a[f"pfp/{what}"]
        if key in a:
            assert (row[row >= 0] < len(a[key])).all()
    # space-point groups
    sp = a["sp/group/pfparticle"]
    assert len(sp) == n_sp and sp.max() < n
    assert a["pfp/n_sp"].sum() == (sp >= 0).sum()
    # slices
    ns = len(a["slice/id"])
    assert a["hits/group/slice"].max() < ns and a["sp/group/slice"].max() < ns
    assert a["slice/n_hits"].sum() == (a["hits/group/slice"] >= 0).sum()
    assert a["slice/n_hits_plane"].shape == (ns, 3) and a["slice/vertex"].shape == (ns, 3)
    assert (a["pfp/slice"] < ns).all()
    assert a["slice/n_pfp"].sum() == (a["pfp/slice"] >= 0).sum()
    top = a["slice/top"]
    assert (top[top >= 0] < n).all() and (par[top[top >= 0]] == -1).all()


@needs_fd
def test_fd_pfp_and_slice_tables(fd_event):
    f, (h, a) = fd_event
    _check_pfp_tables(h, a, len(a["hits/h"]), len(a["sp/xyz"]))
    # the FD neutrino hypothesis is the top-level particle of the slice
    pdg = a["pfp/pdg"]
    assert abs(pdg[a["slice/top"][0]]) in (12, 14, 16) and a["slice/nu"][0] == 1
    # every exported track maps to a track-like PFParticle row
    assert (a["tracks/pfp"] >= 0).all()
    assert (a["pfp/n_hits"][a["tracks/pfp"]] > 0).all()
    assert h["meta"]["pfp"]["n"] == len(pdg)


@needs_pd
def test_pdsp_pfp_and_slice_tables():
    from pylario import EventFile
    from pylarevd.web.export import event_bundle
    f = EventFile(PD)
    h, a = read_bundle(event_bundle(f[3], hit_tag="hitpdune"))
    _check_pfp_tables(h, a, len(a["hits/h"]), len(a["sp/xyz"]))
    assert len(a["pfp/id"]) > 500
    assert "showers/pfp" not in a                  # PDSP showers are unreadable
    assert (a["tracks/pfp"] >= 0).all()


def test_pfp_tables_degrade_without_pfparticles(monkeypatch):
    """Hit/space-point group ids survive when the PFParticle collection cannot be read."""
    from pylario import EventFile
    from pylarevd.web import export
    if not os.path.exists(FD):
        pytest.skip("FD sample missing")
    f = EventFile(FD)
    ev = f[1]
    orig = type(ev)._pick

    def pick(self, classname, tag, prefer=()):
        if classname == "recob::PFParticle":
            raise RuntimeError("unreadable")
        return orig(self, classname, tag, prefer)
    monkeypatch.setattr(type(ev), "_pick", pick)
    h, a = read_bundle(export.event_bundle(ev, hit_tag="hitfd"))
    assert not h["meta"]["available"]["pfp"]
    assert any(n.startswith("pfp:") for n in h["meta"]["notes"])
    assert "pfp/id" not in a
    assert "hits/group/pfparticle" in a and "hits/group/slice" in a
    assert h["meta"]["available"]["slice"] and len(a["slice/id"]) >= 1


# ---- true-particle generator tags and deposit -> trajectory rows (truth filter) -----

def test_rows_of_vectorised_lookup():
    from pylarevd.web.export import _rows_of
    keys = np.array([7, 3, 11, 5, 2], np.int32)
    q = np.array([5, 2, 99, 7, 3, 11, 0, 5, -3], np.int64)
    got = _rows_of(keys, q)
    want = np.array([3, 4, -1, 0, 1, 2, -1, 3, -1], np.int32)
    np.testing.assert_array_equal(got, want)
    assert got.dtype == np.int32
    assert _rows_of(np.zeros(0, np.int32), q).tolist() == [-1] * len(q)
    assert len(_rows_of(keys, np.zeros(0, np.int64))) == 0
    # a value above every key must not run off the end
    assert _rows_of(keys, np.array([1000])).tolist() == [-1]


def test_bundle_version_bumped_for_truth_filter():
    from pylarevd.web import cache
    assert cache.BUNDLE_VERSION >= 3


@needs_fd
def test_fd_mc_generator_and_deposit_rows(fd_event):
    f, (h, a) = fd_event
    m = h["meta"]
    n = len(a["mc/pdg"])
    g = a["mc/generator"]
    assert g.dtype == np.int16 and g.shape == (n,)
    names = m["mc_generators"]
    assert "generator" in names and "Ar39GenInLAr" in names
    assert g.min() >= -1 and g.max() < len(names)
    # the neutrino interaction is exactly the particles made by the "generator" module
    fi = a["mc/from_interaction"].astype(bool)
    gen_idx = names.index("generator")
    assert np.array_equal(g == gen_idx, fi) or (g[fi] == gen_idx).all()
    # deposits: row of the owning trajectory (EM-shower daughters carry a negated track id)
    dm = a["dep/mc"]
    assert dm.dtype == np.int32 and dm.shape == a["dep/track_id"].shape
    ok = dm >= 0
    assert ok.any() and dm.max() < n
    np.testing.assert_array_equal(a["mc/track_id"][dm[ok]], np.abs(a["dep/track_id"][ok]))
    # an unmatched deposit has a track id that is not in mc/*
    absent = ~np.isin(np.abs(a["dep/track_id"]), a["mc/track_id"])
    assert np.array_equal(~ok, absent)
    # interaction flag of a deposit agrees with the one of its trajectory
    np.testing.assert_array_equal(a["dep/from_interaction"][ok], a["mc/from_interaction"][dm[ok]])


@needs_pd
def test_pdsp_mc_generator():
    from pylario import EventFile
    from pylarevd.web.export import event_bundle
    f = EventFile(PD)
    h, a = read_bundle(event_bundle(f[1], hit_tag="hitpdune"))
    names = h["meta"]["mc_generators"]
    assert "cosmicgenerator" in names and "generator" in names
    g = a["mc/generator"]
    assert g.shape == a["mc/pdg"].shape and g.max() < len(names)
    assert (g == names.index("cosmicgenerator")).sum() > 100_000
    assert "dep/mc" not in a            # no SimEnergyDeposit product in this file


@needs_fd
def test_generator_missing_degrades(monkeypatch):
    """No MCParticle-MCTruth association: no mc/generator, a note, everything else unchanged."""
    import dataclasses
    from pylario import EventFile
    from pylarevd.web import export
    f = EventFile(FD)
    ev = f[1]
    orig = type(ev).mc_particles

    def stripped(self, *a, **k):
        mc = orig(self, *a, **k)
        return dataclasses.replace(mc, generator=None, generators=())
    monkeypatch.setattr(type(ev), "mc_particles", stripped)
    h, a = read_bundle(export.event_bundle(ev, hit_tag="hitfd"))
    assert "mc/generator" not in a and "mc_generators" not in h["meta"]
    assert any("mc/generator" in n for n in h["meta"]["notes"])
    assert "mc/from_interaction" in a and "dep/mc" in a
