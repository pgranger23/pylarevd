"""Export a pylario Event / Geometry to EVDB bundles (see docs/WEB_VIEWER.md).

All physics (panel logic, wire projection, flash repair, flash-hit links, PD
walls, truth matching) is delegated to ``pylarevd.display`` / ``pylarevd.light``.
Every accessor can fail on a given file; each section is exported in its own
try/except, flags ``meta.available`` accordingly and records a reason in
``meta.notes``.

CLI::

    python -m pylarevd.web.export FILE [-e all|0-5|3] [-j N] [--hits TAG] [--flash TAG]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

from .bundle import bundle_bytes

#: spacepoint/hit group kinds exported as ``hits/group/<kind>``
GROUP_KINDS = ("track", "shower", "pfparticle", "cluster", "slice")
NO_PANEL = 255
#: extra panel groupings shipped next to the default (orientation): suffix, EventDisplay merge
GROUPINGS = (("_view", "view"), ("_none", "none"))
#: above this many true-trajectory points, mc2d covers from_interaction only
MC2D_FULL_MAX_POINTS = 400_000
#: above this many true particles, labels match against from_interaction ones only
LABEL_FULL_MAX_PARTICLES = 100_000


# --------------------------------------------------------------------------
# helpers


def _f32(a):
    return np.asarray(a, np.float32)


def _stack(chunks, shape_tail, dtype=np.float32):
    chunks = [np.asarray(c) for c in chunks if len(c)]
    if not chunks:
        return np.zeros((0,) + shape_tail, dtype)
    return np.concatenate(chunks).astype(dtype, copy=False)


def _rows_of(keys, query):
    """Row of each ``query`` value in ``keys`` (-1 if absent), vectorised (sort + searchsorted).

    ``keys`` are unique ids (MCParticle track ids); with duplicates the first row wins.
    """
    keys = np.asarray(keys, np.int64)
    query = np.asarray(query, np.int64)
    out = np.full(len(query), -1, np.int32)
    if not len(keys) or not len(query):
        return out
    order = np.argsort(keys, kind="stable")
    sk = keys[order]
    pos = np.searchsorted(sk, query)
    pos[pos >= len(sk)] = len(sk) - 1
    ok = sk[pos] == query
    out[ok] = order[pos[ok]]
    return out


def _ragged(polys):
    """List of (n_i, 3) arrays -> (points f32, offsets u32)."""
    lens = np.array([len(p) for p in polys], np.int64)
    offsets = np.zeros(len(polys) + 1, np.uint32)
    offsets[1:] = np.cumsum(lens)
    pts = (np.concatenate([np.asarray(p, np.float64).reshape(-1, 3) for p in polys])
           if lens.sum() else np.zeros((0, 3)))
    return pts.astype(np.float32), offsets


def project_polylines(d, panel, polys):
    """Project 3-D polylines into one panel, vectorised.

    Same rules as ``EventDisplay._polylines`` (face clipping, break at a gap
    longer than MAX_STEP_CM) but keeps the owner of each run.  Returns
    ``(points (k,2) f32, offsets u32, owner i32)`` -- one run per row of
    ``offsets`` -- or None when nothing lands in the panel.
    """
    from ..display import MAX_STEP_CM
    idx = [i for i, p in enumerate(polys) if len(p) >= 2]
    if not idx:
        return None
    sep = np.full((1, 3), np.nan)
    stacked = np.vstack([np.vstack([np.asarray(polys[i], np.float64), sep]) for i in idx])
    owner_row = np.concatenate([np.full(len(polys[i]) + 1, i, np.int64) for i in idx])
    pr = d.project_points(panel, stacked[:, 0], stacked[:, 1], stacked[:, 2])
    if pr is None:
        return None
    w, x = pr
    valid = np.isfinite(w) & np.isfinite(x)
    with np.errstate(invalid="ignore"):
        step = np.linalg.norm(np.diff(stacked, axis=0), axis=1)
    gap = np.zeros(len(stacked), bool)
    gap[1:] = step > MAX_STEP_CM
    prev_valid = np.r_[False, valid[:-1]]
    new = valid & (~prev_valid | gap)
    run = np.cumsum(new) - 1
    run = np.where(valid, run, -1)
    nrun = int(new.sum())
    if not nrun:
        return None
    counts = np.bincount(run[valid], minlength=nrun)
    keep_run = counts >= 2
    sel = valid & keep_run[np.maximum(run, 0)]
    if not sel.any():
        return None
    first = np.flatnonzero(new)
    kept = np.flatnonzero(keep_run)
    offsets = np.zeros(len(kept) + 1, np.uint32)
    offsets[1:] = np.cumsum(counts[kept])
    pts = np.column_stack([w[sel], x[sel]]).astype(np.float32)
    return pts, offsets, owner_row[first[kept]].astype(np.int32)


def _add_projected(arrays, prefix, key_name, d, polys, panels):
    """Project ``polys`` into every panel; store ``<prefix>/{points,offsets,panel,<key>}``."""
    pts, offs, pan, own = [], [], [], []
    base = 0
    for k, panel in enumerate(panels):
        r = project_polylines(d, panel, polys)
        if r is None:
            continue
        p, o, ow = r
        pts.append(p)
        offs.append(o[:-1].astype(np.int64) + base)
        base += len(p)
        pan.append(np.full(len(ow), k, np.uint8))
        own.append(ow)
    arrays[f"{prefix}/points"] = _stack(pts, (2,))
    off = np.concatenate(offs + [np.array([base])]) if offs else np.zeros(1)
    arrays[f"{prefix}/offsets"] = off.astype(np.uint32)
    arrays[f"{prefix}/panel"] = _stack(pan, (), np.uint8)
    arrays[f"{prefix}/{key_name}"] = _stack(own, (), np.int32)


def _tag_of(product: str) -> str:
    parts = str(product).split("_")
    return parts[1] if len(parts) > 1 else str(product)


def _fmt_exc(exc) -> str:
    msg = str(exc).strip().splitlines()[0] if str(exc).strip() else ""
    return f"{type(exc).__name__}: {msg[:140]}"


# --------------------------------------------------------------------------
# geometry


def geometry_bundle(geometry) -> bytes:
    """Geometry bundle: TPC boxes and photon-detector tiles/walls."""
    from ..light import PhotonDetectorWalls
    meta = {"detector": str(geometry.detector), "tpcs": [], "walls": [],
            "notes": []}
    arrays = {}
    for b in geometry.tpcs:
        meta["tpcs"].append([float(b.xmin), float(b.ymin), float(b.zmin),
                             float(b.xmax), float(b.ymax), float(b.zmax)])
    try:
        pw = PhotonDetectorWalls.from_geometry(geometry)
    except Exception as exc:
        meta["notes"].append(f"no photon-detector walls: {_fmt_exc(exc)}")
        pw = None
    if pw is not None:
        arrays["pd/xyz"] = _f32(np.column_stack([pw.x, pw.y, pw.z]))
        arrays["pd/half"] = _f32(np.column_stack([pw.half_y, pw.half_z]))
        arrays["pd/wall"] = np.asarray(pw.wall, np.uint8)
        lens = np.array([len(c) for c in pw.channels], np.int64)
        off = np.zeros(len(lens) + 1, np.uint32)
        off[1:] = np.cumsum(lens)
        arrays["pd/chan_offsets"] = off
        arrays["pd/channels"] = (np.concatenate([np.asarray(c) for c in pw.channels])
                                 if lens.sum() else np.zeros(0)).astype(np.uint32)
        arrays["pd/chan_tile"] = np.asarray(pw.chan_tile, np.int32)
        meta["projected"] = bool(pw.projected)
        for w in range(pw.n_walls):
            zlo, zhi, ylo, yhi = pw.extent(w)
            meta["walls"].append({"x": float(pw.wall_x[w]), "label": pw.wall_label(w),
                                  "extent": [zlo, zhi, ylo, yhi]})
    return bundle_bytes("geometry", meta, arrays)


# --------------------------------------------------------------------------
# event


def _panel_meta(d, k, panel):
    from ..display import _anode_positions, _drift_bounds
    hx, hy, xlabel, ylabel = d._axes
    a, b = hx[panel.mask], hy[panel.mask]
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    anodes, edges = [], []
    if d.space != "readout":
        try:
            anodes = _anode_positions(d.hits, d.geometry, panel)
            lo, hi = _drift_bounds(d.hits, d.geometry, panel)
            edges = [float(lo), float(hi)] if np.isfinite(lo) else []
        except Exception:
            pass
    vlo = float(min(b.min(), *edges, *anodes)) if len(b) else 0.0
    vhi = float(max(b.max(), *edges, *anodes)) if len(b) else 1.0
    hlo, hhi = (float(a.min()), float(a.max())) if len(a) else (0.0, 1.0)
    pad_h, pad_v = 0.03 * (hhi - hlo or 1.0), 0.03 * (vhi - vlo or 1.0)
    faces = []
    if d.space != "readout":
        # Everything not shipped pre-projected (deposits, ...) can be projected
        # client side exactly as EventDisplay.project_points does: the first face
        # whose boxes contain the point claims it, h = dir[0]*y + dir[1]*z, v = x.
        for p, boxes in d._faces(panel):
            faces.append({"dir": [float(p[0]), float(p[1])],
                          "boxes": [[b.xmin, b.ymin, b.zmin, b.xmax, b.ymax, b.zmax]
                                    for b in boxes]})
    return {"faces": faces, "title": panel.title, "key": int(panel.key), "view": int(panel.view),
            "drift_sign": int(panel.drift_sign), "n_hits": int(panel.mask.sum()),
            "hlabel": xlabel, "vlabel": ylabel,
            "hrange": [hlo - pad_h, hhi + pad_h], "vrange": [vlo - pad_v, vhi + pad_v],
            "anodes": [float(v) for v in anodes], "edges": edges}


def event_bundle(event, *, hit_tag=None, flash_tag=None) -> bytes:
    """Event bundle (``kind: "event"``); see the WEB_VIEWER.md contract."""
    from ..display import EventDisplay
    t_start = time.time()
    ev = event
    arrays: dict = {}
    notes: list[str] = []
    avail = dict.fromkeys(("hits", "spacepoints", "tracks", "showers", "vertices",
                           "truth_mc", "deposits", "optical", "links", "pfp", "slice",
                           "trigger"), False)
    timing: dict = {}
    geometry = ev.geometry
    src = ev._src
    meta = {"id": [int(i) for i in ev.id], "entry": int(ev.entry),
            "file": os.path.basename(src.path), "detector": str(geometry.detector),
            "available": avail, "notes": notes, "timing": timing,
            "summary": [], "groups": {}, "panels": [], "truth": {}}

    def section(name, fn):
        t = time.time()
        try:
            return fn()
        except Exception as exc:
            notes.append(f"{name}: {_fmt_exc(exc)}")
            return None
        finally:
            timing[name] = round(time.time() - t, 3)

    try:
        meta["hit_tags"] = sorted({_tag_of(p) for p in src.hit_products()})
    except Exception:
        meta["hit_tags"] = []
    try:
        meta["flash_tags"] = sorted({_tag_of(p) for p in
                                     src.art.find_product("recob::OpFlash")})
    except Exception:
        meta["flash_tags"] = []
    meta["hit_tag"] = hit_tag
    meta["flash_tag"] = flash_tag

    # ---- reco and truth objects --------------------------------------
    hits = section("hits", lambda: ev.hits(hit_tag))
    tracks = section("tracks", ev.tracks)
    showers = section("showers", ev.showers)
    vertices = section("vertices", ev.vertices)
    spacepoints = section("spacepoints", ev.spacepoints)
    mc = section("mc", lambda: ev.mc_particles(min_points=2))
    deposits = section("deposits", ev.truth_deposits)
    if tracks is not None and not len(tracks):
        tracks = None
    if showers is not None and not len(showers):
        showers = None
    if vertices is not None and not len(vertices):
        vertices = None
    if mc is not None and not len(mc):
        mc = None
    if deposits is not None and not len(deposits):
        deposits = None

    # ---- truth bookkeeping -------------------------------------------
    ids = section("neutrino_track_ids", ev.neutrino_track_ids) if mc is not None or deposits is not None else None
    meta["truth"]["has_split"] = ids is not None
    inter = section("true_interaction", ev.true_interaction)
    tv = section("true_vertex", ev.true_vertex)
    meta["truth"]["vertex"] = ([float(v) for v in tv] if tv is not None else None)
    meta["truth"]["inside"] = None
    if tv is not None:
        con = section("containment", lambda: geometry.containment(tv))
        if con is not None:
            meta["truth"]["inside"] = bool(con.inside)
    meta["truth"]["headline"] = ""

    # ---- hits + panels -------------------------------------------------
    d = None
    extra: dict = {}     # suffix -> EventDisplay for the other panel groupings
    if hits is not None and len(hits):
        d = section("display", lambda: EventDisplay(
            ev, hits=hits, truth=deposits, tracks=tracks, vertices=vertices,
            showers=showers, mc=mc))
    if d is not None:
        avail["hits"] = True
        meta["hit_tag"] = _tag_of(hits.product)
        panels = d.panels
        pidx = np.full(len(hits), NO_PANEL, np.uint8)
        for k, p in enumerate(panels):
            pidx[p.mask] = k
        arrays["hits/panel"] = pidx
        arrays["hits/h"] = _f32(hits.w)
        arrays["hits/v"] = _f32(hits.x)
        arrays["hits/charge"] = _f32(hits.integral)
        arrays["hits/amplitude"] = _f32(hits.amplitude)
        arrays["hits/tick"] = _f32(hits.tick)
        arrays["hits/multiplicity"] = np.clip(hits.multiplicity, 0, 255).astype(np.uint8)
        arrays["hits/channel"] = np.asarray(hits.channel, np.uint32)
        arrays["hits/view"] = np.asarray(hits.view, np.int8)
        arrays["hits/chan_view"] = np.asarray(hits.chan_view, np.int8)
        arrays["hits/drift_sign"] = np.asarray(hits.drift_sign, np.int8)
        meta["panels"] = [_panel_meta(d, k, p) for k, p in enumerate(panels)]
        # the other two panel groupings (merge="view" / "none"), shipped as extra
        # panel sets so the browser switches between them without a refetch
        for sf, mg in GROUPINGS:
            def _extra(sf=sf, mg=mg):
                dx = EventDisplay(ev, hits=hits, truth=deposits, tracks=tracks,
                                  vertices=vertices, showers=showers, mc=mc, merge=mg)
                px = np.full(len(hits), NO_PANEL, np.uint8)
                for k, p in enumerate(dx.panels):
                    px[p.mask] = k
                arrays[f"hits/panel{sf}"] = px
                meta[f"panels{sf}"] = [_panel_meta(dx, k, p) for k, p in enumerate(dx.panels)]
                extra[sf] = dx
            section(f"panels{sf}", _extra)
        # the classic readout view (channel vs tick), one panel per plane
        try:
            from ..display import _readout_panels
            ro = []
            for p in _readout_panels(hits, geometry):
                a, b = hits.channel[p.mask].astype(float), hits.tick[p.mask]
                ro.append({"title": p.title, "view": int(p.view),
                           "hlabel": "channel", "vlabel": "peak time [ticks]",
                           "hrange": [float(a.min()), float(a.max())],
                           "vrange": [float(b.min()), float(b.max())],
                           "n_hits": int(p.mask.sum())})
            meta["panels_readout"] = ro
        except Exception as exc:
            notes.append(f"readout panels: {_fmt_exc(exc)}")
        meta["truth"]["headline"] = d._truth_headline() or ""
        if d.disambiguation_warning:
            notes.append(d.disambiguation_warning)

        # hit -> reco object groups (indices refer to the default collection)
        caps = section("capabilities", ev.capabilities) or {}
        default_tag = _tag_of(src.resolve_hit_product(None))
        n_default = len(ev.hits(None)) if meta["hit_tag"] != default_tag else len(hits)
        if n_default == len(hits):
            if meta["hit_tag"] != default_tag:
                notes.append(f"hit groups: associations index {default_tag}; "
                             f"{meta['hit_tag']} has the same hit count, so index "
                             "alignment is assumed")
            for kind in GROUP_KINDS:
                if caps.get(f"group:{kind}") is False:
                    continue
                g = section(f"group:{kind}", lambda kind=kind: ev.hit_group(kind))
                if g is not None and len(g) == len(hits):
                    arrays[f"hits/group/{kind}"] = np.asarray(g, np.int32)
                    meta["groups"][kind] = int(g.max()) + 1 if (g >= 0).any() else 0
        else:
            notes.append(f"hit groups not exported: associations index "
                         f"{default_tag} ({n_default} hits), not {meta['hit_tag']} "
                         f"({len(hits)} hits)")
        if d.panels:
            meta["summary"] = d.summary().splitlines()
    else:
        panels = []
        if hits is None or not len(hits):
            notes.append("no hits available" if hits is None else "hit collection is empty")
        if inter is not None:
            meta["summary"] = inter.describe().splitlines()
            meta["truth"]["headline"] = inter.headline()

    # ---- space points -------------------------------------------------
    if spacepoints is not None and len(spacepoints):
        avail["spacepoints"] = True
        arrays["sp/xyz"] = _f32(spacepoints.xyz)
        arrays["sp/charge"] = _f32(spacepoints.charge)
        for name in ("amplitude", "tick", "chisq"):
            v = getattr(spacepoints, name, None)
            if v is not None:
                arrays[f"sp/{name}"] = _f32(v)

    # ---- tracks / showers / vertices ----------------------------------
    t0 = time.time()
    if tracks is not None:
        avail["tracks"] = True
        arrays["tracks/points"], arrays["tracks/offsets"] = _ragged(tracks.points)
        arrays["tracks/id"] = np.asarray(tracks.id, np.int32)
        arrays["tracks/chi2"] = _f32(tracks.chi2)
        if d is not None:
            for sf, dd in [("", d)] + list(extra.items()):
                section(f"tracks2d{sf}", lambda sf=sf, dd=dd: _add_projected(
                    arrays, f"tracks2d{sf}", "track", dd, list(tracks.points), dd.panels))
    if showers is not None:
        avail["showers"] = True
        arrays["showers/start"] = _f32(showers.start)
        arrays["showers/dir"] = _f32(showers.direction)
        arrays["showers/length"] = _f32(showers.length)
        arrays["showers/angle"] = _f32(showers.open_angle)
        arrays["showers/id"] = np.asarray(showers.id, np.int32)
        if d is not None:
            def cones(sf, dd):
                polys, owner, pan = [], [], []
                full = dd.showers
                try:
                    for k, panel in enumerate(dd.panels):
                        for j in range(len(full)):
                            dd.showers = full[[j]]
                            for w, x in dd._shower_wx(panel):
                                polys.append(np.column_stack([w, x]))
                                owner.append(j)
                                pan.append(k)
                finally:
                    dd.showers = full
                lens = np.array([len(p) for p in polys], np.int64)
                off = np.zeros(len(polys) + 1, np.uint32)
                off[1:] = np.cumsum(lens)
                arrays[f"showers2d{sf}/points"] = (_f32(np.concatenate(polys)) if polys
                                                   else np.zeros((0, 2), np.float32))
                arrays[f"showers2d{sf}/offsets"] = off
                arrays[f"showers2d{sf}/panel"] = np.asarray(pan, np.uint8)
                arrays[f"showers2d{sf}/shower"] = np.asarray(owner, np.int32)
            for sf, dd in [("", d)] + list(extra.items()):
                section(f"showers2d{sf}", lambda sf=sf, dd=dd: cones(sf, dd))
    if vertices is not None:
        avail["vertices"] = True
        arrays["vtx/xyz"] = _f32(vertices.xyz)
        role = np.zeros(len(vertices), np.uint8)
        roles = section("vertex_roles", ev.vertex_roles)
        if roles:
            for code, key in ((1, "interaction"), (2, "primary_daughters"),
                              (3, "secondary")):
                for i in roles.get(key, []):
                    if i < len(role):
                        role[i] = code
        arrays["vtx/role"] = role
        arrays["vtx/id"] = np.asarray(vertices.id, np.int32)
        if d is not None:
            for sf, dd in [("", d)] + list(extra.items()):
                hv, pan, idx = [], [], []
                for k, panel in enumerate(dd.panels):
                    pr = dd.project_points(panel, vertices.xyz[:, 0], vertices.xyz[:, 1],
                                           vertices.xyz[:, 2])
                    if pr is None:
                        continue
                    ok = np.flatnonzero(np.isfinite(pr[0]) & np.isfinite(pr[1]))
                    hv.append(np.column_stack([pr[0][ok], pr[1][ok]]))
                    pan.append(np.full(len(ok), k, np.uint8))
                    idx.append(ok.astype(np.int32))
                arrays[f"vtx2d{sf}/hv"] = _stack(hv, (2,))
                arrays[f"vtx2d{sf}/panel"] = _stack(pan, (), np.uint8)
                arrays[f"vtx2d{sf}/index"] = _stack(idx, (), np.int32)
    timing["reco_overlays"] = round(time.time() - t0, 3)

    # ---- truth: trajectories + deposits -------------------------------
    if mc is not None:
        avail["truth_mc"] = True
        arrays["mc/points"], arrays["mc/offsets"] = _ragged(mc.points)
        arrays["mc/pdg"] = np.asarray(mc.pdg, np.int32)
        arrays["mc/track_id"] = np.asarray(mc.track_id, np.int32)
        arrays["mc/mother"] = np.asarray(mc.mother, np.int32)
        procs = [str(p) for p in mc.process]
        names = sorted(set(procs))
        lut = {n: i for i, n in enumerate(names)}
        arrays["mc/process"] = np.fromiter((lut[p] for p in procs), np.uint16, len(procs))
        meta["mc_processes"] = names
        fi = (np.isin(mc.track_id, ids).astype(np.uint8) if ids is not None
              else np.ones(len(mc), np.uint8))
        arrays["mc/from_interaction"] = fi
        gen = getattr(mc, "generator", None)
        if gen is not None:
            # which generator module (MCTruth product) made each particle, daughters included;
            # -1 = unknown. Names in meta["mc_generators"].
            arrays["mc/generator"] = np.asarray(gen, np.int16)
            meta["mc_generators"] = [str(g) for g in getattr(mc, "generators", ())]
        else:
            notes.append("no MCParticle-MCTruth association: mc/generator is missing, "
                         "the viewer can only separate the interaction from the rest")
        meta["truth"]["n_mc"] = int(len(mc))
        meta["truth"]["n_from_interaction"] = int(fi.sum())
        if ids is None:
            notes.append("no true vertex: mc/from_interaction is 1 for every particle")
        if d is not None:
            t = time.time()
            npts = int(arrays["mc/offsets"][-1])
            scope = "all"
            if npts > MC2D_FULL_MAX_POINTS and ids is not None:
                scope = "from_interaction"
                sel = np.flatnonzero(fi)
                notes.append(f"mc2d: {npts} trajectory points is too many to project "
                             f"in full; only the {len(sel)} particles with "
                             "from_interaction==1 are projected")
            else:
                sel = np.arange(len(mc))
            meta["truth"]["mc2d_scope"] = scope
            polys = [mc.points[i] for i in sel]

            def mc2d(sf, dd):
                _add_projected(arrays, f"mc2d{sf}", "index", dd, polys, dd.panels)
                # owner is the position in `polys`; make it an index into mc/*
                arrays[f"mc2d{sf}/index"] = sel[arrays[f"mc2d{sf}/index"]].astype(np.int32)
            for sf, dd in [("", d)] + list(extra.items()):
                section(f"mc2d{sf}", lambda sf=sf, dd=dd: mc2d(sf, dd))
            timing["mc2d_s"] = round(time.time() - t, 3)
    if deposits is not None:
        avail["deposits"] = True
        arrays["dep/xyz"] = _f32(np.column_stack([deposits.x, deposits.y, deposits.z]))
        arrays["dep/energy"] = _f32(deposits.edep)
        arrays["dep/track_id"] = np.asarray(deposits.track_id, np.int32)
        arrays["dep/pdg"] = np.asarray(deposits.pdg, np.int32)
        # EM-shower daughters carry a negated TrackID
        arrays["dep/from_interaction"] = (
            np.isin(np.abs(deposits.track_id), ids).astype(np.uint8)
            if ids is not None else np.ones(len(deposits), np.uint8))
        if mc is not None:
            arrays["dep/mc"] = _rows_of(mc.track_id, np.abs(deposits.track_id))
    if d is not None and tv is not None:
        for sf, dd in [("", d)] + list(extra.items()):
            v2 = []
            for panel in dd.panels:
                try:
                    v2.append(dd._true_vertex_wx(panel))
                except Exception:
                    v2.append(None)
            meta["truth"][f"vertex2d{sf}"] = [None if v is None else [v[0], v[1]] for v in v2]

    # ---- particle labels ----------------------------------------------
    if d is not None and (mc is not None) and (tracks is not None or showers is not None
                                                 or (len(mc) <= 3000 or ids is not None)):
        def labels():
            full_mc = d.mc
            if len(full_mc) > LABEL_FULL_MAX_PARTICLES and ids is not None:
                d.mc = full_mc[np.flatnonzero(fi)]
                notes.append("particle labels: reco objects matched to "
                             "from_interaction true particles only")
            try:
                lab = d._matched_particle_labels()
            finally:
                d.mc = full_mc
            if not lab:
                return
            kinds = {"track": 0, "shower": 1, "mc": 2}
            texts = sorted({l["text"] for l in lab})
            lut = {t: i for i, t in enumerate(texts)}
            by_tid = {int(t): i for i, t in enumerate(mc.track_id)}
            arrays["labels/pos"] = _f32([np.asarray(l["pos"], float) for l in lab])
            arrays["labels/text"] = np.array([lut[l["text"]] for l in lab], np.uint32)
            arrays["labels/kind"] = np.array([kinds[l["kind"]] for l in lab], np.uint8)
            arrays["labels/index"] = np.array(
                [by_tid.get(l["tid"], -1) if l["kind"] == "mc" else l["index"]
                 for l in lab], np.int32)
            arrays["labels/track_id"] = np.array([l["tid"] for l in lab], np.int32)
            arrays["labels/pdg"] = np.array([l["pdg"] for l in lab], np.int32)
            meta["label_texts"] = texts
            meta["label_kinds"] = ["track", "shower", "mc"]
        section("labels", labels)
    elif d is not None and mc is not None:
        notes.append("particle labels skipped: no reco tracks/showers and too many "
                     "true trajectories")

    # ---- reconstructed-particle / slice tables (highlighting) ----------
    section("pfp", lambda: _export_reco_objects(
        ev, hits, tracks, showers, spacepoints, vertices, mc, arrays, meta, avail, notes, section))

    # ---- optical ------------------------------------------------------
    def optical():
        from ..light import PE_FLOOR, LightAnimation
        anim = LightAnimation(ev, flash_tag=flash_tag)
        o = anim.op
        fl, hi = anim._split
        t, pe, tile, flash = anim._hits
        links, lnote = anim._links
        arrays["oph/time"] = np.asarray(t, np.float64)
        arrays["oph/pe"] = _f32(pe)
        arrays["oph/channel"] = np.asarray(o.channel, np.int32)[hi]
        arrays["oph/tile"] = np.asarray(tile, np.int32)
        arrays["oph/flash"] = np.asarray(flash, np.int32)
        f = lambda a: np.asarray(a, float)[fl]
        yw, zw = f(o.y_width), f(o.z_width)
        arrays["flash/time"] = f(o.time)
        arrays["flash/pe"] = _f32(f(o.pe))
        arrays["flash/y"] = _f32(f(o.y))
        arrays["flash/z"] = _f32(f(o.z))
        arrays["flash/yw"] = _f32(np.where(np.isfinite(yw) & (yw > 0), yw, 30.0))
        arrays["flash/zw"] = _f32(np.where(np.isfinite(zw) & (zw > 0), zw, 30.0))
        arrays["flash/x"] = _f32(f(o.x))
        nw = anim.walls.n_walls
        share = np.ones((len(fl), nw))
        if links is not None and nw > 1 and len(fl):
            share = np.zeros((len(fl), nw))
            m = (flash >= 0) & (tile >= 0)
            np.add.at(share, (flash[m], anim.walls.wall[tile[m]]), pe[m])
            tot = share.sum(axis=1, keepdims=True)
            share = np.where(tot > 0, share / np.where(tot > 0, tot, 1), 1.0)
        arrays["flash/wall_share"] = _f32(share)
        arrays["light/edges"] = np.asarray(anim.edges, np.float64)
        avail["optical"] = True
        avail["links"] = links is not None
        onotes = [n for n in (lnote, anim._missing_flash.strip()) if n]
        if anim.repaired:
            onotes.append(f"{anim.repaired} flashes had unusable (y, z); recomputed "
                          "from per-detector PE")
        if len(fl) and (not np.isfinite(f(o.x)).any()):
            onotes.append("flash finder left drift position x unset")
        lo, hi_w = anim.window
        meta["optical"] = {
            "t0": float(anim.t0), "quantum": float(anim.quantum),
            "window": [float(lo), float(hi_w)], "pe_floor": float(PE_FLOOR),
            "products": [anim.flash_product], "notes": onotes,
            "n_frames": int(anim.n_frames), "time_scale": anim.time_scale,
            "n_flashes": int(len(fl)), "n_ophits": int(len(hi)),
            "persistence": float(anim.persistence), "n_walls": int(nw)}
        if not meta["flash_tag"]:
            meta["flash_tag"] = _tag_of(anim.flash_product)
        meta["summary"].extend(anim.summary().splitlines())
    section("optical", optical)
    if not avail["optical"]:
        meta["optical"] = {"notes": [n for n in notes if n.startswith("optical")]}

    # ---- online trigger (TPs, TAs, TCs) ----------------------------------
    def trigger():
        from .trigger import export_trigger
        displays = ([("", d)] + list(extra.items())) if d is not None else []
        meta["summary"].extend(export_trigger(ev, hits, displays, arrays, meta))
        avail["trigger"] = True
    if src.art.find_product("dunedaq::trgdataformats::TriggerPrimitive"):
        section("trigger", trigger)

    timing["total"] = round(time.time() - t_start, 3)
    return bundle_bytes("event", meta, arrays)


# --------------------------------------------------------------------------
# PFParticle / slice tables (reconstructed-object highlighting)


_U64_MAX = np.uint64(18446744073709551615)
NEUTRINO_PDGS = (12, 14, 16)


def _first_by_key(left, right, n_left):
    """left -> first right (-1 if none) from an association's key arrays."""
    out = np.full(n_left, -1, np.int64)
    for l, r in zip(left[::-1], right[::-1]):     # reversed so the first one wins
        if 0 <= l < n_left:
            out[l] = r
    return out


def _plane_counts(group, chan_view, n, n_planes=3):
    """(n, n_planes) int32: hits per group id and wire plane."""
    out = np.zeros((n, n_planes), np.int32)
    group = np.asarray(group, np.int64)
    ok = (group >= 0) & (group < n) & (chan_view >= 0) & (chan_view < n_planes)
    if ok.any():
        flat = np.bincount(group[ok] * n_planes + chan_view[ok].astype(np.int64),
                           minlength=n * n_planes)
        out[:] = flat[: n * n_planes].reshape(n, n_planes)
    return out


def _export_reco_objects(ev, hits, tracks, showers, spacepoints, vertices, mc,
                         arrays, meta, avail, notes, section):
    """Per-PFParticle (``pfp/*``) and per-slice (``slice/*``) tables and ``sp/group/*``.

    Group ids in ``hits/group/pfparticle`` / ``sp/group/pfparticle`` are rows of the
    ``pfp/*`` tables (the PFParticle collection index; ``pfp/id`` is Pandora's Self);
    ``hits/group/slice`` / ``sp/group/slice`` ids are rows of the ``slice/*`` tables.
    Every step degrades on its own: with no readable PFParticle the hit and
    space-point groups are still exported and the viewer highlights by those ids.
    """
    art = ev._src.art
    hg = {k: arrays.get(f"hits/group/{k}") for k in GROUP_KINDS}
    n_hits = len(hits) if hits is not None else 0
    chan_view = (np.asarray(hits.chan_view, np.int64) if n_hits else np.zeros(0, np.int64))

    # ---- PFParticle collection -------------------------------------------
    n_pfp = 0
    pfp = None
    try:
        prod = ev._pick("recob::PFParticle", None, ("pandora",))
        raw = art.read(prod, "recob::PFParticle", ev.entry)
        selfid = np.asarray(raw["fSelf"]).astype(np.int64)
        n_pfp = len(selfid)
        parents = np.asarray(raw["fParent"]).astype(np.uint64)
        pdg = np.asarray(raw["fPdgCode"], np.int32)
        by_self = {int(s): i for i, s in enumerate(selfid)}
        parent = np.array([-1 if p == _U64_MAX else by_self.get(int(p), -1) for p in parents],
                          np.int64)
        parent[parent == np.arange(n_pfp)] = -1
        d_off = np.zeros(n_pfp + 1, np.uint32)
        d_idx = []
        for i, d in enumerate(raw["fDaughters"]):
            ix = [by_self[int(x)] for x in np.asarray(d) if int(x) in by_self]
            d_idx.extend(ix)
            d_off[i + 1] = d_off[i] + len(ix)
        top = np.arange(n_pfp, dtype=np.int64)
        for i in range(n_pfp):                       # walk up (guarded against cycles)
            j, steps = i, 0
            while parent[j] >= 0 and steps < 64:
                j = int(parent[j]); steps += 1
            top[i] = j
        pfp = {"self": selfid, "pdg": pdg, "parent": parent, "top": top}
    except Exception as exc:
        notes.append(f"pfp: PFParticles not readable ({_fmt_exc(exc)}); "
                     "highlighting uses the hit/space-point group ids only")

    def assn(left_cls, right_cls, prefer=("pandora",)):
        try:
            a = ev._assns_for(left_cls, right_cls, prefer)
            return np.asarray(a.left_key, np.int64), np.asarray(a.right_key, np.int64)
        except Exception:
            return None

    # ---- slices ---------------------------------------------------------
    n_slice = 0
    slice_id = None
    try:
        sprod = ev._pick("recob::Slice", None, ("pandora",))
        sraw = art.read(sprod, "recob::Slice", ev.entry)
        slice_id = np.asarray(sraw["fID"], np.int32)
        n_slice = len(slice_id)
    except Exception as exc:
        if hg["slice"] is not None or art.find_product("recob::Slice"):
            notes.append(f"slice table: Slices not readable ({_fmt_exc(exc)})")
    for g in (hg["slice"],):
        if g is not None and (g >= 0).any():
            n_slice = max(n_slice, int(g.max()) + 1)
    if slice_id is None or len(slice_id) < n_slice:
        slice_id = np.arange(n_slice, dtype=np.int32)

    pfp_slice = np.full(n_pfp, -1, np.int64)
    if pfp is not None:
        a = assn("recob::PFParticle", "recob::Slice")
        if a is not None:
            pfp_slice = _first_by_key(a[0], a[1], n_pfp)
            pfp_slice[pfp_slice >= n_slice] = -1
            # a descendant without its own slice link inherits the top-level one
            miss = (pfp_slice < 0)
            pfp_slice[miss] = pfp_slice[pfp["top"][miss]]

    # hits: fill a missing slice grouping from the PFParticle one
    if hg["slice"] is None and hg["pfparticle"] is not None and n_slice and pfp is not None:
        g = hg["pfparticle"].astype(np.int64)
        gs = np.where((g >= 0) & (g < n_pfp), pfp_slice[np.clip(g, 0, max(n_pfp - 1, 0))], -1)
        arrays["hits/group/slice"] = gs.astype(np.int32)
        hg["slice"] = arrays["hits/group/slice"]
        meta["groups"]["slice"] = int(n_slice)
        notes.append("hits/group/slice derived from the PFParticle->Slice association")

    # ---- space-point groups ------------------------------------------------
    sp_group = {}
    if spacepoints is not None and len(spacepoints):
        nsp = len(spacepoints)
        a = assn("recob::Hit", "recob::SpacePoint", ("pandora", "spsolve"))
        for kind in GROUP_KINDS:
            g = hg.get(kind)
            if a is None or g is None:
                continue
            lk, rk = a
            good = (rk >= 0) & (rk < nsp) & (lk >= 0) & (lk < len(g))
            rk, lk = rk[good], lk[good]
            has = g[lk] >= 0
            out = np.full(nsp, -1, np.int32)
            out[rk[has]] = g[lk[has]]
            sp_group[kind] = out
        if pfp is not None:
            b = assn("recob::PFParticle", "recob::SpacePoint")
            if b is not None:
                lk, rk = b
                good = (rk >= 0) & (rk < nsp) & (lk >= 0) & (lk < n_pfp)
                direct = np.full(nsp, -1, np.int32)
                direct[rk[good][::-1]] = lk[good][::-1]
                cur = sp_group.get("pfparticle")
                sp_group["pfparticle"] = direct if cur is None else np.where(cur >= 0, cur, direct)
            if "pfparticle" in sp_group and "slice" not in sp_group and n_slice:
                g = sp_group["pfparticle"].astype(np.int64)
                sp_group["slice"] = np.where(g >= 0, pfp_slice[np.clip(g, 0, n_pfp - 1)], -1).astype(np.int32)
        for kind, g in sp_group.items():
            arrays[f"sp/group/{kind}"] = g

    # ---- PFParticle table -------------------------------------------------
    if pfp is not None:
        avail["pfp"] = True
        arrays["pfp/id"] = pfp["self"].astype(np.int32)
        arrays["pfp/pdg"] = pfp["pdg"]
        arrays["pfp/parent"] = pfp["parent"].astype(np.int32)
        arrays["pfp/top"] = pfp["top"].astype(np.int32)
        arrays["pfp/daughter_offsets"] = d_off
        arrays["pfp/daughters"] = np.asarray(d_idx, np.int32)
        arrays["pfp/slice"] = pfp_slice.astype(np.int32)
        sc = np.full(n_pfp, np.nan)
        try:
            ts = ev.pfp_track_scores()
            if ts is not None:
                sc[: min(n_pfp, len(ts))] = ts[:n_pfp]
        except Exception:
            pass
        arrays["pfp/score"] = _f32(sc)
        # PFParticle -> exported track / shower row
        for what, objs, cls, prefer in (("track", tracks, "recob::Track", ("pandoraTrack", "pmtrack")),
                                        ("shower", showers, "recob::Shower", ("pandoraShower",))):
            row = np.full(n_pfp, -1, np.int32)
            if objs is not None and len(objs):
                owner = ev._pfp_of(cls, prefer)
                raw_idx = None
                if owner is not None:
                    keep = ev.pandora_best_match().get(what + "s")
                    good = np.ones(len(keep) if keep is not None else len(owner), bool)
                    if what == "shower":
                        try:
                            sh_prod = ev._pick(cls, None, ("pandoraShower",))
                            ln = np.asarray(art.read(sh_prod, cls, ev.entry)["fLength"], float)
                            good = ~np.isclose(ln, -999.0)
                        except Exception:
                            pass
                    if keep is not None and len(keep) == len(good):
                        good = good & keep
                    raw_idx = np.flatnonzero(good)
                if raw_idx is not None and len(raw_idx) == len(objs):
                    for i, r in enumerate(raw_idx):
                        p = int(owner[r]) if r < len(owner) else -1
                        if 0 <= p < n_pfp and row[p] < 0:
                            row[p] = i
                    arrays[f"{what}s/pfp"] = np.array(
                        [int(owner[r]) if r < len(owner) else -1 for r in raw_idx], np.int32)
                else:
                    notes.append(f"pfp: could not map {what}s to PFParticles")
            arrays[f"pfp/{what}"] = row
        # counts
        gp = hg["pfparticle"]
        if gp is not None:
            pl = _plane_counts(gp, chan_view, n_pfp)
            arrays["pfp/n_hits_plane"] = pl
            arrays["pfp/n_hits"] = np.bincount(gp[(gp >= 0) & (gp < n_pfp)], minlength=n_pfp).astype(np.int32)[:n_pfp]
        else:
            arrays["pfp/n_hits_plane"] = np.zeros((n_pfp, 3), np.int32)
            arrays["pfp/n_hits"] = np.zeros(n_pfp, np.int32)
        sg = sp_group.get("pfparticle")
        arrays["pfp/n_sp"] = (np.bincount(sg[(sg >= 0) & (sg < n_pfp)], minlength=n_pfp)[:n_pfp].astype(np.int32)
                              if sg is not None else np.zeros(n_pfp, np.int32))
        # vertex of each PFParticle (row of vtx/xyz)
        pv = np.full(n_pfp, -1, np.int32)
        if vertices is not None and len(vertices):
            a = assn("recob::PFParticle", "recob::Vertex")
            if a is not None:
                v = _first_by_key(a[0], a[1], n_pfp)
                pv = np.where((v >= 0) & (v < len(vertices)), v, -1).astype(np.int32)
        arrays["pfp/vertex"] = pv
        meta["pfp"] = {"n": int(n_pfp)}

        # matched true particle: reuse the labels/* track and shower matching
        t_mc = np.full(n_pfp, -1, np.int32)
        t_pdg = np.zeros(n_pfp, np.int32)
        t_tid = np.full(n_pfp, -1, np.int32)
        if "labels/kind" in arrays and mc is not None:
            kind, idx = arrays["labels/kind"], arrays["labels/index"]
            tid, lpdg = arrays["labels/track_id"], arrays["labels/pdg"]
            by_tid = {int(t): i for i, t in enumerate(mc.track_id)}
            lab = {}
            for k in range(len(kind)):
                if kind[k] in (0, 1):
                    lab[(int(kind[k]), int(idx[k]))] = k
            for p in range(n_pfp):
                order = (1, 0) if abs(int(pfp["pdg"][p])) == 11 else (0, 1)
                for kk in order:
                    row = arrays["pfp/track" if kk == 0 else "pfp/shower"][p]
                    k = lab.get((kk, int(row))) if row >= 0 else None
                    if k is not None:
                        t_tid[p], t_pdg[p] = tid[k], lpdg[k]
                        t_mc[p] = by_tid.get(int(tid[k]), -1)
                        break
            meta["pfp"]["true_match"] = ("reco track/shower of each PFParticle matched to the "
                                         "nearest true trajectory (same rule as the particle labels)")
        arrays["pfp/true_mc"], arrays["pfp/true_pdg"], arrays["pfp/true_tid"] = t_mc, t_pdg, t_tid

    # ---- slice table ------------------------------------------------------
    if n_slice:
        avail["slice"] = True
        arrays["slice/id"] = slice_id[:n_slice].astype(np.int32)
        gs = hg["slice"]
        arrays["slice/n_hits_plane"] = (_plane_counts(gs, chan_view, n_slice) if gs is not None
                                        else np.zeros((n_slice, 3), np.int32))
        arrays["slice/n_hits"] = (np.bincount(gs[(gs >= 0) & (gs < n_slice)], minlength=n_slice)[:n_slice].astype(np.int32)
                                  if gs is not None else np.zeros(n_slice, np.int32))
        ss = sp_group.get("slice")
        arrays["slice/n_sp"] = (np.bincount(ss[(ss >= 0) & (ss < n_slice)], minlength=n_slice)[:n_slice].astype(np.int32)
                                if ss is not None else np.zeros(n_slice, np.int32))
        s_top = np.full(n_slice, -1, np.int32)
        s_nu = np.zeros(n_slice, np.uint8)
        s_npfp = np.zeros(n_slice, np.int32)
        s_vtx = np.full((n_slice, 3), np.nan, np.float32)
        s_mc = np.full(n_slice, -1, np.int32)
        s_pdg = np.zeros(n_slice, np.int32)
        s_tid = np.full(n_slice, -1, np.int32)
        if pfp is not None:
            vx = arrays.get("vtx/xyz")
            for p in range(n_pfp):
                s = int(pfp_slice[p])
                if s < 0:
                    continue
                s_npfp[s] += 1
                if pfp["parent"][p] < 0:                # top-level PFParticle of the slice
                    if s_top[s] < 0 or abs(int(pfp["pdg"][p])) in NEUTRINO_PDGS:
                        s_top[s] = p
                        s_nu[s] = 1 if abs(int(pfp["pdg"][p])) in NEUTRINO_PDGS else 0
            for s in range(n_slice):
                p = int(s_top[s])
                if p >= 0 and vx is not None and arrays["pfp/vertex"][p] >= 0:
                    s_vtx[s] = vx[int(arrays["pfp/vertex"][p])]
            # truth: the matched PFParticle with the most hits in the slice
            nh = arrays["pfp/n_hits"]
            best = np.full(n_slice, -1, np.int64)
            for p in range(n_pfp):
                s = int(pfp_slice[p])
                if s >= 0 and arrays["pfp/true_tid"][p] >= 0 and (best[s] < 0 or nh[p] > nh[best[s]]):
                    best[s] = p
            for s in range(n_slice):
                if best[s] >= 0:
                    s_mc[s], s_pdg[s], s_tid[s] = (arrays["pfp/true_mc"][best[s]],
                                                   arrays["pfp/true_pdg"][best[s]],
                                                   arrays["pfp/true_tid"][best[s]])
        arrays["slice/top"], arrays["slice/nu"], arrays["slice/n_pfp"] = s_top, s_nu, s_npfp
        arrays["slice/vertex"] = s_vtx
        arrays["slice/true_mc"], arrays["slice/true_pdg"], arrays["slice/true_tid"] = s_mc, s_pdg, s_tid
        meta["slice"] = {"n": int(n_slice),
                         "note": "slice/vertex is the vertex of the slice's top-level PFParticle "
                                 "(NaN when none); slice/true_* is the match of the PFParticle with most hits"}


# --------------------------------------------------------------------------
# CLI


def _parse_entries(spec: str, n: int) -> list[int]:
    if spec == "all":
        return list(range(n))
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return [i for i in out if 0 <= i < n]


def _worker(args):
    from .cache import get_or_build_event
    path, entry, hits, flash = args
    t = time.time()
    data = get_or_build_event(path, entry, hits, flash)
    return entry, len(data), time.time() - t


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m pylarevd.web.export",
                                 description="Fill the pylarevd bundle cache.")
    ap.add_argument("file")
    ap.add_argument("-e", "--entries", default="all", help="all, N, A-B or A,B,C")
    ap.add_argument("-j", "--jobs", type=int, default=1)
    ap.add_argument("--hits", default=None, help="hit product tag")
    ap.add_argument("--flash", default=None, help="flash producer tag")
    ns = ap.parse_args(argv)

    from concurrent.futures import ProcessPoolExecutor
    from pylario import EventFile
    from .cache import cache_dir, canonical, get_or_build_geometry
    path = canonical(ns.file)
    with EventFile(path) as f:
        n = len(f)
    entries = _parse_entries(ns.entries, n)
    t0 = time.time()
    get_or_build_geometry(path)
    jobs = [(path, e, ns.hits, ns.flash) for e in entries]
    if ns.jobs > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=min(ns.jobs, len(jobs))) as ex:
            results = list(ex.map(_worker, jobs))
    else:
        results = [_worker(j) for j in jobs]
    for entry, size, dt in results:
        print(f"entry {entry:4d}  {size / 1e6:8.2f} MB  {dt:7.1f} s", flush=True)
    print(f"{len(results)} events -> {cache_dir(path)}  ({time.time() - t0:.1f} s wall)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
