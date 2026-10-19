"""Server-side exports for the web viewer: vector PDF / PNG snapshots and GIFs.

The browser sends its state (colour, overlays, light parameters, camera) to
``GET /api/export``; :func:`parse_export` validates it in the request thread
(HttpError-style ``ValueError`` -> 400) and :func:`render_export` runs in a
worker process using the same matplotlib renderers as the old Dash app
(``EventDisplay.figure``, ``Display3D.figure`` / ``rotate_gif``,
``OpticalDisplay``, ``FlashDisplay3D``, ``LightAnimation`` /
``ChargeLightAnimation``).
"""
from __future__ import annotations

import io
import math

VIEWS = ("2d", "3d", "light", "charge+light", "2d+3d", "optical", "flash3d")
KINDS = ("pdf", "png", "gif")
GIF_VIEWS = ("3d", "light", "charge+light")
OVERLAYS = ("truth", "rad", "dep", "tracks", "showers", "vtx", "pvtx", "labels", "pd")
OV_DEFAULT = ("truth", "tracks", "showers", "vtx", "pvtx", "pd")
COLOURS_2D = ("integral", "amplitude", "tick", "multiplicity",
              "track", "shower", "slice", "cluster", "pfparticle")
COLOURS_3D = ("integral", "amplitude", "tick", "multiplicity", "track", "shower",
              "pfparticle", "cluster", "slice", "x", "y", "z", "chisq")
DEFAULT_ELEV, DEFAULT_AZIM, DEFAULT_ZOOM = 18.0, -60.0, 1.15
MAX_FRAMES = 600


def _num(q, name, default, lo=None, hi=None, integer=False):
    v = q.get(name)
    if v in (None, ""):
        return default
    try:
        x = float(v)
    except ValueError:
        raise ValueError(f"'{name}' must be a number, got {v!r}")
    if not math.isfinite(x):
        raise ValueError(f"'{name}' must be finite")
    if (lo is not None and x < lo) or (hi is not None and x > hi):
        raise ValueError(f"'{name}' out of range [{lo}, {hi}]: {v}")
    return int(round(x)) if integer else x


def _vec(q, name):
    v = q.get(name)
    if v in (None, ""):
        return None
    try:
        a = [float(s) for s in v.split(",")]
    except ValueError:
        a = []
    if len(a) != 3 or not all(math.isfinite(s) for s in a):
        raise ValueError(f"'{name}' must be three comma-separated numbers")
    return a


def camera_to_mpl(position, target, ref_dist=None):
    """Detector-frame camera -> matplotlib (elev, azim, zoom).

    The matplotlib axes of the 3-D renderer are (z, x, y) = (beam, drift, up),
    so the view vector's components are (v.z, v.x, v.y); the azimuth runs from
    +z toward +x as in ``Display3D.figure``.  ``zoom`` follows the old Dash
    convention (1.15 at the default distance, clipped to 0.65..1.8) where the
    default distance is ``ref_dist`` (the viewer's initial 1.45 x bounding
    diagonal).
    """
    vx, vy, vz = (position[i] - target[i] for i in range(3))   # detector x, y, z
    norm = math.sqrt(vx * vx + vy * vy + vz * vz)
    if norm < 1e-6:
        return DEFAULT_ELEV, DEFAULT_AZIM, DEFAULT_ZOOM
    elev = math.degrees(math.asin(max(-1.0, min(1.0, vy / norm))))
    azim = math.degrees(math.atan2(vx, vz))
    zoom = DEFAULT_ZOOM
    if ref_dist and ref_dist > 0:
        zoom = max(0.65, min(1.80, DEFAULT_ZOOM * ref_dist / norm))
    return elev, azim, zoom


def parse_export(q: dict, default_flash=None) -> dict:
    """Validate query parameters (``q``: name -> single string).  Raises ValueError."""
    p = {}
    p["kind"] = (q.get("kind") or "pdf").lower()
    if p["kind"] not in KINDS:
        raise ValueError(f"kind must be one of {', '.join(KINDS)}")
    p["view"] = q.get("view") or "3d"
    if p["view"] not in VIEWS:
        raise ValueError(f"view must be one of {', '.join(VIEWS)}")
    if p["kind"] == "gif" and p["view"] not in GIF_VIEWS:
        raise ValueError(f"GIF export is available for {', '.join(GIF_VIEWS)} only, "
                         f"not {p['view']!r}")
    color = q.get("color") or "charge"
    p["color"] = "integral" if color == "charge" else color
    p["scale"] = q.get("scale") or "auto"
    if p["scale"] not in ("auto", "log", "linear"):
        raise ValueError("scale must be auto, log or linear")
    p["cmap"] = q.get("cmap") or "turbo"
    p["theme"] = q.get("theme") or "dark"
    from pylarevd.display import COLORMAPS, THEMES
    if p["cmap"] not in COLORMAPS:
        raise ValueError(f"cmap {p['cmap']!r} unknown; use {', '.join(sorted(COLORMAPS))}")
    if p["theme"] not in THEMES:
        raise ValueError(f"theme {p['theme']!r} unknown; use {', '.join(sorted(THEMES))}")
    ov = q.get("ov")
    on = set(OV_DEFAULT if ov is None else [s for s in ov.split(",") if s])
    bad = on - set(OVERLAYS)
    if bad:
        raise ValueError(f"unknown overlay(s): {', '.join(sorted(bad))}")
    p["ov"] = {o: o in on for o in OVERLAYS}
    p["merge2d"] = q.get("merge2d") or "orientation"
    if p["merge2d"] not in ("orientation", "view", "none"):
        raise ValueError("merge2d must be orientation, view or none")
    p["space2d"] = q.get("space2d") or "wire"
    if p["space2d"] not in ("wire", "readout", "physical"):
        raise ValueError("space2d must be wire or readout")
    # light
    p["lclock"] = q.get("lclock") or "log"
    if p["lclock"] not in ("log", "linear"):
        raise ValueError("lclock must be log or linear")
    p["lfill"] = q.get("lfill") or "afterglow"
    if p["lfill"] not in ("afterglow", "cumulative"):
        raise ValueError("lfill must be afterglow or cumulative")
    p["lframes"] = _num(q, "lframes", 120, 2, MAX_FRAMES, integer=True)
    p["lfps"] = _num(q, "lfps", 12.0, 1, 60)
    p["lpers"] = _num(q, "lpers", 0.7, 0.0, 0.999)
    p["lwlo"] = _num(q, "lwlo", -0.25, -1e6, 1e6)
    p["lwhi"] = _num(q, "lwhi", 10.0, -1e6, 1e6)
    if p["lwhi"] <= p["lwlo"]:
        raise ValueError("lwhi must be greater than lwlo")
    p["lt0"] = _num(q, "lt0", None, -1e9, 1e9)
    lt = q.get("lt")
    p["lt"] = None if lt in (None, "", "all") else _num(q, "lt", None, 0, MAX_FRAMES, integer=True)
    # camera and rotation
    pos, tgt = _vec(q, "campos"), _vec(q, "camtgt")
    p["campos"], p["camtgt"] = pos, tgt
    p["camup"] = _vec(q, "camup")
    if (pos is None) != (tgt is None):
        raise ValueError("campos and camtgt go together")
    p["rotstep"] = _num(q, "rotstep", 3, 1, 30, integer=True)
    p["rotfps"] = _num(q, "rotfps", 12, 1, 60, integer=True)
    return p


# ---------------------------------------------------------------- worker side

_FILES: dict = {}


def _event(path, entry):
    from pylario import EventFile
    ef = _FILES.get(path)
    if ef is None:
        ef = _FILES[path] = EventFile(path)
    if not 0 <= entry < len(ef):
        raise ValueError(f"entry {entry} out of range")
    return ef[entry]


def _camera(ev, p):
    if p["campos"] is None:
        return DEFAULT_ELEV, DEFAULT_AZIM, DEFAULT_ZOOM
    ref = None
    try:
        boxes = ev.geometry.tpcs
        dx = max(b.xmax for b in boxes) - min(b.xmin for b in boxes)
        dy = max(b.ymax for b in boxes) - min(b.ymin for b in boxes)
        dz = max(b.zmax for b in boxes) - min(b.zmin for b in boxes)
        ref = 1.45 * math.sqrt(dx * dx + dy * dy + dz * dz)
    except Exception:
        pass
    return camera_to_mpl(p["campos"], p["camtgt"], ref)


def _flags(p):
    o = p["ov"]
    return dict(truth=o["truth"] or o["dep"], tracks=o["tracks"], showers=o["showers"],
                pvtx=o["pvtx"], vtx=o["vtx"], rad=o["rad"], labels=o["labels"])


def _display_3d(ev, p):
    f = _flags(p)
    colour = p["color"] if p["color"] in COLOURS_3D else "integral"
    return ev.display_3d(truth=f["truth"], tracks=f["tracks"], showers=f["showers"],
                         vertices=f["pvtx"] or f["vtx"], pandora_vertex=f["pvtx"],
                         daughter_vertices=f["vtx"], secondary_vertices=f["vtx"],
                         radiologicals=f["rad"], colour_by=colour,
                         colour_scale=p["scale"], particle_symbols=f["labels"],
                         theme=p["theme"], colormap=p["cmap"])


def _display_2d(ev, p, hit_tag):
    f = _flags(p)
    colour = p["color"] if p["color"] in COLOURS_2D else "integral"
    return ev.display(hit_tag, truth=f["truth"], tracks=f["tracks"], showers=f["showers"],
                      vertices=f["pvtx"] or f["vtx"], pandora_vertex=f["pvtx"],
                      daughter_vertices=f["vtx"], secondary_vertices=f["vtx"],
                      colour_by=colour, radiologicals=f["rad"], colour_scale=p["scale"],
                      merge=p["merge2d"],
                      space="readout" if p["space2d"] == "readout" else "physical",
                      particle_symbols=f["labels"], theme=p["theme"], colormap=p["cmap"])


def _light(ev, p, flash_tag, charge):
    from pylarevd.display import display_charge_light, display_light
    kw = dict(time_scale=p["lclock"], fill=p["lfill"], n_frames=p["lframes"],
              persistence=p["lpers"], theme=p["theme"], colormap=p["cmap"])
    if p["lt0"] is not None:
        kw["t0"] = p["lt0"]
    if charge:
        f = _flags(p)
        elev, azim, _ = _camera(ev, p)
        if p["campos"] is None:
            elev, azim = 20.0, -60.0
        d = display_charge_light(ev, flash_tag, truth=f["truth"], tracks=f["tracks"],
                                 vertices=f["pvtx"] or f["vtx"], showers=f["showers"],
                                 radiologicals=f["rad"], elev=elev, azim=azim, **kw)
    else:
        d = display_light(ev, flash_tag, **kw)
    # window is relative to t0 in the viewer; t0 may be automatic
    d._window_arg = (d.t0 + p["lwlo"], d.t0 + p["lwhi"])
    return d


def _fig_bytes(fig, fmt, dpi, **kw):
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    try:
        fig.savefig(buf, format=fmt, dpi=dpi, facecolor=fig.get_facecolor(), **kw)
    finally:
        plt.close(fig)
    return buf.getvalue()


def render_export(path, entry, hit_tag, flash_tag, p) -> dict:
    """Run in a worker.  Returns {"ok": True, "data", "ctype", "filename", "note"}
    or {"ok": False, "status", "error"}."""
    import os
    import time
    t = time.time()
    try:
        import matplotlib
        matplotlib.use("Agg")
        from pylarevd.display import (_vector_text, display_flashes_3d, display_optical)
        kind, view = p["kind"], p["view"]
        ev = _event(path, entry)
        run, sub, evn = ev.id
        note = ""
        if view == "2d+3d":
            view, note = "2d", "2d+3d exports the 2-D panels only"
        if kind == "pdf":
            _vector_text()
        fmt, dpi = ("pdf", 300) if kind == "pdf" else ("png", 150)
        ctype = {"pdf": "application/pdf", "png": "image/png", "gif": "image/gif"}[kind]
        stem = f"pylarevd_run{run}_sub{sub}_ev{evn}_{view.replace('+', '_')}"

        if kind == "gif" and view == "3d":
            elev, azim, zoom = _camera(ev, p)
            d = _display_3d(ev, p)
            data = d.rotate_gif(elev=elev, start_azim=azim, step=p["rotstep"],
                                fps=p["rotfps"], zoom=zoom, pad=0.20,
                                particle_symbols=p["ov"]["labels"])
            stem += "_rotation"
        elif kind == "gif":
            d = _light(ev, p, flash_tag, view == "charge+light")
            data = d.animation_bytes("gif", fps=p["lfps"])
            stem += f"_{flash_tag or 'default'}"
        else:
            if view == "3d":
                elev, azim, zoom = _camera(ev, p)
                fig = _display_3d(ev, p).figure(elev=elev, azim=azim, zoom=zoom, pad=0.20,
                                                particle_symbols=p["ov"]["labels"])
                kw = dict(bbox_inches="tight", pad_inches=0.3)
            elif view == "2d":
                fig = _display_2d(ev, p, hit_tag).figure(particle_symbols=p["ov"]["labels"])
                kw = dict(bbox_inches="tight")
            elif view in ("light", "charge+light"):
                fig = _light(ev, p, flash_tag, view == "charge+light").figure(p["lt"])
                kw = {}
            elif view == "optical":
                fig = display_optical(ev, flash_tag, theme=p["theme"],
                                      colormap=p["cmap"]).figure()
                kw = dict(bbox_inches="tight")
            else:   # flash3d
                elev, azim, _ = _camera(ev, p)
                kw2 = {} if p["campos"] is None else dict(elev=elev, azim=azim)
                fig = display_flashes_3d(ev, flash_tag, theme=p["theme"],
                                         colormap=p["cmap"]).figure(**kw2)
                kw = dict(bbox_inches="tight")
            data = _fig_bytes(fig, fmt, dpi, **kw)
        return {"ok": True, "data": data, "ctype": ctype,
                "filename": f"{stem}.{kind}", "note": note,
                "render_s": round(time.time() - t, 2), "pid": os.getpid()}
    except (ValueError, KeyError, IndexError, OSError) as exc:
        return {"ok": False, "status": 400 if isinstance(exc, ValueError) else 500,
                "error": f"{type(exc).__name__}: {exc}"}
    except Exception as exc:
        return {"ok": False, "status": 500, "error": f"{type(exc).__name__}: {exc}"}
