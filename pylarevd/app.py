"""Interactive event browser.

A small Dash app around the same display code the static images use, adding the
one thing a saved HTML page cannot do: stepping through events without
regenerating anything.

    python -m pylarevd.app reco.root [more.root ...] --port 8050

The server binds to localhost only. From your laptop:

    ssh -N -L 8050:localhost:8050 <your-host>

then open http://localhost:8050
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import numpy as np

import matplotlib
matplotlib.use("Agg")

from urllib.parse import parse_qsl, urlencode

from ._deps import require

require("dash", "the interactive browser")

from dash import Dash, Input, Output, State, dcc, html, no_update

from .artio import ArtReadError
from .theme import COLORMAPS, DEFAULT_COLORMAP, THEMES
from .event import EventFile
from .geometry import Geometry, GeometryError

#: Controls whose value is mirrored into the URL, so a view can be linked.
#: (component id, is-a-checklist)
_URL_CONTROLS = (("file", False), ("entry", False), ("tag", False),
                 ("colour", False), ("scale", False), ("mode", False),
                 ("merge", False), ("theme", False), ("colormap", False),
                 ("truth", True), ("reco", True), ("radio", True))

def url_params(values, event_id=None) -> str:
    """Serialise control values into a query string.

    ``event_id`` is the ``(run, subrun, event)`` of the displayed event.  It is
    recorded alongside the entry number so a link still lands on the right
    event when reopened against a different file.
    """
    params = {}
    for (cid, is_list), value in zip(_URL_CONTROLS, values):
        if value is None:
            continue
        if cid == "file":
            # The full path, not the basename: a link must still resolve
            # against a server that was started with different arguments, or
            # against a file that was opened ad hoc and is in nobody's list.
            params[cid] = value
        elif is_list:
            params[cid] = "1" if value else "0"
        else:
            params[cid] = value
    if event_id is not None:
        params["rse"] = ":".join(str(int(v)) for v in event_id)
    return "?" + urlencode(params)


def file_from_url(raw: str, by_name: dict) -> str | None:
    """Resolve the ``file=`` parameter to a path.

    Accepts a full path (whether or not the server already has it open) or the
    basename of a file the server was started with -- the latter so that links
    written by earlier versions, and hand-shortened ones, keep working.
    """
    if raw in by_name:                     # basename of a file we were given
        return by_name[raw]
    if raw in by_name.values():            # full path we already have open
        return raw
    if _SCHEME.match(raw):                 # root:// & friends, read as-is
        return raw
    if os.path.isabs(raw) or raw.startswith("~") or "/" in raw:
        return normalise_path(raw)         # ad hoc path; the caller opens it
    return None


def url_values(search: str, by_name: dict, resolve=None) -> list:
    """Parse a query string back into control values.

    ``by_name`` maps a basename to a full path; ``resolve`` optionally maps that
    path to an open :class:`EventFile`, which is what allows an ``rse=`` event
    identity to be turned back into an entry number.

    Returns ``None`` for controls the URL does not mention, so a caller can
    leave those alone.
    """
    params = dict(parse_qsl((search or "").lstrip("?")))
    out = []
    path = file_from_url(params.get("file", ""), by_name)
    # An explicit event identity outranks the entry number: entries shift
    # between files, event ids do not.
    if path is not None and resolve is not None and "rse" in params:
        try:
            r, sr, ev = (int(v) for v in params["rse"].split(":"))
            found = resolve(path).index_of(r, sr, ev)
        except Exception:
            found = None          # malformed rse, or the file lacks that event
        if found is not None:
            params["entry"] = str(found)
    for cid, is_list in _URL_CONTROLS:
        if cid not in params:
            out.append(None)
            continue
        raw = params[cid]
        if cid == "file":
            out.append(file_from_url(raw, by_name))
        elif cid == "entry":
            out.append(int(raw) if raw.lstrip("-").isdigit() else None)
        elif is_list:
            out.append(["on"] if raw in ("1", "on", "true") else [])
        else:
            out.append(raw)
    return out


_T = THEMES["dark"]          # the app chrome stays dark; figures follow the toggle
FIG_BG, AXES_BG, FG, FG_MUTED = _T.fig_bg, _T.axes_bg, _T.fg, _T.fg_muted
GRID, LEGEND_BG, LEGEND_EDGE = _T.grid, _T.legend_bg, _T.legend_edge

_CTL = {"backgroundColor": LEGEND_BG, "color": FG,
        "border": f"1px solid {LEGEND_EDGE}"}
_LABEL = {"color": FG_MUTED, "fontSize": "12px", "marginBottom": "4px",
          "display": "block", "fontFamily": "system-ui, sans-serif"}
#: Checkbox text is set explicitly rather than inherited: a disabled option
#: otherwise falls back to the browser's default disabled colour, which is a
#: dark grey that vanishes on a dark panel.
_CHECK = {"fontSize": "12px", "color": FG}
_CHECK_LABEL = {"color": FG, "display": "block", "cursor": "pointer",
                "marginBottom": "2px"}
_CHECK_INPUT = {"marginRight": "5px", "accentColor": LEGEND_EDGE,
                "verticalAlign": "middle"}

_BAND = {"display": "flex", "gap": "12px", "alignItems": "flex-end",
         "backgroundColor": AXES_BG, "padding": "8px 10px",
         "border": f"1px solid {GRID}", "borderRadius": "6px"}


#: Dash gives a disabled checkbox no colour of its own, so the browser default
#: (a dark grey) applies -- unreadable on this panel. State both states.
_INDEX_TEMPLATE = """<!DOCTYPE html>
<html>
  <head>
    {%metas%}    <title>{%title%}</title>
    <link rel="icon" type="image/x-icon" href="assets/favicon.ico">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script>
    {%favicon%}{%css%}
    <style>
      .evd-check label, .evd-check span { color: __FG__; }
      .evd-check label:has(input:disabled),
      .evd-check label:has(input:disabled) span { color: __MUTED__; opacity: .8; }
      .evd-check input:disabled { cursor: not-allowed; }

      /* Dash 4 and React-Select Dropdown styling: Dark theme */
      :root, [data-theme="dark"], #app-container {
        --Dash-Fill-Inverse-Strong: #1e293b !important;
        --Dash-Stroke-Strong: #475569 !important;
        --Dash-Stroke-Weak: #334155 !important;
        --Dash-Text-Primary: #f8fafc !important;
        --Dash-Text-Strong: #ffffff !important;
        --Dash-Text-Weak: #cbd5e1 !important;
        --Dash-Text-Disabled: #64748b !important;
        --Dash-Fill-Interactive-Strong: #0284c7 !important;
        --Dash-Fill-Interactive-Weak: #334155 !important;
        --Dash-Fill-Primary-Hover: #334155 !important;
        --Dash-Fill-Primary-Active: #475569 !important;
      }

      /* Dash 4 Dropdown Elements (Dark Theme) */
      .dash-dropdown,
      .dash-dropdown-content,
      .dash-dropdown-wrapper {
        background-color: #1e293b !important;
        background: #1e293b !important;
        color: #ffffff !important;
        border-color: #475569 !important;
      }
      .dash-dropdown-value,
      .dash-dropdown-value-item,
      .dash-dropdown-trigger-icon,
      .dash-dropdown-value-count,
      .dash-dropdown-option,
      .dash-dropdown-trigger {
        color: #ffffff !important;
        fill: #ffffff !important;
      }
      .dash-dropdown-placeholder {
        color: #94a3b8 !important;
      }
      .dash-dropdown-option:hover,
      .dash-dropdown-option:focus,
      .dash-dropdown-option[data-highlighted] {
        background-color: #334155 !important;
        color: #ffffff !important;
      }
      .dash-dropdown-option[data-state="checked"],
      .dash-dropdown-option.is-selected {
        background-color: #0284c7 !important;
        color: #ffffff !important;
      }

      /* Legacy React-Select Elements (Dark Theme) */
      .Select-control, .dash-dropdown .Select-control {
        background-color: #1e293b !important;
        color: #ffffff !important;
        border: 1px solid #475569 !important;
      }
      .Select-value-label, .dash-dropdown .Select-value-label,
      .Select--single > .Select-control .Select-value,
      .dash-dropdown .Select--single > .Select-control .Select-value {
        color: #ffffff !important;
      }
      .Select-placeholder, .dash-dropdown .Select-placeholder {
        color: #94a3b8 !important;
      }
      .Select-input > input, .dash-dropdown .Select-input > input {
        color: #ffffff !important;
      }
      .Select-menu-outer, .dash-dropdown .Select-menu-outer {
        background-color: #1e293b !important;
        border: 1px solid #475569 !important;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.4) !important;
        z-index: 1000 !important;
      }
      .Select-menu, .dash-dropdown .Select-menu {
        background-color: #1e293b !important;
      }
      .Select-option, .dash-dropdown .Select-option {
        background-color: #1e293b !important;
        color: #ffffff !important;
      }
      .Select-option.is-focused, .dash-dropdown .Select-option.is-focused,
      .Select-option:hover, .dash-dropdown .Select-option:hover {
        background-color: #334155 !important;
        color: #ffffff !important;
      }
      .Select-option.is-selected, .dash-dropdown .Select-option.is-selected {
        background-color: #0284c7 !important;
        color: #ffffff !important;
      }
      .Select-option.is-disabled, .dash-dropdown .Select-option.is-disabled {
        color: #64748b !important;
        background-color: #1e293b !important;
      }

      /* Dash 4 and React-Select Dropdown styling: Light theme */
      [data-theme="light"], #app-container[data-theme="light"] {
        --Dash-Fill-Inverse-Strong: #ffffff !important;
        --Dash-Stroke-Strong: #cbd5e1 !important;
        --Dash-Stroke-Weak: #e2e8f0 !important;
        --Dash-Text-Primary: #0f172a !important;
        --Dash-Text-Strong: #000000 !important;
        --Dash-Text-Weak: #475569 !important;
        --Dash-Text-Disabled: #94a3b8 !important;
        --Dash-Fill-Interactive-Strong: #0284c7 !important;
        --Dash-Fill-Interactive-Weak: #f1f5f9 !important;
        --Dash-Fill-Primary-Hover: #f1f5f9 !important;
        --Dash-Fill-Primary-Active: #e2e8f0 !important;
      }
      [data-theme="light"] .dash-dropdown,
      [data-theme="light"] .dash-dropdown-content,
      [data-theme="light"] .dash-dropdown-wrapper {
        background-color: #ffffff !important;
        background: #ffffff !important;
        color: #000000 !important;
        border-color: #cbd5e1 !important;
      }
      [data-theme="light"] .dash-dropdown-value,
      [data-theme="light"] .dash-dropdown-value-item,
      [data-theme="light"] .dash-dropdown-trigger-icon,
      [data-theme="light"] .dash-dropdown-value-count,
      [data-theme="light"] .dash-dropdown-option,
      [data-theme="light"] .dash-dropdown-trigger {
        color: #000000 !important;
        fill: #000000 !important;
      }
      [data-theme="light"] .dash-dropdown-placeholder {
        color: #64748b !important;
      }
      [data-theme="light"] .dash-dropdown-option:hover,
      [data-theme="light"] .dash-dropdown-option:focus,
      [data-theme="light"] .dash-dropdown-option[data-highlighted] {
        background-color: #f1f5f9 !important;
        color: #000000 !important;
      }
      [data-theme="light"] .dash-dropdown-option[data-state="checked"],
      [data-theme="light"] .dash-dropdown-option.is-selected {
        background-color: #e2e8f0 !important;
        color: #000000 !important;
      }
      [data-theme="light"] .Select-control, [data-theme="light"] .dash-dropdown .Select-control,
      .theme-light .Select-control, .is-light .Select-control {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 1px solid #cbd5e1 !important;
      }
      [data-theme="light"] .Select-value-label, [data-theme="light"] .dash-dropdown .Select-value-label,
      [data-theme="light"] .Select--single > .Select-control .Select-value,
      .theme-light .Select-value-label, .is-light .Select-value-label {
        color: #000000 !important;
      }
      [data-theme="light"] .Select-placeholder, [data-theme="light"] .dash-dropdown .Select-placeholder {
        color: #64748b !important;
      }
      [data-theme="light"] .Select-input > input, [data-theme="light"] .dash-dropdown .Select-input > input {
        color: #000000 !important;
      }
      [data-theme="light"] .Select-menu-outer, [data-theme="light"] .dash-dropdown .Select-menu-outer,
      .theme-light .Select-menu-outer, .is-light .Select-menu-outer {
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15) !important;
      }
      [data-theme="light"] .Select-menu, [data-theme="light"] .dash-dropdown .Select-menu {
        background-color: #ffffff !important;
      }
      [data-theme="light"] .Select-option, [data-theme="light"] .dash-dropdown .Select-option,
      .theme-light .Select-option, .is-light .Select-option {
        background-color: #ffffff !important;
        color: #000000 !important;
      }
      [data-theme="light"] .Select-option.is-focused, [data-theme="light"] .dash-dropdown .Select-option.is-focused,
      [data-theme="light"] .Select-option:hover, [data-theme="light"] .dash-dropdown .Select-option:hover {
        background-color: #f1f5f9 !important;
        color: #000000 !important;
      }
      [data-theme="light"] .Select-option.is-selected, [data-theme="light"] .dash-dropdown .Select-option.is-selected {
        background-color: #e2e8f0 !important;
        color: #000000 !important;
      }
      [data-theme="light"] .Select-option.is-disabled, [data-theme="light"] .dash-dropdown .Select-option.is-disabled {
        color: #94a3b8 !important;
        background-color: #ffffff !important;
      }
    </style>
  </head>
  <body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body>
</html>"""


def _files_store(paths: list[str], geometry: str | None) -> dict[str, EventFile]:
    geom = Geometry(geometry) if geometry else None
    out: dict[str, EventFile] = {}
    skipped: list[tuple[str, str]] = []
    for p in paths:
        try:
            # Deliberately NOT reusing the previous file's geometry: two samples
            # can need different detectors, and the wrong one fails silently
            # (all coordinates NaN). load_geometry() caches by path instead.
            out[p] = EventFile(p, geometry=geom)
        except (ArtReadError, GeometryError, OSError, ValueError) as exc:
            # One unreadable file must not take the whole browser down with it.
            skipped.append((p, str(exc)))
            print(f"skipping {p}: {exc}", file=sys.stderr)
    if paths and not out:
        # Given files and none of them worked: that is an error worth stopping
        # for. Given none at all is a deliberate empty start.
        detail = "\n".join(f"  {p}: {e}" for p, e in skipped)
        raise SystemExit(f"no readable files\n{detail}")
    return out


_SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")

#: CERN EOS instances, by the prefix of the namespace path they serve.
_EOS_REDIRECTORS = (("/eos/user/", "eosuser.cern.ch"),
                    ("/eos/project", "eosproject.cern.ch"),
                    ("/eos/home-", "eosuser.cern.ch"),
                    ("/eos/", "eospublic.cern.ch"))


def normalise_path(path: str) -> str:
    """Expand ``~`` and turn a bare CERN ``/eos`` path into an xrootd URL.

    Nothing mounts ``/eos`` on this host, so a path copied out of a listing or
    a colleague's message would otherwise just be "no such file". uproot reads
    ``root://`` directly, so the path can be rewritten into something that
    actually works instead of rejected.
    """
    p = path.strip()
    if _SCHEME.match(p):
        return p
    p = os.path.expanduser(p)
    if p.startswith("/eos/") and not os.path.isdir("/eos"):
        for prefix, host in _EOS_REDIRECTORS:
            if p.startswith(prefix):
                return f"root://{host}/{p}"       # // between host and path
    return p


def is_remote(path: str) -> bool:
    return bool(_SCHEME.match(path))


def open_file(files: dict[str, EventFile], path: str,
              geometry: Geometry | None = None) -> EventFile:
    """Open ``path`` and add it to ``files``, or return the already-open one.

    Accepts a local path or any URL uproot can read (``root://`` in practice).

    Raises ``ValueError`` with a message meant for the user; every failure mode
    here is something they can act on (wrong path, no permission, not an art
    file), so none of them should surface as a traceback or a blank figure.
    """
    if not path or not path.strip():
        raise ValueError("no path given")
    path = normalise_path(path)
    if path in files:
        return files[path]
    if not is_remote(path):
        # Local-only checks: a remote URL cannot be stat'ed cheaply, and the
        # open below reports its own failures well enough.
        if not os.path.exists(path):
            raise ValueError(f"no such file: {path}")
        if os.path.isdir(path):
            raise ValueError(f"that is a directory, not a file: {path}")
        if not os.access(path, os.R_OK):
            raise ValueError(f"not readable (check permissions): {path}")
    try:
        # geometry=None lets the file choose its own; load_geometry() caches the
        # .npz so this costs nothing after the first open of that detector.
        files[path] = EventFile(path, geometry=geometry)
    except (ArtReadError, GeometryError, OSError, ValueError) as exc:
        raise ValueError(f"could not open {os.path.basename(path)}: {exc}") from None
    return files[path]


def file_options(files) -> list[dict]:
    """Dropdown entries, disambiguating files that share a basename."""
    seen: dict[str, int] = {}
    for p in files:
        seen[os.path.basename(p)] = seen.get(os.path.basename(p), 0) + 1
    out = []
    for p in files:
        base = os.path.basename(p)
        # Two files called reco.root from different passes are otherwise
        # indistinguishable in the list.
        label = base if seen[base] == 1 else \
            os.path.join(os.path.basename(os.path.dirname(p)), base)
        out.append({"label": label, "value": p, "title": p})
    return out


def _parse_reco(reco):
    """Parse reco checkbox selection into (tracks, vertices, pandora_vertex)."""
    if isinstance(reco, (list, tuple, set)):
        want_tracks = "tracks" in reco or "on" in reco
        want_vertices = "vertices" in reco or "on" in reco
        want_pandora = "pandora_vtx" in reco or "pandora" in reco or "on" in reco
    elif isinstance(reco, bool):
        want_tracks = want_vertices = want_pandora = reco
    elif reco == "on":
        want_tracks = want_vertices = want_pandora = True
    else:
        want_tracks = want_vertices = want_pandora = bool(reco)
    return want_tracks, want_vertices, want_pandora


def render(files: dict[str, EventFile], path, entry, tag, colour, scale, mode,
           truth, merge="orientation", reco=None, theme="dark",
           colormap=DEFAULT_COLORMAP, radiologicals=True,
           particle_symbols: bool = False) -> tuple[object, str]:
    """Build the figure for one control state.

    Module-level rather than a closure so it can be exercised without a
    running server.
    """
    import plotly.graph_objects as go
    blank = go.Figure().update_layout(paper_bgcolor=FIG_BG, plot_bgcolor=AXES_BG,
                                      font_color=FG_MUTED)
    if path is None or entry is None:
        return blank, ("no file open — paste a path into the box above "
                       "and press Enter" if not files else "")
    if path not in files:
        return blank, f"ERROR - file not open: {path}"
    want_truth = bool(truth)
    try:
        # inside the try: a stale entry number left over from a file switch
        # raises IndexError here, and must not escape the callback.
        ev = files[path][int(entry)]
        style = dict(theme=theme or "dark", colormap=colormap or DEFAULT_COLORMAP)
        want_tracks, want_vertices, want_pandora = _parse_reco(reco)
        if mode == "flash3d":
            d = ev.display_flashes_3d(**style)
            return d.plotly_figure(height=None), d.summary()
        if mode == "optical":
            d = ev.display_optical(**style)
            return d.plotly_figure(height=None), d.summary()
        if mode == "3d":
            d = ev.display_3d(truth=want_truth,
                              tracks=want_tracks,
                              vertices=want_vertices or want_pandora,
                              pandora_vertex=want_pandora,
                              daughter_vertices=want_vertices,
                              secondary_vertices=want_vertices,
                              radiologicals=radiologicals,
                              colour_by=colour or "integral",
                              colour_scale=scale or "auto",
                              particle_symbols=particle_symbols, **style)
            return d.plotly_figure(height=None), d.summary()
        d = ev.display(tag, truth=want_truth,
                       reco=want_tracks or want_vertices or want_pandora,
                       tracks=want_tracks,
                       vertices=want_vertices or want_pandora,
                       pandora_vertex=want_pandora,
                       daughter_vertices=want_vertices,
                       secondary_vertices=want_vertices,
                       colour_by=colour,
                       radiologicals=radiologicals,
                       colour_scale=scale, merge=merge or "orientation",
                       space="readout" if mode == "readout" else "physical",
                       particle_symbols=particle_symbols,
                       **style)
        note = ""
        if d.tracks is not None:
            note = f"\n  {len(d.tracks)} tracks, {len(d.vertices or [])} vertices"
        return d.plotly_figure(), d.summary() + note
    except (ArtReadError, GeometryError, ValueError, KeyError, IndexError,
            OSError) as exc:
        # IndexError in particular: a stale entry number from a file switch
        # would otherwise escape the callback and 500 the whole page.
        return blank, f"ERROR - {exc}"


def _extract_camera(relayout, fallback=None):
    """Extract Plotly 3D scene camera from either nested or Dash flattened relayoutData."""
    if not relayout or not isinstance(relayout, dict):
        return fallback
    cam = {}
    if "scene.camera" in relayout and isinstance(relayout["scene.camera"], dict):
        cam = dict(relayout["scene.camera"])
    elif "scene" in relayout and isinstance(relayout["scene"], dict) and "camera" in relayout["scene"]:
        cam = dict(relayout["scene"]["camera"])

    for k in ("eye", "center", "up"):
        key = f"scene.camera.{k}"
        if key in relayout and isinstance(relayout[key], dict):
            cam[k] = dict(relayout[key])
        for axis in ("x", "y", "z"):
            flat_key = f"scene.camera.{k}.{axis}"
            if flat_key in relayout:
                if k not in cam or not isinstance(cam[k], dict):
                    cam[k] = {}
                cam[k][axis] = float(relayout[flat_key])

    if "eye" in cam and isinstance(cam["eye"], dict) and "x" in cam["eye"]:
        return cam
    return fallback


def build_app(paths: list[str], geometry: str | None = None,
              allow_open: bool = True) -> Dash:
    files = _files_store(paths, geometry)
    # ONLY an explicit --geometry. Reusing the first file's resolved geometry
    # forced it onto every later one: opening a full-10kt file from a server
    # started on the 1x2x6 sample put 99% of its hits outside the channel map,
    # silently. open_file(geometry=None) lets each file name its own.
    geom = Geometry(geometry) if geometry else None
    app = Dash(__name__, title="pylarevd")
    # Dash gives a disabled checkbox no colour of its own, so the browser
    # default (dark grey) applies -- unreadable on this panel. State both the
    # enabled and disabled colours explicitly.
    app.index_string = _INDEX_TEMPLATE.replace("__FG__", FG).replace(
        "__MUTED__", FG_MUTED)

    app.layout = html.Div(
        id="app-container",
        style={"backgroundColor": FIG_BG, "minHeight": "100vh", "padding": "10px 16px 4px 16px",
               "boxSizing": "border-box", "fontFamily": "system-ui, sans-serif"},
        children=[
            dcc.Location(id="url", refresh=False),
            dcc.Store(id="url-restored", data=False),
            dcc.Store(id="last-event"),
            dcc.Store(id="camera-ref", data=None),
            dcc.Store(id="camera-current", data=None),
            dcc.Store(id="show-symbols", data=False),
            dcc.Download(id="download-snapshot"),
            dcc.Download(id="download-gif"),
            html.H3("pylarevd — LArSoft event display",
                    style={"color": FG, "margin": "0 0 10px 0", "fontWeight": 600}),
            html.Div(
                style={"display": "flex", "gap": "14px", "flexWrap": "wrap",
                       "alignItems": "flex-start", "marginBottom": "10px"},
                children=[
                  html.Div(style=_BAND, children=[
                    html.Div([html.Label("file", style=_LABEL),
                              dcc.Dropdown(
                                  id="file",
                                  options=file_options(files),
                                  value=next(iter(files), None), clearable=False,
                                  style={"width": "340px", **_CTL})]),
                    html.Div([html.Label("event", style=_LABEL),
                              dcc.Dropdown(id="entry", clearable=False,
                                            style={"width": "220px", **_CTL})]),
                    html.Div([html.Label("hit product", style=_LABEL),
                              dcc.Dropdown(id="tag", clearable=False,
                                            style={"width": "220px", **_CTL})]),
                    html.Div([html.Label("go to entry", style=_LABEL),
                              dcc.Input(id="jump", type="number", min=0, step=1,
                                        placeholder="entry #", debounce=True,
                                        style={"width": "90px", **_CTL})]),
                  ]),
                  html.Div(style=_BAND, children=[
                    html.Div([html.Label("colour by", style=_LABEL),
                              dcc.Dropdown(
                                  id="colour",
                                  options=[
                                      {"label": "integral (charge)", "value": "integral"},
                                      {"label": "amplitude", "value": "amplitude"},
                                      {"label": "tick", "value": "tick"},
                                      {"label": "multiplicity",
                                       "value": "multiplicity"},
                                      {"label": "-- by object --",
                                       "value": "__sep__", "disabled": True},
                                      {"label": "track", "value": "track"},
                                      {"label": "shower", "value": "shower"},
                                      {"label": "pfparticle", "value": "pfparticle"},
                                      {"label": "cluster", "value": "cluster"},
                                      {"label": "slice", "value": "slice"},
                                      {"label": "-- by coordinate --",
                                       "value": "__sep2__", "disabled": True},
                                      {"label": "x (drift)", "value": "x"},
                                      {"label": "y (up)", "value": "y"},
                                      {"label": "z (beam)", "value": "z"}],
                                  value="integral", clearable=False,
                                  style={"width": "175px", **_CTL})]),
                    html.Div([html.Label("scale", style=_LABEL),
                              dcc.Dropdown(id="scale",
                                           options=["auto", "log", "linear"],
                                           value="auto", clearable=False,
                                           style={"width": "120px", **_CTL})]),
                    html.Div([html.Label("theme", style=_LABEL),
                              dcc.Dropdown(id="theme", options=sorted(THEMES),
                                           value="dark", clearable=False,
                                           style={"width": "110px", **_CTL})]),
                    html.Div([html.Label("colormap", style=_LABEL),
                              dcc.Dropdown(
                                  id="colormap",
                                  options=[{"label": n, "value": n,
                                            "title": c.note}
                                           for n, c in sorted(COLORMAPS.items())],
                                  value=DEFAULT_COLORMAP, clearable=False,
                                  style={"width": "130px", **_CTL})]),
                  ]),
                  html.Div(style=_BAND, children=[
                    html.Div([html.Label("view", style=_LABEL),
                              dcc.Dropdown(
                                  id="mode",
                                  options=[
                                      {"label": "2D physical", "value": "2d"},
                                      {"label": "2D readout (channel/tick)",
                                       "value": "readout"},
                                      {"label": "3D space points", "value": "3d"},
                                      {"label": "Optical (PDS)", "value": "optical"},
                                      {"label": "Flashes 3D (PDS)",
                                       "value": "flash3d"}],
                                  value="2d", clearable=False,
                                  style={"width": "210px", **_CTL})]),
                    html.Div([html.Label("panels", style=_LABEL),
                              dcc.Dropdown(
                                  id="merge",
                                  options=[
                                      {"label": "by wire orientation",
                                       "value": "orientation"},
                                      {"label": "by view (U/V/Z)", "value": "view"},
                                      {"label": "split every face", "value": "none"}],
                                  value="orientation", clearable=False,
                                  style={"width": "190px", **_CTL})]),
                    html.Div([
                        html.Label("overlays", style=_LABEL),
                        dcc.Checklist(
                            id="truth",
                            options=[{"label": " truth overlay", "value": "on"}],
                            value=[], className="evd-check",
                            style=_CHECK, labelStyle=_CHECK_LABEL,
                            inputStyle=_CHECK_INPUT),
                        dcc.Checklist(
                            id="reco",
                            options=[
                                {"label": " tracks", "value": "tracks"},
                                {"label": " vertices", "value": "vertices"},
                                {"label": " pandora vtx", "value": "pandora_vtx"},
                            ],
                            value=[], className="evd-check",
                            style=_CHECK, labelStyle={**_CHECK_LABEL, "marginRight": "8px"},
                            inputStyle=_CHECK_INPUT),
                        dcc.Checklist(
                            id="radio",
                            options=[{"label": " radiologicals", "value": "on"}],
                            value=[], className="evd-check",
                            style=_CHECK, labelStyle=_CHECK_LABEL,
                            inputStyle=_CHECK_INPUT)]),
                  ]),
                  html.Div(style=_BAND, children=[
                    html.Div([
                        html.Button("◀ prev", id="prev", n_clicks=0,
                                    style={**_CTL, "marginRight": "6px",
                                           "padding": "6px 12px", "cursor": "pointer",
                                           "borderRadius": "4px"}),
                        html.Button("next ▶", id="next", n_clicks=0,
                                    style={**_CTL, "marginRight": "8px",
                                           "padding": "6px 12px",
                                           "cursor": "pointer", "borderRadius": "4px"}),
                        html.Button("📌 fix perspective", id="fix-cam", n_clicks=0,
                                    title="Set/lock current 3D view as reference perspective",
                                    style={**_CTL, "marginRight": "6px",
                                           "padding": "6px 12px", "cursor": "pointer",
                                           "borderRadius": "4px"}),
                        html.Button("🏷️ particle symbols", id="particle-symbols", n_clicks=0,
                                    title="Toggle true particle symbols on tracks and showers (2D & 3D)",
                                    style={**_CTL, "marginRight": "6px",
                                           "padding": "6px 12px", "cursor": "pointer",
                                           "borderRadius": "4px"}),
                        html.Button("📷 snapshot (PDF)", id="snapshot-pdf", n_clicks=0,
                                    title="Save vector PDF from current perspective",
                                    style={**_CTL, "marginRight": "6px", "padding": "6px 12px", "cursor": "pointer",
                                           "borderRadius": "4px"}),
                        html.Button("🎞️ rotate GIF", id="snapshot-gif", n_clicks=0,
                                    title="Export rotating 360° animated GIF of the 3D event",
                                    style={**_CTL, "padding": "6px 12px", "cursor": "pointer",
                                           "borderRadius": "4px"}),
                    ]),
                  ]),
                ]),
            html.Div(style={**_BAND, "marginBottom": "8px"}, children=[
                html.Div(style={"flexGrow": 1}, children=[
                    html.Label("open by path", style=_LABEL),
                    dcc.Input(
                        id="path", type="text", debounce=True,
                        placeholder=("/eos/user/... or root://... or a local path "
                                     "— paste and press Enter"
                                     if allow_open else
                                     "disabled: server is not bound to localhost "
                                     "(pass --allow-remote-open)"),
                        disabled=not allow_open,
                        style={"width": "100%", "boxSizing": "border-box",
                               "padding": "6px 8px", "borderRadius": "4px",
                               "fontFamily": "ui-monospace, monospace",
                               "fontSize": "12px", **_CTL})]),
                html.Button("open", id="open", n_clicks=0, disabled=not allow_open,
                            style={**_CTL, "padding": "6px 14px",
                                   "cursor": "pointer" if allow_open else "default",
                                   "borderRadius": "4px"}),
                dcc.Loading(html.Div(id="path-msg",
                         style={"color": FG_MUTED, "fontSize": "11px",
                                "alignSelf": "center", "minWidth": "200px",
                                "fontFamily": "ui-monospace, monospace"}),
                            type="dot", color=FG_MUTED),

            ]),
            html.Pre(id="summary",
                     style={"color": FG_MUTED, "fontSize": "11px", "margin": "0 0 6px 0",
                            "whiteSpace": "pre-wrap", "fontFamily": "ui-monospace, monospace"}),
            dcc.Loading(dcc.Graph(id="fig", responsive=True,
                                  style={"height": "calc(100vh - 180px)", "minHeight": "550px", "width": "100%"},
                                  config={"scrollZoom": True, "displaylogo": False, "responsive": True}),
                        type="dot", color=FG_MUTED),
        ])

    # ---- callbacks ----------------------------------------------------

    @app.callback(Output("file", "options", allow_duplicate=True),
                  Output("file", "value", allow_duplicate=True),
                  Output("path-msg", "children"),
                  Input("open", "n_clicks"), Input("path", "n_submit"),
                  State("path", "value"), prevent_initial_call=True)
    def _open(_clicks, _submits, path):
        if not allow_open or not path:
            return no_update, no_update, ""
        try:
            open_file(files, path, geom)
        except ValueError as exc:
            return no_update, no_update, f"✗ {exc}"
        # normalise_path, not expanduser: open_file stores under the
        # normalised key, so a bare /eos/... path is filed as its root:// URL.
        # Guessing the key here instead meant the box crashed on exactly the
        # paths the rewriting exists to support.
        full = normalise_path(path)
        n = len(files[full])
        return (file_options(files), full,
                f"✓ {os.path.basename(full)} — {n} event{'s' * (n != 1)}")

    @app.callback(Output("entry", "options"), Output("entry", "value"),
                  Output("tag", "options"), Output("tag", "value"),
                  Output("colour", "options"), Output("mode", "options"),
                  Output("path-msg", "children", allow_duplicate=True),
                  Input("file", "value"),
                  State("entry", "value"), State("tag", "value"),
                  State("last-event", "data"), prevent_initial_call="initial_duplicate")
    def _on_file(path, entry, tag, last):
        if path is None or path not in files:
            return [], None, [], None, no_update, no_update, no_update
        f = files[path]
        # Entry numbers are an artefact of how a file was written. When the user
        # switches files mid-session, follow the *physics* event they were on --
        # otherwise "compare this event across two reco passes" silently lands
        # on a different event, which is what the rse= URL parameter exists to
        # prevent and what this path used to ignore.
        followed = None
        if last and last.get("path") != path:
            found = f.index_of(*last["id"])
            if found is not None:
                entry = found
            else:
                # Do NOT silently keep the old index: it names a different
                # physics event here, and the result was indistinguishable
                # from the feature working.
                entry = 0
                r, sr, ev = last["id"]
                followed = (f"run {r} / subrun {sr} / event {ev} is not in "
                            f"{os.path.basename(path)} — showing entry 0")
        opts = [{"label": f"[{i}]  run {r} / sub {s} / ev {e}", "value": i}
                for i, (r, s, e) in enumerate(f.event_ids)]
        tags = [p.split("_")[1] for p in f.hit_products()]
        # Keep the user where they were when the new file can honour it, rather
        # than snapping back to event 0 and the default tag on every switch.
        entry = entry if isinstance(entry, int) and entry < len(f) else 0
        tag = tag if tag in tags else ("hitfd" if "hitfd" in tags else
                                       (tags[0] if tags else None))

        caps = f[entry].capabilities()
        def mark(label, ok):
            return label if ok else f"{label} (not in this file)"
        colour_opts = (
            [{"label": n, "value": n} for n in
             ("integral", "amplitude", "tick", "multiplicity")]
            + [{"label": "-- by object --", "value": "__sep__", "disabled": True}]
            + [{"label": mark(n, caps.get(f"group:{n}", False)), "value": n,
                "disabled": not caps.get(f"group:{n}", False)}
               for n in ("track", "shower", "pfparticle", "cluster", "slice")])
        has_reco = any(caps[k] for k in ("tracks", "vertices", "showers"))
        mode_opts = [
            {"label": "2D physical", "value": "2d"},
            {"label": "2D readout (channel/tick)", "value": "readout"},
            {"label": mark("3D space points", caps["spacepoints"]), "value": "3d",
             "disabled": not caps["spacepoints"]},
            {"label": mark("Optical (PDS)", caps["optical"]), "value": "optical",
             "disabled": not caps["optical"]},
            {"label": mark("Flashes 3D (PDS)", caps["optical"]), "value": "flash3d",
             "disabled": not caps["optical"]}]
        return (opts, entry, tags, tag, colour_opts, mode_opts,
                followed if followed else no_update)

    @app.callback(Output("truth", "options"), Output("reco", "options"),
                  Output("radio", "options"),
                  Input("file", "value"), Input("entry", "value"),
                  Input("mode", "value"), Input("truth", "value"))
    def _overlay_options(path, entry, mode, truth_on):
        """Label the overlay checkboxes with why they are unavailable.

        Two separate reasons: the file has no such product, or the current view
        does not draw overlays at all. dcc.Checklist has no component-level
        ``disabled`` prop -- it is per option -- so this is where both are said.
        """
        if path is None or entry is None:
            off = lambda lab: [{"label": f" {lab} (no file open)",
                                "value": "on", "disabled": True}]
            return (off("truth overlay"), off("tracks + vertices"),
                    off("radiologicals"))
        try:
            ev = files[path][int(entry)]
            caps = ev.capabilities()
        except Exception:
            return no_update, no_update, no_update
        has_reco = any(caps[k] for k in ("tracks", "vertices", "showers"))
        drawn = mode in ("2d", "3d")        # readout/optical draw no overlays

        def option(label, available):
            if not available:
                return [{"label": f" {label} (not in this file)",
                         "value": "on", "disabled": True}]
            if not drawn:
                return [{"label": f" {label} (not in this view)",
                         "value": "on", "disabled": True}]
            return [{"label": f" {label}", "value": "on"}]

        if not has_reco:
            reco_opts = [{"label": " tracks + vertices (not in this file)", "value": "on", "disabled": True}]
        elif not drawn:
            reco_opts = [{"label": " tracks + vertices (not in this view)", "value": "on", "disabled": True}]
        else:
            reco_opts = [
                {"label": " tracks", "value": "tracks"},
                {"label": " vertices", "value": "vertices"},
                {"label": " pandora vtx", "value": "pandora_vtx"},
            ]

        # Splitting radiologicals off needs a neutrino vertex to trace back to,
        # and only matters while truth is being drawn.
        try:
            separable = ev.neutrino_track_ids() is not None
        except Exception:
            separable = False
        if not drawn:
            # Name the actual blocker. Saying "needs truth overlay" in a view
            # where truth is ITSELF disabled sends the user to a dead end.
            radio = [{"label": " radiologicals (not in this view)",
                      "value": "on", "disabled": True}]
        elif not separable:
            radio = [{"label": " radiologicals (no neutrino to separate from)",
                      "value": "on", "disabled": True}]
        elif not truth_on:
            radio = [{"label": " radiologicals (needs truth overlay)",
                      "value": "on", "disabled": True}]
        else:
            radio = [{"label": " radiologicals", "value": "on"}]
        return (option("truth overlay", caps["truth"]), reco_opts, radio)

    @app.callback(Output("entry", "value", allow_duplicate=True),
                  Input("jump", "value"), State("file", "value"),
                  prevent_initial_call=True)
    def _jump(value, path):
        from dash import no_update
        if value is None or path not in files:
            return no_update
        return max(0, min(int(value), len(files[path]) - 1))

    @app.callback(Output("entry", "value", allow_duplicate=True),
                  Input("prev", "n_clicks"), Input("next", "n_clicks"),
                  State("entry", "value"), State("file", "value"),
                  prevent_initial_call=True)
    def _step(_p, _n, current, path):
        from dash import ctx
        if path not in files:
            return no_update
        n = len(files[path])
        cur = int(current or 0)
        if ctx.triggered_id == "prev":
            return (cur - 1) % n
        return (cur + 1) % n

    @app.callback(Output("prev", "disabled"), Output("next", "disabled"),
                  Output("jump", "disabled"), Input("file", "value"))
    def _enable_stepping(path):
        """Stepping does nothing with no file open; say so rather than no-op."""
        return (path is None,) * 3

    @app.callback(Output("tag", "disabled"), Output("colour", "disabled"),
                  Output("scale", "disabled"), Output("merge", "disabled"),
                  Input("mode", "value"))
    def _enable(mode):  # noqa: D401
        """Grey out every control the selected view ignores."""
        other = mode in ("optical", "flash3d")
        three_d = mode == "3d"
        readout = mode == "readout"
        return (other or three_d), other, (other or readout), (other or readout or three_d)

    @app.callback(Output("fig", "figure"), Output("summary", "children"),
                  Output("summary", "style"), Output("last-event", "data"),
                  Output("camera-current", "data"),
                  Input("file", "value"), Input("entry", "value"), Input("tag", "value"),
                  Input("colour", "value"), Input("scale", "value"),
                  Input("mode", "value"), Input("truth", "value"),
                  Input("merge", "value"), Input("reco", "value"),
                  Input("theme", "value"), Input("colormap", "value"),
                  Input("radio", "value"),
                  Input("camera-ref", "data"),
                  Input("show-symbols", "data"),
                  State("fig", "relayoutData"),
                  State("camera-current", "data"),
                  State("last-event", "data"))
    def _render(path, entry, tag, colour, scale, mode, truth, merge, reco,
                theme, colormap, radio, cam_ref, show_symbols,
                relayout, last_cam, last_ev):
        fig, summary = render(files, path, entry, tag, colour, scale, mode, truth,
                              merge, reco, theme, colormap, bool(radio),
                              particle_symbols=bool(show_symbols))
        base = {"fontSize": "11px", "margin": "0 0 6px 0", "whiteSpace": "pre-wrap",
                "fontFamily": "ui-monospace, monospace"}
        seen = None
        try:
            if path is not None and entry is not None and path in files:
                seen = {"path": path, "id": list(files[path][int(entry)].id)}
        except Exception:
            seen = no_update      # keep the last good event; do not erase it
        failed = summary.startswith("ERROR")
        style = {**base,
                 "color": "#fca5a5" if failed else FG_MUTED,
                 "backgroundColor": "#450a0a" if failed else "transparent",
                 "padding": "6px 8px" if failed else "0",
                 "borderRadius": "4px"}

        # Extract current camera from relayoutData if present (supports flattened and nested keys)
        active_cam = _extract_camera(relayout)
        current_cam = active_cam or last_cam

        # Apply perspective for 3D modes
        cam_to_apply = None
        if cam_ref:
            cam_to_apply = cam_ref
        elif current_cam and mode in ("3d", "flash3d"):
            # Check if event changed: if staying on the same event (e.g. toggling reco, truth, etc.)
            # or if reference is set, keep the perspective intact!
            same_ev = (seen is not None and last_ev is not None and
                       seen.get("path") == last_ev.get("path") and
                       seen.get("id") == last_ev.get("id"))
            if same_ev:
                cam_to_apply = current_cam

        if cam_to_apply and hasattr(fig, "layout") and hasattr(fig.layout, "scene"):
            fig.layout.scene.camera = cam_to_apply
            fig.layout.scene.uirevision = "fixed_perspective"
            fig.layout.uirevision = "fixed_perspective"

        return fig, summary, style, seen, current_cam

    @app.callback(Output("camera-ref", "data"),
                  Output("fix-cam", "children"),
                  Output("fix-cam", "style"),
                  Input("fix-cam", "n_clicks"),
                  State("camera-ref", "data"),
                  State("fig", "relayoutData"),
                  State("camera-current", "data"),
                  prevent_initial_call=True)
    def _toggle_fix_cam(n_clicks, current_ref, relayout, current_cam):
        if current_ref is not None:
            # Release fixed perspective
            return None, "📌 fix perspective", {**_CTL, "marginRight": "6px", "padding": "6px 12px", "cursor": "pointer", "borderRadius": "4px"}

        # Lock perspective
        cam = _extract_camera(relayout) or current_cam or {"eye": {"x": 1.25, "y": 1.25, "z": 1.25}}

        fixed_style = {**_CTL, "marginRight": "6px", "padding": "6px 12px", "cursor": "pointer",
                       "borderRadius": "4px", "backgroundColor": "#0284c7", "color": "#ffffff",
                       "fontWeight": "600", "border": "1px solid #38bdf8"}
        return cam, "✓ perspective fixed", fixed_style

    @app.callback(Output("show-symbols", "data"),
                  Output("particle-symbols", "children"),
                  Output("particle-symbols", "style"),
                  Input("particle-symbols", "n_clicks"),
                  State("show-symbols", "data"),
                  prevent_initial_call=True)
    def _toggle_symbols(n_clicks, current):
        active = not bool(current)
        if active:
            style = {**_CTL, "marginRight": "6px", "padding": "6px 12px", "cursor": "pointer",
                     "borderRadius": "4px", "backgroundColor": "#0284c7", "color": "#ffffff",
                     "fontWeight": "600", "border": "1px solid #38bdf8"}
            return True, "✓ particle symbols", style
        style = {**_CTL, "marginRight": "6px", "padding": "6px 12px", "cursor": "pointer",
                 "borderRadius": "4px"}
        return False, "🏷️ particle symbols", style

    @app.callback(Output("download-snapshot", "data"),
                  Input("snapshot-pdf", "n_clicks"),
                  State("file", "value"), State("entry", "value"), State("tag", "value"),
                  State("colour", "value"), State("scale", "value"),
                  State("mode", "value"), State("truth", "value"),
                  State("merge", "value"), State("reco", "value"),
                  State("theme", "value"), State("colormap", "value"),
                  State("radio", "value"),
                  State("show-symbols", "data"),
                  State("fig", "relayoutData"),
                  State("camera-ref", "data"),
                  State("camera-current", "data"),
                  prevent_initial_call=True)
    def _snapshot(n_clicks, path, entry, tag, colour, scale, mode, truth,
                  merge, reco, theme, colormap, radio, show_symbols,
                  relayout, cam_ref, current_cam):
        import io
        import matplotlib.pyplot as plt

        if not path or entry is None or path not in files:
            return no_update
        try:
            ev = files[path][int(entry)]
            style = dict(theme=theme or "dark", colormap=colormap or DEFAULT_COLORMAP)
            want_truth = bool(truth)

            buf = io.BytesIO()
            run, subrun, ev_num = ev.id
            filename = f"pylarevd_run{run}_sub{subrun}_ev{ev_num}_{mode}.pdf"

            if mode == "3d":
                cam = _extract_camera(relayout) or cam_ref or current_cam

                elev = 18.0
                azim = -60.0
                zoom = 1.15
                if cam and isinstance(cam, dict) and "eye" in cam:
                    eye = cam["eye"]
                    center = cam.get("center", {"x": 0.0, "y": 0.0, "z": 0.0}) or {"x": 0.0, "y": 0.0, "z": 0.0}
                    dx = float(eye.get("x", 1.25)) - float(center.get("x", 0.0))
                    dy = float(eye.get("y", 1.25)) - float(center.get("y", 0.0))
                    dz = float(eye.get("z", 1.25)) - float(center.get("z", 0.0))
                    norm = float(np.sqrt(dx**2 + dy**2 + dz**2))
                    if norm > 1e-6:
                        elev = float(np.rad2deg(np.arcsin(np.clip(dz / norm, -1.0, 1.0))))
                        azim = float(np.rad2deg(np.arctan2(dy, dx)))
                        zoom = float(np.clip(1.15 * (2.165 / norm), 0.65, 1.80))

                want_tracks, want_vertices, want_pandora = _parse_reco(reco)
                d = ev.display_3d(truth=want_truth,
                                  tracks=want_tracks,
                                  vertices=want_vertices or want_pandora,
                                  pandora_vertex=want_pandora,
                                  daughter_vertices=want_vertices,
                                  secondary_vertices=want_vertices,
                                  radiologicals=bool(radio), colour_by=colour or "integral",
                                  colour_scale=scale or "auto",
                                  particle_symbols=bool(show_symbols), **style)
                fig_mpl = d.figure(elev=elev, azim=azim, zoom=zoom, pad=0.20,
                                   particle_symbols=bool(show_symbols))
                fig_mpl.savefig(buf, format="pdf", bbox_inches="tight", pad_inches=0.3, dpi=300)
                plt.close(fig_mpl)
            elif mode == "flash3d":
                d = ev.display_flashes_3d(**style)
                fig_mpl = d.figure()
                fig_mpl.savefig(buf, format="pdf", bbox_inches="tight", dpi=300)
                plt.close(fig_mpl)
            elif mode == "optical":
                d = ev.display_optical(**style)
                fig_mpl = d.figure()
                fig_mpl.savefig(buf, format="pdf", bbox_inches="tight", dpi=300)
                plt.close(fig_mpl)
            else:
                want_tracks, want_vertices, want_pandora = _parse_reco(reco)
                d = ev.display(tag, truth=want_truth,
                               reco=want_tracks or want_vertices or want_pandora,
                               tracks=want_tracks,
                               vertices=want_vertices or want_pandora,
                               pandora_vertex=want_pandora,
                               daughter_vertices=want_vertices,
                               secondary_vertices=want_vertices,
                               colour_by=colour,
                               radiologicals=bool(radio), colour_scale=scale,
                               merge=merge or "orientation",
                               space="readout" if mode == "readout" else "physical",
                               particle_symbols=bool(show_symbols),
                               **style)
                fig_mpl = d.figure(particle_symbols=bool(show_symbols))
                fig_mpl.savefig(buf, format="pdf", bbox_inches="tight", dpi=300)
                plt.close(fig_mpl)

            buf.seek(0)
            data_bytes = buf.getvalue()

            # Also save to local evd_out/ directory as a backup
            try:
                os.makedirs("evd_out", exist_ok=True)
                local_path = os.path.join("evd_out", filename)
                with open(local_path, "wb") as f:
                    f.write(data_bytes)
                print(f"[pylarevd] Snapshot saved to {local_path} ({len(data_bytes)} bytes)")
            except Exception as e:
                print(f"[pylarevd] Note: could not write local backup: {e}", file=sys.stderr)

            return dcc.send_bytes(data_bytes, filename=filename, type="application/pdf")
        except Exception as exc:
            print(f"[pylarevd] Snapshot error: {exc}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            return no_update

    @app.callback(Output("download-gif", "data"),
                  Input("snapshot-gif", "n_clicks"),
                  State("file", "value"), State("entry", "value"), State("tag", "value"),
                  State("colour", "value"), State("scale", "value"),
                  State("mode", "value"), State("truth", "value"),
                  State("merge", "value"), State("reco", "value"),
                  State("theme", "value"), State("colormap", "value"),
                  State("radio", "value"),
                  State("show-symbols", "data"),
                  State("fig", "relayoutData"),
                  State("camera-ref", "data"),
                  State("camera-current", "data"),
                  prevent_initial_call=True)
    def _snapshot_gif(n_clicks, path, entry, tag, colour, scale, mode, truth,
                      merge, reco, theme, colormap, radio, show_symbols,
                      relayout, cam_ref, current_cam):
        if not path or entry is None or path not in files:
            return no_update
        try:
            ev = files[path][int(entry)]
            style = dict(theme=theme or "dark", colormap=colormap or DEFAULT_COLORMAP)
            want_truth = bool(truth)

            run, subrun, ev_num = ev.id
            filename = f"pylarevd_run{run}_sub{subrun}_ev{ev_num}_3d_rotation.gif"

            cam = _extract_camera(relayout) or cam_ref or current_cam
            elev = 18.0
            azim = -60.0
            zoom = 1.15
            if cam and isinstance(cam, dict) and "eye" in cam:
                eye = cam["eye"]
                center = cam.get("center", {"x": 0.0, "y": 0.0, "z": 0.0}) or {"x": 0.0, "y": 0.0, "z": 0.0}
                dx = float(eye.get("x", 1.25)) - float(center.get("x", 0.0))
                dy = float(eye.get("y", 1.25)) - float(center.get("y", 0.0))
                dz = float(eye.get("z", 1.25)) - float(center.get("z", 0.0))
                norm = float(np.sqrt(dx**2 + dy**2 + dz**2))
                if norm > 1e-6:
                    elev = float(np.rad2deg(np.arcsin(np.clip(dz / norm, -1.0, 1.0))))
                    azim = float(np.rad2deg(np.arctan2(dy, dx)))
                    zoom = float(np.clip(1.15 * (2.165 / norm), 0.65, 1.80))

            want_tracks, want_vertices, want_pandora = _parse_reco(reco)
            d = ev.display_3d(truth=want_truth,
                              tracks=want_tracks,
                              vertices=want_vertices or want_pandora,
                              pandora_vertex=want_pandora,
                              daughter_vertices=want_vertices,
                              secondary_vertices=want_vertices,
                              radiologicals=bool(radio), colour_by=colour or "integral",
                              colour_scale=scale or "auto",
                              particle_symbols=bool(show_symbols), **style)
            gif_bytes = d.rotate_gif(elev=elev, start_azim=azim, zoom=zoom, pad=0.20,
                                     particle_symbols=bool(show_symbols))

            try:
                os.makedirs("evd_out", exist_ok=True)
                local_path = os.path.join("evd_out", filename)
                with open(local_path, "wb") as f:
                    f.write(gif_bytes)
                print(f"[pylarevd] Rotating GIF saved to {local_path} ({len(gif_bytes)} bytes)")
            except Exception as e:
                print(f"[pylarevd] Note: could not write local GIF backup: {e}", file=sys.stderr)

            return dcc.send_bytes(gif_bytes, filename=filename, type="image/gif")
        except Exception as exc:
            print(f"[pylarevd] GIF snapshot error: {exc}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            return no_update

    @app.callback(Output("app-container", "data-theme"),
                  Input("theme", "value"))
    def _theme_change(theme_name):
        is_dark = (theme_name or "dark") == "dark"
        return "dark" if is_dark else "light"

    # Left/right arrows step events: scanning many events should not mean
    # aiming at a button every time.
    app.clientside_callback(
        """
        function(_) {
            if (!window.__pylarevd_keys) {
                window.__pylarevd_keys = true;
                document.addEventListener('keydown', function (e) {
                    if (e.target && /INPUT|TEXTAREA/.test(e.target.tagName)) return;
                    if (e.key === 'ArrowRight') {
                        const b = document.getElementById('next'); if (b) b.click();
                    } else if (e.key === 'ArrowLeft') {
                        const b = document.getElementById('prev'); if (b) b.click();
                    }
                });
            }
            return window.dash_clientside.no_update;
        }
        """,
        Output("jump", "placeholder"), Input("file", "value"))

    # ---- URL state ------------------------------------------------------
    # Without this a view can only be shared by describing it in words: nine
    # control values and "you'll know it when you see it".

    by_name = {os.path.basename(p): p for p in files}

    @app.callback(
        [Output(cid, "value", allow_duplicate=True) for cid, _ in _URL_CONTROLS]
        + [Output("url-restored", "data"),
           Output("file", "options", allow_duplicate=True),
           Output("path-msg", "children", allow_duplicate=True)],
        Input("url", "search"), State("url-restored", "data"),
        prevent_initial_call="initial_duplicate")
    def _restore(search, restored):
        """Apply query parameters once, on first load."""
        idle = [no_update] * len(_URL_CONTROLS)
        if restored or not search:
            return idle + [True, no_update, no_update]
        # A link may name a file this server was not started with. Open it here,
        # before anything downstream tries to look it up.
        wanted = file_from_url(dict(parse_qsl(search.lstrip("?"))).get("file", ""),
                               by_name)
        opts, msg = no_update, no_update
        failed = None
        if wanted is not None and wanted not in files:
            if not allow_open:
                failed = f"opening files by path is disabled: {wanted}"
            else:
                try:
                    open_file(files, wanted, geom)
                    opts = file_options(files)
                    msg = f"✓ opened from link: {os.path.basename(wanted)}"
                except ValueError as exc:
                    failed = str(exc)
        parsed = url_values(search, by_name, files.get)
        if failed is not None:
            # Honour the rest of the link anyway. Losing the file should not
            # also lose the theme, view and overlays it was sent with -- but
            # file and entry go together, so drop the entry with it rather than
            # applying an index meant for a file we do not have.
            parsed[0] = parsed[1] = None
            msg = f"✗ {failed}"
        return ([no_update if v is None else v for v in parsed]
                + [True, opts, msg])

    @app.callback(Output("url", "search"),
                  [Input(cid, "value") for cid, _ in _URL_CONTROLS],
                  prevent_initial_call=True)
    def _persist(*values):
        path, entry = values[0], values[1]
        event_id = None
        try:
            src = files.get(path) if path else None
            if src is not None and entry is not None and 0 <= int(entry) < len(src):
                event_id = src[int(entry)].id
        except Exception:
            event_id = None       # a link without rse still works, just by entry
        return url_params(values, event_id)

    app._pylarevd_files = files      # handy for tests and interactive poking
    return app


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("files", nargs="*",
                    help="art-ROOT file(s); optional -- with none, start empty "
                         "and use the 'open by path' box in the browser")
    ap.add_argument("-g", "--geometry", default=None, help="geometry .npz")
    ap.add_argument("--port", type=int, default=8050)
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind address (default localhost only; use an SSH tunnel)")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--allow-remote-open", action="store_true",
                    help="permit opening files by path even when --host exposes "
                         "the server beyond localhost (this lets anyone who can "
                         "reach the port read any file this account can read)")
    a = ap.parse_args(argv)

    loopback = a.host in ("127.0.0.1", "localhost", "::1")
    # Opening files by path turns the browser into a file reader for whoever can
    # reach the port. That is exactly what is wanted on localhost and not at all
    # what is wanted on an exposed bind address.
    allow_open = loopback or a.allow_remote_open
    app = build_app(a.files, a.geometry, allow_open=allow_open)
    node = os.uname().nodename
    fqdn = node if "." in node else f"{node}.cern.ch"
    print(f"\n  pylarevd serving on http://{a.host}:{a.port}", flush=True)
    if a.host in ("127.0.0.1", "localhost"):
        print(f"  from your laptop:  ssh -N -L {a.port}:localhost:{a.port} {fqdn}\n",
              flush=True)
    else:
        print(f"  WARNING: bound to {a.host} - reachable from the network, "
              "not just this host", flush=True)
        print("  open-by-path is %s\n"
              % ("ENABLED (--allow-remote-open): anyone who can reach this port "
                 "can read any file you can" if a.allow_remote_open
                 else "disabled; pass --allow-remote-open to enable"), flush=True)
    app.run(host=a.host, port=a.port, debug=a.debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
