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

from urllib.parse import parse_qsl, urlencode

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
    {%metas%}<title>{%title%}</title>{%favicon%}{%css%}
    <style>
      .evd-check label, .evd-check span { color: __FG__; }
      .evd-check label:has(input:disabled),
      .evd-check label:has(input:disabled) span { color: __MUTED__; opacity: .8; }
      .evd-check input:disabled { cursor: not-allowed; }
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
    if not out:
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


def render(files: dict[str, EventFile], path, entry, tag, colour, scale, mode,
           truth, merge="orientation", reco=None, theme="dark",
           colormap=DEFAULT_COLORMAP, radiologicals=True) -> tuple[object, str]:
    """Build the figure for one control state.

    Module-level rather than a closure so it can be exercised without a
    running server.
    """
    import plotly.graph_objects as go
    blank = go.Figure().update_layout(paper_bgcolor=FIG_BG, plot_bgcolor=AXES_BG,
                                      font_color=FG_MUTED)
    if path is None or entry is None:
        return blank, ""
    if path not in files:
        return blank, f"ERROR - file not open: {path}"
    want_truth = bool(truth)
    try:
        # inside the try: a stale entry number left over from a file switch
        # raises IndexError here, and must not escape the callback.
        ev = files[path][int(entry)]
        style = dict(theme=theme or "dark", colormap=colormap or DEFAULT_COLORMAP)
        if mode == "flash3d":
            d = ev.display_flashes_3d(**style)
            return d.plotly_figure(), d.summary()
        if mode == "optical":
            d = ev.display_optical(**style)
            return d.plotly_figure(), d.summary()
        if mode == "3d":
            d = ev.display_3d(truth=want_truth, tracks=bool(reco),
                              radiologicals=radiologicals, **style)
            return d.plotly_figure(), d.summary()
        d = ev.display(tag, truth=want_truth, reco=bool(reco), colour_by=colour,
                       radiologicals=radiologicals,
                       colour_scale=scale, merge=merge or "orientation",
                       space="readout" if mode == "readout" else "physical",
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


def build_app(paths: list[str], geometry: str | None = None,
              allow_open: bool = True) -> Dash:
    files = _files_store(paths, geometry)
    geom = next(iter(files.values())).geometry
    app = Dash(__name__, title="pylarevd")
    # Dash gives a disabled checkbox no colour of its own, so the browser
    # default (dark grey) applies -- unreadable on this panel. State both the
    # enabled and disabled colours explicitly.
    app.index_string = _INDEX_TEMPLATE.replace("__FG__", FG).replace(
        "__MUTED__", FG_MUTED)

    app.layout = html.Div(
        style={"backgroundColor": FIG_BG, "minHeight": "100vh", "padding": "12px 16px",
               "fontFamily": "system-ui, sans-serif"},
        children=[
            dcc.Location(id="url", refresh=False),
            dcc.Store(id="url-restored", data=False),
            dcc.Store(id="last-event"),
            html.H3("pylarevd — LArSoft event display",
                    style={"color": FG, "margin": "0 0 12px 0", "fontWeight": 600}),
            html.Div(
                style={"display": "flex", "gap": "14px", "flexWrap": "wrap",
                       "alignItems": "flex-start", "marginBottom": "10px"},
                children=[
                  html.Div(style=_BAND, children=[
                    html.Div([html.Label("file", style=_LABEL),
                              dcc.Dropdown(
                                  id="file",
                                  options=file_options(files),
                                  value=next(iter(files)), clearable=False,
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
                                      {"label": "integral", "value": "integral"},
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
                                      {"label": "slice", "value": "slice"}],
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
                            options=[{"label": " tracks + vertices", "value": "on"}],
                            value=[], className="evd-check",
                            style=_CHECK, labelStyle=_CHECK_LABEL,
                            inputStyle=_CHECK_INPUT),
                        dcc.Checklist(
                            id="radio",
                            options=[{"label": " radiologicals", "value": "on"}],
                            value=["on"], className="evd-check",
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
                                    style={**_CTL, "padding": "6px 12px",
                                           "cursor": "pointer", "borderRadius": "4px"}),
                    ]),
                  ]),
                ]),
            html.Div(style={**_BAND, "marginBottom": "10px"}, children=[
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
                html.Div(id="path-msg",
                         style={"color": FG_MUTED, "fontSize": "11px",
                                "alignSelf": "center", "minWidth": "200px",
                                "fontFamily": "ui-monospace, monospace"}),
            ]),
            html.Pre(id="summary",
                     style={"color": FG_MUTED, "fontSize": "11px", "margin": "0 0 8px 0",
                            "whiteSpace": "pre-wrap", "fontFamily": "ui-monospace, monospace"}),
            dcc.Loading(dcc.Graph(id="fig", style={"height": "78vh"},
                                  config={"scrollZoom": True, "displaylogo": False}),
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
        full = os.path.expanduser(path.strip())
        n = len(files[full])
        return (file_options(files), full,
                f"✓ {os.path.basename(full)} — {n} event{'s' * (n != 1)}")

    @app.callback(Output("entry", "options"), Output("entry", "value"),
                  Output("tag", "options"), Output("tag", "value"),
                  Output("colour", "options"), Output("mode", "options"),
                  Input("file", "value"),
                  State("entry", "value"), State("tag", "value"),
                  State("last-event", "data"))
    def _on_file(path, entry, tag, last):
        f = files[path]
        # Entry numbers are an artefact of how a file was written. When the user
        # switches files mid-session, follow the *physics* event they were on --
        # otherwise "compare this event across two reco passes" silently lands
        # on a different event, which is what the rse= URL parameter exists to
        # prevent and what this path used to ignore.
        if last and last.get("path") != path:
            found = f.index_of(*last["id"])
            if found is not None:
                entry = found
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
        return opts, entry, tags, tag, colour_opts, mode_opts

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
            return no_update, no_update, no_update
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

        # Splitting radiologicals off needs a neutrino vertex to trace back to,
        # and only matters while truth is being drawn.
        try:
            separable = ev.neutrino_track_ids() is not None
        except Exception:
            separable = False
        if not separable:
            radio = [{"label": " radiologicals (no neutrino to separate from)",
                      "value": "on", "disabled": True}]
        elif not (drawn and truth_on):
            radio = [{"label": " radiologicals (needs truth overlay)",
                      "value": "on", "disabled": True}]
        else:
            radio = [{"label": " radiologicals", "value": "on"}]
        return (option("truth overlay", caps["truth"]),
                option("tracks + vertices", has_reco), radio)

    @app.callback(Output("entry", "value", allow_duplicate=True),
                  Input("jump", "value"), State("file", "value"),
                  prevent_initial_call=True)
    def _jump(value, path):
        from dash import no_update
        if value is None:
            return no_update
        return max(0, min(int(value), len(files[path]) - 1))

    @app.callback(Output("entry", "value", allow_duplicate=True),
                  Input("prev", "n_clicks"), Input("next", "n_clicks"),
                  State("entry", "value"), State("file", "value"),
                  prevent_initial_call=True)
    def _step(_p, _n, current, path):
        from dash import ctx
        n = len(files[path])
        cur = int(current or 0)
        if ctx.triggered_id == "prev":
            return (cur - 1) % n
        return (cur + 1) % n

    @app.callback(Output("tag", "disabled"), Output("colour", "disabled"),
                  Output("scale", "disabled"), Output("merge", "disabled"),
                  Input("mode", "value"))
    def _enable(mode):  # noqa: D401
        """Grey out every control the selected view ignores.

        A control that is live but inert is worse than one that is greyed out:
        it teaches the user that the panel lies.
        """
        other = mode in ("3d", "optical", "flash3d")
        readout = mode == "readout"
        return other, other, other, (other or readout)

    @app.callback(Output("fig", "figure"), Output("summary", "children"),
                  Output("summary", "style"), Output("last-event", "data"),
                  Input("file", "value"), Input("entry", "value"), Input("tag", "value"),
                  Input("colour", "value"), Input("scale", "value"),
                  Input("mode", "value"), Input("truth", "value"),
                  Input("merge", "value"), Input("reco", "value"),
                  Input("theme", "value"), Input("colormap", "value"),
                  Input("radio", "value"))
    def _render(path, entry, tag, colour, scale, mode, truth, merge, reco,
                theme, colormap, radio):
        fig, summary = render(files, path, entry, tag, colour, scale, mode, truth,
                              merge, reco, theme, colormap, bool(radio))
        base = {"fontSize": "11px", "margin": "0 0 8px 0", "whiteSpace": "pre-wrap",
                "fontFamily": "ui-monospace, monospace"}
        seen = None
        try:
            if path is not None and entry is not None:
                seen = {"path": path, "id": list(files[path][int(entry)].id)}
        except Exception:
            seen = None
        failed = summary.startswith("ERROR")
        style = {**base,
                 "color": "#fca5a5" if failed else FG_MUTED,
                 "backgroundColor": "#450a0a" if failed else "transparent",
                 "padding": "6px 8px" if failed else "0",
                 "borderRadius": "4px"}
        return fig, summary, style, seen

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
    ap.add_argument("files", nargs="+", help="art-ROOT file(s)")
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
