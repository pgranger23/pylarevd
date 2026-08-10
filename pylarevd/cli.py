"""Command-line entry point: ``python -m pylarevd``."""

from __future__ import annotations

import argparse
import os
import sys

from .artio import ArtReadError
from .display import PRESETS
from .theme import COLORMAPS, DEFAULT_COLORMAP, THEMES
from .event import EventFile
from .geometry import GeometryError


def _parse_events(spec: str, n: int) -> list[int]:
    """Parse ``0``, ``0,3,7``, ``2-5`` or ``all`` into entry numbers.

    Rejects anything it cannot honour exactly -- a reversed range or a negative
    index would otherwise quietly draw the wrong events, or nothing at all,
    while still reporting success.
    """
    if spec.strip() == "all":
        return list(range(n))
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part[1:] if part.startswith("-") else "-" in part:
            lo, _, hi = part.partition("-")
            a, b = int(lo), int(hi)
            if a < 0 or b < 0:
                raise ValueError(f"negative event number in range {part!r}")
            if b < a:
                raise ValueError(f"reversed range {part!r} (use {b}-{a})")
            out.extend(range(a, b + 1))
        else:
            v = int(part)
            if v < 0:
                raise ValueError(
                    f"negative event number {v}; entries are counted from 0")
            out.append(v)
    if not out:
        raise ValueError(f"no events selected by {spec!r}")
    return out


_MANIFEST_FIELDS = ("entry", "run", "subrun", "event", "hits", "headline",
                    "png", "html", "status", "error")


def _write_manifest(outdir: str, rows: list[dict]) -> None:
    """A CSV and a linked index for a batch run.

    Without one, working out which of 120 output files was the interesting one
    means re-reading the terminal log and matching filenames by hand, and a
    failed event leaves no record at all.
    """
    import csv
    import html as _html

    with open(os.path.join(outdir, "manifest.csv"), "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_MANIFEST_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in _MANIFEST_FIELDS})

    def cell(row, key):
        value = row.get(key, "")
        if key in ("png", "html") and value:
            return f'<a href="{_html.escape(value)}">{key}</a>'
        return _html.escape(str(value))

    body = "\n".join(
        "<tr class='{}'>{}</tr>".format(
            row.get("status", "ok"),
            "".join(f"<td>{cell(row, k)}</td>" for k in _MANIFEST_FIELDS))
        for row in rows)
    with open(os.path.join(outdir, "index.html"), "w") as fh:
        fh.write(f"""<!doctype html><meta charset=utf-8>
<title>pylarevd batch</title>
<style>
 body{{font:13px system-ui;background:#0f172a;color:#e2e8f0;padding:16px}}
 table{{border-collapse:collapse}} th,td{{padding:4px 9px;text-align:left;
 border-bottom:1px solid #334155}} th{{cursor:pointer}}
 a{{color:#7dd3fc}} tr.failed{{color:#fca5a5}}
</style>
<h2>pylarevd — {len(rows)} events</h2>
<p>Click a column heading to sort.</p>
<table><thead><tr>{''.join(f'<th>{k}</th>' for k in _MANIFEST_FIELDS)}</tr></thead>
<tbody>{body}</tbody></table>
<script>
document.querySelectorAll('th').forEach((th,i)=>th.onclick=()=>{{
  const tb=th.closest('table').tBodies[0];
  const rows=[...tb.rows], dir=th.dataset.d=th.dataset.d==='1'?'':'1';
  rows.sort((a,b)=>{{const x=a.cells[i].innerText,y=b.cells[i].innerText;
    const n=parseFloat(x)-parseFloat(y);
    return (dir?1:-1)*(isNaN(n)?x.localeCompare(y):n);}});
  rows.forEach(r=>tb.appendChild(r));}});
</script>""")


def _unique_path(path: str, taken: set[str]) -> str:
    """Avoid clobbering an existing render.

    Different input files routinely carry the same run/subrun/event numbers, so
    the natural name collides across files. Rather than silently overwrite one
    event's picture with another's, disambiguate with a counter.
    """
    if path not in taken and not os.path.exists(path):
        taken.add(path)
        return path
    stem, ext = os.path.splitext(path)
    i = 2
    while True:
        candidate = f"{stem}_{i}{ext}"
        if candidate not in taken and not os.path.exists(candidate):
            taken.add(candidate)
            return candidate
        i += 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="pylarevd",
        description="Event display for LArSoft art-ROOT files, in pure python.")
    ap.add_argument("file", nargs="?", help="art-ROOT file")
    ap.add_argument("-e", "--events", default="0",
                    help="entries to draw: '0', '0,3,7', '2-5' or 'all' (default 0)")
    ap.add_argument("-g", "--geometry", default=None,
                    help="geometry .npz (default: the one in pylarevd/geom/)")
    ap.add_argument("-t", "--tag", default=None,
                    help="hit producer module label (default: prefer 'hitfd')")
    ap.add_argument("-o", "--outdir", default="evd_out", help="output directory")
    ap.add_argument("--html", action="store_true", help="also write interactive HTML")
    ap.add_argument("--html-cdn", action="store_true",
                    help="with --html, link plotly.js from a CDN instead of "
                         "embedding it: ~20x smaller (17.8 MB -> 1.1 MB per "
                         "event here), but viewing then needs internet")
    ap.add_argument("--only-html", action="store_true", help="write only HTML")
    ap.add_argument("--no-radiologicals", action="store_true",
                    help="with --truth, keep only truth descending from the "
                         "neutrino interaction (drops radiological decays)")
    ap.add_argument("--truth", action="store_true",
                    help="overlay true energy depositions if simulated")
    ap.add_argument("--colour-by", default="integral",
                    metavar="WHAT",
                    choices=["integral", "amplitude", "tick", "multiplicity",
                             "track", "shower", "slice", "cluster", "pfparticle"],
                    help="hit quantity, or the object each hit belongs to "
                         "(2-D and readout views only)")
    ap.add_argument("--mode", default="2d",
                    choices=["2d", "readout", "3d", "optical", "flash3d"],
                    help="which view to render (default: the 2-D physical view)")
    ap.add_argument("--reco", action="store_true",
                    help="overlay reconstructed tracks, vertices and showers")
    ap.add_argument("--merge", default="orientation",
                    choices=["orientation", "view", "none"],
                    help="panel grouping (default: by wire orientation)")
    ap.add_argument("--space", default="physical", choices=["physical", "readout"],
                    help="physical (w vs drift x) or classic readout "
                         "(channel vs tick); --mode readout overrides this")
    ap.add_argument("--colour-scale", default="auto",
                    choices=["auto", "log", "linear"],
                    help="colour axis scaling (auto: log for charge quantities)")
    ap.add_argument("--theme", default="dark", choices=sorted(THEMES),
                    help="dark for screen, light for print (default: dark)")
    ap.add_argument("--colormap", default=DEFAULT_COLORMAP,
                    choices=sorted(COLORMAPS),
                    help="charge colour ramp (default: turbo)")
    ap.add_argument("--preset", default="screen", choices=sorted(PRESETS),
                    help="figure size and font scale target: screen, paper1, "
                         "paper2, slide, poster (default: screen)")
    ap.add_argument("--dpi", type=int, default=None,
                    help="output resolution in dots per inch "
                         "(default: from --preset)")
    ap.add_argument("--check", action="store_true",
                    help="report which dependencies and geometries are "
                         "available, and how to install what is missing")
    ap.add_argument("--list-colormaps", action="store_true",
                    help="describe the available colour ramps and exit")
    ap.add_argument("--list", action="store_true",
                    help="list events and hit products, then exit")
    a = ap.parse_args(argv)

    if a.check:
        from ._deps import report
        print(report())
        return 0

    if a.list_colormaps:
        for name, cm in sorted(COLORMAPS.items()):
            flag = "  " if cm.monotonic else " *"
            print(f"{flag}{name:9s} {cm.note}")
        print("\n  * lightness is not monotonic: fine on screen, but charge "
              "ordering flattens in greyscale")
        return 0

    if a.file is None:
        ap.error("a file is required (or use --check / --list-colormaps)")

    try:
        f = EventFile(a.file, geometry=a.geometry)
    except (ArtReadError, GeometryError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f)
    if a.list:
        print("hit products:")
        for p in f.hit_products():
            print(f"  {p.rstrip('.')}")
        print(f"events: {len(f)}")
        for i, (run, sub, ev) in enumerate(f.event_ids):
            print(f"  [{i}] run {run} subrun {sub} event {ev}")
        return 0

    try:
        entries = _parse_events(a.events, len(f))
    except ValueError as exc:
        print(f"error: --events {a.events!r}: {exc}", file=sys.stderr)
        return 2

    os.makedirs(a.outdir, exist_ok=True)
    failures = 0
    taken: set[str] = set()
    rows: list[dict] = []
    for n_done, i in enumerate(entries, 1):
        if len(entries) > 1:
            print(f"[{n_done}/{len(entries)}] event {i}", flush=True)
        try:
            ev = f[i]
            style = dict(theme=a.theme, colormap=a.colormap, preset=a.preset)
            if a.mode == "3d":
                disp = ev.display_3d(truth=a.truth,
                                     radiologicals=not a.no_radiologicals,
                                     **style)
            elif a.mode == "flash3d":
                disp = ev.display_flashes_3d(**style)
            elif a.mode == "optical":
                disp = ev.display_optical(**style)
            else:
                disp = ev.display(a.tag, truth=a.truth, reco=a.reco,
                                  radiologicals=not a.no_radiologicals,
                                  colour_by=a.colour_by,
                                  colour_scale=a.colour_scale, merge=a.merge,
                                  space="readout" if a.mode == "readout" else a.space,
                                  **style)
            print(disp.summary())
            run, sub, evno = ev.id
            stem = os.path.join(a.outdir, f"r{run}_s{sub}_e{evno}")
            row = {"entry": i, "run": run, "subrun": sub, "event": evno,
                   "hits": len(getattr(disp, "hits", []) or []),
                   "headline": (disp.neutrino.headline() if disp.neutrino else ""),
                   "png": "", "html": "", "status": "ok", "error": ""}
            if not a.only_html:
                row["png"] = os.path.basename(
                    disp.save(_unique_path(f"{stem}.png", taken), dpi=a.dpi))
                print("  ->", os.path.join(a.outdir, row["png"]))
            if a.html or a.only_html:
                row["html"] = os.path.basename(
                    disp.save_html(_unique_path(f"{stem}.html", taken),
                                   bundle_plotlyjs=not a.html_cdn))
                print("  ->", os.path.join(a.outdir, row["html"]))
            rows.append(row)
        except (ArtReadError, GeometryError, IndexError, ValueError) as exc:
            print(f"  event {i}: {exc}", file=sys.stderr)
            # Failures belong in the manifest too: a partly-failed batch left
            # no record at all of which events failed or why.
            rows.append({"entry": i, "run": "", "subrun": "", "event": "",
                         "hits": "", "headline": "", "png": "", "html": "",
                         "status": "failed", "error": str(exc)})
            failures += 1

    if rows:
        _write_manifest(a.outdir, rows)
        print(f"  -> {os.path.join(a.outdir, 'manifest.csv')}")
        print(f"  -> {os.path.join(a.outdir, 'index.html')}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
