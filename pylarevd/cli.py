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
            if not a.only_html:
                print("  ->", disp.save(_unique_path(f"{stem}.png", taken),
                                        dpi=a.dpi))
            if a.html or a.only_html:
                print("  ->", disp.save_html(_unique_path(f"{stem}.html", taken)))
        except (ArtReadError, GeometryError, IndexError, ValueError) as exc:
            print(f"  event {i}: {exc}", file=sys.stderr)
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
