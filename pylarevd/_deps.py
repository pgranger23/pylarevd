"""What pylarevd needs, whether it is here, and how to get it.

Kept in one place so the ``--check`` report, the import-time error messages and
the docs cannot drift apart. A beta tester who hits a missing package should be
told which package, what it is for, and the exact command to fix it -- not a
bare ``ModuleNotFoundError``.
"""

from __future__ import annotations

import importlib
import importlib.metadata as metadata
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class Dep:
    """One dependency: its import name, what it buys you, and how to install it."""

    module: str            # what you `import`
    purpose: str
    pip: str               # what you `pip install`
    required: bool = False
    extra: str | None = None   # the pyproject extra that pulls it in
    note: str = ""

    @property
    def install_hint(self) -> str:
        if self.extra:
            return f'pip install "pylarevd[{self.extra}]"'
        return f"pip install {self.pip}"


DEPENDENCIES: tuple[Dep, ...] = (
    Dep("numpy", "everything", "numpy", required=True),
    Dep("uproot", "reading art-ROOT files", "uproot", required=True),
    Dep("matplotlib", "static images (PNG/PDF/SVG)", "matplotlib"),
    Dep("plotly", "interactive figures and self-contained HTML", "plotly"),
    Dep("dash", "the interactive browser (python -m pylarevd.app)",
        "dash", extra="app"),
    Dep("fsspec_xrootd", "reading root:// and /eos files",
        "fsspec-xrootd", extra="xrootd"),
    Dep("XRootD", "the XRootD client itself", "xrootd", extra="xrootd",
        note="ships with the CVMFS LCG view; pip-installing it needs the "
             "XRootD C++ libraries and a compiler"),
    Dep("pytest", "running the test suite", "pytest", extra="test"),
)


def _version(module: str) -> str:
    for name in (module, module.replace("_", "-")):
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    mod = sys.modules.get(module)
    return getattr(mod, "__version__", "?") if mod else "?"


def status() -> list[tuple[Dep, bool, str]]:
    """(dependency, present, version) for everything pylarevd can use."""
    out = []
    for dep in DEPENDENCIES:
        try:
            importlib.import_module(dep.module)
            out.append((dep, True, _version(dep.module)))
        except Exception:
            out.append((dep, False, ""))
    return out


def require(module: str, what: str) -> None:
    """Raise a message a user can act on when an optional package is absent.

    Called at the point of use, so importing pylarevd never fails because of
    something you were not going to use.
    """
    try:
        importlib.import_module(module)
        return
    except ImportError:
        pass
    dep = next((d for d in DEPENDENCIES if d.module == module), None)
    hint = dep.install_hint if dep else f"pip install {module}"
    extra = f"\n  {dep.note}" if dep and dep.note else ""
    raise SystemExit(
        f"pylarevd: {what} needs the '{module}' package, which is not installed.\n"
        f"  install it with:  {hint}\n"
        f"  or use the CVMFS LCG view, which already has it:\n"
        f"    source /cvmfs/sft.cern.ch/lcg/views/LCG_110/"
        f"x86_64-el9-gcc13-opt/setup.sh{extra}\n"
        f"  run 'python -m pylarevd --check' to see everything at once.")


def report() -> str:
    """Human-readable environment report, for --check and for bug reports."""
    import os

    rows = status()
    width = max(len(d.module) for d, _, _ in rows)
    lines = [f"pylarevd environment",
             f"  python {sys.version.split()[0]}  ({sys.executable})"]
    try:
        from . import __version__
        lines.append(f"  pylarevd {__version__}  ({os.path.dirname(__file__)})")
    except Exception:
        pass
    lines.append("")
    for dep, ok, version in rows:
        mark = "ok     " if ok else ("MISSING" if dep.required else "absent ")
        tag = "  (required)" if dep.required and not ok else ""
        lines.append(f"  [{mark}] {dep.module:<{width}}  {version:<10} "
                     f"{dep.purpose}{tag}")

    missing = [d for d, ok, _ in rows if not ok]
    lines.append("")
    if not missing:
        lines.append("  everything pylarevd can use is installed.")
    else:
        broken = [d for d in missing if d.required]
        if broken:
            lines.append("  REQUIRED packages are missing: "
                         + ", ".join(d.module for d in broken))
        extras = sorted({d.extra for d in missing if d.extra})
        plain = [d.pip for d in missing if not d.extra]
        lines.append("  to install what is absent:")
        if extras:
            lines.append(f'    pip install "pylarevd[{",".join(extras)}]"')
        if plain:
            lines.append(f"    pip install {' '.join(plain)}")
        lines.append("  or source the CVMFS LCG view, which has all of it:")
        lines.append("    source /cvmfs/sft.cern.ch/lcg/views/LCG_110/"
                     "x86_64-el9-gcc13-opt/setup.sh")
        for dep in missing:
            if dep.note:
                lines.append(f"    note: {dep.module} -- {dep.note}")

    # the geometries decide whether a file can be displayed at all
    try:
        import glob
        import numpy as np
        here = os.path.dirname(os.path.abspath(__file__))
        found = sorted(glob.glob(os.path.join(here, "geom", "*.npz")))
        lines.append("")
        if found:
            lines.append("  bundled geometries:")
            for path in found:
                with np.load(path, allow_pickle=False) as z:
                    lines.append(f"    {str(z['detector']):<22} "
                                 f"{int(z['nchannels']):>7} channels   "
                                 f"{os.path.basename(path)}")
        else:
            lines.append("  no geometries found in pylarevd/geom/ -- "
                         "nothing can be displayed until one is exported")
    except Exception as exc:
        lines.append(f"  could not list geometries: {exc}")
    return "\n".join(lines)
