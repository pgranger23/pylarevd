"""pylarevd - a LArSoft event display in pure python.

Reads art-ROOT files directly with uproot (no ROOT, art or LArSoft at display
time) and draws reconstructed hits in physical coordinates.

    from pylarevd import EventFile

    f = EventFile("reco.root")
    ev = f[0]
    ev.display().save("event0.png")        # static
    ev.display().save_html("event0.html")  # interactive

Detector geometry comes from a ``.npz`` exported once per geometry by
:mod:`pylarevd.export_geometry`, which is the only component that needs
LArSoft.

Importing the package itself is deliberately cheap: names are resolved on first
use (PEP 562). ``python -m pylarevd --check`` has to run when numpy or uproot
are missing -- that is the situation it exists to diagnose -- and an eager
``from .artio import ...`` here made it fail with the bare ModuleNotFoundError
it was written to replace.
"""

__version__ = "0.1.0"

#: public name -> the submodule that defines it
_EXPORTS = {
    "ArtFile": "artio", "ArtReadError": "artio",
    "Event": "event", "EventFile": "event", "Hits": "event",
    "MCParticles": "event", "Neutrino": "event", "OpticalActivity": "event",
    "Showers": "event", "SpacePoints": "event", "Tracks": "event",
    "TruthDeposits": "event", "Vertices": "event",
    "Geometry": "geometry", "GeometryError": "geometry",
}

__all__ = sorted(list(_EXPORTS) + ["physics"])


def __getattr__(name: str):
    from importlib import import_module

    if name == "physics":
        return import_module(".physics", __name__)
    if name in _EXPORTS:
        return getattr(import_module("." + _EXPORTS[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return __all__
