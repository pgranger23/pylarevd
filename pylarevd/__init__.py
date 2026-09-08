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

__version__ = "0.2.0"

_PYLAR_EXPORTS = {
    "ArtFile", "ArtReadError", "Event", "EventFile", "Hits",
    "MCParticles", "Neutrino", "OpticalActivity", "Showers",
    "SpacePoints", "Tracks", "TruthDeposits", "Vertices",
    "Geometry", "GeometryError", "physics",
}

_DISPLAY_EXPORTS = {
    "EventDisplay", "Display3D", "OpticalDisplay", "FlashDisplay3D",
    "display", "display_3d", "display_optical", "display_flashes_3d",
}

__all__ = sorted(list(_PYLAR_EXPORTS) + list(_DISPLAY_EXPORTS))


def __getattr__(name: str):
    from importlib import import_module

    if name in _DISPLAY_EXPORTS:
        disp = import_module(".display", __name__)
        return getattr(disp, name)
    if name in _PYLAR_EXPORTS:
        import pylar
        return getattr(pylar, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return __all__
