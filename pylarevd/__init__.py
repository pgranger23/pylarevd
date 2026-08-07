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
"""

from .artio import ArtFile, ArtReadError
from . import physics
from .event import (Event, EventFile, Hits, MCParticles, Neutrino,
                    OpticalActivity, Showers, SpacePoints, Tracks,
                    TruthDeposits, Vertices)
from .geometry import Geometry, GeometryError

__version__ = "0.1.0"
__all__ = [
    "ArtFile", "ArtReadError", "Event", "EventFile", "Geometry", "GeometryError",
    "Hits", "MCParticles", "Neutrino", "OpticalActivity", "Showers",
    "SpacePoints", "Tracks", "TruthDeposits", "Vertices", "physics",
]
