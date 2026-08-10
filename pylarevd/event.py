"""User-facing event/hit API.

Ties an art file (:mod:`pylarevd.artio`) to a detector geometry
(:mod:`pylarevd.geometry`) and exposes hits with physical coordinates already
attached, so plotting code never has to think about wire wrapping or drift
conversion.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from functools import cached_property, lru_cache, wraps

import numpy as np

from .artio import ArtFile, ArtReadError
from . import physics
from .geometry import Geometry, GeometryError

def _channel_view(geometry, channels) -> np.ndarray:
    """View per hit taken from the channel number alone.

    Works even when the WireID is unset, which is the case for hit collections
    that have not been through disambiguation.
    """
    ch = np.asarray(channels, np.int64)
    ok = (ch >= 0) & (ch < len(geometry.channel_view))
    out = np.full(ch.shape, -1, np.int32)
    out[ok] = geometry.channel_view[ch[ok]]
    return out


_HIT_CLASS = "recob::Hit"
_SPACEPOINT_CLASS = "recob::SpacePoint"
_SIMDEP_CLASS = "sim::SimEnergyDeposit"
_TRACK_CLASS = "recob::Track"
_VERTEX_CLASS = "recob::Vertex"
_OPFLASH_CLASS = "recob::OpFlash"
_OPHIT_CLASS = "recob::OpHit"
_MCPART_CLASS = "simb::MCParticle"
_SHOWER_CLASS = "recob::Shower"
_MCTRUTH_CLASS = "simb::MCTruth"
_PFP_CLASS = "recob::PFParticle"
_PFPMETA_CLASS = "larpandoraobj::PFParticleMetadata"

#: Pandora writes both a track and a shower for every PFParticle; its own
#: ``TrackScore`` decides which one is meant. 0.5 is Pandora's boundary, and on
#: the samples here it reproduces the PDG code Pandora assigns (13 vs 11) exactly.
TRACK_SCORE_THRESHOLD = 0.5

_BOGUS = -999.0          # LArSoft's marker for a masked trajectory point


class _Selectable:
    """Boolean-mask / index selection, shared by every product collection.

    ``Hits`` grew this first and the README teaches the idiom, so every
    collection supports it rather than only some.  Fields that are per-object
    (arrays, lists) are subset; scalars like ``label`` carry over.
    """

    _RAGGED: tuple = ()          # list-valued fields, subset element-wise

    def __getitem__(self, mask):
        idx = np.asarray(mask)
        if idx.ndim == 0:
            raise IndexError(
                f"{type(self).__name__} takes a 1-D boolean mask or index "
                f"array; got a scalar ({mask!r}). Use x[[i]] to keep one.")
        if idx.dtype == bool:
            idx = np.flatnonzero(idx)
        values = {}
        for field in self.__dataclass_fields__:
            current = getattr(self, field)
            if field in self._RAGGED:
                values[field] = [current[i] for i in idx]
            elif isinstance(current, np.ndarray):
                values[field] = current[idx]
            else:
                values[field] = current
        return type(self)(**values)


@dataclass
class Hits(_Selectable):
    _RAGGED = ()
    """Reconstructed hits of one event, with physical coordinates.

    Attributes are parallel numpy arrays, one entry per hit:

    Readout quantities as reconstruction stored them:
    ``channel``, ``tick`` (peak time), ``start_tick``, ``end_tick``,
    ``integral`` (charge), ``amplitude``, ``rms``, ``multiplicity``,
    ``goodness``.

    WireID components: ``cryo``, ``tpc``, ``plane``, ``wire``, and ``valid``
    (False when reconstruction never resolved the wire -- such hits get NaN
    coordinates and are excluded from the physical views).

    Derived: ``w`` (continuous wire coordinate, cm), ``x`` (drift coordinate,
    cm), ``view``, ``drift_sign``, ``panel`` (view + drift side),
    ``orientation`` (equal iff the wires are parallel -- this, not the U/V
    label, is what says two sets of hits share a projection) and ``chan_view``
    (view from the channel alone, so it works without a WireID).

    ``product`` names the source collection; ``label`` adds the current count.
    """

    product: str
    channel: np.ndarray
    tick: np.ndarray
    start_tick: np.ndarray
    end_tick: np.ndarray
    integral: np.ndarray
    amplitude: np.ndarray
    rms: np.ndarray
    multiplicity: np.ndarray
    goodness: np.ndarray
    cryo: np.ndarray
    tpc: np.ndarray
    plane: np.ndarray
    wire: np.ndarray
    valid: np.ndarray
    w: np.ndarray
    x: np.ndarray
    view: np.ndarray
    drift_sign: np.ndarray
    panel: np.ndarray
    orientation: np.ndarray   # equal iff the wires are parallel
    chan_view: np.ndarray     # view from the channel alone, no WireID needed

    def __len__(self) -> int:
        return len(self.channel)

    @property
    def label(self) -> str:
        """Product name and the CURRENT hit count.

        Derived rather than stored, so it stays truthful after a selection.
        """
        return f"{self.product} [{len(self)} hits]"

    def __repr__(self) -> str:
        return f"<Hits {self.label!r} n={len(self)}>"

    @property
    def n_bad_geometry(self) -> int:
        """Hits whose WireID could not be located in the geometry."""
        return int(np.count_nonzero(~np.isfinite(self.w)))

    @property
    def channels_split_across_wires(self) -> int:
        """Channels whose hits were assigned to more than one physical wire.

        In a wrapped-wire detector this is the fingerprint of disambiguation:
        one readout channel is bonded to segments on both drift faces, and
        resolving which segment each hit belongs to is exactly what splits a
        channel across wires. A collection that never does it has not been
        disambiguated, and its hits land on an arbitrary one of the candidate
        wires -- metres from the truth, with nothing else to show for it.
        """
        if not len(self):
            return 0
        key = ((self.tpc.astype(np.int64) << 40)
               | (self.plane.astype(np.int64) << 20) | self.wire)
        pairs = np.unique(np.stack([self.channel.astype(np.int64), key]), axis=1)
        _, counts = np.unique(pairs[0], return_counts=True)
        return int((counts > 1).sum())


@dataclass
class SpacePoints(_Selectable):
    """Reconstructed 3-D points.

    ``chisq`` is left unfilled (0) by the Pandora-based builders in practice;
    ``charge`` is summed from the hits associated with each point, and is NaN
    when no Hit<->SpacePoint association is available.
    """

    _RAGGED = ()
    label: str
    xyz: np.ndarray        # (N, 3)
    chisq: np.ndarray
    id: np.ndarray
    charge: np.ndarray

    def __len__(self) -> int:
        return len(self.xyz)

    def __repr__(self) -> str:
        return f"<SpacePoints {self.label!r} n={len(self)}>"


@dataclass
class TruthDeposits(_Selectable):
    _RAGGED = ()
    """True energy depositions from the simulation."""

    label: str
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    edep: np.ndarray
    track_id: np.ndarray
    pdg: np.ndarray

    def __len__(self) -> int:
        return len(self.x)

    def __repr__(self) -> str:
        return f"<TruthDeposits {self.label!r} n={len(self)}>"


@dataclass
class Tracks(_Selectable):
    _RAGGED = ("points",)
    """Reconstructed tracks: one polyline of 3-D points each."""

    label: str
    points: list           # list of (N, 3) float arrays, masked points removed
    id: np.ndarray
    chi2: np.ndarray
    ndof: np.ndarray

    def __len__(self) -> int:
        return len(self.points)

    def __repr__(self) -> str:
        n = sum(len(p) for p in self.points)
        return f"<Tracks {self.label!r} n={len(self)} points={n}>"


@dataclass
class MCParticles(_Selectable):
    _RAGGED = ("points", "process")
    """True particle trajectories from the simulation."""

    label: str
    points: list           # list of (N, 3) float arrays
    pdg: np.ndarray
    track_id: np.ndarray
    mother: np.ndarray
    process: list

    def __len__(self) -> int:
        return len(self.points)

    def __repr__(self) -> str:
        return f"<MCParticles {self.label!r} n={len(self)}>"

    def primaries(self) -> "MCParticles":
        """Particles with no simulated parent."""
        return self[self.mother == 0]



@dataclass
class Neutrino:
    """One true neutrino interaction, as the generator recorded it.

    Energies are GeV and positions cm, matching what LArSoft stores. The
    ``fs_*`` arrays are the *final state*: the particles actually handed to the
    detector simulation (GENIE status 1), which is what the hits in the event
    were made by. Intermediate and nuclear-remnant entries are kept in
    ``all_*`` for completeness but are not what you want to read off a display.
    """

    label: str
    pdg: int                  # incoming neutrino
    energy: float             # GeV
    vertex: np.ndarray        # (3,) cm
    direction: np.ndarray     # (3,) unit
    ccnc: int
    mode: int
    interaction_type: int
    target: int
    hit_nucleon: int
    hit_quark: int
    w: float
    x: float
    y: float
    q2: float
    origin: int
    lepton_pdg: int
    lepton_energy: float
    fs_pdg: np.ndarray
    fs_energy: np.ndarray
    fs_momentum: np.ndarray   # (N, 3) GeV
    all_pdg: np.ndarray
    all_status: np.ndarray
    all_energy: np.ndarray

    @property
    def is_cc(self) -> bool:
        return int(self.ccnc) == physics.CC

    @property
    def nu_name(self) -> str:
        return physics.particle_name(self.pdg)

    @property
    def target_name(self) -> str:
        return physics.particle_name(self.target)

    @property
    def mode_name(self) -> str:
        return physics.mode_name(self.mode)

    def headline(self) -> str:
        """One line: what interacted, how, and at what energy."""
        return (f"{self.nu_name} {physics.current_name(self.ccnc)} "
                f"{self.mode_name}   E = {self.energy:.2f} GeV")

    def final_state_counts(self) -> list[tuple[str, int, float]]:
        """(name, count, summed energy) per species, most energetic first."""
        totals: dict[str, list] = {}
        for pdg, e in zip(self.fs_pdg, self.fs_energy):
            row = totals.setdefault(physics.particle_name(int(pdg)), [0, 0.0])
            row[0] += 1
            row[1] += float(e)
        return sorted(((k, v[0], v[1]) for k, v in totals.items()),
                      key=lambda r: -r[2])

    def describe(self) -> str:
        """Multi-line human summary, as shown on the display."""
        lines = [self.headline()]
        lines.append(f"  target {self.target_name}"
                     f"   hit nucleon {physics.particle_name(self.hit_nucleon)}"
                     f"   {physics.origin_name(self.origin)}")
        lines.append(f"  W = {self.w:.2f} GeV   x = {self.x:.3f}   "
                     f"y = {self.y:.3f}   Q2 = {self.q2:.2f} GeV2")
        lines.append(f"  vertex ({self.vertex[0]:.1f}, {self.vertex[1]:.1f}, "
                     f"{self.vertex[2]:.1f}) cm")
        if self.lepton_pdg and np.isfinite(self.lepton_energy):
            # A tau decays (GENIE status 3) and so never appears in the final
            # state, which for a tau-appearance sample omits the one particle
            # the event is about.
            lines.append(f"  outgoing lepton: "
                         f"{physics.particle_name(self.lepton_pdg)} "
                         f"({self.lepton_energy:.2f} GeV)")
        fs = self.final_state_counts()
        if fs:
            shown = ", ".join(f"{n}x {name} ({e:.2f} GeV)" if n > 1
                              else f"{name} ({e:.2f} GeV)"
                              for name, n, e in fs[:8])
            more = "" if len(fs) <= 8 else f", +{len(fs) - 8} more"
            lines.append(f"  final state: {shown}{more}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"<Neutrino {self.headline()}>"


@dataclass
class Showers(_Selectable):
    _RAGGED = ()
    """Reconstructed showers. Failed fits (all-`-999` placeholders) are dropped,
    so ``direction`` really is a unit vector."""

    label: str
    start: np.ndarray      # (N, 3)
    direction: np.ndarray  # (N, 3) unit vectors
    length: np.ndarray
    open_angle: np.ndarray
    id: np.ndarray

    def __len__(self) -> int:
        return len(self.start)

    def __repr__(self) -> str:
        return f"<Showers {self.label!r} n={len(self)}>"


@dataclass
class Vertices(_Selectable):
    _RAGGED = ()
    label: str
    xyz: np.ndarray        # (N, 3)
    id: np.ndarray

    def __len__(self) -> int:
        return len(self.xyz)

    def __repr__(self) -> str:
        return f"<Vertices {self.label!r} n={len(self)}>"


@dataclass
class OpticalActivity(_Selectable):
    _RAGGED = ()
    """Photon-detector activity: flashes and/or hits, on a common time axis."""

    label: str
    time: np.ndarray          # us
    pe: np.ndarray            # photoelectrons
    y: np.ndarray             # cm - reconstructed for flashes, the seeing
    z: np.ndarray             # cm   detector's own position for hits
    channel: np.ndarray       # -1 for flashes
    is_flash: np.ndarray      # bool
    x: np.ndarray             # cm - NaN when the flash finder left it unset
    y_width: np.ndarray       # cm - flash extent (1 sigma), NaN for hits
    z_width: np.ndarray

    def __len__(self) -> int:
        return len(self.time)

    @property
    def has_x(self) -> bool:
        """Whether any flash carries a reconstructed drift position.

        Most flash finders localise (y, z) only: light alone cannot fix the
        drift coordinate without a matched TPC object, and LArSoft stores the
        unset value as DBL_MAX. A display must say so rather than draw every
        flash at x = 0 as if it were measured.
        """
        return bool(np.isfinite(self.x[self.is_flash]).any())

    def __repr__(self) -> str:
        nf = int(self.is_flash.sum())
        return f"<OpticalActivity {self.label!r} flashes={nf} hits={len(self) - nf}>"


def _memoised(method):
    """Cache an accessor's result per (name, args) on the Event.

    Decoding is expensive -- a radiological sample's MCParticles take tens of
    seconds -- and the interactive browser re-renders on every control change,
    including ones that do not affect the data.
    """
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        key = (method.__name__, args, tuple(sorted(kwargs.items())))
        if key not in self._memo:
            self._memo[key] = method(self, *args, **kwargs)
        return self._memo[key]
    return wrapper


class Event:
    """One event of one file."""

    def __init__(self, source: "EventFile", entry: int):
        self._src = source
        self.entry = entry
        self._memo: dict = {}

    @property
    def geometry(self) -> Geometry:
        return self._src.geometry

    @cached_property
    def id(self) -> tuple[int, int, int]:
        ids = self._src.event_ids
        r = ids[self.entry]
        return int(r["run"]), int(r["subrun"]), int(r["event"])

    def __repr__(self) -> str:
        r, s, e = self.id
        return f"<Event entry={self.entry} run={r} subrun={s} event={e}>"

    # ---- products ------------------------------------------------------

    @_memoised
    def hits(self, tag: str | None = None) -> Hits:
        """Load hits, defaulting to the most disambiguated collection available."""
        product = self._src.resolve_hit_product(tag)
        raw = self._src.art.read(product, _HIT_CLASS, self.entry)
        g = self.geometry

        cryo = raw["fWireID.Cryostat"].astype(np.int32)
        tpc = raw["fWireID.TPC"].astype(np.int32)
        plane = raw["fWireID.Plane"].astype(np.int32)
        wire = raw["fWireID.Wire"].astype(np.int32)
        valid = raw["fWireID.isValid"]

        rows = g.wire_rows(cryo, tpc, plane, wire)
        w = np.where(rows >= 0, g.wire_coord[np.maximum(rows, 0)], np.nan)

        prows = g.plane_rows(cryo, tpc, plane)
        tick = raw["fPeakTime"].astype(np.float64)
        x = g.ticks_to_x(tick, prows) if g.has_drift else np.full(tick.shape, np.nan)
        view = np.where(prows >= 0, g.p_view[np.maximum(prows, 0)], -1)
        dsign = np.where(prows >= 0, g.drift_sign[np.maximum(prows, 0)], 0)

        return Hits(
            product=product.rstrip("."),
            channel=raw["fChannel"].astype(np.int32),
            tick=tick,
            start_tick=raw["fStartTick"].astype(np.int32),
            end_tick=raw["fEndTick"].astype(np.int32),
            integral=raw["fIntegral"].astype(np.float64),
            amplitude=raw["fPeakAmplitude"].astype(np.float64),
            rms=raw["fRMS"].astype(np.float64),
            multiplicity=raw["fMultiplicity"].astype(np.int32),
            goodness=raw["fGoodnessOfFit"].astype(np.float64),
            cryo=cryo, tpc=tpc, plane=plane, wire=wire, valid=valid,
            w=w, x=x, view=view.astype(np.int32), drift_sign=dsign.astype(np.int32),
            panel=g.panel_of(view, dsign),
            orientation=np.where(prows >= 0, g.orientation_key[np.maximum(prows, 0)],
                                 999).astype(np.int64),
            chan_view=_channel_view(g, raw["fChannel"]),
        )

    @_memoised
    def spacepoints(self, tag: str | None = None) -> SpacePoints:
        products = self._src.art.find_product(_SPACEPOINT_CLASS, tag)
        if not products:
            raise ArtReadError(
                "no recob::SpacePoint product"
                + (f" with tag {tag!r}" if tag else "")
                + f" in {os.path.basename(self._src.path)}")
        product = products[0]
        raw = self._src.art.read(product, _SPACEPOINT_CLASS, self.entry)
        xyz = np.column_stack([raw["fXYZ[0]"], raw["fXYZ[1]"], raw["fXYZ[2]"]])
        charge = np.full(len(xyz), np.nan)
        try:
            assns = self._assns_for("recob::Hit", "recob::SpacePoint",
                                    ("pandora", "spsolve"))
            hits = self.hits()
            good = (assns.right_key >= 0) & (assns.right_key < len(xyz)) & \
                   (assns.left_key >= 0) & (assns.left_key < len(hits))
            charge = np.zeros(len(xyz))
            np.add.at(charge, assns.right_key[good], hits.integral[assns.left_key[good]])
        except (ArtReadError, KeyError, ValueError):
            pass
        return SpacePoints(label=product.rstrip("."), xyz=xyz.astype(np.float64),
                           chisq=raw["fChisq"].astype(np.float64),
                           id=raw["fID"].astype(np.int32), charge=charge)

    @_memoised
    @_memoised
    def pfp_track_scores(self) -> "np.ndarray | None":
        """Pandora's ``TrackScore`` per PFParticle, NaN where it has none.

        Indexed by PFParticle key, so it lines up with the PFParticle
        collection. Returns ``None`` when the sample has no Pandora metadata
        (a non-Pandora reconstruction, or one written without it).
        """
        products = self._src.art.find_product(_PFPMETA_CLASS)
        if not products:
            return None
        try:
            raw = self._src.art.read(products[0], _PFPMETA_CLASS, self.entry)
            link = self._assns_for(_PFPMETA_CLASS, _PFP_CLASS, ("pandora",))
        except Exception:
            return None
        maps = raw.get("m_propertiesMap") or []
        # The association gives metadata index -> PFParticle index; do not
        # assume they are written in the same order.
        n_pfp = int(link.right_key.max()) + 1 if len(link.right_key) else 0
        out = np.full(max(n_pfp, len(maps)), np.nan)
        for meta_i, pfp_i in zip(link.left_key, link.right_key):
            if meta_i >= len(maps):
                continue
            props = dict(maps[meta_i])
            if "TrackScore" in props:
                out[pfp_i] = float(props["TrackScore"])
        return out

    def _pfp_of(self, other: str, prefer: tuple[str, ...]) -> "np.ndarray | None":
        """For each object of *other*, the PFParticle it belongs to (-1 if none)."""
        try:
            link = self._assns_for(_PFP_CLASS, other, prefer)
        except Exception:
            return None
        if not len(link.right_key):
            return None
        out = np.full(int(link.right_key.max()) + 1, -1, np.int64)
        out[link.right_key] = link.left_key
        return out

    @_memoised
    def pandora_best_match(self, threshold: float = TRACK_SCORE_THRESHOLD) -> dict:
        """Which tracks and showers are Pandora's *intended* interpretation.

        Pandora runs both a track fit and a shower fit on every PFParticle and
        writes both, so drawing all of each paints every particle twice -- once
        as a line and once as a cone. ``TrackScore`` says which one was meant.

        Returns ``{"tracks": mask, "showers": mask}`` (boolean, indexed by
        object) or empty dict when there is no metadata to decide with, in
        which case callers must draw everything rather than guess.
        """
        scores = self.pfp_track_scores()
        if scores is None:
            return {}
        out = {}
        for key, cls, prefer in (("tracks", _TRACK_CLASS, ("pandoraTrack",)),
                                 ("showers", _SHOWER_CLASS, ("pandoraShower",))):
            owner = self._pfp_of(cls, prefer)
            if owner is None:
                continue
            score = np.full(len(owner), np.nan)
            ok = owner >= 0
            valid = ok & (owner < len(scores))
            score[valid] = scores[owner[valid]]
            is_track = score >= threshold
            # A PFParticle with no score (the neutrino itself) and any object
            # whose owner we could not resolve are kept: better a duplicate
            # than a silently dropped particle.
            unknown = ~np.isfinite(score)
            out[key] = (is_track | unknown) if key == "tracks" else (~is_track | unknown)
        return out

    def pandora_selection_note(self) -> "str | None":
        """One line describing what the best-match filter removed, or None."""
        sel = self.pandora_best_match()
        if not sel:
            return None
        dropped = sum(int((~m).sum()) for m in sel.values())
        if not dropped:
            return None
        kept = ", ".join(f"{int(m.sum())} {k}" for k, m in sorted(sel.items()))
        return (f"Pandora best match: kept {kept}; dropped {dropped} duplicate "
                f"track/shower fits of the same PFParticles")

    @_memoised
    def tracks(self, tag: str | None = None, *, best_match: bool = True) -> Tracks:
        """Reconstructed tracks as 3-D polylines.

        Trajectory points that reconstruction masked out are stored as a -999
        sentinel; they are dropped here so a track draws as a clean line.

        ``best_match`` (default) keeps only the tracks Pandora actually meant:
        it fits every PFParticle both ways and writes both, so taking all of
        them draws each shower-like particle a second time as a line. See
        :meth:`pandora_best_match`. Pass ``False`` for the raw collection.

        Note that this reindexes: with ``best_match`` the positional index no
        longer matches :meth:`hit_group`, which refers to the full collection.
        ``Tracks.id`` is unchanged and is the stable handle.
        """
        product = self._pick(_TRACK_CLASS, tag, prefer=("pandoraTrack", "pmtrack"))
        raw = self._src.art.read(product, _TRACK_CLASS, self.entry)
        polylines = []
        for traj in raw["fTraj"]:
            pts = traj.get("fPositions") or []
            if not pts:
                polylines.append(np.zeros((0, 3)))
                continue
            xyz = np.array([[p["fCoordinates.fX"], p["fCoordinates.fY"],
                             p["fCoordinates.fZ"]] for p in pts], dtype=np.float64)
            polylines.append(xyz[~np.any(np.isclose(xyz, _BOGUS), axis=1)])
        n = len(polylines)
        # Pandora leaves the fit quality unset (-999); report that as NaN
        # rather than a magic number a caller might compare against.
        def optional(key):
            v = np.asarray(raw.get(key, np.full(n, _BOGUS)), float)
            return np.where(np.isclose(v, _BOGUS), np.nan, v)
        ids = np.asarray(raw.get("fID", np.arange(n)))
        chi2, ndof = optional("fChi2"), optional("fNdof")

        keep = self.pandora_best_match().get("tracks") if best_match else None
        if keep is not None and len(keep) == n:
            polylines = [p for p, k in zip(polylines, keep) if k]
            ids, chi2, ndof = ids[keep], chi2[keep], ndof[keep]
        return Tracks(label=product.rstrip("."), points=polylines,
                      id=ids, chi2=chi2, ndof=ndof)

    @_memoised
    def neutrino(self, tag: str | None = None) -> "Neutrino | None":
        """The true neutrino interaction, or ``None`` if the event has none.

        Radiological and cosmic samples carry ``simb::MCTruth`` too, but with
        no neutrino set; those return ``None`` rather than a record of zeros.
        When several generators contributed (a beam interaction overlaid on
        radiologicals), the one with a neutrino wins.
        """
        products = self._src.art.find_product(_MCTRUTH_CLASS)
        if not products:
            return None
        if tag is not None:
            products = [p for p in products if f"_{tag}_" in p] or products
        else:
            # "generator" is the conventional label for the physics generator;
            # the rest of a DUNE sample's MCTruths are radiological chains.
            preferred = [p for p in products if "_generator_" in p]
            products = preferred + [p for p in products if p not in preferred]

        for product in products:
            try:
                raw = self._src.art.read(product, _MCTRUTH_CLASS, self.entry)
            except Exception:
                # A sample can carry dozens of MCTruth products (radiological
                # chains alongside the physics generator); one that will not
                # decode must not hide the one that would.
                continue
            got = self._neutrino_from(raw, product)
            if got is not None:
                return got
        return None

    @staticmethod
    def _neutrino_from(raw: dict, product: str) -> "Neutrino | None":
        """Build a :class:`Neutrino` from one decoded ``simb::MCTruth`` vector."""
        nus = raw.get("fMCNeutrino") or []
        parts = raw.get("fPartList") or []
        flags = raw.get("fNeutrinoSet")
        origins = raw.get("fOrigin")
        for i, nu in enumerate(nus):
            if flags is not None and i < len(flags) and not bool(flags[i]):
                continue                      # radiological: no interaction
            plist = parts[i] if i < len(parts) else []
            pdg = np.array([int(p.get("fpdgCode", 0)) for p in plist], np.int32)
            status = np.array([int(p.get("fstatus", -1)) for p in plist], np.int32)

            def four(p):
                """(position, momentum) of a particle's first trajectory point."""
                traj = p.get("ftrajectory.ftrajectory") or []
                if not traj:
                    return None, None
                pos, mom = traj[0][0], traj[0][1]
                return (np.array([pos["fP.fX"], pos["fP.fY"], pos["fP.fZ"]]),
                        np.array([mom["fP.fX"], mom["fP.fY"], mom["fP.fZ"],
                                  mom["fE"]]))

            energy = np.array([(four(p)[1][3] if four(p)[1] is not None else 0.0)
                               for p in plist])
            # The incoming neutrino is the initial-state lepton; its trajectory
            # point carries both the vertex and the beam energy.
            nu_i = next((j for j in range(len(pdg))
                         if status[j] == 0 and abs(int(pdg[j])) in (12, 14, 16)),
                        None)
            if nu_i is None:
                continue
            vertex, p4 = four(plist[nu_i])
            if p4 is None:
                continue
            mom3 = p4[:3]
            norm = float(np.linalg.norm(mom3)) or 1.0

            fs = status == physics.FINAL_STATE
            fs_p4 = np.array([four(p)[1] if four(p)[1] is not None
                              else np.zeros(4) for p in plist])[fs] \
                if fs.any() else np.zeros((0, 4))
            # Outgoing lepton: identified by the species MCNeutrino names, not
            # by status -- a muon leaves as final state (1) while a tau decays
            # (3), and keying on either alone gets the other wrong.
            lep = int(nu.get("fLepton.fpdgCode", 0))
            mother = np.array([int(p.get("fmother", -1)) for p in plist], np.int32)
            cand = np.flatnonzero((pdg == lep) & (mother == nu_i)
                                  & np.isin(status, (1, 3))) if lep else []
            lep_e = float(energy[cand[0]]) if len(cand) else float("nan")

            return Neutrino(
                label=product.rstrip("."),
                pdg=int(pdg[nu_i]), energy=float(p4[3]),
                vertex=vertex[:3], direction=mom3 / norm,
                ccnc=int(nu.get("fCCNC", -1)), mode=int(nu.get("fMode", -1)),
                interaction_type=int(nu.get("fInteractionType", 0)),
                target=int(nu.get("fTarget", 0)),
                hit_nucleon=int(nu.get("fHitNuc", 0)),
                hit_quark=int(nu.get("fHitQuark", 0)),
                w=float(nu.get("fW", float("nan"))),
                x=float(nu.get("fX", float("nan"))),
                y=float(nu.get("fY", float("nan"))),
                q2=float(nu.get("fQSqr", float("nan"))),
                origin=int(origins[i]) if origins is not None and i < len(origins) else 0,
                lepton_pdg=lep, lepton_energy=lep_e,
                fs_pdg=pdg[fs], fs_energy=energy[fs],
                fs_momentum=fs_p4[:, :3],
                all_pdg=pdg, all_status=status, all_energy=energy)
        return None

    @_memoised
    def neutrino_track_ids(self, tol: float = 2.0) -> "np.ndarray | None":
        """Track ids of every particle descending from the neutrino vertex.

        Radiological generators produce tens of thousands of *primaries*
        scattered through the detector, while everything from the interaction
        traces back to one point. Walking each particle to its root ancestor and
        asking where that ancestor started separates the two cleanly: on the
        sample here the root-start distance is 0 for the interaction and a
        median 773 cm for the background.

        Returns ``None`` when the event has no neutrino, since then there is no
        signal/background split to make.
        """
        nu = self.neutrino()
        if nu is None or not np.isfinite(nu.vertex).all():
            return None
        # min_points=1: a particle that never moved still parents ones that do,
        # and deposits reference it by track id.
        mc = self.mc_particles(min_points=1)   # memoised; display() reuses it
        if not len(mc):
            return None
        index = {int(t): i for i, t in enumerate(mc.track_id)}
        start = np.array([p[0] if len(p) else (np.nan, np.nan, np.nan)
                          for p in mc.points], float)
        roots = np.arange(len(mc))
        for i in range(len(mc)):
            j, steps = i, 0
            while steps < 64:            # guard against a malformed cycle
                mother = int(mc.mother[j])
                if mother == 0 or mother not in index:
                    break
                j = index[mother]
                steps += 1
            roots[i] = j
        d = np.linalg.norm(start[roots] - nu.vertex, axis=1)
        return np.asarray(mc.track_id)[d < tol]

    @_memoised
    def mc_particles(self, tag: str | None = None, *,
                     min_points: int = 2) -> MCParticles:
        """True trajectories, one polyline per simulated particle.

        Radiological samples contain tens of thousands of particles, most of
        them single-step; ``min_points`` drops those so the overlay stays
        readable.
        """
        product = self._pick(_MCPART_CLASS, tag, prefer=("largeant",))
        raw = self._src.art.read(product, _MCPART_CLASS, self.entry)
        pts, pdg, tid, mom, proc = [], [], [], [], []
        for i, traj in enumerate(raw["ftrajectory"]):
            steps = traj.get("ftrajectory") or []
            if len(steps) < min_points:
                continue
            xyz = np.array([[s[0]["fP.fX"], s[0]["fP.fY"], s[0]["fP.fZ"]]
                            for s in steps], dtype=np.float64)
            pts.append(xyz)
            pdg.append(raw["fpdgCode"][i])
            tid.append(raw["ftrackId"][i])
            mom.append(raw["fmother"][i])
            proc.append(raw["fprocess"][i])
        return MCParticles(label=product.rstrip("."), points=pts,
                           pdg=np.asarray(pdg, np.int32),
                           track_id=np.asarray(tid, np.int32),
                           mother=np.asarray(mom, np.int32), process=proc)

    @_memoised
    def showers(self, tag: str | None = None, *, best_match: bool = True) -> Showers:
        """Reconstructed showers, as a start point plus an axis and opening angle."""
        product = self._pick(_SHOWER_CLASS, tag, prefer=("pandoraShower",))
        raw = self._src.art.read(product, _SHOWER_CLASS, self.entry)
        def get(stem):
            """A TVector3 member, however the decoder presented it.

            Fixed-size members come back as flattened columns
            (``fXYZstart.fX``); if the class ever falls to the sequential
            reader they arrive as a list of ``{fX, fY, fZ}`` dicts instead.
            """
            flat = [f"{stem}.f{axis}" for axis in "XYZ"]
            if all(k in raw for k in flat):
                return np.column_stack([raw[k] for k in flat]).astype(float)
            v = raw.get(stem) or []
            return np.array([[d["fX"], d["fY"], d["fZ"]] for d in v], float) \
                if len(v) else np.zeros((0, 3))
        start, direction = get("fXYZstart"), get("fDCosStart")
        length = np.asarray(raw["fLength"], float)
        # Pandora emits a default-constructed placeholder when a shower fit
        # fails, to keep indices aligned with its PFParticles. Its fields are
        # all -999, which would otherwise appear as a shower with a
        # direction vector of norm 1730 pointing out of the detector.
        good = ~(np.any(np.isclose(start, _BOGUS), axis=1)
                 | np.any(np.isclose(direction, _BOGUS), axis=1)
                 | np.isclose(length, _BOGUS))
        # Combined with the placeholder filter, not applied after it: both
        # masks are indexed against the raw collection, and this one would be
        # misaligned if it ran on an already-compacted array.
        keep = self.pandora_best_match().get("showers") if best_match else None
        if keep is not None and len(keep) == len(good):
            good = good & keep
        return Showers(label=product.rstrip("."),
                       start=start[good], direction=direction[good],
                       length=length[good],
                       open_angle=np.asarray(raw["fOpenAngle"], float)[good],
                       id=np.asarray(raw.get("fID", np.arange(len(good))))[good])

    @_memoised
    def vertices(self, tag: str | None = None) -> Vertices:
        """Reconstructed vertices."""
        product = self._pick(_VERTEX_CLASS, tag, prefer=("pandora",))
        raw = self._src.art.read(product, _VERTEX_CLASS, self.entry)
        xyz = np.column_stack([raw["pos_.fCoordinates.fX"], raw["pos_.fCoordinates.fY"],
                               raw["pos_.fCoordinates.fZ"]]).astype(np.float64)
        return Vertices(label=product.rstrip("."), xyz=xyz,
                        id=np.asarray(raw.get("id_", np.arange(len(xyz)))))

    @_memoised
    def optical(self, tag: str | None = None, *, hits: bool = True) -> OpticalActivity:
        """Photon-detector flashes, and optionally the individual hits.

        Flashes carry a reconstructed (y, z); hits do not, so theirs are NaN.
        """
        product = self._pick(_OPFLASH_CLASS, tag, prefer=("opflash",))
        raw = self._src.art.read(product, _OPFLASH_CLASS, self.entry)
        t = np.asarray(raw["fTime"], np.float64)
        pe = np.array([float(np.sum(v)) for v in raw["fPEperOpDet"]]) \
            if "fPEperOpDet" in raw else np.zeros(len(t))
        def sane(key):
            """A member the flash finder may have left at the DBL_MAX sentinel."""
            v = np.asarray(raw.get(key, np.full(len(t), np.nan)), np.float64)
            return np.where(np.abs(v) > 1e30, np.nan, v)

        out = dict(time=t, pe=pe, y=np.asarray(raw["fYCenter"], np.float64),
                   z=np.asarray(raw["fZCenter"], np.float64),
                   channel=np.full(len(t), -1, np.int32),
                   is_flash=np.ones(len(t), bool),
                   x=sane("fXCenter"),
                   y_width=sane("fYWidth"), z_width=sane("fZWidth"))
        label = product.rstrip(".")

        if hits:
            try:
                hp = self._pick(_OPHIT_CLASS, None, prefer=("ophitspe", "ophit"))
                hr = self._src.art.read(hp, _OPHIT_CLASS, self.entry)
                ht = np.asarray(hr["fPeakTime"], np.float64)
                hchan = np.asarray(hr["fOpChannel"], np.int32)
                # A hit carries no reconstructed position, but the detector that
                # saw it does -- so give the hit its detector's (y, z).
                hpos = self.geometry.opdet_positions(hchan)
                out = dict(
                    time=np.concatenate([t, ht]),
                    pe=np.concatenate([pe, np.asarray(hr["fPE"], np.float64)]),
                    y=np.concatenate([out["y"], hpos[:, 1]]),
                    z=np.concatenate([out["z"], hpos[:, 2]]),
                    channel=np.concatenate([out["channel"], hchan]),
                    is_flash=np.concatenate([np.ones(len(t), bool),
                                             np.zeros(len(ht), bool)]),
                    x=np.concatenate([out["x"], hpos[:, 0]]),
                    y_width=np.concatenate([out["y_width"],
                                            np.full(len(ht), np.nan)]),
                    z_width=np.concatenate([out["z_width"],
                                            np.full(len(ht), np.nan)]))
                label = f"{label} + {hp.rstrip('.')}"
            except (ArtReadError, KeyError):
                pass
        return OpticalActivity(label=label, **out)

    #: What a hit can be grouped by, and how to get there from the hit.
    #: Some links are direct; PFParticle goes via clusters, which is how
    #: Pandora actually records it.
    #: Each hop is (left class, right class, direction), where "fwd" maps a
    #: left index to a right one and "rev" the other way. The first hop always
    #: starts from a hit.
    #: Preferred producer per hop. Several modules write the same association
    #: -- pandoraShower also emits Hit<->Track links for its own track
    #: collection -- so the module label has to be pinned, not guessed.
    _ASSN_ROUTES = {
        "track": [("recob::Hit", "recob::Track", "fwd", ("pandoraTrack", "pmtrack"))],
        "shower": [("recob::Hit", "recob::Shower", "fwd", ("pandoraShower",))],
        "slice": [("recob::Hit", "recob::Slice", "fwd", ("pandora",))],
        "cluster": [("recob::Cluster", "recob::Hit", "rev", ("pandora",))],
        "pfparticle": [("recob::Cluster", "recob::Hit", "rev", ("pandora",)),
                       ("recob::Cluster", "recob::PFParticle", "fwd", ("pandora",))],
    }

    @_memoised
    def hit_group(self, colour_by: str = "track", tag: str | None = None) -> np.ndarray:
        """For every hit, the index of the reconstructed object it belongs to.

        Returns -1 where a hit is not associated with anything -- which is most
        of them in a radiological sample, and is itself informative.

        ``colour_by`` is one of ``track``, ``shower``, ``slice``, ``cluster``
        or ``pfparticle`` -- the same vocabulary ``EventDisplay`` uses.
        """
        if colour_by not in self._ASSN_ROUTES:
            raise ValueError(f"colour_by={colour_by!r} is not one of "
                             f"{', '.join(sorted(self._ASSN_ROUTES))}")
        # The associations index whatever hit collection the producer clustered
        # against -- passing a different tag here would size the result against
        # one collection while the indices refer to another, silently
        # mislabelling every hit.
        default = self._src.resolve_hit_product(None)
        if tag is not None and self._src.resolve_hit_product(tag) != default:
            raise ValueError(
                f"hit_group() indices refer to {default.rstrip('.')}, which is "
                f"what reconstruction associated; tag={tag!r} selects a "
                "different collection. Use ev.hits() without a tag alongside it.")
        n_hits = len(self.hits(None))
        route = self._ASSN_ROUTES[colour_by]

        mapping = np.arange(n_hits, dtype=np.int64)
        for left, right, direction, prefer in route:
            hop = self._assns_for(left, right, prefer)
            src = hop.left_key if direction == "fwd" else hop.right_key
            dst = hop.right_key if direction == "fwd" else hop.left_key
            if not len(src):
                return np.full(n_hits, -1, np.int64)
            lookup = np.full(int(src.max()) + 1, -1, np.int64)
            lookup[src] = dst
            out = np.full(n_hits, -1, np.int64)
            ok = (mapping >= 0) & (mapping < len(lookup))
            out[ok] = lookup[mapping[ok]]
            mapping = out
        return mapping

    def _assns_for(self, left: str, right: str, prefer: tuple[str, ...] = ()):
        products = self._src.art.assns_products(left, right)
        if not products:
            raise ArtReadError(
                f"no art::Assns<{left},{right}> in "
                f"{os.path.basename(self._src.path)}")
        for want in prefer:
            for p in products:
                if p.split("art::Assns_")[-1].split("_")[0] == want:
                    return self._src.art.read_assns(p, self.entry)
        if len(products) == 1:
            return self._src.art.read_assns(products[0], self.entry)
        # Several producers write this association and none is the expected
        # one. Picking alphabetically would silently substitute a different
        # algorithm's answer (pandoraShower's own track fit, say).
        raise ArtReadError(
            f"several art::Assns<{left},{right}> producers and none is one of "
            f"{', '.join(prefer)}: "
            f"{', '.join(p.split('art::Assns_')[-1].split('_')[0] for p in products)}")

    def _pick(self, classname: str, tag: str | None, prefer: tuple[str, ...]) -> str:
        products = self._src.art.find_product(classname, tag)
        if not products:
            raise ArtReadError(
                f"no {classname} product"
                + (f" with tag {tag!r}" if tag else "")
                + f" in {os.path.basename(self._src.path)}")
        if tag or len(products) == 1:
            return products[0]
        for want in prefer:
            for p in products:
                if p.split("_")[1] == want:
                    return p
        raise ArtReadError(
            f"several {classname} producers and none is one of "
            f"{', '.join(prefer)}: {', '.join(p.rstrip('.') for p in products)}. "
            "Pass tag=... to choose.")

    @_memoised
    @_memoised
    def deposited_energy(self) -> "float | None":
        """Total true ionisation energy in the detector [MeV], or None.

        This is the *visible* energy: what the simulation actually laid down in
        argon, as opposed to what the neutrino brought in. The gap between the
        two is what makes an uncontained event look empty.
        """
        try:
            dep = self.truth_deposits()
        except Exception:
            return None
        if dep is None or not len(dep):
            return None
        return float(np.sum(dep.edep))

    @_memoised
    def neutrino_visible_energy(self) -> "float | None":
        """True ionisation from the INTERACTION alone [MeV].

        Not the event total: a radiological sample lays down ~3 GeV of
        unrelated decays, which against a sub-GeV neutrino made the "visible
        energy" fraction exceed 100% on 9 of 10 events here (peak 1283%).
        Falls back to the total when there is no neutrino to attribute to.
        """
        try:
            dep = self.truth_deposits()
        except Exception:
            return None
        if dep is None or not len(dep):
            return None
        ids = self.neutrino_track_ids()
        if ids is None:
            return float(np.sum(dep.edep))
        # EM shower daughters carry a negated TrackID
        return float(np.sum(dep.edep[np.isin(np.abs(dep.track_id), ids)]))

    @_memoised
    def truth_deposits(self, tag: str | None = None) -> "TruthDeposits":
        """True energy depositions (``sim::SimEnergyDeposit``), if simulated.

        Useful as an overlay: reconstructed hits should sit on top of these.
        """
        products = self._src.art.find_product(_SIMDEP_CLASS, tag)
        if not products:
            raise ArtReadError(
                "no sim::SimEnergyDeposit product"
                + (f" with tag {tag!r}" if tag else "")
                + f" in {os.path.basename(self._src.path)}")
        # Prefer the combined ionisation-and-scintillation collection. The
        # per-volume collections largeant writes each cover ONE volume, so
        # falling back to a single one would silently present a fraction of the
        # event as the whole of it -- read them all and concatenate instead.
        combined = [p for p in products if "IonAndScint" in p]
        product = combined[0] if combined else None
        chosen = [product] if product else products
        parts = [self._src.art.read(p, _SIMDEP_CLASS, self.entry) for p in chosen]
        col = lambda k: np.concatenate([p[k] for p in parts]) if parts else np.array([])
        label = (chosen[0].rstrip(".") if len(chosen) == 1
                 else f"{len(chosen)} per-volume SimEnergyDeposit collections")
        return TruthDeposits(
            label=label,
            x=col("startPos.fCoordinates.fX").astype(np.float64),
            y=col("startPos.fCoordinates.fY").astype(np.float64),
            z=col("startPos.fCoordinates.fZ").astype(np.float64),
            edep=col("edep").astype(np.float64),
            track_id=col("trackID").astype(np.int32),
            pdg=col("pdgCode").astype(np.int32),
        )

    # ---- display -------------------------------------------------------

    def capabilities(self) -> dict:
        """What this file can actually show, probed once per event.

        Lets a front end disable what will not work instead of letting the user
        discover it as an error -- or worse, as a checkbox that silently does
        nothing.
        """
        art = self._src.art
        caps = {"tracks": bool(art.find_product("recob::Track")),
                "vertices": bool(art.find_product("recob::Vertex")),
                "showers": bool(art.find_product("recob::Shower")),
                "optical": bool(art.find_product("recob::OpFlash")),
                "spacepoints": bool(art.find_product("recob::SpacePoint")),
                "truth": bool(art.find_product("sim::SimEnergyDeposit"))}
        for name, route in self._ASSN_ROUTES.items():
            caps[f"group:{name}"] = all(
                art.assns_products(left, right) for left, right, _d, _p in route)
        return caps

    def _drop_radiologicals(self, deposits, mc):
        """Keep only what descends from the neutrino interaction."""
        ids = self.neutrino_track_ids()
        if ids is None:
            return deposits, mc          # nothing to separate
        if deposits is not None and len(deposits):
            # EM shower daughters are recorded with a negated TrackID
            deposits = deposits[np.isin(np.abs(deposits.track_id), ids)]
        if mc is not None and len(mc):
            mc = mc[np.isin(mc.track_id, ids)]
        return deposits, mc

    def display(self, tag: str | None = None, *, truth: bool = False,
                reco: bool = False, radiologicals: bool = True, **kwargs):
        """Return an :class:`~pylarevd.display.EventDisplay` for this event.

        With ``truth=True`` the true energy depositions are overlaid where the
        file contains them (silently skipped otherwise).

        ``radiologicals=False`` keeps only truth that descends from the
        neutrino interaction. A radiological sample carries tens of thousands
        of unrelated decays -- 33k of 34.6k trajectories in the sample here --
        which is honest but buries the interaction.
        """
        from .display import EventDisplay
        deposits = None
        # Readout axes are channel/tick: there is no position to draw an
        # overlay at, so the drawing code skips them entirely. Loading them
        # anyway cost 6.3 s against 1.0 s for a pixel-identical figure.
        if kwargs.get("space") == "readout":
            truth = reco = False
        if truth:
            try:
                deposits = self.truth_deposits()
            except ArtReadError:
                deposits = None
        got = {}
        for name in ("tracks", "vertices", "showers") if reco else ():
            try:
                got[name] = getattr(self, name)()
            except (ArtReadError, KeyError):
                got[name] = None
        if truth:
            try:
                got["mc"] = self.mc_particles()
            except (ArtReadError, KeyError):
                got["mc"] = None
        if truth and not radiologicals:
            deposits, got["mc"] = self._drop_radiologicals(deposits, got.get("mc"))
        return EventDisplay(self, hits=self.hits(tag), truth=deposits,
                            radiologicals=radiologicals,
                            tracks=got.get("tracks"), vertices=got.get("vertices"),
                            showers=got.get("showers"), mc=got.get("mc"), **kwargs)

    def display_flashes_3d(self, **kwargs):
        """Reconstructed flashes in the detector volume (:class:`FlashDisplay3D`)."""
        from .display import FlashDisplay3D
        return FlashDisplay3D(self, self.optical(), **kwargs)

    def display_optical(self, tag: str | None = None, **kwargs):
        """Return an :class:`~pylarevd.display.OpticalDisplay` for this event.

        Raises :class:`ArtReadError` naming what is missing when the file has no
        reconstructed optical products -- a file may carry raw waveforms only.
        """
        from .display import OpticalDisplay
        try:
            optical = self.optical(tag)
        except ArtReadError as exc:
            raw = self._src.art.find_product("raw::OpDetWaveform")
            hint = (" (the file has raw::OpDetWaveform but no flash "
                    "reconstruction)" if raw else "")
            raise ArtReadError(f"{exc}{hint}") from exc
        return OpticalDisplay(self, optical=optical, **kwargs)

    def display_3d(self, spacepoint_tag: str | None = None, *, truth: bool = False,
                   tracks: bool = True, radiologicals: bool = True, **kwargs):
        """Return a :class:`~pylarevd.display.Display3D` built from space points.

        The tag selects a ``recob::SpacePoint`` producer -- not a hit producer,
        since the 3-D view is built from space points.

        ``radiologicals=False`` keeps only truth descending from the neutrino
        interaction; see :meth:`display`.
        """
        from .display import Display3D
        deposits = None
        if truth:
            try:
                deposits = self.truth_deposits()
            except ArtReadError:
                deposits = None
        polylines = None
        if tracks:
            try:
                polylines = self.tracks()
            except (ArtReadError, KeyError):
                polylines = None
        extra = {}
        for name in ("vertices", "showers"):
            try:
                extra[name] = getattr(self, name)()
            except (ArtReadError, KeyError):
                extra[name] = None
        mc = None
        if truth:
            try:
                mc = self.mc_particles()
            except (ArtReadError, KeyError):
                mc = None
        if truth and not radiologicals:
            deposits, mc = self._drop_radiologicals(deposits, mc)
        return Display3D(self, spacepoints=self.spacepoints(spacepoint_tag),
                         truth=deposits, tracks=polylines, mc=mc,
                         radiologicals=radiologicals, **extra,
                         **kwargs)



@lru_cache(maxsize=8)
def load_geometry(path: str) -> Geometry:
    """Parse a geometry ``.npz`` once and share it between files.

    Callers must not reuse one file's geometry for another file: two samples
    can need different detectors, and using the wrong one fails silently (every
    coordinate becomes NaN) rather than loudly. Caching by path gives the same
    saving without that risk.
    """
    return Geometry(path)


class EventFile:
    """An art file plus the detector geometry needed to interpret it."""

    _EVENT_CACHE = 4          # keep a few events warm for prev/next browsing

    def __init__(self, path: str, geometry: str | Geometry | None = None):
        self.path = str(path)
        self.art = ArtFile(self.path)
        self._events: dict[int, Event] = {}
        self._scan_cache: dict = {}
        if isinstance(geometry, Geometry):
            self.geometry = geometry
        else:
            self.geometry = load_geometry(geometry or self._default_geometry())

    def _default_geometry(self) -> str:
        here = os.path.dirname(os.path.abspath(__file__))
        found = sorted(glob.glob(os.path.join(here, "geom", "*.npz")))
        if not found:
            raise GeometryError(
                "no geometry given and none found in pylarevd/geom/. "
                "Export one with pylarevd.export_geometry (see its docstring).")

        # The file records the Geometry service it was produced with, so match
        # on that rather than guessing. Guessing wrong is not a visible failure:
        # channels simply fall outside the map and every coordinate is NaN.
        wanted = self.art.geometry_config().get("Name")
        if wanted:
            for path in found:
                with np.load(path, allow_pickle=False) as z:
                    if str(z["detector"]) == wanted:
                        return path
            raise GeometryError(
                f"this file was produced with the {wanted!r} geometry, which is "
                f"not exported. Available: "
                f"{', '.join(str(np.load(p, allow_pickle=False)['detector']) for p in found)}.\n"
                f"Export it with pylarevd.export_geometry (see its docstring), "
                f"or pass geometry=... to use one of the above anyway.")

        if len(found) > 1:
            raise GeometryError(
                "several geometries available and the file does not record "
                "which it needs; pass geometry=... explicitly:\n  "
                + "\n  ".join(os.path.basename(f) for f in found))
        return found[0]

    def __repr__(self) -> str:
        return (f"<EventFile {os.path.basename(self.path)} "
                f"events={len(self)} geometry={self.geometry.detector!r}>")

    def __len__(self) -> int:
        return self.art.num_events

    def __getitem__(self, i: int) -> Event:
        n = len(self)
        if i < 0:
            i += n
        if not 0 <= i < n:
            raise IndexError(f"event {i} out of range (0..{n - 1})")
        # Hand back the same Event object so its decoded products stay cached
        # across repeated access (the browser re-indexes on every callback).
        cached = self._events.get(i)
        if cached is None:
            if len(self._events) >= self._EVENT_CACHE:
                self._events.pop(next(iter(self._events)))
            cached = self._events[i] = Event(self, i)
        return cached

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    @cached_property
    def event_ids(self) -> np.ndarray:
        return self.art.event_ids()

    def is_disambiguated(self, tag: str | None = None, *, scan: int = 12) -> "bool | None":
        """Whether a hit collection resolves the wrapped-wire ambiguity.

        Disambiguation splits a wrapped channel across physical wires, so its
        presence is the signature -- but a single event need not exercise it. A
        cosmic track that stays on one drift face splits nothing, which made a
        per-event test call the correctly disambiguated collection "NOT
        disambiguated" on 5 of 10 events. Ask the collection, over several
        events, and stop at the first event that settles it.

        Returns None when the detector has no wrapped channels (nothing to
        disambiguate) or nothing could be read.
        """
        key = ("disambig", tag)
        cached = self._scan_cache.get(key)
        if cached is not None:
            return cached
        if not self.geometry.wrapped_channels:
            self._scan_cache[key] = None
            return None
        verdict = None
        for i in range(min(scan, len(self))):
            try:
                hits = self[i].hits(tag)
            except Exception:
                continue
            if not len(hits):
                continue
            verdict = False if verdict is None else verdict
            if hits.channels_split_across_wires:
                verdict = True
                break
        self._scan_cache[key] = verdict
        return verdict

    def index_of(self, run: int, subrun: int, event: int) -> int | None:
        """Entry number of a given physics event, or ``None`` if absent.

        Entry numbers are an artefact of how a file was written; the same
        physics event sits at a different entry in a different file.  Looking
        events up by identity is what lets a link survive being pointed at
        another file -- e.g. comparing two reconstruction passes.
        """
        ids = self.event_ids
        match = ((ids["run"] == run) & (ids["subrun"] == subrun)
                 & (ids["event"] == event)).nonzero()[0]
        return int(match[0]) if len(match) else None

    def hit_products(self) -> list[str]:
        return self.art.find_product(_HIT_CLASS)

    def resolve_hit_product(self, tag: str | None) -> str:
        products = self.art.find_product(_HIT_CLASS, tag)
        if not products:
            tags = sorted({p.split("_")[1] for p in self.hit_products()})
            raise ArtReadError(
                "no recob::Hit product"
                + (f" with tag {tag!r}" if tag else "")
                + f" in {os.path.basename(self.path)}."
                + (f" Available tags: {', '.join(tags)}" if tags else ""))
        if tag or len(products) == 1:
            return products[0]
        # Prefer a disambiguated collection: its WireIDs are resolved, which is
        # what makes a physically meaningful 2D display possible at all.
        for preferred in ("hitfd", "hitdisam"):
            for p in products:
                if p.split("_")[1] == preferred:
                    return p
        return products[0]
