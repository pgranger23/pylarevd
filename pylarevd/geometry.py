"""Detector geometry for the display, loaded from an exported ``.npz``.

Nothing here is detector-specific: every quantity comes from the file written
by :mod:`pylarevd.export_geometry`, which got it from LArSoft's own channel map.
There are no hard-coded wire angles, pitches or drift constants.

The central operation is turning a hit's ``geo::WireID`` into a *continuous
wire coordinate* ``w``.  In a wrapped-wire APA (DUNE FD-HD) a readout channel is
bonded to segments on both drift faces, so anything binned against *channel*
folds the detector onto itself and a single track breaks into disconnected
pieces.  Instead, for each wire we take its direction in the (y, z) plane::

    u = (uy, uz)            wire direction, canonicalised to uy >= 0
    p = (-uz, uy)           measurement direction, perpendicular to the wire
    w = p . (y, z)          continuous coordinate along the plane [cm]

which reduces to ``w = z`` for vertical collection wires.  Segments that wrap to
the opposite face carry the mirrored angle, so grouping hits by
:attr:`Geometry.orientation_key` -- rather than by plane label -- is what keeps
a panel's ``w`` axis meaningful across both drift volumes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import cached_property

import numpy as np

VIEW_NAMES = {0: "U", 1: "V", 2: "Z", 3: "Y", 4: "X"}


class GeometryError(RuntimeError):
    pass


@dataclass(frozen=True)
class Containment:
    """Where a point sits relative to the TPC active volume."""

    inside: bool
    distance: float        # cm to the nearest face (inward if inside)
    description: str

    def __bool__(self) -> bool:
        return self.inside

    def __str__(self) -> str:
        return self.description


@dataclass(frozen=True)
class TPCBox:
    cryo: int
    tpc: int
    xmin: float
    ymin: float
    zmin: float
    xmax: float
    ymax: float
    zmax: float


class Geometry:
    """Wire positions and drift conversion for one detector geometry."""

    def __init__(self, path: str):
        if not os.path.exists(path):
            raise GeometryError(
                f"geometry file not found: {path}\n"
                "Export one inside the SL7 container with:\n"
                "  ./inlar.sh \"cd pylarevd && python -m pylarevd.export_geometry "
                "--fcl <geo.fcl> --out <out.npz>\"")
        self.path = path
        try:
            z = np.load(path, allow_pickle=False)
            self._z = {k: z[k] for k in z.files}
        except Exception as exc:
            raise GeometryError(f"cannot read geometry {path}: {exc}") from exc

        missing = [k for k in ("detector", "nchannels", "has_drift", "w_cryo",
                               "w_tpc", "w_plane", "w_wire", "w_chan", "w_view",
                               "p_cryo", "p_tpc", "p_plane", "p_view",
                               "p_x0", "p_slope", "p_nx")
                   if k not in self._z]
        if missing:
            raise GeometryError(
                f"{path} is not a pylarevd geometry file (missing "
                f"{', '.join(missing)}). Re-export it with "
                "python -m pylarevd.export_geometry.")

        self.detector = str(self._z["detector"])
        self.nchannels = int(self._z["nchannels"])
        self.has_drift = bool(self._z["has_drift"])

        self.w_cryo = self._z["w_cryo"].astype(np.int32)
        self.w_tpc = self._z["w_tpc"].astype(np.int32)
        self.w_plane = self._z["w_plane"].astype(np.int32)
        self.w_wire = self._z["w_wire"].astype(np.int32)
        self.w_chan = self._z["w_chan"].astype(np.int32)
        self.w_view = self._z["w_view"].astype(np.int32)

        self.p_cryo = self._z["p_cryo"].astype(np.int32)
        self.p_tpc = self._z["p_tpc"].astype(np.int32)
        self.p_plane = self._z["p_plane"].astype(np.int32)
        self.p_view = self._z["p_view"].astype(np.int32)
        self.p_x0 = self._z["p_x0"]
        self.p_slope = self._z["p_slope"]
        self.p_nx = self._z["p_nx"]

        self._build_index()

    def __repr__(self) -> str:
        return (f"<Geometry {self.detector!r} wires={len(self.w_wire)} "
                f"channels={self.nchannels} drift={'yes' if self.has_drift else 'no'}>")

    # ---- indexing ------------------------------------------------------

    def _build_index(self) -> None:
        """Flat lookup from (cryo, tpc, plane, wire) to a wire row."""
        self._n_tpc = int(self.w_tpc.max()) + 1
        self._n_plane = int(self.w_plane.max()) + 1
        self._n_wire = int(self.w_wire.max()) + 1
        self._n_cryo = n_cryo = int(self.w_cryo.max()) + 1
        size = n_cryo * self._n_tpc * self._n_plane * self._n_wire
        self._wire_index = np.full(size, -1, dtype=np.int64)
        self._wire_index[self._key(self.w_cryo, self.w_tpc, self.w_plane, self.w_wire)] = \
            np.arange(len(self.w_wire), dtype=np.int64)

        # plane rows, keyed the same way but without the wire term
        psize = n_cryo * self._n_tpc * self._n_plane
        self._plane_index = np.full(psize, -1, dtype=np.int64)
        pkey = (self.p_cryo * self._n_tpc + self.p_tpc) * self._n_plane + self.p_plane
        self._plane_index[pkey] = np.arange(len(self.p_plane), dtype=np.int64)

    def _key(self, cryo, tpc, plane, wire):
        return (((cryo * self._n_tpc + tpc) * self._n_plane + plane) * self._n_wire + wire)

    def _plane_key(self, cryo, tpc, plane):
        return (cryo * self._n_tpc + tpc) * self._n_plane + plane

    # ---- derived wire quantities --------------------------------------

    @cached_property
    def wire_coord(self) -> np.ndarray:
        """Continuous wire coordinate ``w`` [cm] of every wire segment."""
        uy = self._z["w_y1"] - self._z["w_y0"]
        uz = self._z["w_z1"] - self._z["w_z0"]
        norm = np.hypot(uy, uz)
        bad = norm == 0
        norm[bad] = 1.0
        uy, uz = uy / norm, uz / norm
        # canonicalise the wire direction so mirrored segments do not flip sign
        flip = uy < 0
        uy = np.where(flip, -uy, uy)
        uz = np.where(flip, -uz, uz)
        py, pz = -uz, uy                     # measurement direction
        cy = 0.5 * (self._z["w_y0"] + self._z["w_y1"])
        cz = 0.5 * (self._z["w_z0"] + self._z["w_z1"])
        w = py * cy + pz * cz
        w[bad] = np.nan
        return w

    @cached_property
    def plane_measure_dir(self) -> np.ndarray:
        """(nplanes, 2) measurement direction ``p = (py, pz)`` per readout plane.

        Projecting any 3-D point onto this direction gives the same continuous
        coordinate ``w`` that :attr:`wire_coord` assigns to wires, so truth
        trajectories and reconstructed hits can be overlaid directly.
        """
        uy = self._z["w_y1"] - self._z["w_y0"]
        uz = self._z["w_z1"] - self._z["w_z0"]
        norm = np.hypot(uy, uz)
        norm[norm == 0] = 1.0
        uy, uz = uy / norm, uz / norm
        flip = uy < 0
        uy = np.where(flip, -uy, uy)
        uz = np.where(flip, -uz, uz)

        out = np.full((len(self.p_plane), 2), np.nan)
        rows = self.plane_rows(self.w_cryo, self.w_tpc, self.w_plane)
        # every wire of a plane is parallel, so the first one fixes the direction
        seen = np.zeros(len(self.p_plane), bool)
        for i, r in enumerate(rows):
            if r >= 0 and not seen[r]:
                out[r] = (-uz[i], uy[i])
                seen[r] = True
        return out

    @cached_property
    def plane_wire_angle(self) -> np.ndarray:
        """Wire angle from vertical, in degrees, per readout plane.

        In a wrapped APA the same physical channel appears at +theta on one
        face and -theta on the other, so this -- not the plane's U/V label --
        is what says whether two sets of wires measure the same projection.
        """
        p = self.plane_measure_dir
        # wire direction u = (p_z, -p_y); angle measured from the y axis
        return np.degrees(np.arctan2(-p[:, 0], p[:, 1]))

    @cached_property
    def orientation_key(self) -> np.ndarray:
        """Integer per readout plane; equal iff the wires are parallel.

        Rounded to 0.1 degrees, which is far finer than any real difference
        between APAs but coarse enough to absorb floating-point noise.
        """
        ang = self.plane_wire_angle.copy()
        ang[~np.isfinite(ang)] = 999.0
        return np.round(ang * 10).astype(np.int64)

    @property
    def has_optical(self) -> bool:
        """Whether photon-detector positions were exported with this geometry."""
        return "o_chan" in self._z and len(self._z["o_chan"]) > 0

    @cached_property
    def opdet_xyz(self) -> np.ndarray:
        """(n_opchannels, 3) photon-detector position, indexed by op-channel.

        Optical hits carry only a channel number, so without this the optical
        view can plot nothing but a detector index.  Channels with no detector
        behind them are NaN.
        """
        if not self.has_optical:
            return np.zeros((0, 3))
        chan = self._z["o_chan"].astype(np.int64)
        out = np.full((int(chan.max()) + 1, 3), np.nan)
        out[chan, 0] = self._z["o_x"]
        out[chan, 1] = self._z["o_y"]
        out[chan, 2] = self._z["o_z"]
        return out

    def opdet_positions(self, channels) -> np.ndarray:
        """Positions for an array of optical channel numbers (NaN if unknown)."""
        table = self.opdet_xyz
        ch = np.asarray(channels, np.int64)
        out = np.full((len(ch), 3), np.nan)
        ok = (ch >= 0) & (ch < len(table))
        if ok.any():
            out[ok] = table[ch[ok]]
        return out

    @cached_property
    def channel_view(self) -> np.ndarray:
        """View of every readout channel, indexed by channel number.

        A wrapped channel touches several wires, but all of them lie in the
        same plane, so this is well defined -- and unlike a WireID it is always
        available, which is what makes the readout view work for hit
        collections that have not been disambiguated.
        """
        out = np.full(self.nchannels, -1, np.int32)
        out[self.w_chan] = self.w_view
        return out

    @cached_property
    def drift_sign(self) -> np.ndarray:
        """+1/-1 per readout plane: which way the drift field points.

        Used to keep the two faces of a wrapped APA in separate panels.
        """
        s = np.sign(self.p_slope)
        s[s == 0] = np.sign(self.p_nx[s == 0])
        s[s == 0] = 1.0
        return s.astype(np.int32)

    @cached_property
    def tpcs(self) -> list[TPCBox]:
        z = self._z
        return [TPCBox(int(z["t_cryo"][i]), int(z["t_tpc"][i]),
                       float(z["t_xmin"][i]), float(z["t_ymin"][i]), float(z["t_zmin"][i]),
                       float(z["t_xmax"][i]), float(z["t_ymax"][i]), float(z["t_zmax"][i]))
                for i in range(len(z["t_tpc"]))]

    @cached_property
    def active_boxes(self) -> np.ndarray:
        """(N, 6) array of TPC active volumes as ``[xmin, ymin, zmin, xmax, ymax, zmax]``."""
        z = self._z
        return np.stack([z["t_xmin"], z["t_ymin"], z["t_zmin"],
                         z["t_xmax"], z["t_ymax"], z["t_zmax"]], axis=1)

    @cached_property
    def wrapped_channels(self) -> int:
        """Channels bonded to wire segments on more than one (tpc, plane, wire).

        Non-zero only for wrapped-wire designs such as DUNE FD-HD. It is what
        makes disambiguation necessary, and therefore what makes an
        undisambiguated hit collection detectable.
        """
        key = ((self.w_tpc.astype(np.int64) << 40)
               | (self.w_plane.astype(np.int64) << 20) | self.w_wire)
        pairs = np.unique(np.stack([self.w_chan.astype(np.int64), key]), axis=1)
        _, counts = np.unique(pairs[0], return_counts=True)
        return int((counts > 1).sum())

    def in_active(self, points) -> np.ndarray:
        """Which points lie inside *some* TPC's active volume.

        Tested per TPC rather than against the overall bounding box: a point
        can sit inside the box that encloses the whole detector while being in
        a gap between drift volumes.
        """
        p = np.atleast_2d(np.asarray(points, float))
        box = self.active_boxes
        inside = ((box[None, :, :3] <= p[:, None, :])
                  & (p[:, None, :] <= box[None, :, 3:])).all(axis=2).any(axis=1)
        return inside

    def containment(self, point) -> "Containment":
        """Where a point sits relative to the active volume.

        Reported as the distance to the nearest TPC, plus which face it is
        beyond -- a bare "outside" leaves you guessing whether the vertex
        missed by a centimetre or by three metres.
        """
        p = np.asarray(point, float).reshape(3)
        if not np.isfinite(p).all():
            return Containment(False, float("nan"), "vertex position unknown")
        box = self.active_boxes
        # Per-axis room to each face; negative on any axis means outside that TPC.
        margin = np.minimum(p - box[:, :3], box[:, 3:] - p)
        held = (margin >= 0).all(axis=1)
        if held.any():
            # Distance to the nearest wall is the *smallest* per-axis room; of
            # the TPCs containing the point, report the most generous one.
            room = margin[held].min(axis=1)
            best = int(np.argmax(room))
            gap = float(room[best])
            axis = "xyz"[int(np.argmin(margin[held][best]))]
            return Containment(
                True, gap,
                f"inside the active volume, {gap:.1f} cm from the nearest "
                f"face ({axis})")
        # Outside every TPC: distance to the closest one, and the axis that misses.
        delta = np.maximum(np.maximum(box[:, :3] - p, p - box[:, 3:]), 0.0)
        dist = np.linalg.norm(delta, axis=1)
        j = int(np.argmin(dist))
        d = float(dist[j])
        axis = int(np.argmax(delta[j]))
        name = "xyz"[axis]
        face = (box[j, axis] if p[axis] < box[j, axis] else box[j, axis + 3])
        beyond = "below" if p[axis] < box[j, axis] else "above"
        return Containment(
            False, d,
            f"{d:.1f} cm outside the active volume "
            f"({delta[j][axis]:.1f} cm {beyond} {name} = {face:.0f})")

    # ---- lookups (vectorised) -----------------------------------------

    def wire_rows(self, cryo, tpc, plane, wire) -> np.ndarray:
        """Wire-table rows for arrays of WireID components (-1 where unknown).

        Every component is checked against BOTH ends of its range.  The lower
        bound matters as much as the upper one: LArSoft marks an unset WireID
        with ``std::numeric_limits<unsigned int>::max()``, which becomes -1 once
        the caller narrows it to a signed type, and a negative index would
        otherwise be silently resolved by numpy from the end of the table --
        returning a real, plausible-looking wire for a hit that has none.
        """
        cryo = np.asarray(cryo, np.int64)
        tpc = np.asarray(tpc, np.int64)
        plane = np.asarray(plane, np.int64)
        wire = np.asarray(wire, np.int64)
        ok = ((cryo >= 0) & (cryo < self._n_cryo) &
              (tpc >= 0) & (tpc < self._n_tpc) &
              (plane >= 0) & (plane < self._n_plane) &
              (wire >= 0) & (wire < self._n_wire))
        out = np.full(np.broadcast(cryo, tpc, plane, wire).shape, -1, np.int64)
        if ok.any():
            k = self._key(cryo[ok], tpc[ok], plane[ok], wire[ok])
            out[ok] = self._wire_index[k]
        return out

    def plane_rows(self, cryo, tpc, plane) -> np.ndarray:
        """Readout-plane rows for arrays of PlaneID components (-1 if unknown)."""
        cryo = np.asarray(cryo, np.int64)
        tpc = np.asarray(tpc, np.int64)
        plane = np.asarray(plane, np.int64)
        ok = ((cryo >= 0) & (cryo < self._n_cryo) &
              (tpc >= 0) & (tpc < self._n_tpc) &
              (plane >= 0) & (plane < self._n_plane))
        out = np.full(np.broadcast(cryo, tpc, plane).shape, -1, np.int64)
        if ok.any():
            k = self._plane_key(cryo[ok], tpc[ok], plane[ok])
            out[ok] = self._plane_index[k]
        return out

    def ticks_to_x(self, ticks, plane_rows) -> np.ndarray:
        """Convert TDC ticks to the drift coordinate x [cm], per readout plane."""
        if not self.has_drift:
            raise GeometryError(
                f"{self.path} was exported without detector properties, so ticks "
                "cannot be converted to x. Re-export with DetectorProperties, "
                "LArProperties and DetectorClocks blocks in the fcl.")
        ticks = np.asarray(ticks, np.float64)
        out = np.full(ticks.shape, np.nan)
        good = plane_rows >= 0
        pr = plane_rows[good]
        out[good] = self.p_x0[pr] + self.p_slope[pr] * ticks[good]
        return out

    def panel_of(self, view, drift_sign) -> np.ndarray:
        """Stable panel key per (view, drift side).

        Wrapped induction segments on opposite faces measure different
        projections and must never share an axis.
        """
        return np.asarray(view, np.int64) * 2 + (np.asarray(drift_sign, np.int64) > 0)

    @staticmethod
    def view_name(view: int) -> str:
        return VIEW_NAMES.get(int(view), f"view{int(view)}")
