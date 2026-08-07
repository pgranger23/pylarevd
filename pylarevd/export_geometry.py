"""Export a LArSoft geometry to a portable ``.npz`` the display can read anywhere.

This is the *only* part of pylarevd that needs LArSoft, and it runs once per
detector geometry.  It loads ``libevdgeom.so`` (a small flat-C shim around
``lar::standalone::SetupGeometry``) through ``ctypes`` -- no ROOT, no cling, no
compiled python extension -- and writes every wire endpoint plus the drift
linearisation to a file.

Run it inside the SL7 container::

    ./inlar.sh "python -m pylarevd.export_geometry \\
        --fcl pylarevd/geom/dune10kt_1x2x6.fcl \\
        --out pylarevd/geom/dune10kt_v6_1x2x6.npz"

Everything downstream is pure python and reads only the ``.npz``.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import time

import numpy as np

_DEFAULT_LIB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "shim", "libevdgeom.so")


_U = ctypes.POINTER(ctypes.c_uint32)
_I = ctypes.POINTER(ctypes.c_int32)
_D = ctypes.POINTER(ctypes.c_double)


def _load(libpath: str) -> ctypes.CDLL:
    lib = ctypes.CDLL(libpath)
    lib.evd_geom_error.restype = ctypes.c_char_p
    lib.evd_det_name.restype = ctypes.c_char_p
    for fn in ("evd_n_wires_total", "evd_dump_wires", "evd_n_planes",
               "evd_dump_planes", "evd_dump_tpcs", "evd_n_tpcs_total",
               "evd_n_opchannels", "evd_dump_opchannels"):
        getattr(lib, fn).restype = ctypes.c_long
    # Declaring argtypes makes ctypes check every pointer's element type, so a
    # dtype that stops matching the C++ signature raises instead of silently
    # scribbling over memory.
    lib.evd_dump_wires.argtypes = [_U] * 5 + [_I] * 2 + [_D] * 6
    lib.evd_dump_planes.argtypes = [_U] * 3 + [_I, _U] + [_D] * 8
    lib.evd_dump_tpcs.argtypes = [_U] * 2 + [_D] * 6
    lib.evd_dump_opchannels.argtypes = [_U] * 2 + [_D] * 3
    return lib


_PTR_FOR = {np.dtype(np.uint32): _U, np.dtype(np.int32): _I,
            np.dtype(np.float64): _D}


def _ptr(a: np.ndarray):
    """Typed pointer, so a dtype mismatch fails loudly at the call boundary."""
    try:
        return a.ctypes.data_as(_PTR_FOR[a.dtype])
    except KeyError:
        raise TypeError(f"unsupported array dtype for the shim: {a.dtype}") from None


def export(fcl: str, out: str, libpath: str = _DEFAULT_LIB, verbose: bool = True) -> str:
    t0 = time.time()
    lib = _load(libpath)

    rc = lib.evd_geom_init(fcl.encode())
    if rc != 0:
        raise RuntimeError(f"geometry init failed (rc={rc}): "
                           f"{lib.evd_geom_error().decode(errors='replace')}")

    detector = lib.evd_det_name().decode()
    nchan = lib.evd_n_channels()
    if verbose:
        print(f"[geom] {detector}: {lib.evd_n_cryo()} cryostat(s), "
              f"{lib.evd_n_tpc()} TPC/cryostat, {nchan} channels", flush=True)

    # ---- wires ---------------------------------------------------------
    nw = lib.evd_n_wires_total()
    if nw <= 0:
        raise RuntimeError("no wires reported by the geometry")
    u = lambda: np.empty(nw, np.uint32)          # noqa: E731
    d = lambda: np.empty(nw, np.float64)         # noqa: E731
    cryo, tpc, plane, wire, chan = u(), u(), u(), u(), u()
    view = np.empty(nw, np.int32)
    signal = np.empty(nw, np.int32)
    x0, y0, z0, x1, y1, z1 = d(), d(), d(), d(), d(), d()
    got = lib.evd_dump_wires(*[_ptr(a) for a in (cryo, tpc, plane, wire, chan, view,
                                                 signal, x0, y0, z0, x1, y1, z1)])
    if got != nw:
        raise RuntimeError(f"wire dump returned {got}, expected {nw}")
    if verbose:
        print(f"[geom] {nw} wire segments, {len(np.unique(chan))} distinct channels",
              flush=True)

    # ---- readout planes (drift linearisation) --------------------------
    npl = lib.evd_n_planes()
    pu = lambda: np.empty(npl, np.uint32)        # noqa: E731
    pd = lambda: np.empty(npl, np.float64)       # noqa: E731
    p_cryo, p_tpc, p_plane, p_nwires = pu(), pu(), pu(), pu()
    p_view = np.empty(npl, np.int32)
    p_x0, p_slope = pd(), pd()
    p_cx, p_cy, p_cz, p_nx, p_ny, p_nz = pd(), pd(), pd(), pd(), pd(), pd()
    gotp = lib.evd_dump_planes(*[_ptr(a) for a in (p_cryo, p_tpc, p_plane, p_view,
                                                   p_nwires, p_x0, p_slope,
                                                   p_cx, p_cy, p_cz, p_nx, p_ny, p_nz)])
    if gotp != npl:
        raise RuntimeError(f"plane dump returned {gotp}, expected {npl}")
    has_drift = bool(lib.evd_has_detprop())
    if verbose:
        msg = "with drift conversion" if has_drift else \
              f"WITHOUT drift conversion ({lib.evd_geom_error().decode(errors='replace')[:120]})"
        print(f"[geom] {npl} readout planes, {msg}", flush=True)

    # ---- TPC active volumes -------------------------------------------
    ntpc = lib.evd_n_tpcs_total()
    tu = lambda: np.empty(ntpc, np.uint32)       # noqa: E731
    td = lambda: np.empty(ntpc, np.float64)      # noqa: E731
    t_cryo, t_tpc = tu(), tu()
    t_xmin, t_ymin, t_zmin, t_xmax, t_ymax, t_zmax = td(), td(), td(), td(), td(), td()
    gott = lib.evd_dump_tpcs(*[_ptr(a) for a in (t_cryo, t_tpc, t_xmin, t_ymin, t_zmin,
                                                 t_xmax, t_ymax, t_zmax)])
    if gott != ntpc:
        raise RuntimeError(f"TPC dump returned {gott}, expected {ntpc}")

    # ---- photon detectors ----------------------------------------------
    # Optical hits carry only a channel number; without this map the optical
    # view can plot nothing but a detector index.
    nop = max(lib.evd_n_opchannels(), 0)
    o_chan, o_det = np.empty(nop, np.uint32), np.empty(nop, np.uint32)
    o_x, o_y, o_z = (np.empty(nop, np.float64) for _ in range(3))
    if nop:
        got_op = lib.evd_dump_opchannels(
            *[_ptr(a) for a in (o_chan, o_det, o_x, o_y, o_z)])
        if got_op != nop:            # channels with no detector behind them
            o_chan, o_det = o_chan[:got_op], o_det[:got_op]
            o_x, o_y, o_z = o_x[:got_op], o_y[:got_op], o_z[:got_op]
        if verbose:
            print(f"[geom] {len(o_chan)} optical channels on "
                  f"{len(np.unique(o_det))} photon detectors", flush=True)
    elif verbose:
        print("[geom] no optical channels reported by the geometry", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    np.savez_compressed(
        out,
        detector=detector, nchannels=nchan, has_drift=has_drift, source_fcl=fcl,
        # wires
        w_cryo=cryo, w_tpc=tpc, w_plane=plane, w_wire=wire, w_chan=chan,
        w_view=view, w_signal=signal,
        w_x0=x0, w_y0=y0, w_z0=z0, w_x1=x1, w_y1=y1, w_z1=z1,
        # planes
        p_cryo=p_cryo, p_tpc=p_tpc, p_plane=p_plane, p_view=p_view, p_nwires=p_nwires,
        p_x0=p_x0, p_slope=p_slope,
        p_cx=p_cx, p_cy=p_cy, p_cz=p_cz, p_nx=p_nx, p_ny=p_ny, p_nz=p_nz,
        # tpcs
        t_cryo=t_cryo, t_tpc=t_tpc,
        t_xmin=t_xmin, t_ymin=t_ymin, t_zmin=t_zmin,
        t_xmax=t_xmax, t_ymax=t_ymax, t_zmax=t_zmax,
        # photon detectors
        o_chan=o_chan, o_opdet=o_det, o_x=o_x, o_y=o_y, o_z=o_z,
    )
    if verbose:
        size = os.path.getsize(out) / 1e6
        print(f"[geom] wrote {out} ({size:.1f} MB) in {time.time() - t0:.0f}s", flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fcl", required=True, help="fcl with Geometry/WireReadout blocks")
    ap.add_argument("--out", required=True, help="output .npz")
    ap.add_argument("--lib", default=_DEFAULT_LIB, help="path to libevdgeom.so")
    a = ap.parse_args(argv)
    export(a.fcl, a.out, a.lib)


if __name__ == "__main__":
    main()
