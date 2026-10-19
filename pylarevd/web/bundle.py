"""EVDB v1: the columnar binary container shared by the exporter and the browser.

Layout (little-endian throughout)::

    "EVDB" | uint32 H | H bytes UTF-8 JSON header | pad to 8 | array data

Each array starts on an 8-byte boundary; ``offset`` in the header is relative
to the start of the array data (the first byte after the header padding).
Pure numpy + json + struct.
"""
from __future__ import annotations

import io
import json
import os
import struct

import numpy as np

MAGIC = b"EVDB"
VERSION = 1

DTYPES = {
    "f32": np.dtype("<f4"), "f64": np.dtype("<f8"),
    "i32": np.dtype("<i4"), "u32": np.dtype("<u4"),
    "i16": np.dtype("<i2"), "u16": np.dtype("<u2"),
    "i8": np.dtype("i1"), "u8": np.dtype("u1"),
}
# numpy kind+size -> code, e.g. "f4" -> "f32"
_CODE = {f"{d.kind}{d.itemsize}": k for k, d in DTYPES.items()}


def _code(arr: np.ndarray) -> str:
    key = f"{arr.dtype.kind}{arr.dtype.itemsize}"
    if arr.dtype.kind == "b":
        return "u8"
    if key not in _CODE:
        raise TypeError(f"unsupported dtype {arr.dtype} (allowed: {', '.join(DTYPES)})")
    return _CODE[key]


def _pad(n: int) -> int:
    return (-n) % 8


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


def _clean(o):
    """Make meta strict-JSON: non-finite floats become null."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, np.generic):
        o = o.item()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def bundle_bytes(kind: str, meta: dict, arrays: dict) -> bytes:
    """Serialise ``arrays`` (name -> ndarray) and ``meta`` into an EVDB blob."""
    entries, chunks, pos = {}, [], 0
    for name, a in arrays.items():
        a = np.asarray(a)
        code = _code(a)
        if a.dtype.kind == "b":
            a = a.astype(np.uint8)
        a = np.ascontiguousarray(a, dtype=DTYPES[code])
        entries[name] = {"dtype": code, "shape": list(a.shape), "offset": pos}
        raw = a.tobytes()
        chunks.append(raw)
        pad = _pad(len(raw))
        if pad:
            chunks.append(b"\0" * pad)
        pos += len(raw) + pad
    header = json.dumps({"format": "evdb", "version": VERSION, "kind": kind,
                         "meta": _clean(meta), "arrays": entries},
                        default=_json_default, allow_nan=False,
                        separators=(",", ":")).encode("utf-8")
    head = MAGIC + struct.pack("<I", len(header)) + header
    head += b"\0" * _pad(len(head))
    return head + b"".join(chunks)


def write_bundle(path_or_fileobj, kind: str, meta: dict, arrays: dict) -> int:
    """Write a bundle to a path (atomically not guaranteed) or file object."""
    data = bundle_bytes(kind, meta, arrays)
    if hasattr(path_or_fileobj, "write"):
        path_or_fileobj.write(data)
    else:
        with open(os.fspath(path_or_fileobj), "wb") as fh:
            fh.write(data)
    return len(data)


def read_bundle(src) -> tuple[dict, dict]:
    """Read a bundle from bytes or a path; returns ``(header, arrays)``."""
    if isinstance(src, (bytes, bytearray, memoryview)):
        buf = bytes(src)
    elif hasattr(src, "read"):
        buf = src.read()
    else:
        with open(os.fspath(src), "rb") as fh:
            buf = fh.read()
    if buf[:4] != MAGIC:
        raise ValueError("not an EVDB bundle (bad magic)")
    (hlen,) = struct.unpack_from("<I", buf, 4)
    header = json.loads(buf[8:8 + hlen].decode("utf-8"))
    if header.get("format") != "evdb" or header.get("version") != VERSION:
        raise ValueError(f"unsupported bundle {header.get('format')} v{header.get('version')}")
    start = 8 + hlen
    start += _pad(start)
    arrays = {}
    for name, e in header["arrays"].items():
        dt = DTYPES[e["dtype"]]
        n = int(np.prod(e["shape"], dtype=np.int64))
        off = start + e["offset"]
        arrays[name] = np.frombuffer(buf, dtype=dt, count=n, offset=off
                                     ).reshape(e["shape"])
    return header, arrays
