"""On-disk bundle cache: ``~/.cache/pylarevd/<sha1(path,mtime,size,version)>/...``.

Writes are atomic (temp file + ``os.replace``), so concurrent builders of the
same entry are safe: both build, the last rename wins, readers never see a
partial file.
"""
from __future__ import annotations

import hashlib
import os
import tempfile

_EVENT_FILES: dict = {}     # per-process: path -> EventFile (workers reuse it)

# Bump whenever the bundle contents change, so stale caches are not served.
BUNDLE_VERSION = 3


def _root() -> str:
    return os.environ.get("PYLAREVD_CACHE") or os.path.join(
        os.path.expanduser("~"), ".cache", "pylarevd")


def cache_dir(path) -> str:
    """Cache directory for a data file (created on demand)."""
    p = os.path.abspath(os.fspath(path))
    st = os.stat(p)
    key = hashlib.sha1(f"{p}|{st.st_mtime_ns}|{st.st_size}|v{BUNDLE_VERSION}".encode()).hexdigest()
    d = os.path.join(_root(), key)
    os.makedirs(d, exist_ok=True)
    return d


def _safe(tag) -> str:
    return "".join(c if c.isalnum() or c in "-." else "-" for c in str(tag)) if tag else "auto"


def entry_path(path, entry: int, hit_tag=None, flash_tag=None) -> str:
    return os.path.join(cache_dir(path),
                        f"{int(entry)}_{_safe(hit_tag)}_{_safe(flash_tag)}.evdb")


def _atomic_write(dest: str, data: bytes) -> None:
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", suffix=".evdb", dir=os.path.dirname(dest))
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, dest)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _open(path):
    from pylario import EventFile
    path = os.path.abspath(os.fspath(path))
    f = _EVENT_FILES.get(path)
    if f is None:
        f = _EVENT_FILES[path] = EventFile(path)
    return f


def _read(dest: str):
    try:
        with open(dest, "rb") as fh:
            data = fh.read()
        return data if data[:4] == b"EVDB" else None
    except OSError:
        return None


def get_or_build_event(path, entry: int, hit_tag=None, flash_tag=None) -> bytes:
    dest = entry_path(path, entry, hit_tag, flash_tag)
    data = _read(dest)
    if data is not None:
        return data
    from .export import event_bundle
    data = event_bundle(_open(path)[int(entry)], hit_tag=hit_tag, flash_tag=flash_tag)
    _atomic_write(dest, data)
    return data


def get_or_build_geometry(path) -> bytes:
    dest = os.path.join(cache_dir(path), "geometry.evdb")
    data = _read(dest)
    if data is not None:
        return data
    from .export import geometry_bundle
    data = geometry_bundle(_open(path).geometry)
    _atomic_write(dest, data)
    return data
