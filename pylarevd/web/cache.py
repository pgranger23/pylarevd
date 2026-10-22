"""On-disk bundle cache: ``~/.cache/pylarevd/<sha1(path,mtime,size,version)>/...``.

Writes are atomic (temp file + ``os.replace``), so concurrent builders of the
same entry are safe: both build, the last rename wins, readers never see a
partial file.

Data files may be local paths or URLs (``root://host//eos/...``); a URL is keyed
by the size and modification time the remote server reports.
"""
from __future__ import annotations

import hashlib
import os
import re
import tempfile
import time

_EVENT_FILES: dict = {}     # per-process: path -> EventFile (workers reuse it)

_SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")
_XROOTD = ("root", "roots", "xroot", "xroots")
_REMOTE_STAT_TTL = 60.0     # s; the stat is needed on every request, the file rarely changes
_remote_stats: dict = {}    # per-process: url -> (monotonic time, (mtime_ns, size))

# Bump whenever the bundle contents change, so stale caches are not served.
BUNDLE_VERSION = 4


def _root() -> str:
    return os.environ.get("PYLAREVD_CACHE") or os.path.join(
        os.path.expanduser("~"), ".cache", "pylarevd")


def is_remote(path) -> bool:
    return bool(_SCHEME.match(os.fspath(path)))


def canonical(path) -> str:
    """Absolute local path, or the URL unchanged (``abspath`` would mangle it)."""
    p = os.fspath(path)
    return p if is_remote(p) else os.path.abspath(p)


def _remote_stat(url: str) -> tuple:
    """(mtime_ns, size) of a remote file, remembered for ``_REMOTE_STAT_TTL`` seconds."""
    now = time.monotonic()
    hit = _remote_stats.get(url)
    if hit is not None and now - hit[0] < _REMOTE_STAT_TTL:
        return hit[1]
    if url.split("://", 1)[0].lower() in _XROOTD:
        from XRootD.client import URL, FileSystem
        u = URL(url)
        st, info = FileSystem(f"{u.protocol}://{u.hostid}").stat(
            "/" + u.path.lstrip("/"), timeout=60)
        if not st.ok or info is None:
            raise OSError(f"cannot stat {url}: {st.message.strip()}")
        res = (int(info.modtime) * 1_000_000_000, int(info.size))
    else:                                   # anything else fsspec can reach (https, s3, ...)
        import fsspec
        fs, p = fsspec.core.url_to_fs(url)
        d = fs.info(p)
        m = d.get("mtime") or d.get("LastModified") or d.get("last_modified") or 0
        m = m.timestamp() if hasattr(m, "timestamp") else float(m)
        res = (int(m * 1e9), int(d.get("size") or 0))
    _remote_stats[url] = (now, res)
    return res


def file_stamp(path) -> tuple:
    """(mtime_ns, size) of a local path or a URL."""
    p = canonical(path)
    if is_remote(p):
        return _remote_stat(p)
    st = os.stat(p)
    return st.st_mtime_ns, st.st_size


def cache_dir(path) -> str:
    """Cache directory for a data file (created on demand)."""
    p = canonical(path)
    mtime_ns, size = file_stamp(p)
    key = hashlib.sha1(f"{p}|{mtime_ns}|{size}|v{BUNDLE_VERSION}".encode()).hexdigest()
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
    path = canonical(path)
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
    f = _open(path)
    try:
        data = event_bundle(f[int(entry)], hit_tag=hit_tag, flash_tag=flash_tag)
    finally:
        # The bundle is all that is kept: the decoded event and the decompressed
        # baskets would stay warm in this worker for nothing -- four ProtoDUNE-HD
        # events held ~4.8 GB per worker, ~0.6 GB once released.
        release = getattr(f, "release", None)          # pylario with EventFile.release
        if release is not None:
            release(int(entry))
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
