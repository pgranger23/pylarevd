"""Stdlib HTTP server for the pylarevd web viewer (see docs/WEB_VIEWER.md).

Static frontend + JSON/EVDB API.  Event decoding is CPU-bound Python, so builds
run in a ``ProcessPoolExecutor``; the finished bundle (and its gzip) is written
into the on-disk cache by the worker and the request thread just reads it.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import mimetypes
import multiprocessing
import os
import re
import sys
import threading
import time
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

STATIC_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
GZIP_LEVEL = 3

_SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")
_EOS_REDIRECTORS = (("/eos/user/", "eosuser.cern.ch"),
                    ("/eos/home-", "eosuser.cern.ch"),
                    ("/eos/experiment/", "eospublic.cern.ch"),
                    ("/eos/", "eospublic.cern.ch"))
_DOT_SEGMENT = re.compile(r"(^|/)\.\.?(/|$)")

MIME = {".js": "text/javascript", ".mjs": "text/javascript", ".css": "text/css",
        ".html": "text/html; charset=utf-8", ".json": "application/json",
        ".evdb": "application/octet-stream", ".svg": "image/svg+xml",
        ".png": "image/png", ".txt": "text/plain; charset=utf-8",
        ".map": "application/json", ".wasm": "application/wasm"}


class HttpError(Exception):
    def __init__(self, status: int, msg: str):
        super().__init__(msg)
        self.status = status


def normalise_path(path: str) -> str:
    """Same rule as ``pylarevd.app.normalise_path`` (copied: app imports dash)."""
    p = path.strip()
    if _SCHEME.match(p):
        return p
    p = os.path.expanduser(p)
    if p.startswith("/eos/") and not os.path.isdir("/eos"):
        for prefix, host in _EOS_REDIRECTORS:
            if p.startswith(prefix):
                return f"root://{host}/{p}"
    return p


def _url_norm(url: str) -> str:
    """``root://Host//eos//a/`` -> ``root://host/eos/a/`` (for prefix comparisons only)."""
    scheme, rest = url.split("://", 1)
    host, _, p = rest.partition("/")
    return f"{scheme.lower()}://{host.lower()}/" + re.sub(r"/+", "/", p).lstrip("/")


def under_prefix(path: str, prefixes) -> bool:
    """Is ``path`` (normalised) inside one of the directories ``prefixes``?

    Local paths are compared after resolving symlinks and ``..``; URLs must not
    contain ``.``/``..`` segments, a query or a fragment.
    """
    if _SCHEME.match(path):
        tail = path.split("://", 1)[1]
        if _DOT_SEGMENT.search(tail.partition("/")[2]) or "?" in path or "#" in path:
            return False
        n = _url_norm(path)
        return any(n.startswith(_url_norm(pre).rstrip("/") + "/")
                   for pre in prefixes if _SCHEME.match(pre))
    real = os.path.realpath(path)
    for pre in prefixes:
        if _SCHEME.match(pre):
            continue
        r = os.path.realpath(normalise_path(pre))
        if os.path.commonpath([real, r]) == r:
            return True
    return False


def expand_inputs(paths) -> list:
    """Command-line inputs -> data files: a directory (local, or a URL ending in '/')
    stands for the ``*.root`` files directly inside it."""
    out = []
    for raw in paths:
        p = normalise_path(raw)
        if _SCHEME.match(p) and p.endswith("/"):
            from XRootD.client import URL, FileSystem
            u = URL(p)
            st, listing = FileSystem(f"{u.protocol}://{u.hostid}").dirlist(
                "/" + u.path.lstrip("/"), timeout=60)
            if not st.ok:
                print(f"  could not list {p}: {st.message.strip()}", file=sys.stderr, flush=True)
                continue
            out += [p + e.name for e in sorted(listing, key=lambda e: e.name)
                    if e.name.endswith(".root")]
        elif not _SCHEME.match(p) and os.path.isdir(p):
            out += [os.path.join(p, n) for n in sorted(os.listdir(p)) if n.endswith(".root")]
        else:
            out.append(p)
    return out


# ---------------------------------------------------------------- workers

def _gz_path(dest: str) -> str:
    return dest + ".gz"


def _write_gz(dest: str) -> None:
    from . import cache
    with open(dest, "rb") as fh:
        raw = fh.read()
    cache._atomic_write(_gz_path(dest), gzip.compress(raw, GZIP_LEVEL, mtime=0))


def _build_job(kind: str, path: str, entry: int, hit_tag, flash_tag, cache_root=None) -> dict:
    """Runs in a worker process.  Returns timing only; data goes via the cache.

    ``cache_root`` is the server's cache directory: workers come from a process-wide
    forkserver, whose environment may predate a later change of PYLAREVD_CACHE.
    """
    from . import cache
    if cache_root:
        os.environ["PYLAREVD_CACHE"] = cache_root
    t = time.time()
    if kind == "geometry":
        cache.get_or_build_geometry(path)
        dest = os.path.join(cache.cache_dir(path), "geometry.evdb")
    else:
        cache.get_or_build_event(path, entry, hit_tag, flash_tag)
        dest = cache.entry_path(path, entry, hit_tag, flash_tag)
    t_build = time.time() - t
    if not os.path.exists(_gz_path(dest)):
        _write_gz(dest)
    return {"build_s": round(t_build, 3), "total_s": round(time.time() - t, 3)}


# ---------------------------------------------------------------- state

class FileInfo:
    def __init__(self, path: str):
        from pylario import EventFile
        self.path = path
        self.name = os.path.basename(path.rstrip("/")) or path
        self.ef = EventFile(path)
        self.n_events = len(self.ef)
        self.detector = str(self.ef.geometry.detector)
        self._ids = None
        self._tags = None
        self.lock = threading.Lock()

    def ids(self):
        with self.lock:
            if self._ids is None:
                a = self.ef.event_ids
                self._ids = [{"entry": i, "run": int(a["run"][i]),
                              "subrun": int(a["subrun"][i]),
                              "event": int(a["event"][i])} for i in range(len(a))]
            return self._ids

    def default_tags(self) -> tuple:
        """(hit_tag, flash_tag) used when the request leaves them out."""
        with self.lock:
            if self._tags is None:
                self._tags = (self._pick_hit(), self._pick_flash())
            return self._tags

    def _tag_list(self, products):
        out = []
        for p in products:
            parts = str(p).split("_")
            tag = parts[1] if len(parts) > 1 else str(p)
            if tag not in out:
                out.append(tag)
        return out

    def hit_tags(self):
        try:
            return self._tag_list(self.ef.hit_products())
        except Exception:
            return []

    def flash_tags(self):
        try:
            return self._tag_list(self.ef.art.find_product("recob::OpFlash"))
        except Exception:
            return []

    def _pick_hit(self):
        tags = self.hit_tags()
        for t in ("hitfd", "hitpdune"):
            if t in tags:
                return t
        for t in tags:               # something else that is disambiguated
            if t == "gaushit":
                continue
            try:
                if self.ef.is_disambiguated(t):
                    return t
            except Exception:
                pass
        for t in tags:
            if t != "gaushit":
                return t
        return tags[0] if tags else None

    def _pick_flash(self):
        tags = self.flash_tags()
        if "opflash" in tags:
            return "opflash"
        return tags[0] if tags else None


class Job:
    __slots__ = ("key", "future", "explicit", "t0", "label")

    def __init__(self, key, explicit, label):
        self.key, self.explicit, self.t0, self.label = key, explicit, time.time(), label
        self.future: Future = Future()


class App:
    def __init__(self, files=(), workers=None, prefetch=2, allow_open=True, open_prefixes=()):
        self.files: dict[str, FileInfo] = {}
        self.by_name: dict[str, str] = {}
        self.allow_open = allow_open                  # any path this account can read
        self.open_prefixes = tuple(open_prefixes)     # otherwise only files under these
        self.workers = max(1, workers or min(8, os.cpu_count() or 1))
        self.prefetch = max(0, prefetch)
        self.lock = threading.RLock()
        self.jobs: dict[tuple, Job] = {}            # in flight (queued prefetch excluded)
        self.queue: collections.deque = collections.deque()   # pending prefetch keys
        self.bg: collections.deque = collections.deque()      # --precompute backlog (lowest priority)
        self.counters = collections.Counter()
        self.last_times: collections.deque = collections.deque(maxlen=20)
        self._pool = None
        self.t_start = time.time()
        self.startup_errors: list[str] = []
        for f in files:
            try:
                self.open_file(f)
            except HttpError as exc:      # keep serving the others (e.g. EOS not reachable yet)
                self.startup_errors.append(f"{f}: {exc}")
                print(f"  could not open {f}: {exc}", file=sys.stderr, flush=True)

    # ---- files
    def may_open(self) -> bool:
        return self.allow_open or bool(self.open_prefixes)

    def open_file(self, path: str, restricted: bool = False) -> FileInfo:
        """``restricted``: a request from a browser, allowed only under ``open_prefixes``
        unless ``allow_open``."""
        path = normalise_path(path)
        if not path:
            raise HttpError(400, "no path given")
        if restricted and not self.allow_open and not under_prefix(path, self.open_prefixes):
            raise HttpError(403, "this server only opens files under: "
                                 + ", ".join(self.open_prefixes))
        if not _SCHEME.match(path):
            path = os.path.abspath(path)
            if not os.path.exists(path):
                raise HttpError(404, f"no such file: {path}")
            if os.path.isdir(path):
                raise HttpError(400, f"that is a directory, not a file: {path}")
            if not os.access(path, os.R_OK):
                raise HttpError(403, f"not readable: {path}")
        elif _DOT_SEGMENT.search(path.split("://", 1)[1].partition("/")[2]):
            raise HttpError(400, f"'.' or '..' in {path}")
        with self.lock:
            if path in self.files:
                return self.files[path]
        try:
            info = FileInfo(path)
        except Exception as exc:
            raise HttpError(400, f"could not open {os.path.basename(path)}: {exc}")
        with self.lock:
            self.files.setdefault(path, info)
            self.by_name.setdefault(info.name, path)
            return self.files[path]

    def lookup(self, ref) -> FileInfo:
        if not ref:
            if len(self.files) == 1:
                return next(iter(self.files.values()))
            raise HttpError(400, "missing 'file' parameter")
        with self.lock:
            if ref in self.files:
                return self.files[ref]
            if ref in self.by_name:
                return self.files[self.by_name[ref]]
            n = normalise_path(ref)
            if n in self.files:
                return self.files[n]
            if os.path.isabs(n) and os.path.abspath(n) in self.files:
                return self.files[os.path.abspath(n)]
        raise HttpError(404, f"file not open: {ref}")

    # ---- pool
    def pool(self):
        with self.lock:
            if self._pool is None:
                self._pool = ProcessPoolExecutor(
                    self.workers, mp_context=multiprocessing.get_context("forkserver"))
            return self._pool

    def shutdown(self):
        with self.lock:
            if self._pool is not None:
                self._pool.shutdown(wait=False, cancel_futures=True)

    # ---- cache paths
    def dest_for(self, key) -> str:
        from . import cache
        kind, path, entry, hit, flash = key
        if kind == "geometry":
            return os.path.join(cache.cache_dir(path), "geometry.evdb")
        return cache.entry_path(path, entry, hit, flash)

    def _submit(self, job: Job):
        """Caller holds self.lock."""
        key = job.key
        self.jobs[key] = job
        self.counters["builds"] += 1
        from .cache import _root
        try:
            fut = self.pool().submit(_build_job, *key, _root())
        except BrokenProcessPool:
            self._pool = None
            fut = self.pool().submit(_build_job, *key, _root())
        fut.add_done_callback(lambda f, j=job: self._done(j, f))

    def _done(self, job: Job, fut):
        try:
            res = fut.result()
            job.future.set_result(res)
            self.last_times.append({"job": job.label, "build_s": res.get("build_s"),
                                    "wait_s": round(time.time() - job.t0, 2),
                                    "prefetch": not job.explicit})
        except BaseException as exc:
            if isinstance(exc, BrokenProcessPool):
                with self.lock:
                    self._pool = None
            self.counters["errors"] += 1
            job.future.set_exception(exc)
        with self.lock:
            self.jobs.pop(job.key, None)
            self._pump()

    def _pump(self):
        """Start queued prefetches while spare capacity exists. Holds lock."""
        while self.queue or self.bg:
            running = len(self.jobs)
            pre = sum(1 for j in self.jobs.values() if not j.explicit)
            explicit = running - pre
            if explicit > 0 or running >= self.workers:
                return              # explicit work pending: prefetch yields
            if self.queue:
                limit = self.prefetch if self.workers == 1 else min(self.prefetch, self.workers - 1)
            else:
                limit = max(1, self.workers - 1)
            if pre >= max(1, limit):
                return
            key = (self.queue or self.bg).popleft()
            if key in self.jobs or os.path.exists(self.dest_for(key)) and \
                    os.path.exists(self.dest_for(key) + ".gz"):
                continue
            self._submit(Job(key, False, self._label(key)))

    @staticmethod
    def _label(key):
        return "geometry" if key[0] == "geometry" else f"{key[1].rsplit('/', 1)[-1]}:{key[2]}"

    def ensure(self, key) -> str:
        """Return the cache path of the finished raw bundle, building if needed."""
        dest = self.dest_for(key)
        gz = dest + ".gz"
        if os.path.exists(dest) and os.path.exists(gz):
            self.counters["hits"] += 1
            return dest
        with self.lock:
            job = self.jobs.get(key)
            if job is None:
                try:
                    self.queue.remove(key)
                except ValueError:
                    pass
                job = Job(key, True, self._label(key))
                self._submit(job)
            else:
                self.counters["joined"] += 1
                job.explicit = True      # someone is waiting on it now
        try:
            job.future.result()
        except BrokenProcessPool:
            raise HttpError(500, "worker process died while decoding; retry")
        except Exception as exc:
            raise HttpError(500, f"decode failed: {type(exc).__name__}: {exc}")
        return dest

    def prefetch_around(self, info: FileInfo, entry: int, hit, flash):
        if not self.prefetch:
            return
        with self.lock:
            self.queue.clear()
            for e in (entry + 1, entry + 2, entry - 1):
                if 0 <= e < info.n_events:
                    k = ("event", info.path, e, hit, flash)
                    if k not in self.jobs:
                        self.queue.append(k)
            self._pump()

    def precompute(self, info: FileInfo):
        hit, flash = info.default_tags()
        with self.lock:
            for e in range(info.n_events):
                self.bg.append(("event", info.path, e, hit, flash))
            self._pump()

    def status(self) -> dict:
        from . import cache
        with self.lock:
            inflight = [{"job": j.label, "explicit": j.explicit,
                         "age_s": round(time.time() - j.t0, 1)} for j in self.jobs.values()]
            queued = [self._label(k) for k in self.queue]
            n_bg = len(self.bg)
        files = []
        for info in list(self.files.values()):
            cached = []
            try:
                for n in os.listdir(cache.cache_dir(info.path)):
                    m = re.match(r"(\d+)_(.+)_(.+)\.evdb$", n)
                    if m:
                        cached.append(int(m.group(1)))
            except OSError:
                pass
            files.append({"path": info.path, "n_events": info.n_events,
                          "cached_entries": sorted(set(cached))})
        return {"uptime_s": round(time.time() - self.t_start, 1), "workers": self.workers,
                "prefetch": self.prefetch, "in_flight": inflight, "queued_prefetch": queued, "precompute_backlog": n_bg,
                "counters": dict(self.counters), "recent": list(self.last_times),
                "files": files, "startup_errors": self.startup_errors,
                "allow_open": self.may_open(),
                "open_prefixes": list(self.open_prefixes) if not self.allow_open else []}


# ---------------------------------------------------------------- HTTP

def _int(q, name, default=None):
    v = q.get(name, [None])[0]
    if v in (None, ""):
        if default is None:
            raise HttpError(400, f"missing '{name}' parameter")
        return default
    try:
        return int(v)
    except ValueError:
        raise HttpError(400, f"'{name}' must be an integer")


def _tag(q, name):
    v = q.get(name, [None])[0]
    return v or None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    app: App = None  # type: ignore
    server_version = "pylarevd-web"

    def log_message(self, fmt, *args):   # replaced by _log
        pass

    # -- plumbing
    def _log(self, status, nbytes, note=""):
        dt = (time.time() - self._t0) * 1000
        sys.stderr.write(f"{time.strftime('%H:%M:%S')} {self.command} {self.path[:120]} "
                         f"-> {status} {nbytes}B {dt:.0f}ms{(' ' + note) if note else ''}\n")

    def _send(self, status, body: bytes, ctype, headers=None, note=""):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)
        self._log(status, len(body), note)

    def _json(self, obj, status=200):
        self._send(status, json.dumps(obj).encode(), "application/json",
                   {"Cache-Control": "no-store"})

    def _error(self, status, msg):
        try:
            self._json({"error": msg}, status)
        except OSError:
            pass

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        self._t0 = time.time()
        try:
            u = urlsplit(self.path)
            q = parse_qs(u.query, keep_blank_values=True)
            route = u.path
            if route.startswith("/api/"):
                self._api(route, q)
            else:
                self._static(route)
        except HttpError as e:
            self._error(e.status, str(e))
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:      # never a stack trace page
            self._error(500, f"{type(e).__name__}: {e}")

    def do_POST(self):
        self._t0 = time.time()
        try:
            u = urlsplit(self.path)
            n = int(self.headers.get("Content-Length") or 0)
            if n > 1 << 20:
                raise HttpError(413, "request too large")
            raw = self.rfile.read(n) if n else b""
            if u.path != "/api/open":
                raise HttpError(404, "not found")
            if not self.app.may_open():
                raise HttpError(403, "opening files by path is disabled when bound "
                                     "beyond localhost (use --allow-open-under DIR "
                                     "or --allow-remote-open)")
            try:
                body = json.loads(raw or b"{}")
                path = body["path"]
                if not isinstance(path, str):
                    raise TypeError
            except Exception:
                raise HttpError(400, 'expected JSON body {"path": "..."}')
            info = self.app.open_file(path, restricted=True)
            self._json(self._file_entry(info))
        except HttpError as e:
            self._error(e.status, str(e))
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            self._error(500, f"{type(e).__name__}: {e}")

    # -- api
    @staticmethod
    def _file_entry(info):
        return {"path": info.path, "name": info.name, "n_events": info.n_events,
                "detector": info.detector}

    def _api(self, route, q):
        app = self.app
        if route == "/api/status":
            return self._json(app.status())
        if route == "/api/files":
            with app.lock:
                infos = list(app.files.values())
            return self._json([self._file_entry(i) for i in infos])
        info = app.lookup(_tag(q, "file"))
        if route == "/api/export":
            return self._export(info, q)
        if route == "/api/events":
            return self._json(info.ids())
        if route == "/api/geometry":
            dest = app.ensure(("geometry", info.path, 0, None, None))
            return self._bundle(dest)
        if route == "/api/event":
            entry = _int(q, "entry")
            if not 0 <= entry < info.n_events:
                raise HttpError(400, f"entry {entry} out of range (0..{info.n_events - 1})")
            dh, df = info.default_tags()
            hit = _tag(q, "hits") or dh
            flash = _tag(q, "flash") or df
            if hit and info.hit_tags() and hit not in info.hit_tags():
                raise HttpError(400, f"no hit product with tag {hit!r}; "
                                     f"available: {', '.join(info.hit_tags())}")
            if flash and info.flash_tags() and flash not in info.flash_tags():
                raise HttpError(400, f"no flash product with tag {flash!r}; "
                                     f"available: {', '.join(info.flash_tags())}")
            cached = os.path.exists(app.dest_for(("event", info.path, entry, hit, flash)) + ".gz")
            dest = app.ensure(("event", info.path, entry, hit, flash))
            try:
                self._bundle(dest, note=f"{'cached' if cached else 'built'} hits={hit} flash={flash}")
            finally:
                app.prefetch_around(info, entry, hit, flash)
            return None
        raise HttpError(404, f"unknown endpoint {route}")

    def _export(self, info, q):
        from . import render
        app = self.app
        entry = _int(q, "entry")
        if not 0 <= entry < info.n_events:
            raise HttpError(400, f"entry {entry} out of range (0..{info.n_events - 1})")
        dh, df = info.default_tags()
        hit = _tag(q, "hits") or dh
        flash = _tag(q, "flash") or df
        if hit and info.hit_tags() and hit not in info.hit_tags():
            raise HttpError(400, f"no hit product with tag {hit!r}")
        if flash and info.flash_tags() and flash not in info.flash_tags():
            raise HttpError(400, f"no flash product with tag {flash!r}")
        flat = {k: v[0] for k, v in q.items() if v}
        try:
            p = render.parse_export(flat)
        except ValueError as exc:
            raise HttpError(400, str(exc))
        try:
            fut = app.pool().submit(render.render_export, info.path, entry, hit, flash, p)
        except BrokenProcessPool:
            app._pool = None
            fut = app.pool().submit(render.render_export, info.path, entry, hit, flash, p)
        app.counters["exports"] += 1
        try:
            res = fut.result()
        except BrokenProcessPool:
            with app.lock:
                app._pool = None
            raise HttpError(500, "worker process died while rendering; retry")
        if not res["ok"]:
            raise HttpError(res["status"], res["error"])
        hdr = {"Content-Disposition": f'attachment; filename="{res["filename"]}"',
               "Cache-Control": "no-store", "X-Export-Seconds": str(res["render_s"]),
               "Access-Control-Expose-Headers": "Content-Disposition, X-Export-Note"}
        if res["note"]:
            hdr["X-Export-Note"] = res["note"]
        self._send(200, res["data"], res["ctype"], hdr,
                   f"export {p['kind']} {p['view']} {res['render_s']}s")

    def _bundle(self, dest, note=""):
        gz_ok = "gzip" in (self.headers.get("Accept-Encoding") or "")
        hdr = {"Cache-Control": "no-store", "Vary": "Accept-Encoding"}
        try:
            if gz_ok:
                if not os.path.exists(dest + ".gz"):
                    _write_gz(dest)
                with open(dest + ".gz", "rb") as fh:
                    body = fh.read()
                hdr["Content-Encoding"] = "gzip"
            else:
                with open(dest, "rb") as fh:
                    body = fh.read()
        except OSError as e:
            raise HttpError(500, f"cache read failed: {e}")
        self._send(200, body, "application/octet-stream", hdr,
                   note + (" gzip" if gz_ok else " raw"))

    # -- static
    def _static(self, route):
        if route in ("", "/"):
            route = "/index.html"
        rel = route[len("/static/"):] if route.startswith("/static/") else route.lstrip("/")
        full = os.path.realpath(os.path.join(STATIC_ROOT, rel))
        root = os.path.realpath(STATIC_ROOT)
        if not (full == root or full.startswith(root + os.sep)) or "\0" in rel:
            raise HttpError(404, "not found")
        if not os.path.isfile(full):
            raise HttpError(404, f"not found: {route}")
        st = os.stat(full)
        etag = f'"{st.st_mtime_ns:x}-{st.st_size:x}"'
        hdr = {"ETag": etag, "Cache-Control": "no-cache",
               "Last-Modified": self.date_time_string(st.st_mtime)}
        if self.headers.get("If-None-Match") == etag:
            self.send_response(304)
            for k, v in hdr.items():
                self.send_header(k, v)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return self._log(304, 0)
        ext = os.path.splitext(full)[1].lower()
        ctype = MIME.get(ext) or mimetypes.guess_type(full)[0] or "application/octet-stream"
        with open(full, "rb") as fh:
            body = fh.read()
        self._send(200, body, ctype, hdr)


def make_server(files=(), host="127.0.0.1", port=8050, workers=None, prefetch=2,
                allow_remote_open=False, open_prefixes=()) -> tuple:
    loopback = host in ("127.0.0.1", "localhost", "::1")
    app = App(files, workers=workers, prefetch=prefetch,
              allow_open=loopback or allow_remote_open, open_prefixes=open_prefixes)
    handler = type("BoundHandler", (Handler,), {"app": app})
    httpd = ThreadingHTTPServer((host, port), handler)
    httpd.daemon_threads = True
    return httpd, app


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m pylarevd.web",
                                 description="pylarevd browser event viewer")
    ap.add_argument("files", nargs="*",
                    help="art-ROOT file(s): local paths or root:// URLs; a directory "
                         "(a URL ending in '/') stands for the .root files inside it")
    ap.add_argument("--port", type=int, default=8050)
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind address (default localhost only; use an SSH tunnel)")
    ap.add_argument("--workers", type=int, default=None,
                    help="decode processes (default min(8, cpu_count))")
    ap.add_argument("--prefetch", type=int, default=2,
                    help="max concurrent background prefetch builds (0 disables)")
    ap.add_argument("--allow-remote-open", action="store_true",
                    help="permit opening files by path even when --host exposes the "
                         "server beyond localhost (anyone who can reach the port can "
                         "read any file this account can read)")
    ap.add_argument("--allow-open-under", action="append", default=[], metavar="DIR",
                    help="let browsers open files by path under this directory (local or "
                         "root:// URL; repeatable), whatever --host is")
    ap.add_argument("--precompute", action="store_true",
                    help="build the cache for every event of the given files in the background")
    a = ap.parse_intermixed_args(argv)      # files may follow options

    httpd, app = make_server(expand_inputs(a.files), a.host, a.port, a.workers, a.prefetch,
                             a.allow_remote_open, a.allow_open_under)
    node = os.uname().nodename
    fqdn = node if "." in node else f"{node}.cern.ch"
    print(f"\n  pylarevd web viewer on http://{a.host}:{a.port}  "
          f"({app.workers} workers, prefetch {app.prefetch})", flush=True)
    if a.host in ("127.0.0.1", "localhost"):
        print(f"  from your laptop:  ssh -N -L {a.port}:localhost:{a.port} {fqdn}\n", flush=True)
    else:
        print(f"  WARNING: bound to {a.host} - reachable from the network, not just this host")
        print("  open-by-path is %s\n" % (
            "ENABLED (--allow-remote-open): anyone who can reach this port can read any "
            "file you can" if a.allow_remote_open
            else "limited to " + ", ".join(a.allow_open_under) if a.allow_open_under
            else "disabled; pass --allow-open-under DIR or --allow-remote-open to enable"),
            flush=True)
    if a.precompute:
        for info in list(app.files.values()):
            app.precompute(info)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        app.shutdown()
        httpd.server_close()
    return 0
