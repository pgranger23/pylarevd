"""End-to-end tests of the web viewer server (pylarevd.web.server)."""
import gzip
import json
import os
import socket
import threading
import time
import urllib.error
import urllib.request

import pytest

FD = "/home/pigrange/projects/data/reco/atmnu_radio_18643_1_20260727T121224Z_reco.root"
PD = ("/home/pigrange/projects/data/pdsp/PDSPProd4a_protoDUNE_sp_reco_stage1_p1GeV_35ms_"
      "sce_datadriven_18803601_67_20210414T032355Z.root")
have_fd = pytest.mark.skipif(not os.path.exists(FD), reason="FD sample absent")
have_pd = pytest.mark.skipif(not os.path.exists(PD), reason="PDSP sample absent")

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _free_port():
    for p in range(8700, 8800):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    pytest.skip("no free port")


def get(base, path, headers=None, data=None, method=None):
    req = urllib.request.Request(base + path, headers=headers or {}, data=data, method=method)
    try:
        with _OPENER.open(req, timeout=300) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


class Srv:
    def __init__(self, files, host="127.0.0.1", **kw):
        from pylarevd.web.server import make_server
        self.httpd, self.app = make_server(files, host, _free_port(), **kw)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.t = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.t.start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.app.shutdown()


@pytest.fixture(scope="module")
def cache_env(tmp_path_factory):
    d = tmp_path_factory.mktemp("evdbcache")
    old = os.environ.get("PYLAREVD_CACHE")
    os.environ["PYLAREVD_CACHE"] = str(d)
    yield d
    if old is None:
        os.environ.pop("PYLAREVD_CACHE", None)
    else:
        os.environ["PYLAREVD_CACHE"] = old


@pytest.fixture(scope="module")
def srv(cache_env):
    files = [p for p in (FD,) if os.path.exists(p)]
    s = Srv(files, workers=2, prefetch=0)
    yield s
    s.close()


def test_static_and_errors(srv):
    st, h, b = get(srv.base, "/")
    assert st == 200 and h["Content-Type"].startswith("text/html") and b"<" in b
    assert "no-cache" in h["Cache-Control"] and "ETag" in h
    st2, _, _ = get(srv.base, "/", {"If-None-Match": h["ETag"]})
    assert st2 == 304
    st, h, _ = get(srv.base, "/static/main.js")
    if st == 200:
        assert h["Content-Type"].startswith("text/javascript")
    for bad in ("/static/../server.py", "/../server.py", "/%2e%2e/server.py",
                "/static/%2e%2e/%2e%2e/web/server.py"):
        assert get(srv.base, bad)[0] == 404
    st, h, b = get(srv.base, "/api/nothing")
    assert st == 404 and "error" in json.loads(b)
    assert get(srv.base, "/api/event?file=nope&entry=0")[0] == 404
    assert get(srv.base, "/api/status")[0] == 200


def test_open_refused_non_loopback(cache_env):
    s = Srv([], host="0.0.0.0", workers=1)
    try:
        st, _, b = get(s.base, "/api/open", data=json.dumps({"path": FD}).encode(),
                       headers={"Content-Type": "application/json"}, method="POST")
        assert st == 403 and "error" in json.loads(b)
    finally:
        s.close()


def test_open_bad_requests(srv):
    def post(body):
        return get(srv.base, "/api/open", data=body, method="POST")
    assert post(b"not json")[0] == 400
    assert post(json.dumps({"path": "/no/such/file.root"}).encode())[0] == 404
    assert post(json.dumps({"path": "/tmp"}).encode())[0] == 400


@have_fd
def test_files_events_geometry(srv):
    st, _, b = get(srv.base, "/api/files")
    files = json.loads(b)
    assert st == 200 and files[0]["path"] == FD and files[0]["n_events"] >= 1
    assert files[0]["detector"]
    ev = json.loads(get(srv.base, "/api/events?file=" + os.path.basename(FD))[2])
    assert len(ev) == files[0]["n_events"] and {"entry", "run", "subrun", "event"} <= set(ev[0])
    from pylarevd.web import bundle
    st, h, b = get(srv.base, "/api/geometry?file=" + FD, {"Accept-Encoding": "gzip"})
    assert st == 200 and h.get("Content-Encoding") == "gzip"
    assert int(h["Content-Length"]) == len(b)
    header, arrays = bundle.read_bundle(gzip.decompress(b))
    assert header["kind"] == "geometry" and header["meta"]["detector"]
    # open-by-path of an already open file is idempotent
    st, _, b = get(srv.base, "/api/open", data=json.dumps({"path": FD}).encode(), method="POST")
    assert st == 200 and json.loads(b)["path"] == FD


@have_fd
def test_event_bundle_gzip_and_errors(srv):
    from pylarevd.web import bundle
    q = "/api/event?file=%s&entry=0" % FD
    t = time.time()
    st, h, raw = get(srv.base, q)                      # no Accept-Encoding: raw
    assert st == 200 and "Content-Encoding" not in h and raw[:4] == b"EVDB"
    assert int(h["Content-Length"]) == len(raw)
    st, h, gz = get(srv.base, q, {"Accept-Encoding": "gzip, deflate"})
    assert h.get("Content-Encoding") == "gzip" and int(h["Content-Length"]) == len(gz)
    assert gzip.decompress(gz) == raw
    header, arrays = bundle.read_bundle(raw)
    assert header["kind"] == "event" and header["meta"]["entry"] == 0
    assert header["meta"]["hit_tag"] == "hitfd"
    t = time.time()
    get(srv.base, q, {"Accept-Encoding": "gzip"})
    assert time.time() - t < 2.0                               # cached
    n = srv.app.files[FD].n_events
    assert get(srv.base, f"/api/event?file={FD}&entry={n}")[0] == 400
    assert get(srv.base, f"/api/event?file={FD}&entry=-1")[0] == 400
    assert get(srv.base, f"/api/event?file={FD}&entry=abc")[0] == 400
    assert get(srv.base, f"/api/event?file={FD}")[0] == 400
    st, _, b = get(srv.base, f"/api/event?file={FD}&entry=0&hits=nonexistent")
    assert st == 400 and "error" in json.loads(b)


@have_fd
def test_default_tags(srv):
    hit, flash = srv.app.files[FD].default_tags()
    assert hit == "hitfd" and flash in (None, "opflash")


@have_pd
def test_default_tag_pdsp_not_gaushit(cache_env):
    from pylarevd.web.server import FileInfo
    info = FileInfo(PD)
    hit, _ = info.default_tags()
    assert hit != "gaushit" and hit == "hitpdune"


@have_fd
def test_concurrent_duplicate_builds_once(cache_env):
    n = None
    s = Srv([FD], workers=3, prefetch=0)
    try:
        n = s.app.files[FD].n_events
        entry = n - 1
        before = s.app.counters["builds"]
        out = []
        def go():
            out.append(get(s.base, f"/api/event?file={FD}&entry={entry}&flash=opflash",
                           {"Accept-Encoding": "gzip"}))
        ths = [threading.Thread(target=go) for _ in range(5)]
        [t.start() for t in ths]
        [t.join() for t in ths]
        assert all(o[0] == 200 for o in out) and len({o[2] for o in out}) == 1
        # one build, unless the entry was already in the (shared) cache
        assert s.app.counters["builds"] - before <= 1
        st = json.loads(get(s.base, "/api/status")[2])
        assert st["files"][0]["n_events"] == n and not st["in_flight"]
    finally:
        s.close()


@have_fd
def test_prefetch_fills_neighbours(cache_env):
    s = Srv([FD], workers=3, prefetch=2)
    try:
        n = s.app.files[FD].n_events
        if n < 3:
            pytest.skip("need >= 3 events")
        get(s.base, f"/api/event?file={FD}&entry=0")
        from pylarevd.web import cache
        want = [cache.entry_path(FD, e, "hitfd", "opflash") + ".gz" for e in (1, 2)]
        t = time.time()
        while time.time() - t < 240 and not all(os.path.exists(w) for w in want):
            time.sleep(1)
        assert all(os.path.exists(w) for w in want)
    finally:
        s.close()
