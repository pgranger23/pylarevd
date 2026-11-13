"""Regression tests for the web-backend bug fixes (server, export, trigger, render, cache)."""
import socket
import types

import numpy as np
import pytest

from test_web_server import FD, Srv, cache_env, get, have_fd  # noqa: F401  (fixture)


@pytest.fixture(scope="module")
def bare(cache_env):
    """A server with no file open: enough for request parsing, HEAD and static files."""
    s = Srv([], workers=1, prefetch=0)
    yield s
    s.close()


def _raw(base, request: bytes, timeout=10) -> bytes:
    """Send ``request`` verbatim (no client-side path normalisation); return the reply."""
    host, port = base.rsplit("/", 1)[-1].split(":")
    with socket.create_connection((host, int(port)), timeout=timeout) as s:
        s.sendall(request)
        out = b""
        while True:
            chunk = s.recv(65536)
            if not chunk:
                return out
            out += chunk


def _status(reply: bytes) -> int:
    return int(reply.split(b" ", 2)[1])


# ---- POST /api/open: Content-Length -------------------------------------------------

@pytest.mark.parametrize("value", [b"-5", b"abc", b"1.5"])
def test_bad_content_length_is_a_400_and_does_not_hang(bare, value):
    # a negative length used to reach rfile.read(-1), which waits for the client to close
    reply = _raw(bare.base, b"POST /api/open HTTP/1.1\r\nHost: x\r\nContent-Length: " + value
                 + b"\r\n\r\n")
    assert _status(reply) == 400 and b"bad Content-Length" in reply


def test_oversized_post_closes_the_connection(bare):
    reply = _raw(bare.base, b"POST /api/open HTTP/1.1\r\nHost: x\r\nContent-Length: 99999999"
                 b"\r\n\r\n")
    assert _status(reply) == 413 and b"Connection: close" in reply


# ---- HEAD ---------------------------------------------------------------------------

def test_head_static_matches_get_without_a_body(bare):
    st, h, b = get(bare.base, "/index.html")
    st2, h2, b2 = get(bare.base, "/index.html", method="HEAD")
    assert st == st2 == 200 and b2 == b""
    assert h2["Content-Length"] == str(len(b)) and h2["ETag"] == h["ETag"]
    assert h2["Content-Type"] == h["Content-Type"]


def test_head_cheap_api_routes_answer(bare):
    for route in ("/api/status", "/api/files"):
        st, h, b = get(bare.base, route, method="HEAD")
        assert st == 200 and b == b"" and int(h["Content-Length"]) > 0


@pytest.mark.parametrize("route", ["/api/event?entry=0", "/api/geometry", "/api/export?entry=0",
                                   "/api/events"])
def test_head_heavy_api_routes_are_refused(bare, route):
    builds = bare.app.counters["builds"]
    st, h, b = get(bare.base, route, method="HEAD")
    assert st == 405 and h["Allow"] == "GET" and b == b""
    assert bare.app.counters["builds"] == builds and bare.app.counters["exports"] == 0


# ---- static files: percent-decoding and traversal --------------------------------------

def test_static_routes_are_percent_decoded(bare):
    st, _, b = get(bare.base, "/%69ndex.html")           # %69 = 'i'
    st2, _, b2 = get(bare.base, "/index.html")
    assert st == st2 == 200 and b == b2


@pytest.mark.parametrize("path", [
    "/static/%2e%2e/server.py", "/%2e%2e/server.py", "/%2E%2E%2Fserver.py",
    "/static/%2e%2e%2f%2e%2e%2fcache.py", "/..%2fserver.py", "/%2e/index.html",
    "/static/%2Fetc%2Fpasswd", "/index.html%00.js", "/%ff%fe.js",
])
def test_encoded_traversal_is_refused(bare, path):
    reply = _raw(bare.base, f"GET {path} HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n"
                 .encode())
    assert _status(reply) == 404, reply[:200]
    assert b"import" not in reply and b"root:" not in reply


# ---- status counters ------------------------------------------------------------------

def test_cache_hit_counter_is_updated_under_the_lock(bare, monkeypatch):
    """``ensure`` counted cache hits without the lock; check it now takes it."""
    from pylarevd.web import server
    monkeypatch.setattr(server, "_built", lambda dest: True)
    monkeypatch.setattr(bare.app, "dest_for", lambda key: "/nonexistent")
    seen = []

    class Spy:
        def __init__(self, lock):
            self.lock = lock

        def __enter__(self):
            seen.append("lock")
            return self.lock.__enter__()

        def __exit__(self, *a):
            return self.lock.__exit__(*a)
    monkeypatch.setattr(bare.app, "lock", Spy(bare.app.lock))
    before = bare.app.counters["hits"]
    assert bare.app.ensure(("event", "x", 0, None, None)) == "/nonexistent"
    assert bare.app.counters["hits"] == before + 1 and seen == ["lock"]


# ---- export: PFParticle -> slice mapping -----------------------------------------------

def test_slice_of_pfp_checks_the_range():
    from pylarevd.web.export import _slice_of_pfp
    pfp_slice = np.array([0, 1, -1], np.int64)
    out = _slice_of_pfp(np.array([0, 1, 2, 3, 7, -1]), pfp_slice)
    # ids 3 and 7 have no PFParticle: -1, not the last PFParticle's slice
    assert out.tolist() == [0, 1, -1, -1, -1, -1] and out.dtype == np.int32


def test_slice_of_pfp_without_pfparticles():
    from pylarevd.web.export import _slice_of_pfp
    out = _slice_of_pfp(np.array([0, 2, -1], np.int32), np.zeros(0, np.int64))
    assert out.tolist() == [-1, -1, -1]
    assert _slice_of_pfp(np.zeros(0, np.int32), np.zeros(0, np.int64)).shape == (0,)


# ---- product tag rule / trigger ---------------------------------------------------------

def test_tag_of():
    from pylarevd.web._tags import tag_of
    assert tag_of("recob::Hits_hitfd__Reco") == "hitfd"
    assert tag_of("hitfd") == "hitfd"
    assert tag_of("a_b") == "b"


def test_disambiguated_hits_skips_a_label_without_underscore():
    from pylarevd.web.trigger import _disambiguated_hits
    good = types.SimpleNamespace(channels_split_across_wires=True)
    asked = []

    def hits(tag):
        asked.append(tag)
        if tag != "hitfd":
            raise KeyError(tag)
        return good
    ev = types.SimpleNamespace(
        geometry=types.SimpleNamespace(wrapped_channels=True), hits=hits,
        _src=types.SimpleNamespace(hit_products=lambda: ["weird", "recob::Hits_hitfd__Reco"]))
    assert _disambiguated_hits(ev, None) is good and asked == ["weird", "hitfd"]


# ---- render: the decoded event is released ------------------------------------------------

@have_fd
def test_render_export_releases_the_event(cache_env, monkeypatch):
    from pylarevd.web import cache, render
    released = []
    ef = cache._open(FD)
    orig = ef.release
    monkeypatch.setattr(ef, "release", lambda entry=None: (released.append(entry), orig(entry)))
    p = render.parse_export({"kind": "png", "view": "optical"})
    res = render.render_export(cache.canonical(FD), 0, None, None, p)
    assert res["ok"], res
    assert released == [0] and 0 not in ef._events
    assert not hasattr(render, "_FILES")            # no second, never-released EventFile
    # a failed export releases too
    res = render.render_export(cache.canonical(FD), 10**6, None, None, p)
    assert not res["ok"] and released == [0, 10**6]


# ---- cache: magic --------------------------------------------------------------------------

def test_cache_read_uses_the_bundle_magic(tmp_path):
    from pylarevd.web import bundle, cache
    good, bad = tmp_path / "a.evdb", tmp_path / "b.evdb"
    good.write_bytes(bundle.bundle_bytes("event", {}, {"x": np.arange(3, dtype=np.int32)}))
    bad.write_bytes(b"EVDX" + b"\0" * 12)
    assert cache._read(str(good))[:4] == bundle.MAGIC and cache._read(str(bad)) is None
