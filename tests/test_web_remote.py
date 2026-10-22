"""Remote (root://) data files and the --allow-open-under limit of the web viewer server."""
import gzip
import json
import os
import time
from urllib.parse import quote

import pytest

from test_web_server import FD, Srv, cache_env, get, have_fd  # noqa: F401  (fixture)

# e.g. root://eosuser.cern.ch//eos/user/<u>/<user>/some_reco.root (needs a Kerberos ticket)
XRD = os.environ.get("PYLAREVD_XROOTD_FILE")
have_xrd = pytest.mark.skipif(not XRD, reason="set PYLAREVD_XROOTD_FILE to a root:// art file")


def post_open(base, path):
    return get(base, "/api/open", data=json.dumps({"path": path}).encode(),
               headers={"Content-Type": "application/json"}, method="POST")


def test_under_prefix(tmp_path):
    from pylarevd.web.server import under_prefix
    d = tmp_path / "data"
    (d / "sub").mkdir(parents=True)
    (tmp_path / "secret.root").write_bytes(b"")
    os.symlink(tmp_path / "secret.root", d / "link.root")
    pre = [str(d)]
    assert under_prefix(str(d / "a.root"), pre)
    assert under_prefix(str(d / "sub" / "b.root"), pre)
    assert not under_prefix(str(tmp_path / "secret.root"), pre)
    assert not under_prefix(str(d / ".." / "secret.root"), pre)
    assert not under_prefix(str(d / "link.root"), pre)          # symlink out of the tree
    assert not under_prefix(str(d) + "x/a.root", pre)           # sibling sharing the prefix
    eos = ["root://eosuser.cern.ch//eos/user/p/pigrange/rockmu/"]
    assert under_prefix("root://eosuser.cern.ch//eos/user/p/pigrange/rockmu/x.root", eos)
    assert under_prefix("root://EOSUSER.cern.ch/eos//user/p/pigrange/rockmu/x.root", eos)
    assert not under_prefix("root://eosuser.cern.ch//eos/user/p/pigrange/rockmu/../x.root", eos)
    assert not under_prefix("root://eosuser.cern.ch//eos/user/p/pigrange/rockmux/x.root", eos)
    assert not under_prefix("root://other.cern.ch//eos/user/p/pigrange/rockmu/x.root", eos)
    assert not under_prefix("root://eosuser.cern.ch//eos/user/p/pigrange/rockmu/x.root?a=b", eos)
    assert not under_prefix(str(d / "a.root"), eos)             # local path, URL-only prefixes


def test_canonical_and_url_cache_key(cache_env):
    from pylarevd.web import cache
    url = "root://eosuser.cern.ch//eos/user/x/someone/f.root"
    assert cache.canonical(url) == url
    assert cache.canonical("rel/f.root") == os.path.abspath("rel/f.root")
    cache._remote_stats[url] = (time.monotonic(), (1_000_000_000, 123))
    d1 = cache.cache_dir(url)
    assert os.path.isdir(d1) and d1.startswith(str(cache_env))
    cache._remote_stats[url] = (time.monotonic(), (2_000_000_000, 123))   # file replaced
    assert cache.cache_dir(url) != d1
    with pytest.raises(OSError):        # not remembered -> a real stat, which fails here
        cache._remote_stat("root://eosuser.cern.ch//eos/user/x/someone/does-not-exist.root")


def test_expand_inputs_local_dir(tmp_path):
    from pylarevd.web.server import expand_inputs
    for n in ("b.root", "a.root", "notes.txt"):
        (tmp_path / n).write_bytes(b"")
    assert expand_inputs([str(tmp_path), "/x/y.root"]) == [
        str(tmp_path / "a.root"), str(tmp_path / "b.root"), "/x/y.root"]
    assert expand_inputs(["/eos/user/p/pigrange/f.root"])[0].startswith(
        ("root://eosuser.cern.ch/", "/eos/"))                   # /eos mounted or not


def test_open_limited_to_prefixes(cache_env, tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    os.symlink("/etc/hostname", allowed / "out.root")          # link out of the tree
    pre = [str(allowed), os.path.dirname(FD)]
    s = Srv([], host="0.0.0.0", workers=1, prefetch=0, open_prefixes=pre)
    try:
        st = json.loads(get(s.base, "/api/status")[2])
        assert st["allow_open"] is True and st["open_prefixes"] == pre
        assert post_open(s.base, "/etc/hostname")[0] == 403
        assert post_open(s.base, str(allowed / "out.root"))[0] == 403
        assert post_open(s.base, str(allowed) + "/../../../../etc/hostname")[0] == 403
        assert post_open(s.base, str(allowed / "missing.root"))[0] == 404
        if os.path.exists(FD):
            stc, _, b = post_open(s.base, FD)
            assert stc == 200 and json.loads(b)["n_events"] > 0
    finally:
        s.close()


def test_open_unrestricted_on_loopback_reports_no_prefixes(cache_env):
    s = Srv([], workers=1, prefetch=0, open_prefixes=["/somewhere"])
    try:
        st = json.loads(get(s.base, "/api/status")[2])
        assert st["allow_open"] is True and st["open_prefixes"] == []
    finally:
        s.close()


@have_xrd
def test_remote_file_over_xrootd(cache_env):
    s = Srv([XRD], workers=1, prefetch=0)
    try:
        files = json.loads(get(s.base, "/api/files")[2])
        assert files[0]["path"] == XRD and files[0]["n_events"] > 0
        st, h, b = get(s.base, "/api/geometry?file=" + quote(XRD, safe=""))
        assert st == 200
        st, h, b = get(s.base, "/api/event?file=" + quote(XRD, safe="") + "&entry=0",
                       {"Accept-Encoding": "gzip"})
        assert st == 200
        raw = gzip.decompress(b) if h.get("Content-Encoding") == "gzip" else b
        assert raw[:4] == b"EVDB"
        st = json.loads(get(s.base, "/api/status")[2])
        assert st["files"][0]["cached_entries"] == [0]
    finally:
        s.close()


def test_unreadable_inputs_do_not_stop_the_server(cache_env, tmp_path):
    s = Srv([str(tmp_path / "missing.root")], workers=1, prefetch=0)
    try:
        st = json.loads(get(s.base, "/api/status")[2])
        assert st["files"] == [] and "missing.root" in st["startup_errors"][0]
        assert json.loads(get(s.base, "/api/files")[2]) == []
    finally:
        s.close()
