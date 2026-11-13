"""The viewer server must not leave its decode pool behind when it is stopped."""
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.parse
import urllib.request

import pytest

FD = os.environ.get("PYLAREVD_ATMNU",
                    "/home/pigrange/projects/data/reco/atmnu_radio_18643_1_20260727T121224Z_reco.root")
needs_fd = pytest.mark.skipif(not os.path.exists(FD), reason="FD sample absent")
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _forkservers():
    out = subprocess.run(["ps", "-ww", "-u", str(os.getuid()), "-o", "pid=,cmd="],
                         capture_output=True, text=True).stdout
    return {int(l.split()[0]) for l in out.splitlines() if "multiprocessing.forkserver" in l}


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _run_and_stop(sig, tmp_path, settle):
    port = _free_port()
    before = _forkservers()
    env = dict(os.environ, PYLAREVD_CACHE=str(tmp_path))
    proc = subprocess.Popen([sys.executable, "-m", "pylarevd.web", FD, "--port", str(port),
                             "--workers", "1", "--prefetch", "0"],
                            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(240):
            try:
                _OPENER.open(base + "/api/status", timeout=2).read()
                break
            except Exception:
                time.sleep(0.5)
        q = urllib.parse.quote(FD, safe="")
        _OPENER.open(f"{base}/api/event?file={q}&entry=2", timeout=600).read()   # starts the pool
        started = _forkservers() - before
        assert started, "the build did not start a pool"
    finally:
        proc.send_signal(sig)
        proc.wait(60)
    deadline = time.time() + settle
    left = started
    while time.time() < deadline:
        left = _forkservers() & started
        if not left:
            break
        time.sleep(0.5)
    for pid in left:                     # never leak from the test itself
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    return left


@needs_fd
@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT])
def test_stopping_the_server_stops_its_pool(sig, tmp_path):
    assert not _run_and_stop(sig, tmp_path, settle=10)


@needs_fd
def test_pool_workers_exit_when_the_server_is_killed(tmp_path):
    assert not _run_and_stop(signal.SIGKILL, tmp_path, settle=20)


def test_a_broken_pool_is_shut_down_not_just_forgotten():
    from pylarevd.web.server import App

    class Pool:
        calls = []

        def shutdown(self, **kw):
            self.calls.append(kw)
    app = App([], workers=1, prefetch=0)
    app._pool = pool = Pool()
    app._drop_pool()
    assert app._pool is None and pool.calls == [{"wait": False, "cancel_futures": True}]
