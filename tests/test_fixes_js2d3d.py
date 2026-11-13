"""Regression tests for fixes in the 2-D / 3-D web views (view2d.js, view3d.js; fmt now lives in util.js).

The browser-level fixes (labels in flash3d, 3-D categorical colours, wheel hover, tooltip placement)
are covered by the Playwright script regress/extra/js2d3d/fixes_check.py; the pure helpers are run
here under node.
"""
import json
import pathlib
import re
import shutil
import subprocess

import pytest

STATIC = pathlib.Path(__file__).resolve().parents[1] / "pylarevd" / "web" / "static"
NODE = shutil.which("node")


def _function_source(path, name):
    """Source of the top-level `function name(...) {...}` in a JS file (brace matched)."""
    src = path.read_text()
    m = re.search(rf"^(?:export )?function {name}\(", src, re.M)
    assert m, f"{name} not found in {path.name}"
    i = src.index("{", m.end())
    depth = 0
    for j in range(i, len(src)):
        depth += {"{": 1, "}": -1}.get(src[j], 0)
        if depth == 0:
            return src[m.start():j + 1].removeprefix("export ")
    raise AssertionError("unbalanced braces")


@pytest.mark.parametrize("module", ["view2d.js", "view3d.js"])
def test_views_use_the_shared_fmt(module):
    """Both views format numbers with util.js's fmt, not a local copy that could drift."""
    src = (STATIC / module).read_text()
    assert re.search(r"import \{[^}]*\bfmt\b[^}]*\} from './util\.js'", src)
    assert not re.search(r"^function fmt\(", src, re.M)


@pytest.mark.skipif(NODE is None, reason="node not available")
def test_fmt_keeps_integer_zeros():
    fn = _function_source(STATIC / "util.js", "fmt")
    values = [100, 120, 200, -500, 10, 50, 0, 0.5, 1.25, 12.5, 0.0123, 999, 1000, 2.5e-3]
    js = fn + f"\nconsole.log(JSON.stringify({json.dumps(values)}.map(fmt)));"
    out = json.loads(subprocess.run([NODE, "-e", js], capture_output=True, text=True, check=True).stdout)
    assert out == ["100", "120", "200", "-500", "10", "50", "0", "0.5", "1.25", "12.5", "0.0123",
                   "999", "1.0e+3", "2.5e-3"]
