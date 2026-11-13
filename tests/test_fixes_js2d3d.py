"""Regression tests for fixes in the 2-D / 3-D web views (pylarevd/web/static/view2d.js, view3d.js).

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
    m = re.search(rf"^function {name}\(", src, re.M)
    assert m, f"{name} not found in {path.name}"
    i = src.index("{", m.end())
    depth = 0
    for j in range(i, len(src)):
        depth += {"{": 1, "}": -1}.get(src[j], 0)
        if depth == 0:
            return src[m.start():j + 1]
    raise AssertionError("unbalanced braces")


@pytest.mark.skipif(NODE is None, reason="node not available")
@pytest.mark.parametrize("module", ["view2d.js", "view3d.js"])
def test_fmt_keeps_integer_zeros(module):
    fn = _function_source(STATIC / module, "fmt")
    values = [100, 120, 200, -500, 10, 50, 0, 0.5, 1.25, 12.5, 0.0123, 999, 1000, 2.5e-3]
    js = fn + f"\nconsole.log(JSON.stringify({json.dumps(values)}.map(fmt)));"
    out = json.loads(subprocess.run([NODE, "-e", js], capture_output=True, text=True, check=True).stdout)
    assert out == ["100", "120", "200", "-500", "10", "50", "0", "0.5", "1.25", "12.5", "0.0123",
                   "999", "1.0e+3", "2.5e-3"]
