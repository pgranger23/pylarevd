"""``python -m pylarevd``.

``--check`` is answered before importing anything heavy: it exists to diagnose
missing dependencies, so it must not need them itself. Importing .cli pulls in
artio -> numpy/uproot, which is exactly what a tester without them lacks.
"""

import sys

if "--check" in sys.argv[1:]:
    from ._deps import report

    print(report())
    raise SystemExit(0)

from .cli import main

raise SystemExit(main())
