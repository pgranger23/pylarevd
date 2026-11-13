"""Product label -> module tag, the one rule shared by the server, exporter and trigger code.

Kept dependency-free so the server can use it without importing the exporter.
"""


def tag_of(product) -> str:
    """``"recob::Hits_hitfd__Reco"`` -> ``"hitfd"``; a label without ``_`` is its own tag."""
    parts = str(product).split("_")
    return parts[1] if len(parts) > 1 else str(product)
