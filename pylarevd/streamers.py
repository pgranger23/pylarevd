"""Sequential ROOT deserialiser for variable-length members.

:mod:`pylarevd.artio` decodes member-wise vectors as numpy slices, which is
fast but requires every member to have a fixed on-disk size.  That covers hits
and space points; it does not cover the products that carry containers --
``recob::Track`` (a trajectory of N points), ``recob::Shower``,
``recob::PFParticle``, ``simb::MCParticle``, ``recob::Wire``.

Those still arrive member-wise, so the columnar structure holds: what changes is
that a variable member's column cannot be strided over.  Empirically (verified
byte-exactly against dunesw output) such a column is written as::

    byte count (4) | version (2) | [count (4)][elements] x N objects

i.e. one header for the whole column, then a length-prefixed run per object.
This module walks that layout with a cursor instead of slicing it.

Nested objects carry their own byte-count prefix, which is read rather than
predicted -- so unlike the fast path this code never has to guess whether a
class uses a 6- or 10-byte header.
"""

from __future__ import annotations

import re
import struct
import weakref
from dataclasses import dataclass

import numpy as np

# C++ spelling -> (numpy dtype, on-disk size). ROOT writes big-endian.
_PRIMITIVES: dict[str, tuple[str, int]] = {
    "bool": (">?", 1), "char": (">i1", 1), "signed char": (">i1", 1),
    "unsigned char": (">u1", 1), "short": (">i2", 2), "unsigned short": (">u2", 2),
    "int": (">i4", 4), "unsigned int": (">u4", 4),
    "long": (">i8", 8), "unsigned long": (">u8", 8),
    "long long": (">i8", 8), "unsigned long long": (">u8", 8),
    "Long64_t": (">i8", 8), "ULong64_t": (">u8", 8),
    "float": (">f4", 4), "double": (">f8", 8),
    "Double32_t": (">f4", 4),      # stored as a float unless a range is declared
    "Float16_t": (">f2", 2),
    "size_t": (">u8", 8),
}

_BASE, _OBJECT, _ANY, _TOBJECT_BASE, _STL = 0, 61, 62, 66, 500
_OBJ_HEADER = 6
_TOBJECT_SIZE = 10             # version (2) + fUniqueID (4) + fBits (4)
_BYTECOUNT_MASK = 0x40000000


class StreamerError(RuntimeError):
    """Raised when a byte stream does not match the streamer description."""


#: Key added to a decoded object when some member could not be modelled.
UNPARSED = "__unparsed__"

# Caches, scoped PER FILE. Keying them by class name alone would let one file's
# schema decide how another file's identically-named class is read -- and since
# a partial memo is never retried, that silently loses members. A weak map also
# means closing a file drops its caches.
_FILE_CACHE: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def _caches(file) -> dict:
    try:
        entry = _FILE_CACHE.get(file)
    except TypeError:                       # not weak-referenceable
        entry = None
    if entry is None:
        entry = {"partial": {}, "why": {}, "elements": {}}
        try:
            _FILE_CACHE[file] = entry
        except TypeError:
            pass
    return entry


def reset_caches(file=None) -> None:
    """Forget cached streamer info and partial-parse decisions."""
    if file is None:
        _FILE_CACHE.clear()
    else:
        _FILE_CACHE.pop(file, None)


class Cursor:
    """A position in a buffer, with primitive readers."""

    __slots__ = ("buf", "pos")

    def __init__(self, buf: bytes, pos: int = 0):
        self.buf = buf
        self.pos = pos

    def skip(self, n: int) -> None:
        self.pos += n

    def u1(self) -> int:
        v = self.buf[self.pos]
        self.pos += 1
        return v

    def u2(self) -> int:
        v = struct.unpack_from(">H", self.buf, self.pos)[0]
        self.pos += 2
        return v

    def u4(self) -> int:
        v = struct.unpack_from(">I", self.buf, self.pos)[0]
        self.pos += 4
        return v

    def array(self, dtype: str, count: int, size: int) -> np.ndarray:
        a = np.frombuffer(self.buf, dtype, count, self.pos)
        self.pos += count * size
        return a.astype(np.dtype(dtype).newbyteorder("="), copy=True)

    def header(self) -> tuple[int, int]:
        """Read a (byte count, version) prefix; returns (end position, version)."""
        raw = self.u4()
        version = self.u2()
        if raw & _BYTECOUNT_MASK:
            return self.pos - _OBJ_HEADER + 4 + (raw & ~_BYTECOUNT_MASK), version
        return -1, version          # no byte count available

    def string(self) -> str:
        """ROOT's length-prefixed string: one byte, or 255 then four."""
        n = self.u1()
        if n == 255:
            n = self.u4()
        s = self.buf[self.pos:self.pos + n]
        self.pos += n
        return s.decode("utf-8", "replace")


# ---- type-name parsing -----------------------------------------------------

def _split_template(arg: str) -> list[str]:
    """Split a template argument list at top-level commas."""
    out, depth, cur = [], 0, ""
    for ch in arg:
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


def parse_type(typename: str) -> tuple[str, list[str]]:
    """``vector<pair<int,float> >`` -> ``("vector", ["pair<int,float>"])``."""
    t = typename.strip().rstrip("*").strip()
    m = re.match(r"^([A-Za-z_][\w:]*)\s*<(.*)>$", t, re.S)
    if not m:
        return t, []
    return m.group(1), _split_template(m.group(2))


_SEQUENCES = {"vector", "list", "deque", "set", "multiset", "unordered_set"}


# ---- readers ---------------------------------------------------------------

def read_value(file, typename: str, cur: Cursor):
    """Read one value of *typename* at the cursor."""
    base = typename.strip()
    if base in _PRIMITIVES:
        dt, size = _PRIMITIVES[base]
        return cur.array(dt, 1, size)[0]
    if base in ("string", "std::string", "TString"):
        return cur.string()

    kind, args = parse_type(base)
    kind = kind.replace("std::", "")
    if kind in _SEQUENCES:
        return read_sequence(file, args[0], cur, headered=True)
    if kind == "pair":
        return (read_bare(file, args[0], cur), read_bare(file, args[1], cur))
    if kind == "map":
        return read_sequence(file, f"pair<{args[0]},{args[1]}>", cur, headered=True)
    return read_object(file, base, cur)


def read_bare(file, typename: str, cur: Cursor):
    """Read a value with no object prefix (as inside a std::pair)."""
    base = typename.strip()
    if base in _PRIMITIVES:
        dt, size = _PRIMITIVES[base]
        return cur.array(dt, 1, size)[0]
    if base in ("string", "std::string", "TString"):
        return cur.string()
    kind, _args = parse_type(base)
    if kind.replace("std::", "") in _SEQUENCES | {"pair", "map"}:
        return read_value(file, base, cur)
    # Class members of a pair carry their own prefix in every case seen so far,
    # but fall back to a bare read rather than assume it.
    save = cur.pos
    try:
        return read_object(file, base, cur)
    except (StreamerError, struct.error, ValueError, IndexError):
        cur.pos = save
        return read_object(file, base, cur, headered=False)


def _as_pair(o):
    """Normalise a decoded std::pair to a 2-tuple.

    The element-wise path yields a tuple already; the member-wise path yields a
    dict whose halves are either plain ``first``/``second`` keys (POD members)
    or flattened ``first.<member>``/``second.<member>`` ones (object members).
    Callers should not have to know which layout the file used.
    """
    if not isinstance(o, dict):
        return o
    first, second = o.get("first"), o.get("second")
    a, b = {}, {}
    for key, value in o.items():
        if key.startswith("first."):
            a[key[len("first."):]] = value
        elif key.startswith("second."):
            b[key[len("second."):]] = value
    return (a or first, b or second)


def read_collection_header(cur: Cursor) -> tuple[int, bool, int]:
    """Read a container prefix: (element count, value class is foreign, end).

    When the member-wise bit is set the value class's version follows, and a
    version of 0 means that class is foreign, so a 4-byte checksum stands in
    for it.

    ``end`` is where the container's byte count says it finishes.  That is the
    only reliable way to tell whether its elements carry per-element headers:
    the foreign flag does NOT decide it (``ROOT::Math::PositionVector3D``
    elements are headered, ``lar::sparse_vector::datarange_t`` elements are
    not, and both report version 0).
    """
    end, version = cur.header()
    foreign = False
    if version & 0x4000:
        value_version = cur.u2()
        if value_version == 0:
            cur.skip(4)         # checksum in place of a version
            foreign = True
    return cur.u4(), foreign, end


def read_sequence(file, elem_type: str, cur: Cursor, *, headered: bool):
    """Read a length-prefixed run of *elem_type*.

    ``headered`` says whether the container carries its own prefix.  Inside a
    member-wise column it does not -- the column has a single shared prefix and
    each object contributes only its count.
    """
    end = -1
    if headered:
        n, _foreign, end = read_collection_header(cur)
    else:
        n = cur.u4()
    et = elem_type.strip()
    # Every element costs at least one byte, so a count larger than the bytes
    # remaining cannot be real -- it is a misread of a wrong framing guess.
    # Rejecting it here matters: without the check the reader faithfully parses
    # the phantom elements until the buffer runs out, which turned one
    # simb::MCTruth (15 trajectories) into 2.35 million TLorentzVector reads and
    # never finished.
    budget = (end if end >= 0 else len(cur.buf)) - cur.pos
    if n < 0 or n > max(budget, 0):
        raise StreamerError(
            f"vector<{et}>: element count {n} exceeds the {budget} bytes left")
    if et in _PRIMITIVES:                       # fast path: contiguous PODs
        dt, size = _PRIMITIVES[et]
        return cur.array(dt, n, size)

    if n == 0:
        return []
    kind = parse_type(et)[0].replace("std::", "")
    composite = kind in _SEQUENCES | {"pair", "map"}

    # How class elements are framed is not derivable from streamer info, so try
    # each layout and keep the one that lands exactly on the container's end.
    # Member-wise (column per member) first: that is what ROOT writes when the
    # member-wise bit is set, and it is the only reading that recovers e.g.
    # multi-ROI wire waveforms.
    start = cur.pos
    # "memberwise" = one column per member (what ROOT writes for the member-wise
    # bit, including for std::pair elements, whose two halves become two
    # columns); "elementwise" = each element read whole, in order.
    attempts = [("memberwise", "mw"), ("elementwise", "ew")] if composite else \
               [("memberwise", "mw"), ("headerless", False), ("headered", True)]
    last_error = None
    for label, how in attempts:
        cur.pos = start
        try:
            if how == "mw":
                out = read_memberwise_elements(file, et, cur, n)
            elif how == "ew":
                out = [read_value(file, et, cur) for _ in range(n)]
            else:
                out = [read_object(file, et, cur, headered=how) for _ in range(n)]
        except (StreamerError, struct.error, ValueError, IndexError) as exc:
            last_error = exc
            continue
        if end < 0 or cur.pos == end:
            # A pair reaches us as {"first","second"} from the member-wise path
            # and as a tuple from the element-wise one; normalise so callers
            # never have to care which layout the file happened to use.
            if kind == "pair":
                out = [_as_pair(o) for o in out]
            return out
        last_error = StreamerError(
            f"{et} as {label}: ended at {cur.pos}, container ends at {end}")
    raise StreamerError(f"vector<{et}>: cannot parse ({last_error})")


def read_memberwise_elements(file, classname: str, cur: Cursor, n: int) -> list:
    """Read *n* objects of *classname* stored member-wise: one column per member.

    This is ROOT's real member-wise layout, applied recursively -- the same rule
    the top-level product vector uses.  A base class contributes its own columns
    inline; a nested-object member's column holds *n* objects written
    object-wise, each with its own prefix (which is why ``geo::WireID`` is 41
    bytes inside ``recob::Hit``, and why a ``PositionVector3D`` column holds
    22-byte ``Cartesian3D`` objects).
    """
    columns: dict = {}
    for el in _elements(file, classname):
        ftype = el.ftype
        name = el.name
        if ftype == _TOBJECT_BASE or (ftype == _BASE and name == "TObject"):
            cur.skip(_TOBJECT_SIZE * n)
        elif ftype == _BASE:
            base_rows = read_memberwise_elements(file, name, cur, n)
            for key in (base_rows[0].keys() if base_rows else ()):
                columns[key] = [row[key] for row in base_rows]
        elif ftype == _STL:
            columns[name] = read_memberwise_column(file, el, cur, n)
        elif ftype in (_OBJECT, _ANY):
            # Flatten nested members to "outer.inner" exactly as read_object
            # does, so a caller sees the same keys whichever layout the file
            # used.
            objs = [read_object(file, el.typename, cur) for _ in range(n)]
            for key in (objs[0].keys() if objs else ()):
                columns[f"{name}.{key}"] = [o[key] for o in objs]
        else:
            from .artio import _ROOT_TYPES
            if ftype not in _ROOT_TYPES:
                raise StreamerError(f"{name}: unsupported fType={ftype}")
            dt, size = _ROOT_TYPES[ftype]
            length = el.array_length
            if length <= 1:
                columns[name] = cur.array(dt, n, size)
            else:
                columns[name] = cur.array(dt, n * length, size).reshape(n, length)
    return [{k: v[i] for k, v in columns.items()} for i in range(n)]


def read_object(file, classname: str, cur: Cursor, *, headered: bool = True) -> dict:
    """Read one object of *classname*, returning a flat dict of members.

    The object's byte count is read from the stream and used to check that
    parsing consumed exactly the right number of bytes, so a layout mistake
    surfaces here rather than as corrupt numbers downstream.

    Whether a by-value member carries its own prefix is not recorded in
    streamer info, so both readings are tried and the one that fits the byte
    count is kept.  If neither does, the byte count still says exactly where
    the object ends: we keep the members read up to the first unmodelled one,
    resynchronise, and flag the shortfall under :data:`UNPARSED`.  Losing a
    flag word must not cost us a whole trajectory.
    """
    end = -1
    if headered:
        end, version = cur.header()
        # Version 0 means the class is foreign: a 4-byte checksum stands in for
        # the version. Read that from the stream rather than predicting it.
        if version == 0:
            cur.skip(4)

    body = cur.pos
    elements = _elements(file, classname)

    memo = _caches(file)

    # Always attempt the FULL parse, for every object.
    #
    # This used to be skipped once a partial-parse limit had been memoised for
    # the class -- which made decoding order-dependent and silently lossy: the
    # same class can be framed differently at different call sites (simb::MCParticle
    # parses whole inside MCTruth::fPartList but not inside MCNeutrino::fNu), so
    # one failure taught the reader to truncate *every* later object of that
    # class. Visiting one event then cost every subsequent event its truth.
    # The memo below now short-circuits only the expensive probe, never the
    # parse itself, so a readable object is always read in full.
    last_error = None
    for nested_headered in (True, False):
        cur.pos = body
        out: dict = {}
        try:
            for el in elements:
                _read_element(file, el, cur, out, end, nested_headered)
        except (StreamerError, struct.error, ValueError, IndexError) as exc:
            last_error = exc
            continue
        # Require the byte count to be consumed EXACTLY. Accepting
        # cur.pos <= end silently tolerates a truncated parse, which is how
        # whole members (recob::Wire's waveforms) went missing while every
        # object still looked like it decoded cleanly.
        if end < 0 or cur.pos == end:
            return out
        last_error = StreamerError(
            f"{classname}: consumed {cur.pos - body} bytes, "
            f"byte count says {end - body}")
        if end < 0:
            break
    if end < 0:
        raise StreamerError(f"{classname}: cannot parse ({last_error})")

    # This object genuinely does not parse. Fall back to reading the leading
    # members and resynchronising on the byte count. How many are readable is a
    # property of the class, so that much is worth remembering.
    limit = memo["partial"].get(classname)
    if limit is None:
        limit = _probe_limit(file, classname, elements, cur.buf, body, end)
        memo["partial"][classname] = limit
        memo["why"][classname] = str(last_error)

    out = {}
    cur.pos = body
    try:
        for el in elements[:limit]:
            _read_element(file, el, cur, out, end)
    except (StreamerError, struct.error, ValueError, IndexError):
        out = {}
    cur.pos = end
    out[UNPARSED] = memo["why"].get(classname, "unmodelled member")
    return out


def _probe_limit(file, classname: str, elements, buf: bytes, body: int,
                 end: int) -> int:
    """How many leading members of *classname* can be read before one fails."""
    probe = Cursor(buf, body)
    n = 0
    for el in elements:
        mark = probe.pos
        try:
            _read_element(file, el, probe, {}, end)
            if probe.pos > end:
                raise StreamerError("overran")
        except Exception:
            probe.pos = mark
            break
        n += 1
    return n


def _is_checksummed(file, classname: str) -> bool:
    try:
        info = file.streamer_named(classname)
    except Exception:
        return False
    return info is not None and int(info.class_version) == 1


@dataclass(frozen=True)
class ElementSpec:
    """The streamer facts we actually use, extracted once per class.

    Reading them off uproot's model object instead costs a ``member()`` lookup
    per element *per object*: that walk dominated the decode of nested products
    (``simb::MCTruth`` did not finish in ten minutes; it takes well under a
    second here).
    """

    name: str
    typename: str
    ftype: int
    array_length: int

    @classmethod
    def of(cls, el) -> "ElementSpec":
        """Adopt a raw uproot streamer element (no-op if already a spec)."""
        if isinstance(el, cls):
            return el
        return cls(name=el.name, typename=el.typename,
                   ftype=int(el.member("fType")),
                   array_length=int(el.member("fArrayLength") or 0))


def _elements(file, classname: str):
    """Streamer elements for *classname*, memoised per file.

    Without the memo this is re-resolved on every object at every nesting
    depth -- 125k lookups to decode eleven tracks, most of the decode time.
    """
    cache = _caches(file)["elements"]
    hit = cache.get(classname)
    if hit is None:
        info = file.streamer_named(classname)
        if info is None:
            raise StreamerError(f"no streamer info for {classname!r}")
        hit = cache[classname] = [ElementSpec.of(el) for el in info.elements]
    return hit


def _read_nested(file, typename: str, cur: Cursor, budget_end: int,
                 prefer_headered: bool = True) -> dict:
    """Read a nested object whose header may or may not be present.

    Whether ROOT writes a prefix for a by-value member depends on the member's
    type in ways streamer info does not record, and the obvious test -- does the
    next word look like a byte count? -- is useless here, because a plausible
    coordinate such as 36.85 has the byte-count bit set in its float encoding.

    So try it with a header and fall back without one, using the *enclosing*
    object's byte count to tell the two apart. That is authoritative: a wrong
    guess overruns a known boundary.
    """
    save = cur.pos
    try:
        sub = read_object(file, typename, cur, headered=prefer_headered)
        if budget_end >= 0 and cur.pos > budget_end:
            raise StreamerError("overran the enclosing object")
        return sub
    except (StreamerError, struct.error, ValueError, IndexError):
        cur.pos = save
        return read_object(file, typename, cur, headered=not prefer_headered)


def _read_element(file, el, cur: Cursor, out: dict, budget_end: int = -1,
                  nested_headered: bool = True) -> None:
    ftype = el.ftype
    name = el.name
    typename = el.typename

    if ftype == _TOBJECT_BASE:
        cur.skip(_TOBJECT_SIZE)
        return
    if ftype == _BASE:
        if name in ("TObject",):
            cur.skip(_TOBJECT_SIZE)
            return
        out.update(_read_nested(file, name, cur, budget_end, nested_headered))
        return
    if ftype == _STL:
        out[name] = read_value(file, typename, cur)
        return
    if ftype in (_OBJECT, _ANY):
        for k, v in _read_nested(file, typename, cur, budget_end,
                                 nested_headered).items():
            out[f"{name}.{k}"] = v
        return

    # plain data, possibly a fixed-size array
    from .artio import _ROOT_TYPES
    if ftype not in _ROOT_TYPES:
        raise StreamerError(f"{name}: unsupported streamer fType={ftype} ({typename})")
    dt, size = _ROOT_TYPES[ftype]
    length = el.array_length
    if length <= 1:
        out[name] = cur.array(dt, 1, size)[0]
    else:
        out[name] = cur.array(dt, length, size)


def read_memberwise_column(file, el, cur: Cursor, n: int) -> list:
    """Read one variable-length member for all *n* objects of a member-wise vector.

    The column carries a single (byte count, version) prefix; each object then
    contributes its own length-prefixed run.
    """
    el = ElementSpec.of(el)      # artio passes raw uproot elements
    ftype = el.ftype

    if ftype != _STL:
        # A nested object member is written object-wise within its column:
        # no shared prefix, but every object keeps its own byte-count header
        # (this is what makes geo::WireID 41 bytes inside recob::Hit).
        return [read_object(file, el.typename, cur) for _ in range(n)]

    # An STL member's column has one shared prefix, then a length-prefixed run
    # per object with no per-object header. The prefix carries the value-class
    # version (and a checksum when that is 0) exactly as a standalone container
    # does -- skipping those misaligns every object after the first.
    end, version = cur.header()
    if version & 0x4000:
        if cur.u2() == 0:
            cur.skip(4)
    kind, args = parse_type(el.typename)
    kind = kind.replace("std::", "")
    if el.typename.strip() in ("string", "std::string"):
        values = [cur.string() for _ in range(n)]
    elif kind in _SEQUENCES:
        values = [read_sequence(file, args[0], cur, headered=False)
                  for _ in range(n)]
    elif kind == "map":
        values = [read_sequence(file, f"pair<{args[0]},{args[1]}>", cur,
                                headered=False) for _ in range(n)]
    else:
        raise StreamerError(f"{el.name}: unsupported STL container {el.typename}")

    if end >= 0:
        if cur.pos > end:
            raise StreamerError(
                f"{el.name}: read {cur.pos - end} bytes past the end of the column")
        cur.pos = end
    return values
