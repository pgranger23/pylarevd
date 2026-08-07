"""Pure-python reader for art-ROOT (LArSoft) event files.

art writes ``std::vector<T>`` data products *member-wise* (ROOT's
``kStreamedMemberWise`` bit, 0x4000).  uproot's automatic model assumes the
object-wise layout and raises ``ValueError: ... has the wrong number of bytes``,
so we decode the member-wise stream ourselves.

Member-wise is columnar: after the header, member 1 is stored contiguously for
all N objects, then member 2, and so on.  Decoding is therefore a handful of
numpy views with no per-object work, which is what makes an interactive display
practical without ROOT.

The layout is derived from the streamer info stored in the file itself, so this
generalises to any product whose members are PODs or nested POD structs
(recob::Hit, recob::SpacePoint, ...) rather than being hard-coded per class.

On-disk layout of one member-wise vector entry::

    byte count (4) | version|0x4000 (2) | value-class version (2) | N (4)
    [ member 1 for all N objects ][ member 2 for all N objects ] ...

A *flat* member's block is a plain contiguous array of N values.  A *nested
object* member (e.g. ``geo::WireID``) is written object-wise within its block,
and every class in its inheritance chain contributes a 6-byte
``(byte count, version)`` prefix -- so its sub-members are read with a stride.
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass

import numpy as np
import uproot


def _patch_fsspec_numpy_offsets() -> None:
    """Let uproot read over XRootD (``root://``).

    ``TBranch.basket_chunk_cursor`` takes the basket offset straight out of the
    ``fBasketSeek`` numpy array, so it hands uproot's source a ``numpy.int64``.
    Local sources slice happily with that; XRootD's binding does not, and the
    read dies with ``TypeError: integer argument expected for offset``.  The
    coercion below is what uproot omits -- it cannot change behaviour, since
    ``int()`` on a python int is the identity.

    Applied once, at import, and only if the installed uproot still needs it.
    """
    try:
        from uproot.source.fsspec import FSSpecSource
    except ImportError:                      # older uproot, no fsspec backend
        return
    if getattr(FSSpecSource, "_pylarevd_int_coercion", False):
        return
    chunk, chunks = FSSpecSource.chunk, FSSpecSource.chunks

    def _chunk(self, start, stop, **kw):
        return chunk(self, int(start), int(stop), **kw)

    def _chunks(self, ranges, notifications):
        return chunks(self, [(int(a), int(b)) for a, b in ranges], notifications)

    FSSpecSource.chunk, FSSpecSource.chunks = _chunk, _chunks
    FSSpecSource._pylarevd_int_coercion = True


_patch_fsspec_numpy_offsets()

# ROOT TStreamerElement fType codes -> (numpy dtype, on-disk size in bytes).
# ROOT always writes big-endian.
_ROOT_TYPES: dict[int, tuple[str, int]] = {
    1: (">i1", 1),    # char
    2: (">i2", 2),    # short
    3: (">i4", 4),    # int
    4: (">i8", 8),    # long
    5: (">f4", 4),    # float
    6: (">i4", 4),    # counter
    8: (">f8", 8),    # double
    # Double32_t is a double in memory but written as a 32-bit float on disk,
    # unless the member declares a packing range (handled explicitly below).
    9: (">f4", 4),
    11: (">u1", 1),   # unsigned char
    12: (">u2", 2),   # unsigned short
    13: (">u4", 4),   # unsigned int
    14: (">u8", 8),   # unsigned long
    15: (">u4", 4),   # bits
    16: (">i8", 8),   # Long64_t
    17: (">u8", 8),   # ULong64_t
    18: (">?", 1),    # bool
    19: (">f2", 2),   # Float16_t
}

_BASE = 0                 # TStreamerBase
_TOBJECT_BASE = 66        # a TObject base: a fixed 10-byte block, not a member
_TOBJECT_SIZE = 10        # version (2) + fUniqueID (4) + fBits (4)
_OBJECT_ANY = (61, 62)    # kObject / kAny: nested object stored by value
_OBJ_HEADER = 6           # byte count (4) + version (2)
# "Foreign" classes -- those without a ClassDef, such as every ROOT::Math
# vector type -- are written with a 4-byte checksum after the version, because
# their version number alone does not identify the layout.  ROOT reports their
# class version as 1, which is what we key on.
_FOREIGN_HEADER = 10      # byte count (4) + version (2) + checksum (4)
_MEMBERWISE_BIT = 0x4000
_VEC_HEADER = 12          # bytecount 4 + version 2 + class version 2 + count 4


class ArtReadError(RuntimeError):
    """Raised when an art product cannot be decoded."""


class VariableLayout(Exception):
    """Internal: this class has a member with no fixed on-disk size."""

    def __init__(self, classname: str, member: str):
        super().__init__(f"{classname}.{member} is variable-length")
        self.classname = classname
        self.member = member


@dataclass(frozen=True)
class Column:
    """One scalar member, addressed relative to its enclosing block."""

    name: str      # dotted C++ path, e.g. "fWireID.Wire"
    dtype: str     # big-endian numpy dtype string
    offset: int    # byte offset within the enclosing block's per-object record
    size: int


@dataclass(frozen=True)
class Block:
    """One top-level member of the vector's element type.

    In member-wise order this occupies ``n * itemsize`` consecutive bytes.
    ``itemsize == columns[0].size`` and a single column means a flat member.
    """

    name: str
    itemsize: int
    columns: tuple[Column, ...]

    @property
    def is_flat(self) -> bool:
        return len(self.columns) == 1 and self.columns[0].size == self.itemsize


@dataclass(frozen=True)
class RecordLayout:
    classname: str
    blocks: tuple[Block, ...]

    @property
    def itemsize(self) -> int:
        return sum(b.itemsize for b in self.blocks)


def _streamer_info(file, classname: str):
    try:
        info = file.streamer_named(classname)
    except Exception as exc:  # pragma: no cover - file dependent
        raise ArtReadError(f"no streamer info for {classname!r}") from exc
    if info is None:
        raise ArtReadError(f"no streamer info for {classname!r}")
    return info


def _streamer_elements(file, classname: str):
    return _streamer_info(file, classname).elements


def _header_size(file, classname: str, ctx: dict) -> int:
    """Bytes of prefix a nested object of *classname* carries on disk.

    A class written with a checksum instead of a version has a 10-byte prefix
    rather than 6.  ROOT reports such classes at version 1 -- but so do plenty
    of ordinary versioned classes (``TObject``, ``TNamed``, ``TArray`` all sit
    at version 1 with a real ClassDef), so version alone cannot decide it.

    Rather than guess, every version-1 class is recorded as *ambiguous* and the
    choice is settled against the actual on-disk record size in
    :meth:`ArtFile._resolve_layout`.  ``ctx["foreign"]`` carries the hypothesis
    currently being tested; ``None`` means "assume all of them are foreign",
    which is the common case and is tried first.
    """
    try:
        version = int(_streamer_info(file, classname).class_version)
    except ArtReadError:
        return _OBJ_HEADER
    if version != 1:
        return _OBJ_HEADER
    ctx["ambiguous"].add(classname)
    foreign = ctx["foreign"]
    if foreign is None or classname in foreign:
        return _FOREIGN_HEADER
    return _OBJ_HEADER


_DOUBLE32 = 9


def _element_scalars(el, path: str, base_offset: int) -> tuple[list[Column], int]:
    """Columns and byte width for one POD streamer element.

    Fixed-size arrays (``Double32_t fXYZ[3]``) become one column per index,
    named ``fXYZ[0]``, ``fXYZ[1]``, ...
    """
    ftype = int(el.member("fType"))
    dt, sz = _ROOT_TYPES[ftype]

    if ftype == _DOUBLE32:
        # A declared range means the value is bit-packed into an integer, which
        # needs the range to invert. Refuse rather than return wrong numbers.
        factor = float(el.member("fFactor", none_if_missing=True) or 0.0)
        if factor != 0.0:
            raise ArtReadError(
                f"{path}: packed Double32_t (fFactor={factor}) is not supported")

    length = int(el.member("fArrayLength") or 0)
    if length <= 1:
        return [Column(path, dt, base_offset, sz)], sz
    cols = [Column(f"{path}[{i}]", dt, base_offset + i * sz, sz) for i in range(length)]
    return cols, sz * length


def _nested_columns(file, classname: str, prefix: str,
                    ctx: dict) -> tuple[list[Column], int]:
    """Columns and total size of a nested object written object-wise.

    The object and each of its base classes carry their own header.
    """
    columns: list[Column] = []
    offset = _header_size(file, classname, ctx)
    for el in _streamer_elements(file, classname):
        ftype = int(el.member("fType"))
        if ftype == _TOBJECT_BASE or (ftype == _BASE and el.name == "TObject"):
            # Fixed-size: treating it as variable would push perfectly
            # columnar classes (anything holding a TVector3) onto the slow path.
            offset += _TOBJECT_SIZE
        elif ftype == _BASE:
            sub, size = _nested_columns(file, el.name, prefix, ctx)
            columns += [Column(c.name, c.dtype, offset + c.offset, c.size) for c in sub]
            offset += size
        elif ftype in _ROOT_TYPES:
            cols, width = _element_scalars(el, f"{prefix}{el.name}", offset)
            columns += cols
            offset += width
        elif ftype in _OBJECT_ANY:
            sub, size = _nested_columns(file, el.typename, f"{prefix}{el.name}.", ctx)
            columns += [Column(c.name, c.dtype, offset + c.offset, c.size) for c in sub]
            offset += size
        else:
            raise VariableLayout(classname, el.name)
    return columns, offset


def build_layout(file, classname: str,
                 foreign: set[str] | None = None) -> tuple[RecordLayout, set[str]]:
    """Derive the member-wise on-disk layout of ``std::vector<classname>``.

    Returns the layout and the set of version-1 classes whose header size could
    not be determined from streamer info alone (see :func:`_header_size`).
    """
    ctx = {"foreign": foreign, "ambiguous": set()}
    blocks: list[Block] = []
    for el in _streamer_elements(file, classname):
        ftype = int(el.member("fType"))
        name = el.name
        if ftype in _ROOT_TYPES:
            cols, width = _element_scalars(el, name, 0)
            blocks.append(Block(name, width, tuple(cols)))
        elif ftype in _OBJECT_ANY or ftype == _BASE:
            cls = el.typename if ftype in _OBJECT_ANY else el.name
            prefix = f"{name}." if ftype in _OBJECT_ANY else ""
            cols, size = _nested_columns(file, cls, prefix, ctx)
            blocks.append(Block(name, size, tuple(cols)))
        else:
            # A variable-length member (an STL container, or an object holding
            # one) has no fixed stride, so the whole product must go through
            # the sequential reader instead of the columnar fast path.
            raise VariableLayout(classname, name)
    return RecordLayout(classname, tuple(blocks)), ctx["ambiguous"]


def _skip_object(buf: bytes, pos: int) -> int:
    """Skip a whole ROOT object using its byte-count prefix."""
    bytecount = struct.unpack(">I", buf[pos:pos + 4])[0] & ~0x40000000
    return pos + 4 + bytecount


def _parse_event_auxiliary(buf: bytes) -> tuple[int, int, int]:
    """Extract (run, subrun, event) from a serialised art::EventAuxiliary."""
    pos = _OBJ_HEADER                 # into EventAuxiliary
    pos = _skip_object(buf, pos)      # past processHistoryID_ (Hash<2>, holds a string)
    pos += _OBJ_HEADER                # into EventID
    pos += _OBJ_HEADER                # into SubRunID
    pos += _OBJ_HEADER                # into RunID
    run, subrun, event = struct.unpack(">III", buf[pos:pos + 12])
    return int(run), int(subrun), int(event)


def _native(a: np.ndarray) -> np.ndarray:
    """Byte-swap to native order so downstream numpy/plotly is fast."""
    return a.astype(a.dtype.newbyteorder("="), copy=False)


class ArtFile:
    """An art-ROOT file, read without ROOT, art or LArSoft."""

    _BASKET_CACHE = 8         # branches in flight during a render, roughly

    def __init__(self, path: str):
        self.path = str(path)
        try:
            self._file = uproot.open(self.path)
        except (OSError, ValueError, KeyError) as exc:
            # uproot raises FileNotFoundError/OSError for a missing or non-ROOT
            # file. Callers contract on ArtReadError, so translate rather than
            # letting a raw traceback escape.
            raise ArtReadError(f"cannot open {self.path}: {exc}") from exc
        if "Events" not in self._file:
            raise ArtReadError(f"{self.path}: no 'Events' tree - not an art file?")
        self.events = self._file["Events"]
        self._layouts: dict[str, RecordLayout] = {}
        self._ambiguous: dict[str, set[str]] = {}
        self._baskets: dict[tuple, object] = {}

    def close(self) -> None:
        """Release the underlying file handle."""
        self._baskets.clear()
        self._file.close()

    def __enter__(self) -> "ArtFile":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"<ArtFile {self.path!r} nevents={self.num_events}>"

    # ---- introspection -------------------------------------------------

    @property
    def num_events(self) -> int:
        return self.events.num_entries

    def geometry_config(self) -> dict[str, str]:
        """The ``Geometry`` service settings the file was produced with.

        art embeds the job's full fhicl configuration in the file as a SQLite
        database (``RootFileDB``), so the detector is recorded in the data
        rather than having to be inferred from channel numbers or a filename.
        Returns e.g. ``{"Name": "dune10kt_v6", "GDML": "dune10kt_v6_refactored.gdml"}``,
        or ``{}`` if the file does not carry one.
        """
        try:
            key = self._file.key("RootFileDB")
            source = self._file.file.source
            start = int(key.data_cursor.index)
            blob = bytes(source.chunk(
                start, start + int(key.data_compressed_bytes)).raw_data[:])
        except Exception:
            return {}
        if not blob.startswith(b"SQLite format 3\x00"):
            return {}       # compressed or a layout we do not know; not fatal
        import sqlite3
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
            tmp.write(blob)
            tmp.flush()
            try:
                db = sqlite3.connect(tmp.name)
                rows = db.execute("select PSetBlob from ParameterSets").fetchall()
            except sqlite3.Error:
                return {}
            finally:
                try:
                    db.close()
                except Exception:
                    pass
        for (raw,) in rows:
            text = raw if isinstance(raw, str) else raw.decode("utf-8", "replace")
            if 'service_type:"Geometry"' not in text:
                continue
            out = {}
            for field in ("Name", "GDML"):
                m = re.search(field + r':"([^"]+)"', text)
                if m:
                    out[field] = m.group(1)
            if out:
                return out
        return {}

    def products(self, pattern: str | None = None) -> list[str]:
        names = [k for k in self.events.keys(recursive=False) if "_" in k]
        if pattern:
            rx = re.compile(pattern)
            names = [n for n in names if rx.search(n)]
        return sorted(names)

    def find_product(self, classname: str, tag: str | None = None) -> list[str]:
        """Branches holding ``std::vector<classname>``.

        art names them ``<class>s_<module>_<instance>_<process>.``
        """
        stem = classname + "s_"
        out = [n for n in self.events.keys(recursive=False) if n.startswith(stem)]
        if tag:
            out = [n for n in out if n[len(stem):].split("_")[0] == tag]
        return sorted(out)

    def event_ids(self) -> np.ndarray:
        """(run, subrun, event) per entry.

        uproot cannot deserialise ``art::EventAuxiliary`` (its bundled model is
        a different version from the one art writes), so the record is walked
        directly.  ROOT prefixes every object with its own byte count, which
        makes the walk robust to the variable-length process-history hash and
        to version differences in the surrounding classes.

        Layout::

            EventAuxiliary  hdr | Hash<2> hdr+string | EventID hdr
              -> SubRunID hdr -> RunID hdr | run (4) | subRun (4) | event (4)

        Returns zeros for any entry that cannot be parsed rather than failing:
        event identity is display metadata, not something to lose hits over.
        """
        branch = self.events["EventAuxiliary"]
        n = branch.num_entries
        out = np.zeros(n, dtype=[("run", "i4"), ("subrun", "i4"), ("event", "i4")])
        for i in range(n):
            try:
                out[i] = _parse_event_auxiliary(self._entry_bytes(branch, i))
            except Exception:
                out[i] = (0, 0, i)
        return out

    # ---- decoding ------------------------------------------------------

    def layout(self, classname: str) -> RecordLayout:
        if classname not in self._layouts:
            lay, amb = build_layout(self._file.file, classname)
            self._layouts[classname] = lay
            self._ambiguous[classname] = amb
        return self._layouts[classname]

    def _resolve_layout(self, classname: str, n: int, nbytes: int) -> RecordLayout:
        """Pick the layout whose record size matches the bytes actually stored.

        Version-1 classes are ambiguous between a 6- and a 10-byte header
        (see :func:`_header_size`).  The on-disk size settles it: try each
        hypothesis and keep the one that reproduces the entry length exactly.
        The common case -- all of them foreign -- is what `layout()` already
        built, so this normally costs nothing.
        """
        lay = self.layout(classname)
        if n == 0 or _VEC_HEADER + n * lay.itemsize == nbytes:
            return lay

        ambiguous = sorted(self._ambiguous.get(classname, ()))
        if not ambiguous or len(ambiguous) > 12:
            return lay          # nothing to vary, or too many to search
        matches = []
        for bits in range(1 << len(ambiguous)):
            subset = {c for i, c in enumerate(ambiguous) if bits >> i & 1}
            cand, _ = build_layout(self._file.file, classname, foreign=subset)
            if _VEC_HEADER + n * cand.itemsize == nbytes:
                matches.append((subset, cand))
        if len(matches) == 1:
            subset, cand = matches[0]
            self._layouts[classname] = cand
            return cand
        return lay              # ambiguous or unresolvable; decode() will raise

    def _obj_branch(self, product: str):
        base = product.rstrip(".")
        try:
            return self.events[f"{base}./{base}.obj"]
        except uproot.KeyInFileError as exc:
            raise ArtReadError(f"{product}: no '.obj' sub-branch") from exc

    def read_assns(self, product: str, entry: int) -> "Assns":
        """Decode an ``art::Assns`` product into paired index arrays."""
        from . import streamers as st
        raw = self._entry_bytes(self._obj_branch(product), entry)
        cur = st.Cursor(raw, 0)
        cur.header()                    # the Assns object itself
        cur.header()                    # art::detail::AssnsBase (empty)
        left_key, left_id, nl = _read_ptr_vector(raw, cur)
        right_key, right_id, nr = _read_ptr_vector(raw, cur)
        if nl != nr:
            raise ArtReadError(
                f"{product}: association sides disagree ({nl} vs {nr} pointers)")
        return Assns(product.rstrip("."), left_key, right_key, left_id, right_id)

    def assns_products(self, left: str | None = None, right: str | None = None,
                       *, plain_first: bool = True) -> list[str]:
        """Association branches, optionally filtered by the classes they link.

        art names them ``<L><R><D>art::Assns_<module>_<instance>_<process>.``
        With ``plain_first`` the ``void`` (no extra data) variants come first --
        an Assns carrying data has a third vector this reader does not model.
        """
        out = []
        for name in self.events.keys(recursive=False):
            if "art::Assns" not in name:
                continue
            head = name.split("art::Assns")[0]
            if left is not None:
                if not head.startswith(left):
                    continue
                head_rest = head[len(left):]
            else:
                head_rest = head
            if right is not None:
                if not head_rest.startswith(right):
                    continue
                head_rest = head_rest[len(right):]
            out.append(name)
        if plain_first:
            out.sort(key=lambda n: (0 if "voidart::Assns" in n else 1, n))
        else:
            out.sort()
        return out

    def read(self, product: str, classname: str, entry: int) -> dict:
        """Decode one event's ``std::vector<classname>``.

        Returns a dict of parallel numpy columns for fixed-size classes; for
        classes carrying containers, the variable members come back as lists
        (one entry per object) alongside the numpy columns.
        """
        raw = self._entry_bytes(self._obj_branch(product), entry)
        return self.decode(raw, classname, where=f"{product}[{entry}]")

    def _basket(self, branch, basket_i: int):
        """One decompressed basket, remembered between calls.

        ``branch.basket()`` re-reads every time. On a local file that is a
        memmap slice and costs nothing, but over xrootd it is a network
        round-trip -- so walking 120 entries of one branch became 120 sequential
        fetches (~6 s) where one fetch would do. Baskets hold many entries, so
        a handful of slots also covers prev/next browsing.
        """
        key = (id(branch), basket_i)
        hit = self._baskets.get(key)
        if hit is None:
            if len(self._baskets) >= self._BASKET_CACHE:
                self._baskets.pop(next(iter(self._baskets)))
            hit = self._baskets[key] = branch.basket(basket_i)
        return hit

    def _entry_bytes(self, branch, entry: int) -> bytes:
        offsets = branch.entry_offsets
        if entry < 0 or entry >= offsets[-1]:
            raise IndexError(f"entry {entry} out of range (0..{offsets[-1] - 1})")
        basket_i = int(np.searchsorted(offsets, entry, side="right") - 1)
        basket = self._basket(branch, basket_i)
        local = entry - offsets[basket_i]
        if basket.byte_offsets is None:
            raise ArtReadError("basket has no entry offsets; cannot split entries")
        start = int(basket.byte_offsets[local])
        stop = int(basket.byte_offsets[local + 1])
        return bytes(memoryview(basket.data)[start:stop])

    def decode(self, buf: bytes, classname: str, *, where: str = "?") -> dict[str, np.ndarray]:
        if len(buf) < _VEC_HEADER:
            raise ArtReadError(f"{where}: truncated entry ({len(buf)} bytes)")
        version = struct.unpack(">H", buf[4:6])[0]
        if not version & _MEMBERWISE_BIT:
            raise ArtReadError(
                f"{where}: vector is object-wise (version={version}); this reader "
                "handles the member-wise layout art normally writes")
        n = struct.unpack(">I", buf[8:12])[0]

        try:
            layout = self._resolve_layout(classname, n, len(buf))
        except VariableLayout:
            return self._decode_sequential(buf, classname, n, where=where)
        expected = _VEC_HEADER + n * layout.itemsize
        if expected != len(buf):
            raise ArtReadError(
                f"{where}: layout mismatch for {classname} - {n} objects x "
                f"{layout.itemsize} B + {_VEC_HEADER} B header = {expected} B, "
                f"but the entry is {len(buf)} B")

        out: dict[str, np.ndarray] = {}
        pos = _VEC_HEADER
        for blk in layout.blocks:
            width = n * blk.itemsize
            if blk.is_flat:
                col = blk.columns[0]
                out[col.name] = _native(
                    np.frombuffer(buf, dtype=col.dtype, count=n, offset=pos))
            else:
                # Nested object: strided read past the per-object headers.
                raw = np.frombuffer(buf, dtype=np.uint8, count=width, offset=pos)
                rec = raw.reshape(n, blk.itemsize)
                for col in blk.columns:
                    chunk = rec[:, col.offset:col.offset + col.size]
                    out[col.name] = _native(
                        np.ascontiguousarray(chunk).view(col.dtype).reshape(n))
            pos += width
        return out

    def _decode_sequential(self, buf: bytes, classname: str, n: int,
                           *, where: str) -> dict:
        """Member-wise decode for classes carrying variable-length members.

        Still columnar -- fixed members keep the fast numpy path -- but a column
        without a fixed stride is walked object by object.
        """
        from . import streamers as st

        cur = st.Cursor(buf, _VEC_HEADER)
        out: dict = {}
        file = self._file.file
        for el in _streamer_elements(file, classname):
            ftype = int(el.member("fType"))
            name = el.name
            fixed = None
            if ftype in _ROOT_TYPES:
                fixed = _element_scalars(el, name, 0)
            elif ftype in (_BASE,) + _OBJECT_ANY:
                try:
                    cls = el.typename if ftype in _OBJECT_ANY else name
                    prefix = f"{name}." if ftype in _OBJECT_ANY else ""
                    ctx = {"foreign": None, "ambiguous": set()}
                    fixed = _nested_columns(file, cls, prefix, ctx)
                except VariableLayout:
                    fixed = None

            if fixed is not None:
                cols, width = fixed
                raw = np.frombuffer(buf, np.uint8, n * width, cur.pos)
                rec = raw.reshape(n, width)
                for col in cols:
                    chunk = rec[:, col.offset:col.offset + col.size]
                    out[col.name] = _native(
                        np.ascontiguousarray(chunk).view(col.dtype).reshape(n))
                cur.skip(n * width)
            else:
                out[name] = st.read_memberwise_column(file, el, cur, n)

        if cur.pos != len(buf):
            raise ArtReadError(
                f"{where}: sequential decode of {classname} consumed "
                f"{cur.pos} of {len(buf)} bytes")
        return out


# ---- associations ----------------------------------------------------------
#
# art::Assns<L, R, D> streams as two parallel vectors of art::Ptr -- the left
# and right sides of each association. An art::Ptr is dictionary-emitted as
# pair<art::RefCore, unsigned long>, and the vector is member-wise, so the pair
# is stored COLUMNAR: every RefCore first, then every key.
#
#     [ RefCore x N ][ key x N ]
#
# with each RefCore laid out as::
#
#     header (6) | ProductID header (6) | ProductID value (4) | transients (6)
#
# Both columns are fixed-stride, so an association decodes as two numpy views.
# The key is the index into the referenced collection; the ProductID says which
# collection that is.

_REFCORE_STRIDE = 22
_REFCORE_PRODUCTID_OFFSET = 12
_KEY_SIZE = 8


class Assns:
    """Decoded ``art::Assns``: parallel index arrays linking two collections."""

    __slots__ = ("product", "left_key", "right_key", "left_id", "right_id")

    def __init__(self, product, left_key, right_key, left_id, right_id):
        self.product = product
        self.left_key = left_key        # index into the left collection
        self.right_key = right_key      # index into the right collection
        self.left_id = left_id          # art ProductID of the left collection
        self.right_id = right_id

    def __len__(self) -> int:
        return len(self.left_key)

    def __repr__(self) -> str:
        return f"<Assns {self.product!r} pairs={len(self)}>"

    def group_by_right(self, n_right: int) -> list[np.ndarray]:
        """Left indices for each right object -- e.g. the hits of each track.

        Raises if ``n_right`` is too small to hold every association, rather
        than silently dropping the ones that do not fit.
        """
        if len(self.right_key) and self.right_key.max() >= n_right:
            raise ValueError(
                f"n_right={n_right} but the largest right index is "
                f"{int(self.right_key.max())}; entries would be dropped")
        order = np.argsort(self.right_key, kind="stable")
        keys = self.right_key[order]
        vals = self.left_key[order]
        edges = np.searchsorted(keys, np.arange(n_right + 1))
        return [vals[edges[i]:edges[i + 1]] for i in range(n_right)]

    def right_of_left(self, n_left: int, fill: int = -1) -> np.ndarray:
        """For each left object, one right index.

        Where a left object has several rights the last wins; use
        :meth:`group_by_right` or :attr:`left_key`/:attr:`right_key` directly
        when the multiplicity matters (35% of hits have more than one space
        point in a typical event).
        """
        out = np.full(n_left, fill, np.int64)
        inside = (self.left_key >= 0) & (self.left_key < n_left)
        out[self.left_key[inside]] = self.right_key[inside]
        return out

    def multiplicity(self) -> int:
        """How many left objects are associated with more than one right."""
        _, counts = np.unique(self.left_key, return_counts=True)
        return int((counts > 1).sum())


def _read_ptr_vector(buf: bytes, cur) -> tuple[np.ndarray, np.ndarray, int]:
    """Read one vector<art::Ptr> at the cursor: (keys, product ids, count)."""
    from .streamers import read_collection_header
    n, _foreign, _end = read_collection_header(cur)
    width = n * (_REFCORE_STRIDE + _KEY_SIZE)
    if cur.pos + width > len(buf):
        raise ArtReadError(
            f"association vector of {n} pointers overruns the entry "
            f"({cur.pos + width} > {len(buf)} bytes)")
    rec = np.frombuffer(buf, np.uint8, n * _REFCORE_STRIDE,
                        cur.pos).reshape(n, _REFCORE_STRIDE)
    ids = np.ascontiguousarray(
        rec[:, _REFCORE_PRODUCTID_OFFSET:_REFCORE_PRODUCTID_OFFSET + 4]
    ).view(">u4").reshape(n)
    keys = np.frombuffer(buf, ">u8", n, cur.pos + n * _REFCORE_STRIDE)
    cur.skip(width)
    return (keys.astype(np.int64), ids.astype(np.int64), n)
