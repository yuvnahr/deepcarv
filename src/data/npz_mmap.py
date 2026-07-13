"""
src/data/npz_mmap.py
=====================
Memory-mapped access to arrays stored inside an **uncompressed** ``.npz`` file.

Why this exists
---------------
``np.load("file.npz")["x"]`` materialises the entire array in RAM. For FFT-75 at
4096-byte fragments the training split alone is ~25 GB raw, which does not fit in
a Kaggle instance (~13 GB). Loading it and then upcasting (e.g. ``astype(int16)``)
doubles that again.

An ``.npz`` is just a ZIP container of ``.npy`` members. When written with
``np.savez`` (**not** ``savez_compressed``), those members are stored
uncompressed (``ZIP_STORED``), which means the raw array bytes sit contiguously
inside the file and can be memory-mapped directly with ``np.memmap``.

The OS then pages in only the slices actually read, so a DataLoader touching one
batch at a time keeps resident memory near-flat regardless of dataset size. This
turns "can't train on the full dataset" into "trains on the full dataset with
negligible RAM".

Caveats
-------
* Only works for **uncompressed** npz members. ``savez_compressed`` output cannot
  be mmapped; :func:`mmap_npz_member` returns ``None`` in that case so callers can
  fall back to a normal load (or re-save the file uncompressed).
* Reads are lazy: random access across a huge file is I/O bound. Keep the file on
  fast local disk (on Kaggle, ``/kaggle/working``, not ``/kaggle/input``, if you
  need repeated random reads — though input is usually fine).
"""

from __future__ import annotations

import struct
import zipfile
from pathlib import Path

import numpy as np

# Local file header: signature + 26 bytes of fixed fields, then name + extra.
_LOCAL_HEADER_STRUCT = "<IHHHHHIIIHH"
_LOCAL_HEADER_SIZE = 30
_LOCAL_HEADER_SIGNATURE = 0x04034B50


def _npy_data_offset(npz_path: Path, member_name: str, info: zipfile.ZipInfo) -> tuple[int, tuple[int, ...], np.dtype]:
    """Return (byte offset of raw array data, shape, dtype) for an npz member.

    Walks the ZIP local file header to find where the embedded ``.npy`` begins,
    then parses the ``.npy`` header to find where its raw data begins.
    """
    with open(npz_path, "rb") as fh:
        fh.seek(info.header_offset)
        raw = fh.read(_LOCAL_HEADER_SIZE)
        if len(raw) != _LOCAL_HEADER_SIZE:
            raise ValueError(f"Truncated ZIP local header in {npz_path}")
        fields = struct.unpack(_LOCAL_HEADER_STRUCT, raw)
        signature, name_len, extra_len = fields[0], fields[9], fields[10]
        if signature != _LOCAL_HEADER_SIGNATURE:
            raise ValueError(f"Bad ZIP local header signature in {npz_path}")

        npy_start = info.header_offset + _LOCAL_HEADER_SIZE + name_len + extra_len
        fh.seek(npy_start)

        version = np.lib.format.read_magic(fh)
        if version == (1, 0):
            shape, _fortran, dtype = np.lib.format.read_array_header_1_0(fh)
        elif version == (2, 0):
            shape, _fortran, dtype = np.lib.format.read_array_header_2_0(fh)
        else:
            raise ValueError(f"Unsupported .npy format version {version} in {member_name}")

        data_offset = fh.tell()

    return data_offset, shape, dtype


def _resolve_member(archive: zipfile.ZipFile, member: str) -> str | None:
    """Find a member by name, case-insensitively.

    FFT-75 files in the wild use both ``X.npy`` and ``x.npy``; matching only one
    case previously made the other silently unloadable.
    """
    target = f"{member}.npy".lower()
    for name in archive.namelist():
        if name.lower() == target:
            return name
    return None


def mmap_npz_member(npz_path: str | Path, member: str) -> np.memmap | None:
    """Memory-map one array from an uncompressed ``.npz``.

    Parameters
    ----------
    npz_path:
        Path to the ``.npz`` file.
    member:
        Array name inside the archive (e.g. ``"x"`` for the member ``x.npy``).

    Returns
    -------
    np.memmap | None
        A read-only memmap view of the array, or ``None`` if the member is
        compressed (and therefore not mmappable) or absent. Returning ``None``
        rather than raising lets callers fall back to a normal ``np.load``.
    """
    npz_path = Path(npz_path)

    try:
        with zipfile.ZipFile(npz_path) as archive:
            member_file = _resolve_member(archive, member)
            if member_file is None:
                return None
            info = archive.getinfo(member_file)
            if info.compress_type != zipfile.ZIP_STORED:
                # Compressed member: bytes are not contiguous on disk.
                return None
    except zipfile.BadZipFile:
        return None

    data_offset, shape, dtype = _npy_data_offset(npz_path, member_file, info)
    return np.memmap(npz_path, dtype=dtype, mode="r", offset=data_offset, shape=shape)


def is_mmappable(npz_path: str | Path, members: tuple[str, ...] = ("x", "y")) -> bool:
    """Whether every requested member of ``npz_path`` can be memory-mapped."""
    npz_path = Path(npz_path)
    if not npz_path.exists():
        return False
    try:
        with zipfile.ZipFile(npz_path) as archive:
            for member in members:
                member_file = _resolve_member(archive, member)
                if member_file is None:
                    return False
                if archive.getinfo(member_file).compress_type != zipfile.ZIP_STORED:
                    return False
    except zipfile.BadZipFile:
        return False
    return True


def rewrite_uncompressed(src: str | Path, dst: str | Path) -> Path:
    """Re-save a (possibly compressed) npz as an uncompressed, mmappable one.

    Useful one-off utility when a dataset was written with ``savez_compressed``:
    the compressed form trades disk space for the ability to stream it, which is
    the wrong trade when RAM — not disk — is the bottleneck.
    """
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with np.load(src) as data:
        arrays = {name: data[name] for name in data.files}
    np.savez(dst, **arrays)  # savez => ZIP_STORED => mmappable
    return dst
