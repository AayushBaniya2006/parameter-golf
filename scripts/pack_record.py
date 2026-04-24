#!/usr/bin/env python3
"""Pack readable source → LZMA+b85 wrapper matching PR #1493's format.

Usage: python3 scripts/pack_record.py <in.py> <out.py>
"""
import base64 as B
import lzma as L
import sys


WRAPPER_PREFIX = (
    'import lzma as L,base64 as B\n'
    'exec(L.decompress(B.b85decode("'
)
WRAPPER_SUFFIX = (
    '"),format=L.FORMAT_RAW,filters=[{"id":L.FILTER_LZMA2}]))'
)


def pack(in_path: str, out_path: str) -> int:
    src = open(in_path, "rb").read()
    compressed = L.compress(
        src,
        format=L.FORMAT_RAW,
        filters=[{"id": L.FILTER_LZMA2, "preset": 9 | L.PRESET_EXTREME}],
    )
    encoded = B.b85encode(compressed).decode("ascii")
    wrapper = WRAPPER_PREFIX + encoded + WRAPPER_SUFFIX
    with open(out_path, "w") as f:
        f.write(wrapper)
    return len(wrapper)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: pack_record.py <in.py> <out.py>", file=sys.stderr)
        sys.exit(1)
    n = pack(sys.argv[1], sys.argv[2])
    print(f"wrote {n} bytes to {sys.argv[2]}")
