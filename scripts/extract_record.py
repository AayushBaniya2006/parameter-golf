#!/usr/bin/env python3
"""Extract LZMA-wrapped record train_gpt.py to readable source.

Usage: python3 scripts/extract_record.py <in.py> <out.py>
"""
import base64 as B
import lzma as L
import re
import sys


def extract(in_path: str, out_path: str) -> int:
    wrapped = open(in_path).read()
    m = re.search(r'B\.b85decode\("([^"]+)"\)', wrapped)
    if not m:
        raise ValueError(f"Not a b85+lzma-wrapped record: {in_path}")
    payload = m.group(1)
    source_bytes = L.decompress(
        B.b85decode(payload),
        format=L.FORMAT_RAW,
        filters=[{"id": L.FILTER_LZMA2}],
    )
    with open(out_path, "wb") as f:
        f.write(source_bytes)
    return len(source_bytes)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: extract_record.py <in.py> <out.py>", file=sys.stderr)
        sys.exit(1)
    n = extract(sys.argv[1], sys.argv[2])
    print(f"wrote {n} bytes to {sys.argv[2]}")
