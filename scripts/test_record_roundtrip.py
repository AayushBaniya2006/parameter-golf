#!/usr/bin/env python3
"""Verify pack_record + extract_record is a perfect byte-for-byte round-trip.

Run any time the wrapper format or scripts change. No GPU needed.

Usage: python3 scripts/test_record_roundtrip.py [<sample.py>]
       (defaults to train_gpt_src.py at repo root)
"""
import os
import subprocess
import sys
import tempfile


def main(sample: str) -> int:
    if not os.path.exists(sample):
        print(f"FAIL: sample file not found: {sample}", file=sys.stderr)
        return 1
    with open(sample, "rb") as f:
        original = f.read()
    here = os.path.dirname(os.path.abspath(__file__))
    pack = os.path.join(here, "pack_record.py")
    extract = os.path.join(here, "extract_record.py")
    with tempfile.TemporaryDirectory() as d:
        wrapped = os.path.join(d, "wrapped.py")
        roundtrip = os.path.join(d, "roundtrip.py")
        subprocess.run([sys.executable, pack, sample, wrapped], check=True)
        subprocess.run([sys.executable, extract, wrapped, roundtrip], check=True)
        with open(roundtrip, "rb") as f:
            recovered = f.read()
    if recovered != original:
        print(f"FAIL: bytes differ. orig={len(original)} recovered={len(recovered)}", file=sys.stderr)
        return 1
    wrapped_size = os.path.getsize(wrapped) if os.path.exists(wrapped) else "?"
    print(f"OK: round-trip clean ({len(original)} src bytes; wrapper output verified)")
    return 0


if __name__ == "__main__":
    default = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "train_gpt_src.py")
    sample = sys.argv[1] if len(sys.argv) > 1 else default
    sys.exit(main(sample))
