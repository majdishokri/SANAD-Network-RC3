"""Verify all bit-for-bit copied frozen RC3 source/model files."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def verify():
    manifest=json.loads((ROOT/"SOURCE_SHA256.json").read_text())
    for filename,expected in manifest["frozen_files_sha256"].items():
        actual=hashlib.sha256((ROOT/filename).read_bytes()).hexdigest()
        if actual!=expected:
            raise RuntimeError(f"Frozen source/model SHA mismatch: {filename}")
    return len(manifest["frozen_files_sha256"])

if __name__=="__main__":
    print(f"PASS: {verify()} frozen source and model files match the original release.")
