#!/usr/bin/env python3
"""Verify original frozen SANAD RC3 model and runtime in the public distribution.

The immutable study manifest FREEZE_MANIFEST.json is retained unchanged.
Only the public landing README.md and .gitignore differ from the original
frozen commit. Unlike the historical whole-tree verifier, this public-mode
verifier allows additive docs, security workflows, and accelerated_v1/ while
requiring every original protected runtime, study and model byte to match its
original pinned SHA-256.

For historical *whole-tree* exact freeze, check out original commit
182e77b485cceb770ee6570ba332205c73823c91 and run its verify_freeze.py.
"""
from pathlib import Path
import gzip,hashlib,json

ROOT=Path(__file__).resolve().parent
MUTABLE_PUBLIC_PRESENTATION={".gitignore","README.md","verify_freeze.py"}
PROTECTED_PREFIXES=("model/","internal/","prospective/","qualification/","reports/")
PROTECTED_INDIVIDUAL={"rc3.py","requirements.txt","docs/FREEZE_POLICY.md"}


def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as src:
        for part in iter(lambda:src.read(1048576),b""):
            h.update(part)
    return h.hexdigest()

def verify():
    m=json.loads((ROOT/"FREEZE_MANIFEST.json").read_text())
    required=m["files_sha256"]
    if m["decision_threshold"]!=0.5:
        raise RuntimeError("The original RC3 operating point changed")
    if m["frozen_model_sha256"]!="0a90835bb57a56437fc09211902b581bb22377676f91dce68f309c92e3a95bf4":
        raise RuntimeError("Unexpected base RC3 model identity")
    checked=0
    for name,expected in required.items():
        file=ROOT/name
        if not file.is_file():
            raise RuntimeError(f"MISSING ORIGINAL FROZEN FILE: {name}")
        if name in MUTABLE_PUBLIC_PRESENTATION:
            continue
        actual=sha256(file)
        if actual!=expected:
            raise RuntimeError(f"PINNED ORIGINAL FILE CHANGED: {name}")
        checked+=1

    # Additive public resources must not silently extend/replace the frozen
    # signal model or frozen prospective evidence whose SHA set was preregistered.
    for p in ROOT.rglob("*"):
        if not p.is_file() or ".git" in p.parts or "__pycache__" in p.parts:
            continue
        path=p.relative_to(ROOT).as_posix()
        if (path.startswith(PROTECTED_PREFIXES) or path in PROTECTED_INDIVIDUAL):
            if path not in required:
                raise RuntimeError(f"UNMANIFESTED FILE IN FROZEN SCIENTIFIC SCOPE: {path}")

    compressed=ROOT/"model/rc3_candidate.joblib.gz"
    if sha256(compressed)!=m["compressed_model_sha256"]:
        raise RuntimeError("Compressed RC3 model SHA mismatch")
    raw=gzip.decompress(compressed.read_bytes())
    if hashlib.sha256(raw).hexdigest()!=m["frozen_model_sha256"]:
        raise RuntimeError("Unpacked RC3 model SHA mismatch")
    print(f"PUBLIC RELEASE INTEGRITY PASS — {checked} original protected files pinned, "
          "RC3 threshold 0.50 and model identical")
    print("Scope: main public documentation differs from immutable historical freeze; "
          "no new simulated or field performance claim")

if __name__=="__main__":
    verify()
