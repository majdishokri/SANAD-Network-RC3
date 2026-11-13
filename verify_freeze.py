"""Verify exact SHA-256 bytes for the frozen SANAD RC3 GitHub snapshot."""
import json, hashlib, gzip
from pathlib import Path

ROOT=Path(__file__).resolve().parent
def sha256(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):
            h.update(chunk)
    return h.hexdigest()

def verify():
    m=json.loads((ROOT/'FREEZE_MANIFEST.json').read_text())
    for name,expected in m['files_sha256'].items():
        f=ROOT/name
        if not f.is_file(): raise RuntimeError(f'MISSING: {name}')
        got=sha256(f)
        if got!=expected: raise RuntimeError(f'SHA mismatch: {name}; expected {expected}; got {got}')
    files={p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file() and '.git' not in p.parts and '__pycache__' not in p.parts and p.suffix not in ('.pyc','.pyo') and p.name!='FREEZE_MANIFEST.json'}
    expected=set(m['files_sha256'])
    unexpected=files-expected
    if unexpected:raise RuntimeError(f'Unmanifested files: {sorted(unexpected)}')
    assert m['decision_threshold']==0.5
    archive=ROOT/'model'/'rc3_candidate.joblib.gz'
    raw=gzip.decompress(archive.read_bytes())
    if hashlib.sha256(raw).hexdigest()!=m['frozen_model_sha256']:
        raise RuntimeError('Original RC3 model bytes do not match frozen digest')
    print(f"FROZEN RC3 PASS — {len(expected)} files verified, threshold 0.50")

if __name__=='__main__': verify()
