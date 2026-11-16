"""SANAD QA1: input-contract, atomic provenance and archive safety checks.

QA1 is an engineering correction, *not* a qualified change to the frozen v34
research candidate. Full scientific qualification requires new, unused seeds.
"""
from __future__ import annotations
import hashlib
import json
import os
import stat
import tempfile
import zipfile
from pathlib import Path
from typing import Mapping
import numpy as np

N_REPORTS=300
from geometry_config import N_ARRAYS, N_ELEMENTS


def canonical_digest(data: Mapping) -> str:
    """Hash a JSON record excluding its own integrity key."""
    payload={k:v for k,v in data.items() if k!='record_sha256'}
    return hashlib.sha256(json.dumps(payload,sort_keys=True,
                separators=(',',':'),allow_nan=False,ensure_ascii=True).encode()).hexdigest()


def seal_record(record: Mapping) -> dict:
    body=dict(record)
    body['record_sha256']=canonical_digest(body)
    return body


def verify_record(record: Mapping) -> None:
    if not isinstance(record,dict) or not isinstance(record.get('record_sha256'),str):
        raise ValueError('Record missing QA integrity digest')
    if record['record_sha256']!=canonical_digest(record):
        raise ValueError('Record SHA-256 mismatch; checkpoint must not be reused')


def atomic_record(path, record: Mapping):
    """Atomically commit canonical, sealed JSON in the destination filesystem."""
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    record=seal_record(record)
    fd,tmp=tempfile.mkstemp(prefix=f'.{path.name}.',suffix='.part',dir=path.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf8') as f:
            json.dump(record,f,sort_keys=True,indent=2,allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp,path)
    except BaseException:
        if os.path.exists(tmp):os.unlink(tmp)
        raise
    return record


def read_record(path):
    with open(path,encoding='utf8') as f:r=json.load(f)
    verify_record(r)
    return r


def safe_extract_source(zip_path, dest, *,max_files=120,max_member=3_000_000,max_total=25_000_000):
    """Reject path traversal, links, duplicate destinations and archive bombs.

    Source bundles use flat ASCII filenames only. Never writes outside `dest`.
    """
    dest=Path(dest)
    dest.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        members=z.infolist()
        if not members or len(members)>max_files:
            raise ValueError('Invalid number of ZIP members')
        seen=set()
        total=0
        for m in members:
            name=m.filename
            p=Path(name)
            if (m.is_dir() or p.name!=name or p.is_absolute() or '\\' in name
                or name.startswith('.') or not name.isascii() or not name.replace('_','').replace('-','').replace('.','').isalnum()
                or name in seen):
                raise ValueError('Unsafe or duplicate source member')
            seen.add(name)
            kind=(m.external_attr >> 16) & 0o170000
            if kind not in (0,stat.S_IFREG):
                raise ValueError('Symlink or nonregular source member')
            total+=m.file_size
            if m.file_size>max_member or total>max_total:
                raise ValueError('ZIP member size limit exceeded')
        for m in members:
            out=dest/m.filename
            with z.open(m) as src,open(out,'xb') as dst:
                count=0
                while True:
                    chunk=src.read(1024*1024)
                    if not chunk:break
                    count+=len(chunk)
                    if count>max_member:raise ValueError('ZIP inflated beyond limit')
                    dst.write(chunk)
    return sorted(seen)


def validate_trace(trace: Mapping, stream: str, *, require_truth: bool = True) -> None:
    """Enforce evidence and ground-truth schema at the measurement boundary.

    Allows v11 *candidate* names even when v11 confirmation is false, as in
    the original generator. Truth fields never enter `EvidenceFusionV34.step`.
    """
    if stream not in ('healthy','injected'):
        raise ValueError('Unsupported stream')
    keys=[f'{stream}_{s}' for s in ('v11','v11_named','feats')]
    if require_truth:keys += [f'{stream}_{s}' for s in ('onset','expected')]
    if not all(k in trace for k in keys):
        raise ValueError('Missing evidence or truth field')
    raw,names,feats=[np.asarray(trace[k]) for k in keys[:3]]
    if raw.shape!=(N_REPORTS,N_ARRAYS) or names.shape!=(N_REPORTS,N_ARRAYS) or feats.shape!=(2,N_REPORTS,N_ARRAYS,3):
        raise ValueError('Incorrect evidence/truth dimensions')
    if (not np.issubdtype(raw.dtype,np.number) or not np.isin(raw,[0,1]).all()
        or not np.issubdtype(names.dtype,np.number) or not np.isfinite(names).all()
        or not np.equal(names,np.floor(names)).all() or (names<0).any() or (names>N_ELEMENTS).any()):
        raise ValueError('Invalid raw v11 flags / element identities')
    if (not np.issubdtype(feats.dtype,np.number) or
        np.isinf(feats).any()):
        raise ValueError('Invalid spatial evidence')
    finite=np.isfinite(feats)
    if np.any(np.logical_and(finite.any(-1),~finite.all(-1))):
        raise ValueError('Partially missing spatial feature; require complete triplets')
    active=finite.all(-1)
    score=feats[:,:,:,0];dom=feats[:,:,:,1];element=feats[:,:,:,2]
    if ((score[active]<0).any() or (dom[active]<0).any() or
        (element[active]<1).any() or (element[active]>N_ELEMENTS).any() or
        not np.equal(element[active],np.floor(element[active])).all()):
        raise ValueError('Invalid spatial feature value or element ID')
    if not require_truth:return
    onset,expected=[np.asarray(trace[f'{stream}_{k}']) for k in ('onset','expected')]
    if onset.shape!=(N_ARRAYS,) or expected.shape!=(N_ARRAYS,):
        raise ValueError('Incorrect truth dimensions')
    if (not np.issubdtype(onset.dtype,np.number) or not np.issubdtype(expected.dtype,np.number)
        or not np.isfinite(onset).all() or not np.isfinite(expected).all()
        or not np.equal(onset,np.floor(onset)).all()
        or not np.equal(expected,np.floor(expected)).all()):
        raise ValueError('Invalid ground truth')
    if stream=='healthy':
        if not ((onset==-1).all() and (expected==-1).all()):
            raise ValueError('Healthy controls include fault labels')
    else:
        faults=expected>0
        if (faults.sum()!=2 or not np.all((expected[faults]>=1)&(expected[faults]<=N_ELEMENTS))
            or not np.all((onset[faults]>=0)&(onset[faults]<N_REPORTS))
            or not np.all((onset[~faults]==-1)&(expected[~faults]==-1))):
            raise ValueError('Injected truth must include exactly two plausible fault cases')
