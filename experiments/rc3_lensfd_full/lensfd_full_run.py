"""Full-archive LensFD source audit and frozen RC3 compatibility gate.

Read every byte of every regular archive member and every element in every numeric
HDF5 dataset. No sampling, no silent truncation, no fabricated RC3 features.
Source: Rice University LensFD (Chen et al., IEEE TVT 2023).
License: Rice grants non-exclusive/non-transferable use; raw data must not be redistributed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import shutil
import tarfile
import tempfile
from datetime import datetime, timezone
from itertools import product
from collections import Counter
from pathlib import Path
from typing import Iterator

import numpy as np

BLOCK_BYTES = 16 * 1024 * 1024
OFFICIAL_URL = 'https://renew.rice.edu/datasets/ashu/traces/LensFD/data_lensFD.tgz'
EXPECTED_FROZEN_COMMIT = '182e77b485cceb770ee6570ba332205c73823c91'
EXPECTED_MODEL_SHA256 = '0a90835bb57a56437fc09211902b581bb22377676f91dce68f309c92e3a95bf4'


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def hash_file(p: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(p).open('rb') as file:
        for b in iter(lambda: file.read(chunk_bytes), b''):
            h.update(b)
    return h.hexdigest()


def atomic_json(path: Path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w', encoding='utf8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)


def _json_line(path: Path, obj):
    with Path(path).open('a', encoding='utf8') as f:
        f.write(json.dumps(obj, ensure_ascii=False, allow_nan=False) + '\n')
        f.flush(); os.fsync(f.fileno())


def iter_dataset_slices(shape, itemsize, target_bytes=BLOCK_BYTES) -> Iterator[tuple]:
    """Cover the complete N-D logical HDF5 space exactly once, at bounded memory."""
    if not shape:
        yield ()
        return
    blocks = list(map(int, shape))
    if any(k <= 0 for k in blocks):
        return
    while math.prod(blocks) * max(1, itemsize) > target_bytes:
        largest = max(range(len(blocks)), key=lambda i: blocks[i])
        if blocks[largest] == 1:
            raise MemoryError('Single HDF5 element larger than target block')
        blocks[largest] = (blocks[largest] + 1) // 2
    ranges = [range(0, n, w) for n,w in zip(shape, blocks)]
    for start in product(*ranges):
        yield tuple(slice(s, min(s+b,n)) for s,b,n in zip(start,blocks,shape))


def _scan_numeric(ds) -> dict:
    dtype = ds.dtype
    count = int(math.prod(ds.shape)) if ds.shape else 1
    record = dict(shape=list(ds.shape), dtype=str(dtype), values=count)
    if dtype.kind not in ('b','i','u','f','c'):
        record.update(numeric_analyzed=False, values_scanned=0, reason='Non-plain-numeric HDF5 dtype; complete source bytes still SHA-256 verified')
        return record
    total = 0
    zero = 0
    finite = 0
    abs_sum = 0.0
    sq_sum = 0.0
    mn = float('inf'); mx = float('-inf')
    for sl in iter_dataset_slices(ds.shape, int(dtype.itemsize)):
        vals = np.asarray(ds[sl] if sl else ds[()]).reshape(-1)
        total += len(vals)
        valid = np.isfinite(vals)
        finite += int(np.count_nonzero(valid))
        good = vals[valid]
        if good.size:
            mag = np.abs(good.astype(np.complex128 if np.iscomplexobj(good) else np.float64))
            zero += int(np.count_nonzero(mag == 0))
            abs_sum += float(np.sum(mag, dtype=np.float64))
            sq_sum += float(np.sum(mag * mag, dtype=np.float64))
            mn = min(mn, float(np.min(mag)))
            mx = max(mx, float(np.max(mag)))
    if total != count:
        raise RuntimeError(f'Numeric coverage incomplete {ds.name}: {total}/{count}')
    record.update(numeric_analyzed=True,values_scanned=total,finite_values=finite,
                  nonfinite_values=total-finite,zero_values=zero,
                  mean_abs=abs_sum/finite if finite else None,
                  rms_abs=math.sqrt(sq_sum/finite) if finite else None,
                  min_abs=mn if finite else None,max_abs=mx if finite else None)
    return record


def inspect_hdf5(path: Path):
    import h5py
    rows=[]; n_numeric=0; n_analyzed=0; total_values=0
    with h5py.File(path, 'r') as f:
        def visit(name, item):
            nonlocal n_numeric,n_analyzed,total_values
            if not isinstance(item,h5py.Dataset):
                return
            stats=_scan_numeric(item)
            stats['dataset_path']='/'+name
            stats['attribute_names']=sorted(list(item.attrs.keys()))
            rows.append(stats)
            if stats['numeric_analyzed']:
                n_numeric+=1
                n_analyzed+=stats['values_scanned']
            total_values+=stats['values']
        f.visititems(visit)
        root_attrs=sorted(list(f.attrs.keys()))
    return {'hdf5_datasets':rows,'hdf5_dataset_count':len(rows),
            'numeric_dataset_count':n_numeric,'numeric_values_scanned':n_analyzed,
            'total_logical_values':total_values,'root_attribute_names':root_attrs}


def _safe_member_name(name):
    p=Path(name)
    if p.is_absolute() or '..' in p.parts or '\\' in name or not name or name.startswith('~'):
        raise ValueError('Unsafe archive member path')
    return str(p)


def _load_journal(journal):
    if not journal.exists():return {}
    records={}
    with journal.open(encoding='utf8') as f:
        for line in f:
            if line.strip():
                row=json.loads(line)
                # Append-only retry journal; most recent attempt for each member wins.
                records[row['name']]=row
    return records


def run_full_archive(archive: str | Path, output: str | Path, scratch_dir: str | Path | None = None, strict=True) -> dict:
    archive=Path(archive)
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    if not archive.is_file():raise FileNotFoundError(str(archive))
    arch_size=archive.stat().st_size
    arch_sha=hash_file(archive)  # every byte of the ~18-GB archive is hashed
    lock=output/'SOURCE_LOCK.json'
    if lock.exists():
        previous=json.loads(lock.read_text())
        if (previous['archive_sha256'],previous['archive_size']) != (arch_sha,arch_size):
            raise RuntimeError('Archive has changed; refusing to mix old and new checkpoint outputs')
    else:
        atomic_json(lock, {'official_source':OFFICIAL_URL, 'archive_size':arch_size,
                           'archive_sha256':arch_sha, 'created_utc':utcnow()})
    journal=output/'members.jsonl'
    prior=_load_journal(journal)
    seen=set(); errors=[]; dirs=0; fresh=0
    # stream tar/gzip, one member at a time; no extraction of the full archive
    with tarfile.open(archive,'r|gz') as tar:
        for member in tar:
            name=_safe_member_name(member.name)
            if member.isdir():
                dirs+=1
                continue
            if name in seen: raise RuntimeError(f'Duplicate regular archive member: {name}')
            seen.add(name)
            if name in prior and prior[name]['status']=='OK':
                if prior[name]['size']!=member.size:
                    raise RuntimeError(f'Invalid resume checkpoint: {name}')
                continue
            row={'name':name,'size':member.size,'mtime':member.mtime,'status':'ERROR'}
            tempname=None
            try:
                if not member.isfile():
                    # symbolic links carry no independently supplied data payload
                    row.update(status='ERROR',error='Unsupported non-regular archive object')
                else:
                    free=shutil.disk_usage(str(scratch_dir or tempfile.gettempdir())).free
                    if member.size+1024*1024*200>free:
                        raise OSError(f'Insufficient scratch disk for {name}, need {member.size:,}, free {free:,}')
                    inp=tar.extractfile(member)
                    if inp is None: raise RuntimeError('No member stream')
                    dig=hashlib.sha256()
                    with tempfile.NamedTemporaryFile(prefix='lensfd_',suffix=Path(name).suffix,
                        dir=str(scratch_dir) if scratch_dir else None,delete=False) as temp:
                        tempname=temp.name
                        remaining=member.size
                        while remaining:
                            data=inp.read(min(8*1024*1024,remaining))
                            if not data: raise EOFError(f'Truncated tar member {name}')
                            temp.write(data);dig.update(data);remaining-=len(data)
                    row['sha256']=dig.hexdigest()
                    suffix=Path(name).suffix.lower()
                    if suffix in ('.hdf5','.h5'):
                        row.update(inspect_hdf5(Path(tempname)))
                        row['analysis']='FULL_HDF5_NUMERIC_SCAN'
                    else:
                        row['analysis']='FULL_BYTE_SHA256_ONLY'
                        row['warning']='No numeric HDF5 decode available for this file type'
                    row['status']='OK'
            except Exception as exc:
                row.update(status='ERROR',error=f'{type(exc).__name__}: {exc}')
                errors.append({'name':name,'error':row['error']})
            finally:
                if tempname is not None:
                    try:Path(tempname).unlink(missing_ok=True)
                    except Exception:pass
            _json_line(journal,row)
            prior[name]=row
            fresh+=1
            print(f"[{len(seen)}] {row['status']:5} {name}  bytes={member.size:,}",flush=True)
    regular=[v for v in prior.values() if v['name'] in seen]
    complete=(len(regular)==len(seen) and all(v['status']=='OK' for v in regular))
    source_groups=Counter(Path(r['name']).name.split('-')[0].lower() for r in regular if r['status']=='OK')
    summary={'status':'COMPLETE_FULL_ARCHIVE_AUDIT' if complete else 'INCOMPLETE',
        'official_source':OFFICIAL_URL,'finished_utc':utcnow(),
        'archive_sha256':arch_sha,'archive_bytes_hashed':arch_size,
        'archive_members_regular':len(seen),'archive_members_directories':dirs,
        'source_filename_prefix_counts':dict(sorted(source_groups.items())),
        'members_checked':len(regular),'members_ok':sum(r['status']=='OK' for r in regular),
        'member_bytes_accounted':sum(r['size'] for r in regular if r['status']=='OK'),
        'hdf5_dataset_count':sum(r.get('hdf5_dataset_count',0) for r in regular if r['status']=='OK'),
        'hdf5_numeric_values_scanned':sum(r.get('numeric_values_scanned',0) for r in regular if r['status']=='OK'),
        'hdf5_total_logical_values':sum(r.get('total_logical_values',0) for r in regular if r['status']=='OK'),
        'hdf5_nonnumeric_dataset_count':sum(r.get('hdf5_dataset_count',0)-r.get('numeric_dataset_count',0) for r in regular if r['status']=='OK'),
        'bytes_sampled':0, 'inference_source':'RAW_REAL_RF_MEASUREMENTS',
        'rc3_qualification_on_lensfd':'NOT_YET_TESTED',
        'errors':errors + [{'name':r['name'],'error':r.get('error','unresolved')} for r in regular if r['status']!='OK' and r['name'] not in [e['name'] for e in errors]]}
    atomic_json(output/'FULL_AUDIT_SUMMARY.json',summary)
    if strict and not complete:raise RuntimeError('Full archive audit INCOMPLETE; see FULL_AUDIT_SUMMARY.json')
    return summary


def rc3_contract_report(output_dir:str|Path, frozen_repository_path:str|Path|None=None):
    """Fail closed: LensFD raw IQ is NOT the frozen RC3 300x7 observation trace.

    RC3 requires real upstream evidence streams produced by the frozen Sionna
    feature-generation pathway, not arbitrary tensor reshaping or synthetic rows.
    """
    reasons=[
        'RC3 input is an evidence NPZ of 300 reporting periods x 7 arrays with precomputed v11, fast/strict and temporal delta observations, not raw HDF5 IQ/CSI.',
        'LensFD self-interference recording covers up to 32 antenna ports in any run; seven eight-element RC3 arrays must not be fabricated from asynchronous recordings.',
        'LensFD source does not supply the frozen RC3 temporal feature stream, matching clock, 7-array baselines, or a machine-readable verified fault-label map by default.',
        'Transforming raw IQ/CSI into plausible arrays or random RC3 observation features would not constitute an RC3 evaluation.',
    ]
    out={'dataset':'Rice LensFD','mode':'FROZEN_RC3','decision_threshold':0.50,
         'expected_frozen_commit':EXPECTED_FROZEN_COMMIT,
         'model_sha256':EXPECTED_MODEL_SHA256,'input_contract':'NPZ [2,300,7,3] plus [300,7,3] and v11 [300,7] per stream',
         'adapter_status':'NO_VALIDATED_LENSFD_TO_RC3_FEATURE_ADAPTER',
         'rc3_predictions_produced':0,'rc3_field_fault_detection_rate':None,
         'reasons':reasons,'integrity_check_status':'SEPARATE_FROM_LENSFD_INFERENCE'}
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    atomic_json(output_dir/'RC3_COMPATIBILITY.json',out)
    return out


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--archive',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--scratch',default='/content')
    args=parser.parse_args()
    print(json.dumps(run_full_archive(args.archive,args.output,args.scratch),indent=2))
    print(json.dumps(rc3_contract_report(args.output),indent=2))
