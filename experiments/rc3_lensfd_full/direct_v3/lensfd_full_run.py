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
import subprocess
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


def validate_archive_structure(archive: str | Path) -> dict:
    """Exhaustively validate gzip CRC and TAR member framing without extracting data.

    GNU tar streams through all bytes, so this checks the large final member too.
    Merely hashing an incomplete file does NOT prove completeness.
    """
    archive = Path(archive)
    check = subprocess.run(['tar', '-tzf', str(archive)], stdout=subprocess.DEVNULL,
                           stderr=subprocess.PIPE, text=True, errors='replace')
    result = {'archive': str(archive), 'archive_size': archive.stat().st_size,
              'tar_exit_code': check.returncode,
              'complete': check.returncode == 0,
              'stderr_tail': check.stderr[-2400:]}
    return result


def archive_preflight(archive, output, *, require_complete=True):
    """Prevent a slow HDF5 scan from ending on a truncated archive."""
    output = Path(output); output.mkdir(parents=True,exist_ok=True)
    report = validate_archive_structure(archive)
    atomic_json(output/'ARCHIVE_PREFLIGHT.json', report)
    if not report['complete'] and require_complete:
        raise RuntimeError('LensFD TAR/GZIP integrity FAILED before HDF5 scanning. '
                           'Inspect ARCHIVE_PREFLIGHT.json; repair/re-download the '
                           'source archive, then rerun. Existing members.jsonl '
                           'and completed HDF5 statistics are preserved. '
                           + report['stderr_tail'][-500:])
    return report


def _archive_summary(*, archive, arch_sha, seen, prior, dirs, errors, source_changed):
    regular = [r for r in prior.values() if r['name'] in seen]
    ok = [r for r in regular if r['status'] == 'OK']
    complete = (len(regular) == len(seen) and len(ok) == len(seen) and not errors)
    source_groups = Counter(Path(r['name']).name.split('-')[0].lower() for r in ok)
    return {'status':'COMPLETE_FULL_ARCHIVE_AUDIT' if complete else 'INCOMPLETE',
       'official_source':OFFICIAL_URL, 'finished_utc':utcnow(),
       'archive_sha256':arch_sha, 'archive_bytes_hashed':archive.stat().st_size,
       'archive_members_regular':len(seen), 'archive_members_directories':dirs,
       'source_filename_prefix_counts':dict(sorted(source_groups.items())),
       'members_checked':len(regular), 'members_ok':len(ok),
       'member_bytes_accounted':sum(r['size'] for r in ok),
       'hdf5_dataset_count':sum(r.get('hdf5_dataset_count',0) for r in ok),
       'hdf5_numeric_values_scanned':sum(r.get('numeric_values_scanned',0) for r in ok),
       'hdf5_total_logical_values':sum(r.get('total_logical_values',0) for r in ok),
       'hdf5_nonnumeric_dataset_count':sum(r.get('hdf5_dataset_count',0)-r.get('numeric_dataset_count',0) for r in ok),
       'bytes_sampled':0, 'inference_source':'RAW_REAL_RF_MEASUREMENTS',
       'rc3_qualification_on_lensfd':'NOT_YET_TESTED',
       'archive_replacement_detected':bool(source_changed),
       'errors':errors + [{'name':r['name'],'error':r.get('error','unresolved')}
                          for r in regular if r['status']!='OK' and r['name'] not in {e['name'] for e in errors}]}


def run_full_archive(archive: str | Path, output: str | Path, scratch_dir: str | Path | None = None,
                     strict=True, *, preflight=True) -> dict:
    """Audit every regular TAR member; preserve checkpoints, including on TAR read errors.

    Reused source: completed HDF5 member checks are not repeated.
    Replaced source: re-hash cached members BEFORE reusing their HDF5 statistics.
    No sample limits, no silent skipping.  The frozen RC3 model is not changed.
    """
    archive = Path(archive); output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if not archive.is_file(): raise FileNotFoundError(str(archive))
    if preflight: archive_preflight(archive, output)
    arch_size=archive.stat().st_size
    arch_sha=hash_file(archive)
    lock=output/'SOURCE_LOCK.json'
    previous=json.loads(lock.read_text()) if lock.exists() else None
    source_changed=bool(previous and
        (previous['archive_sha256'],previous['archive_size'])!=(arch_sha,arch_size))
    # Do NOT discard old journal simply because source was repaired; verify each
    # cached member's SHA against the replacement archive before trusting stats.
    journal=output/'members.jsonl'; prior=_load_journal(journal)
    seen=set();errors=[];dirs=0;tar_error=None
    scratch=Path(scratch_dir or tempfile.gettempdir()); scratch.mkdir(parents=True,exist_ok=True)
    try:
        with tarfile.open(archive,'r|gz') as tar:
            for member in tar:
                name=_safe_member_name(member.name)
                if member.isdir(): dirs+=1; continue
                if name in seen: raise RuntimeError(f'Duplicate archive member {name}')
                seen.add(name)
                old=prior.get(name)
                if old and old['status']=='OK' and old['size']==member.size and not source_changed:
                    continue
                row={'name':name, 'size':member.size,'mtime':member.mtime,'status':'ERROR',
                     'archive_sha256':arch_sha}
                tempname=None
                try:
                    if not member.isfile():
                        raise ValueError(f'Unsupported non-regular archive object: {member.type!r}')
                    inp=tar.extractfile(member)
                    if inp is None:raise RuntimeError('Could not access archive member stream')
                    # If source archive was repaired, verify old 112+ completed members
                    # by hashing them directly without extracting / re-scanning HDF5.
                    if old and old['status']=='OK' and old['size']==member.size and source_changed:
                        dig=hashlib.sha256(); remaining=member.size
                        while remaining:
                            b=inp.read(min(8*1024*1024,remaining))
                            if not b: raise EOFError(f'Truncated archive member {name}')
                            dig.update(b); remaining-=len(b)
                        if dig.hexdigest()!=old.get('sha256'):
                            raise RuntimeError('PREVIOUS_OK_FILE_CHANGED; cannot reuse its HDF5 statistics. '
                                               'Use a fresh audit folder for this different source.')
                        row={**old,'archive_sha256':arch_sha,
                             'cache_verified_against_repaired_archive':True}
                    else:
                        free=shutil.disk_usage(str(scratch)).free
                        margin=256*1024*1024
                        if member.size+margin>free:
                            raise OSError('INSUFFICIENT_SCRATCH_DISK: '
                               f'{name} is {member.size:,} bytes ({member.size/2**30:.2f} GiB); '
                               f'/content has only {free:,} bytes ({free/2**30:.2f} GiB) free; '
                               'select a larger-disk Colab runtime. NO SUBSAMPLING.')
                        dig=hashlib.sha256()
                        with tempfile.NamedTemporaryFile(prefix='lensfd_',
                            suffix=Path(name).suffix,dir=str(scratch),delete=False) as temp:
                            tempname=temp.name; remaining=member.size
                            while remaining:
                                b=inp.read(min(8*1024*1024,remaining))
                                if not b: raise EOFError(f'Truncated archive member {name}')
                                temp.write(b);dig.update(b);remaining-=len(b)
                        row['sha256']=dig.hexdigest()
                        if Path(name).suffix.lower() in ('.hdf5','.h5'):
                            row.update(inspect_hdf5(Path(tempname)))
                            row['analysis']='FULL_HDF5_NUMERIC_SCAN'
                        else:
                            row['analysis']='FULL_BYTE_SHA256_ONLY'
                            row['warning']='Non-HDF5 member fully SHA-256 hashed'
                        row['status']='OK'
                except Exception as exc:
                    row.update(status='ERROR',error=f'{type(exc).__name__}: {exc}')
                    errors.append({'name':name,'error':row['error']})
                finally:
                    if tempname is not None:Path(tempname).unlink(missing_ok=True)
                _json_line(journal,row); prior[name]=row
                print(f'[{len(seen)}] {row["status"]:5} {name}  bytes={member.size:,}',flush=True)
    except (tarfile.TarError, EOFError, OSError, RuntimeError, ValueError) as exc:
        tar_error={'name':'__ARCHIVE_STREAM__','error':f'{type(exc).__name__}: {exc}'}
        errors.append(tar_error)
    summary=_archive_summary(archive=archive, arch_sha=arch_sha, seen=seen,
                             prior=prior,dirs=dirs,errors=errors,source_changed=source_changed)
    if tar_error: summary['fatal_tar_stream_error']=tar_error['error']
    atomic_json(output/'FULL_AUDIT_SUMMARY.json',summary)
    if summary['status']=='COMPLETE_FULL_ARCHIVE_AUDIT':
        if source_changed and previous:
            atomic_json(output/'SOURCE_LOCK_PREVIOUS.json',previous)
        atomic_json(lock, {'official_source':OFFICIAL_URL,'archive_size':arch_size,
                           'archive_sha256':arch_sha,'verified_complete_utc':utcnow()})
    if strict and summary['status']!='COMPLETE_FULL_ARCHIVE_AUDIT':
        detail='; '.join(x['error'] for x in summary['errors'][-3:])
        raise RuntimeError('Full LensFD archive audit INCOMPLETE. See '
                           'FULL_AUDIT_SUMMARY.json and members.jsonl. '+detail)
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
