"""Full-source LensFD direct-directory recovery when publisher .tgz is truncated.

No partial-archive promotion, missing-record substitution, or sampling. Files
are enumerated from the publisher's directory listing, downloaded individually,
SHA-256 verified, and HDF5 numeric values scanned with bounded memory. Previously
completed 112+ archive members may be reused with explicitly marked provenance.

The dataset is Rice University property; never redistribute source recordings.
"""
from __future__ import annotations
import hashlib
import json
import os
import re
import shutil
import time
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlsplit, unquote, quote

import requests
from lensfd_full_run import inspect_hdf5, hash_file, atomic_json, _load_journal, _json_line, utcnow

DATASET_ROOTS = (
    'https://renew.rice.edu/datasets/ashu/traces/LensFD/data_LensFD/',
    'https://renew-wireless.org/datasets/ashu/traces/LensFD/data_LensFD/',
)
SOURCE_TAR = 'https://renew.rice.edu/datasets/ashu/traces/LensFD/data_lensFD.tgz'
SESSION_HEADERS = {'Accept-Encoding': 'identity', 'User-Agent': 'SANAD-LensFD-full-corpus-audit/3.0'}


class LinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() == 'a':
            for k, v in attrs:
                if k.lower() == 'href' and v is not None:
                    self.links.append(v)


def crawl_publisher(root: str, session=None, max_dirs=250) -> dict:
    """Enumerate every published regular file via server directory listings.

    Do not claim full coverage if listing is disabled, truncated, or ambiguous.
    Do not cross hosts, traverse parent paths, or count HTML index pages as data.
    """
    root = root.rstrip('/') + '/'
    p_root = urlsplit(root)
    if p_root.scheme not in ('https','http'): raise ValueError('Invalid scheme')
    session = session or requests.Session()
    pending = [root]
    visited = set()
    files = {}
    while pending:
        url = pending.pop()
        if url in visited: continue
        if len(visited) >= max_dirs: raise RuntimeError('Too many source subdirectories; refusing incomplete crawl')
        visited.add(url)
        r = session.get(url, timeout=(20,90), headers=SESSION_HEADERS)
        r.raise_for_status()
        ctype = r.headers.get('Content-Type', '').lower()
        if 'html' not in ctype and not r.text.lstrip().lower().startswith(('<!doctype html','<html')):
            raise RuntimeError(f'Publisher directory index unavailable: {url}, Content-Type={ctype}')
        if len(r.content) > 15 * 1024 * 1024:
            raise RuntimeError('Oversized directory index; cannot verify completeness')
        links = LinkParser(); links.feed(r.text)
        good = 0
        for href in links.links:
            if href in ('../','./','#') or href.startswith(('../','?','#','mailto:')):
                continue
            resolved = urlsplit(urljoin(url, href))
            if (resolved.scheme, resolved.netloc) != (p_root.scheme, p_root.netloc):
                continue
            if resolved.query or resolved.fragment:
                continue
            target = resolved._replace(query='',fragment='').geturl()
            if not target.startswith(root):
                continue
            rel = unquote(target[len(root):])
            if not rel or rel.startswith('/') or '/..' in rel or '\\' in rel:
                continue
            if target.endswith('/'):
                if target != url and target not in visited:
                    pending.append(target);good += 1
                continue
            if rel in files and files[rel] != target:
                raise RuntimeError(f'Ambiguous listing entry: {rel}')
            files[rel] = target
            good += 1
        if not good and url == root:
            raise RuntimeError(f'Directory listing blocked or empty: {url}; no archive re-download attempted')
    if not files: raise RuntimeError('Publisher returned no data files; cannot assert full coverage')
    if not any(Path(k).suffix.lower() in ('.hdf5','.h5') for k in files):
        raise RuntimeError('No HDF5 files in direct publisher listing; stop rather than fabricate full scan')
    return dict(root=root,directories=len(visited),files=dict(sorted(files.items())))


def discover_all(output_dir, roots=DATASET_ROOTS, session=None):
    """Try official directory sources. Stop without touching the broken .tgz."""
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    attempts=[]
    for root in roots:
        try:
            manifest=crawl_publisher(root,session=session)
            manifest['source_tar_reference']=SOURCE_TAR
            manifest['file_count']=len(manifest['files'])
            manifest['hdf5_file_count']=sum(Path(x).suffix.lower() in ('.hdf5','.h5') for x in manifest['files'])
            manifest['discovered_utc']=utcnow()
            atomic_json(output_dir/'DIRECT_SOURCE_MANIFEST.json',manifest)
            return manifest
        except Exception as e:
            attempts.append({'root':root,'error':f'{type(e).__name__}: {e}'})
    atomic_json(output_dir/'DIRECT_DISCOVERY_ERROR.json',{'attempts':attempts,'status':'NO_COMPLETE_SOURCE_INDEX'})
    raise RuntimeError('No accessible publisher directory listing; see DIRECT_DISCOVERY_ERROR.json. '
                       'Do not re-download a known-bad .tgz.')


def probe_remote(url,session):
    """Inspect source size and range support without downloading payload."""
    try:
        r=session.head(url,timeout=(15,60),allow_redirects=True,headers=SESSION_HEADERS)
        if not r.ok: return None,None
        length=int(r.headers['Content-Length']) if r.headers.get('Content-Length','').isdigit() else None
        return length,r.headers.get('ETag')
    except requests.RequestException:
        return None,None


def download_full_file(url,dest,session=None,max_attempts=8):
    """Verified size (where advertised), retry with Range; never claim partial as full."""
    session=session or requests.Session()
    dest=Path(dest);dest.parent.mkdir(parents=True,exist_ok=True)
    part=dest.with_name(dest.name+'.part')
    expected,etag=probe_remote(url,session)
    last=None
    for attempt in range(1,max_attempts+1):
        offset=part.stat().st_size if part.exists() else 0
        if expected is not None and offset>expected:
            part.unlink();offset=0
        if expected is not None and offset==expected and offset>0:
            part.replace(dest)
            return {'size':expected,'sha256':hash_file(dest),'downloaded_from':url,'verified_bytes':expected}
        head=dict(SESSION_HEADERS)
        if offset:
            head['Range']=f'bytes={offset}-'
            if etag and not etag.startswith('W/'):
                head['If-Range']=etag
        try:
            with session.get(url,headers=head,stream=True,timeout=(25,180)) as resp:
                if resp.status_code in (403,404):
                    raise FileNotFoundError(f'Publisher responded {resp.status_code}: {url}')
                resp.raise_for_status()
                if offset:
                    cr=resp.headers.get('Content-Range','')
                    if resp.status_code != 206 or not cr.startswith(f'bytes {offset}-'):
                        # Never append a full 200 response to prior partial contents.
                        part.unlink(missing_ok=True)
                        last='Range not honored; restarting without mixing bytes'
                        continue
                if expected is None:
                    cr=resp.headers.get('Content-Range','')
                    if '/' in cr and cr.rsplit('/',1)[-1].isdigit():
                        expected=int(cr.rsplit('/',1)[-1])
                    elif not offset and resp.headers.get('Content-Length','').isdigit():
                        expected=int(resp.headers['Content-Length'])
                mode='ab' if offset else 'wb'
                with part.open(mode) as f:
                    for block in resp.iter_content(8*1024*1024):
                        if block: f.write(block)
                n=part.stat().st_size
                if expected is not None and n>expected:
                    part.unlink()
                    raise RuntimeError(f'Publisher file exceeded claimed size: {n}>{expected}')
                if expected is not None and n<expected:
                    last=f'Partial transfer {n}/{expected}; retry Range'
                    continue
                if expected is None and not resp.headers.get('Content-Length'):
                    raise RuntimeError(f'Cannot verify complete transfer without publisher byte count: {url}')
                part.replace(dest)
                return {'size':dest.stat().st_size,'sha256':hash_file(dest),'downloaded_from':url,
                        'verified_bytes':dest.stat().st_size}
        except (requests.RequestException, OSError) as exc:
            last=f'{type(exc).__name__}: {exc}'
            time.sleep(min(attempt,4))
    raise RuntimeError(f'Individual-file download failed after {max_attempts} attempts: {url}; {last}')


def run_all_direct(*,manifest,results,scratch,old_archive_journal=None,
                   strict=True,revalidate_previous_archive_members=False,session=None):
    """Process entire direct-file manifest and merge prior successful audits safely."""
    results=Path(results);results.mkdir(parents=True,exist_ok=True)
    scratch=Path(scratch);scratch.mkdir(parents=True,exist_ok=True)
    session=session or requests.Session()
    old=_load_journal(Path(old_archive_journal)) if old_archive_journal else {}
    direct_journal=results/'direct_members.jsonl'
    done=_load_journal(direct_journal)
    files=manifest['files']; errors=[]
    if len(files)!=len(set(files)):raise RuntimeError('Duplicate manifest names')
    # In the tar the root was "data_LensFD". Preserve exact relative path mapping.
    old_by_rel={r.removeprefix('data_LensFD/'):v for r,v in old.items()
                if r.startswith('data_LensFD/') and v.get('status')=='OK'}
    # A previously verified archive file absent from the publisher listing indicates
    # the direct listing and archive may not describe the same corpus.
    missing_old=set(old_by_rel)-set(files)
    if missing_old:
        atomic_json(results/'DIRECT_MANIFEST_MISMATCH.json',{'missing_archive_prefix_members':sorted(missing_old)})
        raise RuntimeError(f'Direct index missing {len(missing_old)} previously verified archived files. '
                           'Cannot attest to full publisher corpus.')
    for idx,(rel,url) in enumerate(sorted(files.items()),start=1):
        if rel in done and done[rel].get('status')=='OK' and not revalidate_previous_archive_members:
            continue
        prev=old_by_rel.get(rel)
        if prev and not revalidate_previous_archive_members and prev.get('sha256'):
            size,etag=probe_remote(url,session)
            if size is not None and size!=prev.get('size'):
                raise RuntimeError(f'Source mismatch for previously verified member: {rel}; '
                                   f'direct={size} archive={prev.get("size")}')
            row={**prev,'name':rel,'source':SOURCE_TAR,'status':'OK',
                 'reused_verified_archive_prefix':True,
                 'direct_publisher_size_confirmed':size is not None,
                 'direct_file_sha256_revalidated':False,
                 'explanation':'This member was fully scanned and hashed inside readable publisher archive prefix.'}
            _json_line(direct_journal,row);done[rel]=row
            print(f'[{idx}/{len(files)}] REUSE {rel}',flush=True)
            continue
        row={'name':rel,'url':url,'status':'ERROR','source':'direct_publisher'}
        local=scratch/Path(rel).name
        # Unique work file name even when different directories have same base filename
        local=scratch/(hashlib.sha256(rel.encode()).hexdigest()[:14]+'-'+Path(rel).name)
        try:
            # File is downloaded individually, never a subset or a range sample.
            expected,_=probe_remote(url,session)
            disk=shutil.disk_usage(scratch).free
            if expected is not None and disk<expected+256*1024*1024:
                raise OSError(f'INSUFFICIENT_SCRATCH_DISK {rel}: required '
                              f'{expected/2**30:.2f} GiB + margin; available {disk/2**30:.2f} GiB. '
                              'Use a larger disk runtime. Never sample.')
            downloaded=download_full_file(url,local,session=session)
            row.update(size=downloaded['size'],sha256=downloaded['sha256'])
            if prev and revalidate_previous_archive_members and prev.get('sha256')!=row['sha256']:
                raise RuntimeError(f'Previously verified archive member differs from direct publisher: {rel}')
            if prev and revalidate_previous_archive_members and prev.get('sha256')==row['sha256']:
                row.update({k:v for k,v in prev.items() if k in ('hdf5_datasets','hdf5_dataset_count',
                          'numeric_dataset_count','numeric_values_scanned','total_logical_values','root_attribute_names')})
                row['analysis']='REVALIDATED_SHA_REUSE_HDF5_STATISTICS'
            elif Path(rel).suffix.lower() in ('.hdf5','.h5'):
                row.update(inspect_hdf5(local))
                row['analysis']='FULL_HDF5_NUMERIC_SCAN'
            else:
                row['analysis']='FULL_FILE_SHA256'
            row['status']='OK'
        except Exception as exc:
            row['error']=f'{type(exc).__name__}: {exc}'
            errors.append({'name':rel,'error':row['error']})
        finally:
            local.unlink(missing_ok=True)
        _json_line(direct_journal,row)
        done[rel]=row
        print(f'[{idx}/{len(files)}] {row["status"]} {rel}',flush=True)
        # Stop on first failure: avoid expensive partial work on already-bad source.
        if row['status']!='OK' and strict:break
    covered=sum(r.get('status')=='OK' for rel,r in done.items() if rel in files)
    full=covered==len(files)
    summary={'status':'COMPLETE_ALL_PUBLISHED_DIRECT_FILES' if full else 'INCOMPLETE',
             'official_root':manifest['root'],'manifest_count':len(files),
             'manifest_hdf5_count':manifest['hdf5_file_count'],
             'files_ok':covered,'files_missing_or_failed':len(files)-covered,
             'numeric_values_scanned':sum(r.get('numeric_values_scanned',0) for rel,r in done.items() if rel in files),
             'reused_verified_archive_prefix':sum(bool(r.get('reused_verified_archive_prefix')) for rel,r in done.items() if rel in files),
             'source_archive_is_known_broken':True,
             'no_sampling':True,'errors':errors, 'finished_utc':utcnow()}
    atomic_json(results/'DIRECT_FULL_DATA_SUMMARY.json',summary)
    if strict and not full: raise RuntimeError('Full direct LensFD audit INCOMPLETE; see DIRECT_FULL_DATA_SUMMARY.json: '+str(errors[-1:] if errors else []))
    return summary
