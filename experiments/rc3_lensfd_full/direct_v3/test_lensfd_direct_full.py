import hashlib
import json
import os
from pathlib import Path
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import h5py
import numpy as np
import pytest

from lensfd_direct_full import (crawl_publisher,discover_all,run_all_direct,download_full_file)


class Quiet(SimpleHTTPRequestHandler):
    def log_message(self,*a):pass

@pytest.fixture
def host(tmp_path):
    public=tmp_path/'web';pub=public/'data_LensFD';(pub/'outdoor').mkdir(parents=True)
    a=pub/'outdoor'/'a.hdf5'
    with h5py.File(a,'w') as f:
        f.create_dataset('csi',data=np.ones((17,7,3),dtype=np.complex64))
        f.create_dataset('iq',data=np.arange(40,dtype=np.float64))
    with h5py.File(pub/'b.hdf5','w') as f:
        f.create_dataset('q',data=np.array([1,2,3,4],dtype=int))
    (pub/'LICENSE').write_text('dataset license')
    s=ThreadingHTTPServer(('127.0.0.1',0),partial(Quiet,directory=str(public)))
    t=Thread(target=s.serve_forever,daemon=True);t.start()
    yield f'http://127.0.0.1:{s.server_port}/data_LensFD/',pub
    s.shutdown();t.join()


def test_crawl_all_publisher_files(host):
    root,_=host
    m=crawl_publisher(root)
    assert sorted(m['files'])==['LICENSE','b.hdf5','outdoor/a.hdf5']
    assert m['directories']==2


def test_full_direct_no_sampling(host,tmp_path):
    root,_=host
    m=discover_all(tmp_path/'results',roots=[root])
    assert m['hdf5_file_count']==2
    d=run_all_direct(manifest=m,results=tmp_path/'results',scratch=tmp_path/'scratch')
    assert d['status']=='COMPLETE_ALL_PUBLISHED_DIRECT_FILES'
    assert d['files_ok']==3
    assert d['numeric_values_scanned']==17*7*3+40+4
    again=run_all_direct(manifest=m,results=tmp_path/'results',scratch=tmp_path/'scratch')
    assert again['files_ok']==3
    assert len((tmp_path/'results'/'direct_members.jsonl').read_text().splitlines())==3


def test_reuse_archive_112_style(host,tmp_path):
    root,pub=host
    m=discover_all(tmp_path/'results',roots=[root])
    p=pub/'outdoor'/'a.hdf5'
    from lensfd_full_run import inspect_hdf5
    old={**inspect_hdf5(p),'name':'data_LensFD/outdoor/a.hdf5','sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
         'size':p.stat().st_size,'status':'OK'}
    journal=tmp_path/'prior_members.jsonl';journal.write_text(json.dumps(old)+'\n')
    d=run_all_direct(manifest=m,results=tmp_path/'results',scratch=tmp_path/'scratch',old_archive_journal=journal)
    assert d['status']=='COMPLETE_ALL_PUBLISHED_DIRECT_FILES'
    assert d['reused_verified_archive_prefix']==1
    assert d['numeric_values_scanned']==17*7*3+40+4


def test_incomplete_listing_not_claimed_complete(host,tmp_path):
    root,pub=host
    m=discover_all(tmp_path/'results',roots=[root])
    # Record a prior artifact not present on directory listing: cross-source mismatch.
    journal=tmp_path/'old.jsonl'
    journal.write_text(json.dumps({'name':'data_LensFD/nonexistent.hdf5','status':'OK','sha256':'a'*64,'size':100})+'\n')
    with pytest.raises(RuntimeError,match='missing'):
        run_all_direct(manifest=m,results=tmp_path/'results',scratch=tmp_path/'scratch',old_archive_journal=journal)


def test_missing_hdf5_refused(tmp_path):
    pub=tmp_path/'web'/'data_LensFD';pub.mkdir(parents=True)
    (pub/'LICENSE').write_text('hi')
    s=ThreadingHTTPServer(('127.0.0.1',0),partial(Quiet,directory=str(tmp_path/'web')))
    t=Thread(target=s.serve_forever,daemon=True);t.start()
    try:
        with pytest.raises(RuntimeError,match='No HDF5'):
            crawl_publisher(f'http://127.0.0.1:{s.server_port}/data_LensFD/')
    finally:
        s.shutdown();t.join()
