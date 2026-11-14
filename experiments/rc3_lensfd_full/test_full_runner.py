import io
import json
import tarfile
from pathlib import Path
import h5py
import numpy as np
import pytest
from lensfd_full_run import iter_dataset_slices, run_full_archive, rc3_contract_report


def make_archive(tmp_path):
    src=tmp_path/'dataset.hdf5'
    with h5py.File(src,'w') as f:
        f.create_dataset('complex_csi',data=np.array([[1+2j,3+4j],[0,1-1j],[np.nan+0j,4+2j]],dtype=np.complex64))
        f.create_dataset('metadata/info',data=np.array([2,3,4],dtype=np.int64))
        f.create_dataset('scalar',data=np.array(42.,dtype=np.float32))
        f.create_dataset('blank',shape=(0,4),dtype='f')
    b=tmp_path/'new_archive.tgz'
    with tarfile.open(b,'w:gz') as t:
        for name in ['a.hdf5','b.hdf5']:
            t.add(src,arcname=name)
        content=b'LICENSE file was fully read and SHA-256 hashed.'
        ti=tarfile.TarInfo('info/LICENSE');ti.size=len(content)
        t.addfile(ti,io.BytesIO(content))
    return b


def test_slice_partition_exact():
    a=np.zeros((7,19,11),int)
    for s in iter_dataset_slices(a.shape,16,target_bytes=100):a[s]+=1
    assert np.all(a==1)
    assert list(iter_dataset_slices((),4))==[()]
    assert list(iter_dataset_slices((0,3),4))==[]


def test_full_archive_every_byte_and_every_numeric_value(tmp_path):
    source=make_archive(tmp_path)
    out=tmp_path/'out'
    result=run_full_archive(source,out,tmp_path)
    assert result['status']=='COMPLETE_FULL_ARCHIVE_AUDIT'
    assert result['archive_members_regular']==3
    assert result['members_ok']==3
    assert result['hdf5_dataset_count']==8
    assert result['hdf5_numeric_values_scanned']==2*(6+3+1)
    rows=[json.loads(s) for s in (out/'members.jsonl').read_text().splitlines()]
    assert sum('sha256' in s for s in rows)==3
    assert (out/'SOURCE_LOCK.json').exists()
    assert rc3_contract_report(out)['rc3_predictions_produced']==0
    # Source archive fully reused on rerun, all members accounted for.
    again=run_full_archive(source,out,tmp_path)
    assert again['members_checked']==3
    assert len((out/'members.jsonl').read_text().splitlines())==3


def test_resume_after_error(tmp_path, monkeypatch):
    source=make_archive(tmp_path)
    out=tmp_path/'out'
    import lensfd_full_run as m
    native=m.inspect_hdf5
    count={'value':0}
    def maybe_fail(p):
        if count['value']==0:
            count['value']+=1
            raise OSError('transient read error')
        return native(p)
    monkeypatch.setattr(m,'inspect_hdf5',maybe_fail)
    with pytest.raises(RuntimeError,match='INCOMPLETE'):
        m.run_full_archive(source,out,tmp_path)
    monkeypatch.setattr(m,'inspect_hdf5',native)
    done=m.run_full_archive(source,out,tmp_path)
    assert done['status']=='COMPLETE_FULL_ARCHIVE_AUDIT'
    assert done['members_checked']==3
