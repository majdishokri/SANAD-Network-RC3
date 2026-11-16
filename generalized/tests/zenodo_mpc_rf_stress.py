"""Measured-channel-derived RF stress test for original RC3 math.
SOURCE: Niigata University, Calist & Kim, DOI 10.5281/zenodo.21213882.
The Zenodo MPC data are REAL propagation measurements. The input covariance is
RECONSTRUCTED from measured path angles/powers for a hypothetical half-wave ULA.
'Stations' are distinct MOVING RECEIVER route segments, NOT simultaneously
measured base stations. Faults are synthetically injected using original
glr_compare.pipeline, so this is NOT real hardware/field qualification.
No threshold/model/decision-rule fitting is performed.
"""
from __future__ import annotations
import hashlib, io, json, os, sys, zipfile
from pathlib import Path
import numpy as np
from scipy.io import loadmat

HERE=Path(__file__).resolve().parents[1]
RUNTIME=HERE/'runtime'
sys.path.insert(0,str(RUNTIME))
sys.path.insert(0,str(HERE))
from geometry_config import N_ARRAYS,N_ELEMENTS
from rc3 import FrozenRC3  # import generalized RC3 BEFORE legacy glr_compare prepends '.'
from v35_replay import extract_seed_v35

DOI="10.5281/zenodo.21213882"
ZIP_MD5="a19578d10f2edab687136bee42274bf4"
SEED=410001


def measured_blocks(archive:Path, reports:int=300):
    digest=hashlib.md5(archive.read_bytes()).hexdigest()
    if digest!=ZIP_MD5:
        raise RuntimeError("Zenodo data MD5 mismatch: "+digest)
    blocks=[]
    counts={}
    with zipfile.ZipFile(archive) as z:
        candidates=sorted(
            x for x in z.namelist() if x.endswith('_all_Param.mat'))
        if not candidates:raise RuntimeError("Measured multipath MAT records missing")
        for name in candidates:
            obj=loadmat(io.BytesIO(z.read(name)),squeeze_me=True,struct_as_record=False)
            series=obj["MPC"]
            real_observations=[]
            for sample in series:
                a=np.asarray(sample)
                if (a.ndim!=2 or a.shape[1]<4 or a.shape[0]<4
                    or not np.iscomplexobj(a) or not np.isfinite(a).all()):
                    continue
                amplitude=np.asarray(a[:,3],complex)
                angle=np.asarray(a[:,1].real,float)
                keep=(np.abs(amplitude)>0)&np.isfinite(angle)&(np.abs(angle)<=180)
                if keep.sum()>=4:
                    real_observations.append((angle[keep],amplitude[keep]))
            counts[name]=len(real_observations)
            for i in range(0,len(real_observations)-reports+1,reports):
                blocks.append((name,real_observations[i:i+reports]))
    if len(blocks)<N_ARRAYS:
        raise ValueError(
            f"Only {len(blocks)} disjoint measured route segments with {reports} "
            f"valid snapshots; requested {N_ARRAYS}. Cannot create extra independent stations.")
    return blocks[:N_ARRAYS],counts


def covariance_from_measured_mpc(angles_deg,complex_path_coeffs,elements):
    """Test INPUT ADAPTER only: incoherent path-power ULA spatial covariance.
    No detector equations are altered. This is not a directly measured R matrix.
    """
    az=np.deg2rad(angles_deg)
    power=np.abs(complex_path_coeffs)**2
    power=power/power.sum()
    steering=np.exp(1j*np.pi*np.sin(az[:,None])*np.arange(elements)[None,:])
    cov=(steering.conj().T*power[None,:])@steering
    cov+=0.05*np.eye(elements)
    return cov.astype(np.complex128)


def run(archive, output):
    blocks,counts=measured_blocks(archive)
    R=np.empty((300,N_ARRAYS,N_ELEMENTS,N_ELEMENTS),np.complex128)
    routes=[]
    for station,(route,samples) in enumerate(blocks):
        routes.append(route)
        for t,(angles,coeffs) in enumerate(samples):
            R[t,station]=covariance_from_measured_mpc(angles,coeffs,N_ELEMENTS)
    assert np.isfinite(R).all()
    assert np.allclose(R,np.swapaxes(R.conj(),-1,-2),rtol=1e-5,atol=1e-7)
    eig=np.linalg.eigvalsh(R.reshape(-1,N_ELEMENTS,N_ELEMENTS))
    if eig.min()<=0:raise RuntimeError("Non-positive covariance")
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    covpath=output/'measured_derived_covariance.npz'
    evpath=output/'injected_evidence.npz'
    np.savez_compressed(covpath,**{f's{SEED}':R})
    extract_seed_v35(covpath,RUNTIME,SEED,evpath)
    with np.load(evpath,allow_pickle=False) as z:
        ev={k:z[k] for k in z.files}
    classifier=FrozenRC3()
    assert 'generalized' in sys.modules['rc3'].__file__, sys.modules['rc3'].__file__
    pred={s:np.asarray(classifier.predict_trace(ev,s)['alerts'],bool)
          for s in ('healthy','injected')}
    labels={s:np.asarray(ev[f'{s}_expected'])>0
            for s in ('healthy','injected')}
    assert labels['healthy'].sum()==0 and labels['injected'].sum()==2
    stats={
      'source_doi':DOI,'source_md5':ZIP_MD5,
      'station_count':N_ARRAYS,'elements_per_station':N_ELEMENTS,
      'measured_route_segments_used':routes,
      'measured_samples_per_station':300,
      'available_valid_measured_samples_by_route':counts,
      'model_retrained':False,'threshold':0.5,
      'fault_type':'synthetic gain/phase faults from frozen RC3 pipeline',
      'fault_n':int(labels['injected'].sum()),
      'tp':int((pred['injected']&labels['injected']).sum()),
      'healthy_n':int(len(pred['healthy'])),
      'healthy_fp':int(pred['healthy'].sum()),
      'neighbor_n':int((~labels['injected']).sum()),
      'neighbor_fp':int((pred['injected']&~labels['injected']).sum()),
      'recall':float((pred['injected']&labels['injected']).sum()/labels['injected'].sum()),
      'healthy_far':float(pred['healthy'].mean()),
      'provenance':'Measured multipath path parameters; ULA covariance reconstructed, not measured',
      'caveat':'Non-simultaneous moving receiver segments as pseudo-stations; not 3GPP UMa or real hardware deployment',
      'independent_qualified':False
    }
    (output/'metrics.json').write_text(json.dumps(stats,indent=2,sort_keys=True))
    print('RF_STRESS_RESULT='+json.dumps(stats,sort_keys=True),flush=True)
    return stats

if __name__=='__main__':
    archive=Path(sys.argv[1] if len(sys.argv)>1 else '/tmp/urban_mimo.zip')
    out=Path(sys.argv[2] if len(sys.argv)>2 else '/tmp/rc3_zenodo_stress')
    run(archive,out)
