"""SANAD RC3 benchmark on direct measured CSI from DICHASUS / DaRUS.

Source: doi:10.18419/DARUS-4062, V3, dichasus-ad25.tfrecords.
Real measured, phase-coherent 64-channel CSI; synthetic drift faults. No model
retraining or threshold changes. Four PHYSICAL sites with two distinct 8-element
ULA rows per site. Seven-row inference does NOT mean seven sites.
No hardware-fault truth or verified fault-free measurement labels are available.
Partial HTTP Range input has a SHA-256; full-file MD5 does not apply to the slice.
Sampling at 48 ms ≠ the 15-minute report period of original RC3 qualification.
"""
from __future__ import annotations
import hashlib, json, os, struct, sys, time
from pathlib import Path
import numpy as np
import requests

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from rc3 import FrozenRC3
sys.path.insert(0,str(HERE/'runtime'))
from geometry_config import N_ARRAYS,N_ELEMENTS
from v35_replay import extract_seed_v35

DOI="doi:10.18419/DARUS-4062"
DATASETS={
  "ad25":dict(id=664466,size=596950014,md5="3e907ad72ae6aab1aa44ef53ba60a3ef"),
  "ad21":dict(id=664462,size=1553015076,md5="c89c49a4a01e8b344b3e33920ee5b04f"),
}
DATASET=os.environ.get("DICHASUS_DATASET","ad25")
if DATASET not in DATASETS:raise ValueError("Unlisted DICHASUS recording")
FILE_ID=DATASETS[DATASET]['id']
FILE_NAME=f"dichasus-{DATASET}.tfrecords"
FILE_SIZE=DATASETS[DATASET]['size']
DATA_MD5=DATASETS[DATASET]['md5']
FREQ_HZ=3.440e9
SAMPLE_INTERVAL_MS=48
FIRST_BYTES=180_000_000
SEED_START=510000

# Official channel assignments from the DICHASUS antenna configuration page.
# Two horizontal 8-element ULA rows per physical site, zero-based channel IDs.
SITE_ROWS=[
    [[58,12,24,29,16,2,33,38],[34,9,36,32,22,62,39,10]],
    [[26,35,59,53,19,20,17,54],[1,48,61,44,5,49,6,60]],
    [[31,15,4,14,30,47,11,3],[56,23,25,41,27,0,8,40]],
    [[57,45,42,28,63,55,46,43],[7,18,52,21,13,37,50,51]],
]

def group_rows(n):
    if N_ELEMENTS!=8:raise ValueError("Dataset physical ULA rows contain eight elements, no extrapolation")
    if n==4:return [(s,0,SITE_ROWS[s][0]) for s in range(4)]
    if not 5<=n<=8:raise ValueError("Only 4 physical sites or 5-8 subarrays are supported")
    rows=[(s,r,SITE_ROWS[s][r]) for s in range(4) for r in range(2)]
    return rows[:n]

def download_prefix(path):
    url=f"https://darus.uni-stuttgart.de/api/access/datafile/{FILE_ID}"
    path=Path(path)
    if not path.exists():
        response=requests.get(url,headers={"Range":f"bytes=0-{FIRST_BYTES-1}"},
                              stream=True,timeout=(30,180))
        if response.status_code!=206:
            raise RuntimeError(f"Refuse full download: HTTP {response.status_code}")
        expected=f"bytes 0-{FIRST_BYTES-1}/{FILE_SIZE}"
        if response.headers.get("Content-Range")!=expected:
            raise RuntimeError("HTTP Range mismatch: "+str(response.headers.get("Content-Range")))
        with path.open("wb") as out:
            for block in response.iter_content(chunk_size=2**20):
                if block:out.write(block)
        response.close()
    if path.stat().st_size!=FIRST_BYTES:
        raise ValueError("Truncated/oversized CSI byte range")
    return hashlib.sha256(path.read_bytes()).hexdigest()

def _varint(buf,idx):
    shift=0;val=0
    while True:
        if idx>=len(buf):raise ValueError("Truncated protobuf varint")
        b=buf[idx];idx+=1
        val|=(b&127)<<shift
        if b<128:return val,idx
        shift+=7
        if shift>63:raise ValueError("Invalid protobuf varint")

def _fields(buf):
    d={};idx=0
    while idx<len(buf):
        tag,idx=_varint(buf,idx)
        field=tag>>3;kind=tag&7
        if kind==0: val,idx=_varint(buf,idx)
        elif kind==1: val=buf[idx:idx+8];idx+=8
        elif kind==2:
            size,idx=_varint(buf,idx)
            if size>len(buf)-idx:raise ValueError("Invalid protobuf field length")
            val=buf[idx:idx+size];idx+=size
        elif kind==5: val=buf[idx:idx+4];idx+=4
        else:raise ValueError(f"Unsupported protobuf wire type {kind}")
        d.setdefault(field,[]).append(val)
    return d

def extract_csi(example):
    # tf.train.Example: features(1), map entries(1), key(1), Feature(2).
    outer=_fields(example)
    entries=_fields(outer[1][0]).get(1,[])
    for entry in entries:
        kv=_fields(entry)
        name=kv[1][0].decode()
        if name!='csi':continue
        feat=_fields(kv[2][0])
        bytelist=_fields(feat[1][0])
        tensor=_fields(bytelist[1][0])
        if tensor.get(1,[None])[0]!=1:raise ValueError("Expected TensorFlow float32 CSI TensorProto")
        vals=tensor.get(4,[])
        if len(vals)!=1:raise ValueError("Expected TensorProto.tensor_content")
        v=np.frombuffer(vals[0],dtype='<f4')
        if v.size!=64*1024*2:raise ValueError(f"Unexpected CSI dimensions: {v.size}")
        return v.reshape(64,1024,2)
    raise ValueError("No CSI field in TFRecord example")

def records(path,limit=300):
    count=0
    with open(path,'rb') as fh:
        while count<limit:
            head=fh.read(12)
            if len(head)<12:break
            n=struct.unpack('<Q',head[:8])[0]
            if n>8_000_000:raise ValueError("TFRecord too large")
            payload=fh.read(n)
            tail=fh.read(4)
            if len(payload)!=n or len(tail)!=4:break
            yield extract_csi(payload)
            count+=1
    if count!=limit:raise RuntimeError(f"Only {count}/{limit} complete records")

def measured_covariance(csi,channel_ids):
    h=np.asarray(csi[channel_ids],np.float64)
    z=h[...,0]+1j*h[...,1]
    R=z@z.conj().T/z.shape[1]
    scale=float(np.real(np.trace(R)))/len(channel_ids)
    if scale<=0 or not np.isfinite(scale):raise ValueError("Bad measured CSI power")
    return R

def build_measured_covariance(path,selected):
    R=np.empty((300,len(selected),8,8),dtype=np.complex128)
    for t,x in enumerate(records(path)):
        for j,(site,row,ids) in enumerate(selected):
            R[t,j]=measured_covariance(x,ids)
        if (t+1)%100==0:print("MEASURED_CSI",t+1,flush=True)
    if not np.isfinite(R).all():raise ValueError("Nonfinite measured covariance")
    if not np.allclose(R,R.swapaxes(-1,-2).conj(),rtol=1e-6,atol=1e-8):
        raise ValueError("Covariance Hermitian check failed")
    return R

def conventional_baseline(vectors,kind):
    """Fixed classical reference benchmarks on EXACT glr_compare calibration vectors.

    Calibration uses only samples 60..119 before injected faults may start.
    No thresholds are chosen from labeled test outcomes. Outputs are per-array
    terminal 40 alarm flags, same scoring window as frozen RC3.
    """
    from collections import deque
    N=len(vectors);M=N_ELEMENTS
    alerts=np.zeros(N,dtype=bool)
    for station in range(N):
        seq=vectors[station]
        train=np.array([x for x in seq[60:120] if x is not None],dtype=float)
        if len(train)<30:raise ValueError("Insufficient clean baseline calibration")
        center=np.median(train,axis=0)
        residual=train-center
        unit=np.maximum(1.4826*np.median(np.abs(residual),axis=0),
                        np.r_[np.full(M,0.05),np.full(M,0.5)])
        Sigma=np.cov(residual.T)+np.diag(unit**2*0.01)
        inv=np.linalg.pinv(Sigma,hermitian=True)
        last12=deque(maxlen=12)
        ewma=0.;cusum=0.
        for t in range(120,300):
            v=seq[t]
            if v is None:
                last12.append(False)
                continue
            err=np.asarray(v,dtype=float)-center
            if kind=="GLR-1":
                whitened=np.linalg.cholesky(inv+1e-10*np.eye(2*M)).T@err
                scores=whitened[:M]**2+whitened[M:]**2
                instant=bool(np.max(scores)>13.815510557964274)
            elif kind=="EWMA":
                u=err/unit
                score=float(np.max(np.hypot(u[:M],u[M:])))
                ewma=0.2*score+0.8*ewma
                instant=ewma>4.0
            elif kind=="CUSUM":
                u=err/unit
                score=float(np.max(np.hypot(u[:M],u[M:])))
                cusum=max(0.,cusum+score-2.0)
                instant=cusum>20.0
            else:
                raise ValueError(kind)
            last12.append(instant)
            # Identical 8-of-12 persistence for all classical comparators.
            if t>=260 and sum(last12)>=8:
                alerts[station]=True
    return alerts

def run(path,out):
    if N_ARRAYS not in (4,7,8):raise ValueError("Only geometry 4x8/7x8/8x8 specified")
    selected=group_rows(N_ARRAYS)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    sha=download_prefix(path)
    print("PARTIAL_CSI_SHA256",sha,flush=True)
    R=build_measured_covariance(path,selected)
    covpath=out/'source.npz';evpath=out/'evidence.npz'
    trials=int(os.environ.get("SANAD_RF_TRIALS","2"))
    clf=FrozenRC3()
    methods=('RC3','GLR-1','EWMA','CUSUM')
    tally={m:dict(tp=0,nfault=0,control_alert=0,ncontrol=0,neighbor_alert=0,nneighbor=0) for m in methods}
    traces=[]
    for seed in range(SEED_START,SEED_START+trials):
        np.savez_compressed(covpath,**{f's{seed}':R})
        extract_seed_v35(covpath,HERE/'runtime',seed,evpath)
        with np.load(evpath,allow_pickle=False) as z:ev={k:z[k] for k in z.files}
        truth=np.asarray(ev['injected_expected'])>0
        assert truth.sum()==2
        prediction={'RC3':{stream:np.asarray(clf.predict_trace(ev,stream)['alerts'],bool)
                            for stream in ('healthy','injected')}}
        # The original pipeline is deterministic given seed; these exact same
        # calibration vectors were used in extract_seed_v35 above.
        from glr_compare import pipeline
        control_v,control_faults=pipeline(R,900000+seed,with_faults=False)
        injected_v,actual_faults=pipeline(R,1000000+seed,with_faults=True)
        assert not control_faults and set(actual_faults)==set(np.flatnonzero(truth))
        for name in ('GLR-1','EWMA','CUSUM'):
            prediction[name]={
                'healthy':conventional_baseline(control_v,name),
                'injected':conventional_baseline(injected_v,name)}
        row={'seed':seed}
        for name in methods:
            h=prediction[name]['healthy'];q=prediction[name]['injected']
            n_tp=int((q&truth).sum());n_h=int(h.sum());n_neigh=int((q&~truth).sum())
            row[name]={'tp':n_tp,'control_alerts':n_h,'neighbor_alerts':n_neigh}
            ct=tally[name];ct['tp']+=n_tp;ct['nfault']+=2
            ct['control_alert']+=n_h;ct['ncontrol']+=N_ARRAYS
            ct['neighbor_alert']+=n_neigh;ct['nneighbor']+=N_ARRAYS-2
        traces.append(row);print("DICHASUS_TRIAL="+json.dumps(row),flush=True)
    result=dict(doi=DOI,file_id=FILE_ID,source_filename=FILE_NAME,
                source_full_file_md5=DATA_MD5,prefix_bytes=FIRST_BYTES,
                prefix_sha256=sha,geometry=[N_ARRAYS,N_ELEMENTS],
                source_real_rf_measurement=True,
                selected_rows=[{'physical_site':s,'row':r,'element_port_ids':ids} for s,r,ids in selected],
                physical_site_count=4,independent_station_count=4,
                period_interpretation="one 48-ms spaced CSI snapshot per RC3 report; NOT 15-minute operational reports",
                fault_truth="original RC3 synthetic injections, no verified hardware fault annotations",
                healthy_truth="unlabeled measured control, not verified fault-free",
                algorithm_frozen=True,rc3_threshold=0.5,trials=traces,
                methods=tally,
                comparator_status="GLR-1, EWMA, CUSUM use identical RC3 front-end calibration vectors and fault/missingness draws",
                comparator_prespecified={"GLR-1":"full covariance whitening; chi2(2) 99.9 percentile 13.8155",
                                       "EWMA":"alpha 0.2; z_max 4.0",
                                       "CUSUM":"k=2.0; h=20.0",
                                       "all":"8 of last 12 reports; learning 60..119; score any in reports 260..299; no tuning"},
                qualified=False)
    (out/'summary.json').write_text(json.dumps(result,indent=2))
    print("DICHASUS_RESULT="+json.dumps(result),flush=True)
    return result

if __name__=='__main__':
    run(Path(os.environ.get("CSI_PREFIX_PATH",f"/tmp/dichasus-{DATASET}-prefix.tfrecords")),
        Path(os.environ.get("RESULT_DIR","/tmp/dichasus_result")))
