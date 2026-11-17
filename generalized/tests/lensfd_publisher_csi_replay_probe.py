"""Run ORIGINAL Rice LensFD CSI pipeline over genuine first 24 pilot frames.

Diagnostic only. Does processing yield exact zero at port 68 when native raw
pilot IQ is nonzero? Never calls frozen RC3, and not hardware fault labels.
Source: https://renew.rice.edu/datasets/ashu/traces/LensFD/
"""
from __future__ import annotations
import contextlib,hashlib,io,json,os,sys,tempfile,zipfile
from pathlib import Path
from urllib.request import urlopen,Request
import h5py,numpy as np

ROOT="https://renew.rice.edu/datasets/ashu/traces/LensFD/"
RAW="data_LensFD/indoor/indoor-a2c-static/raw/no-indoor/no-indoor-spot-1.hdf5"
MAT="data_LensFD/indoor/indoor-a2c-static/processed/no-indoor-pro/no-indoor-spot-1.mat"
CODE="code_LensFD.zip"

def get(url,dest,max_bytes):
    h=hashlib.sha256()
    n=0
    with urlopen(Request(url,headers={"User-Agent":"SANAD-RC3-RiceCSI-source-replay"}),timeout=160) as src,open(dest,"wb") as out:
        if src.status!=200 or not src.url.startswith("https://renew.rice.edu/"):
            raise RuntimeError("Wrong publisher origin")
        for block in iter(lambda:src.read(1024*1024),b""):
            n+=len(block)
            if n>max_bytes:raise ValueError("Source unexpectedly oversized")
            h.update(block);out.write(block)
    return {"bytes":n,"sha256":h.hexdigest()}

def main():
    with tempfile.TemporaryDirectory() as work:
        d=Path(work)
        code=d/"code.zip";raw=d/"raw.hdf5";mat=d/"published.mat"
        codehash=get(ROOT+CODE,code,1024*1024)
        with zipfile.ZipFile(code) as z:
            for item in z.namelist():
                if item.startswith("/") or ".." in Path(item).parts:
                    raise ValueError("Unsafe publisher ZIP")
            z.extractall(d)
        sys.path.insert(0,str(d/"code_LensFD"))
        # Publisher's original Python module, no modification of its formulas.
        from hdf5_lib import hdf5_lib
        from hdf5process import process_hdf5
        import scipy.io
        rawhash=get(ROOT+RAW,raw,600_000_000)
        mathash=get(ROOT+MAT,mat,5_000_000)
        radio=hdf5_lib(str(raw),n_frames_to_inspect=24,n_fr_insp_st=0,sub_sample=1)
        data_pilot,data_noise=process_hdf5(radio)
        recomputed=np.asarray(data_pilot["chan_full"])
        pub=np.asarray(scipy.io.loadmat(mat)["chan_full"])
        if recomputed.shape!=(2,80,52) or pub.shape!=(2,80,52):
            raise ValueError("Unexpected processed CSI shape")
        result={
          "publisher_source_code_sha256":codehash["sha256"],
          "raw_hdf5":rawhash,
          "published_mat":mathash,
          "raw_frames_used_for_partial_replay":24,
          "publisher_original_pipeline_unmodified":True,
          "port_info":{},
          "hardware_failure_label":"UNVERIFIED",
          "frozen_RC3_not_run":True
        }
        for port in (66,67):
            x=recomputed[:,port,:];y=pub[:,port,:]
            result["port_info"][str(port+1)]={
              "recomputed_first24_rms":float(np.sqrt(np.nanmean(np.abs(x)**2))),
              "published_allframes_rms":float(np.sqrt(np.nanmean(np.abs(y)**2))),
              "recomputed_first24_exact_zero":bool(np.all(x==0)),
              "published_allframes_exact_zero":bool(np.all(y==0)),
              "recomputed_first24_finite_fraction":float(np.isfinite(x).mean())
            }
        print("PUBLISHER_PIPELINE_SOURCE_REPLAY="+json.dumps(result,allow_nan=False),flush=True)
if __name__=="__main__":main()
