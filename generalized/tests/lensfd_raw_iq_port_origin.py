"""Source-native Rice LensFD: raw Pilot_Samples vs processed CSI port zeros.

Runs on one 80-port A2C recording with zero at CSI port 68, one showing
zero at CSI port 67. The two are same physical system under different lens
conditions, NOT independent hardware trials. This script does not label a
physical hardware failure or execute frozen RC3.
"""
from __future__ import annotations
import os, json, hashlib, math, tempfile
from pathlib import Path
from urllib.request import Request, urlopen
import numpy as np
import h5py

ROOT="https://renew.rice.edu/datasets/ashu/traces/LensFD/data_LensFD/"
TRIALS={
  "indoor_no_lens_zero_port68":"indoor/indoor-a2c-static/raw/no-indoor/no-indoor-spot-1.hdf5",
  "indoor_small_lens_zero_port67":"indoor/indoor-a2c-static/raw/sm-indoor/sm-indoor-spot-1.hdf5",
}
MAX_SOURCE_BYTES=600_000_000

def retrieve(name,path):
    url=ROOT+name
    h=hashlib.sha256()
    total=0
    with urlopen(Request(url,headers={"User-Agent":"SANAD-RC3-FaultOrigin-Research"}),timeout=120) as src,open(path,"wb") as dst:
        if src.status!=200:raise ValueError(f"Unexpected source status {src.status}")
        if "renew.rice.edu" not in src.url:raise ValueError("Wrong publisher")
        while True:
            b=src.read(1024*1024)
            if not b:break
            total+=len(b)
            if total>MAX_SOURCE_BYTES:raise ValueError("Oversized raw HDF5; refuse")
            h.update(b);dst.write(b)
    return {"sha256":h.hexdigest(),"bytes":total,"official_url":url}

def all_datasets(f):
    out=[]
    def visit(k,item):
        if isinstance(item,h5py.Dataset):
            out.append({"key":k,"shape":list(item.shape),"dtype":str(item.dtype),
                        "bytes":int(item.size*item.dtype.itemsize)})
    f.visititems(visit)
    return out

def probe_antenna_axis(ds):
    if ds.ndim!=5:raise ValueError(f"Expected 5d raw Pilot_Samples, got {ds.shape}")
    if ds.shape[3]!=80:raise ValueError(f"Expected physical 80-port axis3, got {ds.shape}")
    # Pilot_Samples has n_frames,n_cells,n_user,n_bs_antenna,n_iq.
    frames=min(24,ds.shape[0]); data=np.asarray(ds[:frames])
    if data.shape[4]%2:raise ValueError("Unpaired I/Q")
    I=data[...,0::2].astype(np.float64)
    Q=data[...,1::2].astype(np.float64)
    energy=np.mean((I**2+Q**2),axis=(0,1,2,4))
    rms=np.sqrt(energy)
    pos=rms[rms>0]
    ref=float(np.median(pos)) if len(pos) else 0
    if not ref:raise ValueError("Zero raw all-antenna reference")
    result={
      "frames_audited":int(frames),
      "pilot_iq_dimensions":list(data.shape),
      "raw_median_port_rms":ref,
      "pilot_rms_port67":float(rms[66]),
      "pilot_rms_port68":float(rms[67]),
      "pilot_rms_relative_db_port67":None if not rms[66] else float(20*np.log10(rms[66]/ref)),
      "pilot_rms_relative_db_port68":None if not rms[67] else float(20*np.log10(rms[67]/ref)),
      "raw_iq_all_exact_zero_port67":bool(not np.any(data[:,:,:,66,:])),
      "raw_iq_all_exact_zero_port68":bool(not np.any(data[:,:,:,67,:])),
      "raw_zero_port67_fraction":float(np.mean(data[:,:,:,66,:]==0)),
      "raw_zero_port68_fraction":float(np.mean(data[:,:,:,67,:]==0)),
    }
    return result

def run():
    reports={}
    with tempfile.TemporaryDirectory() as d:
        for key,name in TRIALS.items():
            path=Path(d)/"source.hdf5"
            source=retrieve(name,path)
            with h5py.File(path,"r") as f:
                datasets=all_datasets(f)
                possible=[x["key"] for x in datasets if "pilot" in x["key"].lower()
                          and x["shape"] and len(x["shape"])==5 and x["shape"][3]==80]
                if len(possible)!=1:
                    reports[key]={"source":source,"datasets":datasets[:30],
                                  "status":"ABSTAIN_AMBIGUOUS_PILOT_DATASET"}
                    continue
                probe=probe_antenna_axis(f[possible[0]])
                reports[key]={"source":source,"pilot_dataset":possible[0],
                              "probe":probe,"status":"MEASURED_SOURCE_RAW_IQ_AUDITED"}
            path.unlink(missing_ok=True)
    conclusion={
      "status":"RAW_IQ_RELATIVE_ENERGY_VS_PROCESSED_CSI",
      "hardware_fault_identity":"NOT_VERIFIED",
      "rc3_field_inferences":0,
      "raw_source_data_redistributed":False,
      "scientific_limit":"Nonzero raw receiver energy can be thermal noise or unrelated interference; even low energy cannot alone prove a hardware defect.",
      "records":reports,
    }
    print("LENSFD_RAW_VS_CSI_AUDIT="+json.dumps(conclusion,allow_nan=False),flush=True)
    return conclusion
if __name__=="__main__":run()
