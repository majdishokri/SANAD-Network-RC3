"""Directly evaluate RF pilot coherence and synchronization on ORIGINAL Rice LensFD IQ.

Not SANAD RC3 field validation. No fault labels fabricated. Downloads/uses each
raw Rice HDF5 ephemerally and runs the original untouched publisher CSI formulas.
Outputs descriptive numeric summaries only, not original CSI arrays or RF data.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import tempfile
import zipfile
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import numpy as np
import h5py
import scipy.io

BASE="https://renew.rice.edu/datasets/ashu/traces/LensFD/"
CAPTURES=[
    {
      "id":"no_indoor_spot1",
      "raw":"data_LensFD/indoor/indoor-a2c-static/raw/no-indoor/no-indoor-spot-1.hdf5",
      "mat":"data_LensFD/indoor/indoor-a2c-static/processed/no-indoor-pro/no-indoor-spot-1.mat",
      "expected_raw_sha":"a793135d1e653ff9bc301e79febbaebf9fc3eb794bf10d561730a5967a73dd43",
      "expected_mat_sha":"d60f4726e334bb7dd284cf686fd71330ea545b66a0db1752056e8605679e024c",
    },
    {
      "id":"sm_indoor_spot1",
      "raw":"data_LensFD/indoor/indoor-a2c-static/raw/sm-indoor/sm-indoor-spot-1.hdf5",
      "mat":"data_LensFD/indoor/indoor-a2c-static/processed/sm-indoor-pro/sm-indoor-spot-1.mat",
      "expected_raw_sha":"c00b7691ae6055aca4ef5342d1558800d49009629652fe4a45f5bfef6422dddf",
      "expected_mat_sha":"4a16686207d0ff3f7d038cfdaf2182ad33be9bad9a408d217efd861051367dd3",
    }
]
CODE_SHA="3aaf2b65024753fa680b096999a104ca249234cec8a427e4488e7ea11ca1e164"

def fetch_publisher(remote,dest,expected,max_bytes):
    url=BASE+remote
    n=0
    digest=hashlib.sha256()
    with urlopen(Request(url,headers={"User-Agent":"SANAD-LensFD-pilot-pathology-v1"}),timeout=180) as r:
        u=urlsplit(r.url)
        if r.status!=200 or (u.scheme,u.hostname)!=("https","renew.rice.edu"):
            raise RuntimeError("Unexpected publisher status/redirect")
        with Path(dest).open("wb") as f:
            for chunk in iter(lambda:r.read(1024*1024),b""):
                n+=len(chunk)
                if n>max_bytes:
                    raise RuntimeError("Publisher file larger than validated budget")
                digest.update(chunk)
                f.write(chunk)
    if digest.hexdigest()!=expected:
        raise ValueError("SOURCE_HASH_MISMATCH; refuse evaluation")
    return {"sha256":digest.hexdigest(),"bytes":n}

def rms_from_iq(pilot,indices):
    if pilot.ndim!=5 or pilot.shape[3]!=80:
        raise ValueError(f"Source IQ shape invalid: {pilot.shape}")
    s=np.take(np.asarray(pilot),indices,axis=3)
    i=s[...,::2].astype(np.float64)
    q=s[...,1::2].astype(np.float64)
    return np.sqrt(np.mean(i*i+q*q,axis=(0,1,2,4)))

def fin_rms(v):
    a=np.asarray(v)
    x=np.abs(a[np.isfinite(a)])
    return float(np.sqrt(np.mean(x*x))) if x.size else None

def relative_db(a,b):
    if a is None or b is None or b<=0:
        return None
    if a==0:
        return None
    return float(20*np.log10(a/b))

def measures(pilot_data,raw,publication,frame_count):
    computed=np.asarray(pilot_data["chan_full"])
    p=np.asarray(publication)
    if computed.shape!=p.shape or p.shape!=(2,80,52):
        raise ValueError("Mismatch publisher array shape/port order")
    csi=np.asarray(pilot_data["csi"])
    if csi.ndim!=6 or csi.shape[3]!=80:
        raise ValueError(f"Original CSI unexpected shape {csi.shape}")
    mf=np.asarray(pilot_data["m_filt"])
    sf=np.asarray(pilot_data["sf_start"])
    if mf.ndim!=5 or mf.shape[3]!=80 or sf.ndim!=4 or sf.shape[3]!=80:
        raise ValueError(f"Original matcher shape {mf.shape}, offset shape {sf.shape}")
    raw_pilots=raw["Data/Pilot_Samples"]
    if raw_pilots.shape[0]<frame_count:
        raise ValueError("Source HDF5 has insufficient actual frames")
    indices=list(range(80))
    raw_rms=rms_from_iq(raw_pilots[:frame_count],indices)
    if raw_rms.shape!=(80,):
        raise ValueError(f"raw_rms shape: {raw_rms.shape}")
    ref=float(np.median(raw_rms[raw_rms>0]))
    # Match-filter quality: peak normalized correlation and fraction of samples
    # whose alignment offset is exactly the algorithm's clipped-zero fallback.
    pmax=np.nanmax(mf,axis=-1)  # frames, cells, users, antennas
    med_match=np.nanmedian(pmax,axis=(0,1,2))
    med_ref=float(np.nanmedian(med_match[np.isfinite(med_match)]))
    out={}
    for idx in (65,66,67,68):
        # CSI before the per-frame phase reference and averaging.
        x=csi[:,:,:,idx,:,:]
        raw_csi_rms=fin_rms(x)
        reduced_rms=fin_rms(computed[:,idx,:])
        published_rms=fin_rms(p[:,idx,:])
        # Relative to the median available 80-port power, not a hardware threshold.
        out[str(idx+1)]={
           "zero_based_index":idx,
           "raw_300_frame_iq_rms_counts":float(raw_rms[idx]),
           "raw_iq_relative_to_80port_median_db":relative_db(float(raw_rms[idx]),ref),
           "pre_average_pilot_csi_rms":raw_csi_rms,
           "reconstructed_phase_normalized_mean_csi_rms":reduced_rms,
           "published_mat_csi_rms":published_rms,
           "normalized_csi_cancellation_db":relative_db(reduced_rms,raw_csi_rms),
           "publisher_mat_exact_zero":bool(np.all(p[:,idx,:]==0)),
           "reconstructed_exact_zero":bool(np.all(computed[:,idx,:]==0)),
           "matched_filter_peak_median":float(med_match[idx]),
           "matched_filter_peak_vs_80port_median_db":relative_db(float(med_match[idx]),med_ref),
           "extraction_clipped_to_sample_zero_frac":float(np.mean(sf[:,:,:,idx]==0)),
           "preaverage_csi_finite_fraction":float(np.isfinite(x).mean()),
        }
    return {
      "native_frame_count":int(frame_count),
      "source_recorded_frame_count":int(raw_pilots.shape[0]),
      "CSI_dims":list(csi.shape),
      "matched_filter_dims":list(mf.shape),
      "frame_start_dims":list(sf.shape),
      "published_vs_reconstruction_complex_RMSE":float(np.sqrt(np.nanmean(np.abs(computed-p)**2))),
      "source_raw_IQ_median_port_RMS":ref,
      "ports":out,
    }

def run(mode="both"):
    with tempfile.TemporaryDirectory() as tmp:
        folder=Path(tmp)
        code=folder/"code_LensFD.zip"
        fetch_publisher("code_LensFD.zip",code,CODE_SHA,1024*1024)
        with zipfile.ZipFile(code) as z:
            for member in z.infolist():
                if member.is_dir():continue
                if member.filename.startswith("/") or ".." in Path(member.filename).parts:
                    raise ValueError("Unsafe zip path")
                if member.file_size>2_000_000:raise ValueError("Unsafe source")
            z.extractall(folder)
        sys.path.insert(0,str(folder/"code_LensFD"))
        from hdf5_lib import hdf5_lib
        from hdf5process import process_hdf5
        reports=[]
        for case in CAPTURES:
            if mode!="both" and case["id"]!=mode:continue
            try:
                dest=folder/"capture.hdf5"
                metadata=fetch_publisher(case["raw"],dest,case["expected_raw_sha"],600_000_000)
                output=folder/"publisher.mat"
                matmeta=fetch_publisher(case["mat"],output,case["expected_mat_sha"],3_000_000)
                with h5py.File(dest) as source:
                    radio=hdf5_lib(str(dest),n_frames_to_inspect=300,n_fr_insp_st=0,sub_sample=1)
                    pilot_data,_=process_hdf5(radio)
                    pub=scipy.io.loadmat(output)["chan_full"]
                    diag=measures(pilot_data,source,pub,300)
                reports.append({
                    "case":case["id"],
                    "raw_hdf5":metadata,
                    "publisher_mat":matmeta,
                    "diagnostics":diag
                })
            finally:
                (folder/"capture.hdf5").unlink(missing_ok=True)
                (folder/"publisher.mat").unlink(missing_ok=True)
    result={
      "stage":"MEASURED_RF_RAW_PILOT_CONSISTENCY",
      "publisher_original_processing_math_modified":False,
      "native_original_rice_rf":True,
      "original_Rice_source_code_SHA256":CODE_SHA,
      "physical_failed_hardware_certified":False,
      "frozen_RC3_inferences":0,
      "source_data_redistributed":False,
      "interpretation":"Coherent pilot loss versus total RF power only. Low signal may be fading, pilot alignment or signal chain defect.",
      "reports":reports
    }
    print("RICE_PILOT_PATHOLOGY_FORENSIC="+json.dumps(result,allow_nan=False,sort_keys=True),flush=True)
    return result

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--case",choices=("both",*[x["id"] for x in CAPTURES]),default="both")
    args=ap.parse_args()
    run(args.case)
