"""Eight-configuration authentic Rice LensFD raw RF integrity + SDR metadata audit.

Source directly from renew.rice.edu. Each original HDF5 file is SHA256
fingerprinted and deleted from runner after measurement; DO NOT publish the
original IQ / configuration source. No physical defect labels and no RC3
inferences are produced. Each original 401-ish capture contains 80 ports,
and multiple lens settings do NOT create independent physical stations.
"""
from __future__ import annotations
import hashlib,json,sys,time
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import urlopen,Request

import h5py
import numpy as np
from generalized.real_rf_v1.raw_iq_quality import analyze_receiver_iq

ROOT="https://renew.rice.edu/datasets/ashu/traces/LensFD/data_LensFD/"
LENSES=("no","sm","md","lg")
CASES=[
   {"env":env,"lens":lens,"raw":f"{where}/{where}-a2c-static/raw/{lens}-{region}/{lens}-{region}-spot-1.hdf5"}
   for env,where,region in (
       ("indoor","indoor","indoor"),
       ("stadium","outdoor","stadium")
   )
   for lens in LENSES
]
FIELDS=("BS_SDR_ID","BS_HUB_ID","BS_CH_PER_RADIO","ANT_OFFSET","ANT_NUM",
        "ANT_TOTAL","BS_RX_GAIN_A","BS_RX_GAIN_B","BS_SDR_NUM_PER_CELL",
        "BS_ANT_NUM_PER_CELL","BS_FRAME_SCHED","FREQ","RATE","RECIPROCAL_CALIB",
        "FULL_MATRIX_MEAS","BS_SDR_ID","BS_SDR_NUM_PER_CELL")

def fetch(src,dest):
    url=ROOT+src
    h=hashlib.sha256()
    n=0
    with urlopen(Request(url,headers={"User-Agent":"SANAD-RC3-Rice-Hardware-Map-RF-Audit"}),timeout=150) as r:
        u=urlsplit(r.url)
        if (u.scheme,u.hostname)!=("https","renew.rice.edu") or r.status!=200:
            raise ValueError("Origin check failed; refusing RF input")
        with dest.open("wb") as w:
            while True:
                b=r.read(1024*1024)
                if not b:break
                n+=len(b)
                if n>680_000_000:raise ValueError("Publisher RF file over expected limit")
                h.update(b);w.write(b)
    return {"bytes":n,"sha256":h.hexdigest()}

def norm_meta(o):
    if isinstance(o,(bytes,np.bytes_)):
        return o.decode("utf-8","replace").strip("\x00")[:120]
    if isinstance(o,(np.integer,np.floating)):
        return o.item()
    if isinstance(o,(int,float,bool,str)):
        return o
    a=np.asarray(o)
    if a.ndim==0:return norm_meta(a.item())
    if a.size<=32:return [norm_meta(x) for x in a.reshape(-1)]
    return {"shape":list(a.shape),"dtype":str(a.dtype),"first":[norm_meta(x) for x in a.reshape(-1)[:12]]}

def get_meta(f):
    result={"root_groups":list(f.keys())}
    for key in ("", "Data", "Metadata"):
        node=f if key=="" else f.get(key)
        if node is None:continue
        fields={}
        for fld in FIELDS:
            if fld in node.attrs:
                fields[fld]=norm_meta(node.attrs[fld])
        if "BS_SDR_ID" in node.attrs and "BS_CH_PER_RADIO" in node.attrs:
            # These are recorder metadata declarations of SDR hardware IDs.
            # The mapping from antenna acquisition index to SDR board is
            # provisional until independently corroborated with wiring.
            radios=np.asarray(node.attrs["BS_SDR_ID"]).reshape(-1)
            channels=int(np.asarray(node.attrs["BS_CH_PER_RADIO"]).reshape(-1)[0])
            fields["SDR_board_count"]=int(radios.size)
            fields["candidate_radio_for_ports67_68_one_based"]={
                "hardware_id_list_index_zero_based":(67-1)//channels,
                "nominal_radio_one_based":1+(67-1)//channels,
                "candidate_SDR_id":norm_meta(radios[(67-1)//channels]),
                "mapping_attested_by_wiring":False,
            }
        result[key or "file_attrs"]=fields
    return result

def assess(f):
    if "Data/Pilot_Samples" not in f:raise ValueError("No original pilot RF data")
    ds=f["Data/Pilot_Samples"]
    if ds.ndim!=5 or ds.shape[3]!=80:
        raise ValueError("Mismatch: expected one 80-port measured Rice receiver")
    k=min(24,int(ds.shape[0]))
    block=ds[:k,:,:,:,:]
    if block.dtype.kind!="i" or block.dtype.itemsize not in (2,4):
        raise ValueError("Unexpected raw RF integer storage type")
    findings=[]
    for port in range(80):
        v=np.ascontiguousarray(block[:,:,:,port,:])
        quality=analyze_receiver_iq(v).as_json()
        if quality["status"].startswith("ABSTAIN_"):
            full=np.asarray(ds[:,:,:,port,:])
            fqual=analyze_receiver_iq(full).as_json()
            findings.append({
                "port_one_based":port+1,
                "first24_quality":quality,
                "all_native_frames_quality":fqual,
            })
    # Always inspect targeted adjacent source indexes 66/67 even when not flagged.
    adj={}
    for idx in (66,67):
        v=np.asarray(ds[:,:,:,idx,:])
        adj[str(idx+1)]=analyze_receiver_iq(v).as_json()
    return {
      "actual_source_frames":int(ds.shape[0]),
      "raw_pilot_shape":list(ds.shape),
      "raw_pilot_dtype":str(ds.dtype),
      "pilot_native_frames_initially_audited":k,
      "invalid_near_constant_ports":findings,
      "paired_ports_67_68":adj,
      "fail_closed_field_inference":True,
    }

def main():
    result=[]
    out=Path("/tmp/lensfd-eight-raw-hardware-sample.hdf5")
    try:
        for i,case in enumerate(CASES,1):
            src=None
            try:
                src=fetch(case["raw"],out)
                with h5py.File(out,"r") as f:
                    meta=get_meta(f)
                    summary=assess(f)
                row={**case,"source":src,"metadata":meta,"raw":summary}
                result.append(row)
                print(f"SOURCE_NATIVE_VERIFIED {i}/{len(CASES)} {case['env']}/{case['lens']} sha256={src['sha256']} muted_ports={[q['port_one_based'] for q in summary['invalid_near_constant_ports']]}",flush=True)
            finally:
                out.unlink(missing_ok=True)
    finally:
        out.unlink(missing_ok=True)
    if len(result)!=8:raise ValueError("Incomplete 8 original RF sessions")
    report={
       "source":"Rice University LensFD published originals",
       "kind":"EIGHT_AUTHENTIC_ARRAY_TO_CLIENT_RAW_RF_CAPTURES",
       "real_data":True,
       "distinct_raw_HDF5_reported_BS_HUB_IDs":sorted(set(
           (r["metadata"]["Data"].get("BS_HUB_ID") or ["UNSPECIFIED"])[0]
           for r in result
       )),
       "distinct_raw_HDF5_BS_HUB_ID_count":len(set(
           (r["metadata"]["Data"].get("BS_HUB_ID") or ["UNSPECIFIED"])[0]
           for r in result
       )),
       "simultaneous_physical_site_timestamps_verified":False,
       "different_recording_hubs_not_equal_5_independent_rc3_sites":True,
       "frozen_RC3_model_unchanged":True,
       "frozen_RC3_field_predictions":0,
       "hardware_fault_labels_verified":False,
       "input_quality_gate_presets_not_independently_validated":True,
       "raw_IQ_redistributed":False,
       "raw_capture_files_examined":len(result),
       "original_source_bytes_downloaded":sum(x["source"]["bytes"] for x in result),
       "results":result,
    }
    print("LENSFD_8_CONFIG_HARDWARE_METADATA_AUDIT="+json.dumps(report,sort_keys=True,allow_nan=False),flush=True)
if __name__=="__main__":main()
