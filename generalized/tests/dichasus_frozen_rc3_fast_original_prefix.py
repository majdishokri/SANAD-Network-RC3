"""Faster original measured CSI prefix (300 genuine frames) frozen-RC3 shadow.

This does NOT claim full file MD5 or physical-hardware 75-hour evaluation.
Checks anchored 48MiB prefix SHA256 + every native TFRecord CRC32C;
retains true original timestamps and 4 physical 16-element array identities.
Fixed 7 disjoint 8-element ROWS from FOUR physical sites purely for numerical
7x8 model compatibility (not seven independent sites).
"""
from __future__ import annotations
import argparse,hashlib,json,os,struct,time
from pathlib import Path
from urllib.parse import urlsplit
import requests
import numpy as np
from generalized.tests.dichasus_real_csi_covariance_shadow import get_metadata,extract_records
from generalized.tests.dichasus_frozen_rc3_measured_shadow import (
    FrozenRC3, GROUPS, SCENARIOS, WINDOW_STEPS, ONSET,
    source_evidence,inject_digital,infer
)

PREFIX=160*1024*1024
CHECK_FIRST=48*1024*1024
CHECK_SHA="2514ca7d55751e08ce2362e93fdec2b7f214a4a9b596b81aade06c88b2be4041"
SOURCE_MD5_META="3e907ad72ae6aab1aa44ef53ba60a3ef"
SPEC_SHA="ddc951c7d9f65ae22206eccdce556be1a76877e176853e62a4b620892d4fea50"

def fetch_prefix(session,source_file):
    if source_file.get("md5")!=SOURCE_MD5_META:
        raise ValueError("Publisher source metadata changed")
    source=f"https://darus.uni-stuttgart.de/api/access/datafile/{source_file['id']}"
    pieces=[];n=0;step=8*1024*1024;fails=0
    while n<PREFIX:
        end=min(n+step,PREFIX)-1
        try:
            with session.get(source,headers={"Range":f"bytes={n}-{end}","User-Agent":"SANAD-RC3-Fast-Original-CSI"},timeout=(30,40),stream=True) as r:
                r.raise_for_status()
                if r.status_code!=206:
                    raise ValueError(f"Source range rejected: {r.status_code}")
                if urlsplit(r.url).hostname not in {"darus.uni-stuttgart.de"} and not (urlsplit(r.url).hostname or "").endswith(".uni-stuttgart.de"):
                    raise ValueError("Unexpected nonpublisher redirect")
                header=r.headers.get("Content-Range","")
                expected=f"bytes {n}-{end}/{source_file['filesize']}"
                if header!=expected:
                    raise ValueError(f"Unverified source range {header!r}; expected {expected!r}")
                before=n
                for c in r.iter_content(1<<20):
                    if not c:continue
                    if n+len(c)>end+1:raise ValueError("Publisher exceeded exact range")
                    pieces.append(c);n+=len(c)
                if n!=end+1:
                    raise requests.exceptions.ChunkedEncodingError("Native source range interrupted")
                fails=0
        except (requests.exceptions.RequestException,):
            fails+=1
            if fails>8:raise RuntimeError("Source prefix download repeatedly interrupted")
            print(f"RETRY_ORIGINAL_CSI_PREFIX offset={n} attempts={fails}",flush=True)
            time.sleep(min(fails,3))
    b=b"".join(pieces)
    if len(b)!=PREFIX or hashlib.sha256(b[:CHECK_FIRST]).hexdigest()!=CHECK_SHA:
        raise ValueError("Publisher measured RF prefix checksum mismatch")
    return b,{"source_prefix_bytes":len(b),"source_first48MiB_SHA256_verified":True,
       "full_file_MD5_verified":False,"source_prefix_SHA256":hashlib.sha256(b).hexdigest(),
       "source_original_native_CRC32C_verification_required":True}

def evaluate(outdir,scenarios):
    session,files,spec,sha=get_metadata()
    if sha!=SPEC_SHA:raise ValueError("Physical antenna mapping source version changed")
    blob,src=fetch_prefix(session,files["dichasus-ad25.tfrecords"])
    physical=[np.asarray(spec["antennas"][p]["assignments"],dtype=int)[row] for p,row in GROUPS]
    if len({int(v) for arr in physical for v in arr})!=56:raise ValueError("Duplicate antennas")
    times=[];cov=[]
    for _,t,H,_ in extract_records(blob,require_complete=False):
        if len(cov)>=300:break
        if not np.isfinite(H).all():raise ValueError("Publisher CSI invalid")
        arr=[]
        for ids in physical:
            h=np.asarray(H[ids],np.complex128)
            R=h@h.conj().T/1024
            if not np.isfinite(R).all() or np.linalg.eigvalsh(R).min()< -1e-7:
                raise ValueError("Invalid physical covariance")
            arr.append(R)
        times.append(float(t));cov.append(arr)
    if len(cov)!=300 or not np.all(np.diff(times)>0):
        raise RuntimeError("Failed to retrieve 300 actual time-ordered original CSI snapshots")
    C=np.asarray(cov)
    model=FrozenRC3()
    original,t_clean=infer(model,C)
    records=[]
    for scenario in scenarios:
        for group in range(7):
            antenna=(group+2)%8
            y,duration=infer(model,inject_digital(C,group,antenna,scenario))
            records.append({
              "scenario":scenario,"target_group":group,"antenna_1based":antenna+1,
              "target_alert":bool(y["alerts"][group]),
              "matched_clean_target_alert":bool(original["alerts"][group]),
              "new_target_alert":bool(y["alerts"][group] and not original["alerts"][group]),
              "target_probability":float(y["probabilities"][group]),
              "clean_target_probability":float(original["probabilities"][group]),
              "paired_probability_delta":float(y["probabilities"][group]-original["probabilities"][group]),
              "alerts_in_other_groups":int(sum(y["alerts"])-int(y["alerts"][group]))
            })
        a=[x for x in records if x["scenario"]==scenario]
        print(f"RC3_TRUE_MODEL_DIGITAL_SHADOW scenario={scenario} target_alerts={sum(x['target_alert'] for x in a)}/7 new_alerts={sum(x['new_target_alert'] for x in a)}/7",flush=True)
    summary={
      "source":"Original DaRUS publisher source / 10.18419/DARUS-4062",
      "data_source":"REAL_MEASURED_COMPLEX_CSI_NOT_SIMULATION",
      "source_fingerprint":src,
      "original_measured_frames":300,
      "native_elapsed_seconds":float(times[-1]-times[0]),
      "real_physical_arrays":4,
      "model_input_groups":7,
      "grouping_physically_independent":False,
      "original_RC3_model_modified":False,
      "threshold":.50,
      "frozen_RC3_real_inference_executed":True,
      "clean_alerts":int(sum(original["alerts"])),
      "clean_decisions":7,
      "clean_alarms_not_verified_hardware_FAR":True,
      "digital_injection_onset_native_frame":ONSET,
      "digital_synthetic_cases":len(records),
      "faults_are_hardware":False,
      "hardware_fault_recall":None,
      "hardware_false_alarm_rate":None,
      "field_qualifying":False,
      "scenario_results":{
        name:{
          "cases":sum(x["scenario"]==name for x in records),
          "target_alerts":sum(x["target_alert"] for x in records if x["scenario"]==name),
          "new_alerts_relative_to_matched_clean":sum(x["new_target_alert"] for x in records if x["scenario"]==name),
          "mean_probability_delta":float(np.mean([x["paired_probability_delta"] for x in records if x["scenario"]==name])),
          "untargeted_group_alerts":sum(x["alerts_in_other_groups"] for x in records if x["scenario"]==name)
        } for name in scenarios
      },
      "matched_case_records":records
    }
    target=Path(outdir);target.mkdir(parents=True,exist_ok=True)
    (target/"original_measured_frozen_RC3_fast_shadow.json").write_text(json.dumps(summary,indent=2,allow_nan=False))
    print("SANAD_RC3_FAST_ORIGINAL_CSI_SHADOW="+json.dumps({k:v for k,v in summary.items() if k!="matched_case_records"},sort_keys=True,allow_nan=False),flush=True)
    return summary

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out-dir",default="out/fast_shadow")
    p.add_argument("--scenarios",nargs="+",default=["port_loss"],choices=list(SCENARIOS))
    a=p.parse_args()
    evaluate(a.out_dir,tuple(a.scenarios))
