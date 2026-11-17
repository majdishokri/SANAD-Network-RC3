"""Frozen SANAD RC3 endpoint metrics on original measured RF + DIGITAL faults.

RESEARCH SHADOW ONLY, not field qualification. DICHASUS ad25 physically has
FOUR 16-element planar arrays, NOT seven independent RC3 sites. We use 7
disjoint 8-channel rows (from four true arrays) ONLY as seven model inputs,
and 300 REAL subsecond CSI observations as a 300-step *numerical* window,
NOT as 75 h of genuine 15-minute reports. No training or model changes.

Fixed original pretrained RC3 model, threshold 0.5 and original calibration
front end are executed directly. Known digital single-port gain+phase and
port-loss perturbations are injected into truly measured complex covariance
AFTER frame 160; no fake RF is generated and no device is actually faulted.
Claims: snapshot-window synthetic-on-measured-RF sensitivity and controls,
not independent hardware fault recall/FAR, TTD or localization.
"""
from __future__ import annotations

import argparse,csv,hashlib,json,mmap,os,sys,tempfile,time
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import requests
from generalized.tests.dichasus_real_csi_covariance_shadow import (
    get_metadata,extract_records
)

# Preserve frozen original 7x8 geometry. Never retune, retrain or change 0.50.
if (os.environ.get("SANAD_STATIONS","7"),os.environ.get("SANAD_ELEMENTS","8"))!=("7","8"):
    raise RuntimeError("Require ORIGINAL RC3 geometry 7x8; refusing reinterpretation")
os.environ["SANAD_STATIONS"]="7"
os.environ["SANAD_ELEMENTS"]="8"

from generalized.rc3 import FrozenRC3
from generalized.real_rf_v1.rc3_bridge import calibration_vectors_from_covariance,ENGINE
from v32_replay import evidence_stream
from temporal_v35 import extract_delta
from qa_integrity import validate_trace

SOURCE_NAME="dichasus-ad25.tfrecords"
SOURCE_BYTES=596950014
MD5="3e907ad72ae6aab1aa44ef53ba60a3ef"
SHA256="bbd4ff14fc30993bd865cdd37dc53f652c4f4a1beb54e4a5ab0c31d1d083cc74"
SPEC_SHA256="ddc951c7d9f65ae22206eccdce556be1a76877e176853e62a4b620892d4fea50"
WINDOW_STEPS=300
WINDOW_STARTS=(0,300,600)
ONSET=160
RAMP=80

# Seven disjoint 8-channel *rows*, from FOUR authentic 2x8 planar arrays.
# This is NOT seven independent receiver sites.
GROUPS=((0,0),(0,1),(1,0),(1,1),(2,0),(2,1),(3,0))
SCENARIOS={
  "small_ramp":{"gain_db":-0.5,"phase_degrees":5.0,"ramp_frames":80},
  "large_ramp":{"gain_db":-2.0,"phase_degrees":12.0,"ramp_frames":80},
  "port_loss":{"gain_db":-20.0,"phase_degrees":0.0,"ramp_frames":0},
}

def download_source(path,info,session):
    """Reliable strict-byte-range download; require full original source hashes.

    Short 16MB ranges prevent long-lived dead connections. After a recoverable
    network error the next request starts at the exact previously written
    byte offset. No samples are generated, cached, or faked.
    """
    import re
    import time
    if info.get("md5")!=MD5 or int(info.get("filesize",-1))!=SOURCE_BYTES:
        raise ValueError("Original DaRUS metadata/source hash changed")
    source=f"https://darus.uni-stuttgart.de/api/access/datafile/{info['id']}"
    step=16*1024*1024
    offset=0
    md5=hashlib.md5()
    sha=hashlib.sha256()
    recoveries=0
    retries_without_progress=0
    with open(path,"wb") as dest:
        while offset<SOURCE_BYTES:
            before=offset
            end=min(SOURCE_BYTES-1,offset+step-1)
            req_headers={
                "Range":f"bytes={offset}-{end}",
                "User-Agent":"SANAD-RC3-DICHASUS-FROZEN-SHADOW-RESUMABLE/1.0"
            }
            try:
                with session.get(source,stream=True,timeout=(30,50),
                                 headers=req_headers,allow_redirects=True) as r:
                    r.raise_for_status()
                    host=urlsplit(r.url).hostname or ""
                    if host!="darus.uni-stuttgart.de" and not host.endswith(".uni-stuttgart.de"):
                        raise ValueError("Unverified DaRUS data host")
                    if r.status_code!=206:
                        raise ValueError(f"Publisher did not honor source range at offset {offset}: HTTP {r.status_code}")
                    header=r.headers.get("Content-Range","")
                    m=re.fullmatch(r"bytes\s+(\d+)-(\d+)/(\d+)",header)
                    if not m or (int(m[1]),int(m[2]),int(m[3]))!=(offset,end,SOURCE_BYTES):
                        raise ValueError(f"Unverified HTTP Content-Range: {header!r}")
                    for piece in r.iter_content(chunk_size=1<<20):
                        if not piece:continue
                        if offset+len(piece)>end+1:
                            raise ValueError("Publisher response exceeds requested exact source range")
                        dest.write(piece)
                        md5.update(piece)
                        sha.update(piece)
                        offset+=len(piece)
                    if offset!=end+1:
                        raise requests.exceptions.ChunkedEncodingError(
                           f"Partial original range ended {offset}, expected {end+1}")
            except (requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout,
                    requests.exceptions.ChunkedEncodingError) as exc:
                recoveries+=1
                retries_without_progress=0 if offset>before else retries_without_progress+1
                print(f"RETRY_ORIGINAL_DARUS_SOURCE offset={offset} error={type(exc).__name__} recovered={recoveries}",flush=True)
                if retries_without_progress>=6 or recoveries>24:
                    raise RuntimeError("Original DaRUS source repeatedly disconnected") from exc
                time.sleep(min(2.0,.25*retries_without_progress))
            if offset==before and retries_without_progress>=6:
                raise RuntimeError("Original source never made progress")
    if offset!=SOURCE_BYTES or md5.hexdigest()!=MD5 or sha.hexdigest()!=SHA256:
        raise ValueError("Full original RF source SHA256/MD5 verification failed")
    return {
        "downloaded_bytes":offset,"publisher_MD5_whole_file_verified":True,
        "full_SHA256_verified":True,"source_sha256":sha.hexdigest(),
        "publisher_original_http_range_recoveries":recoveries,
        "source_is_complete_without_synthetic_bytes":True,
    }

def measure_covariances(path,spec):
    indices=[]
    names=[]
    for parent,row in GROUPS:
        x=np.asarray(spec["antennas"][parent]["assignments"],dtype=int)
        if x.shape!=(2,8):raise ValueError("Original publisher antenna mapping changed")
        indices.append(np.asarray(x[row],dtype=int))
        names.append(spec["antennas"][parent]["name"])
    if sorted(int(i) for arr in indices for i in arr)!=sorted(set(int(i) for arr in indices for i in arr)):
        raise ValueError("Antenna groups overlap")
    if len({int(k) for arr in indices for k in arr})!=56:
        raise ValueError("Expected 56 UNIQUE authentic source channels")
    Ts=[];Cs=[];min_eig_ratio=1.0
    with open(path,"rb") as f,mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as buf:
        for _,t,H,_snr in extract_records(buf,require_complete=True):
            if not np.isfinite(H).all():raise ValueError("Nonfinite real CSI; do not impute")
            site=[]
            for ids in indices:
                v=np.asarray(H[ids,:],dtype=np.complex128)  # losslessly promote source float32 I/Q before complex covariance accumulation
                R=(v@v.conj().T)/1024.0
                if not np.isfinite(R).all():raise ValueError("Invalid covariance")
                er=np.linalg.norm(R-R.conj().T)
                if er>1e-5*max(np.linalg.norm(R),1e-20):
                    raise ValueError("Measured covariance failed Hermitian gate")
                eig=np.linalg.eigvalsh(R)
                ratio=float(eig.min()/max(float(abs(eig).max()),1e-20))
                min_eig_ratio=min(min_eig_ratio,ratio)
                if ratio<-1e-5:raise ValueError("Measured covariance failed PSD gate")
                site.append(R)
            Ts.append(float(t));Cs.append(site)
    covariance=np.asarray(Cs,dtype=np.complex128)
    ts=np.asarray(Ts,dtype=np.float64)
    if covariance.shape!=(1137,7,8,8):raise ValueError("Wrong native DICHASUS capture shape")
    if not np.all(np.diff(ts)>0):raise ValueError("Publisher native source clock invalid")
    return covariance,ts,{
        "source_antenna_indices_zero_based":[a.tolist() for a in indices],
        "parent_real_array_names":names,
        "authentic_physical_array_count":4,
        "numerically_grouped_rc3_inputs":7,
        "physical_sites_independent_per_original_rc3_contract":False,
        "original_snapshots":len(ts),
        "native_elapsed_seconds":float(ts[-1]-ts[0]),
        "native_median_spacing_s":float(np.median(np.diff(ts))),
        "covariances_evaluated":len(ts)*7,
        "smallest_normalized_covariance_eigen_ratio":min_eig_ratio,
    }

def source_evidence(cov):
    if cov.shape!=(300,7,8,8):raise ValueError("No invented frame counts")
    # EXACT original raw covariance -> calibration proxy -> original v11/v29/v35.
    vectors=calibration_vectors_from_covariance(cov)
    raw,named,feats=evidence_stream(vectors,cal_dir=ENGINE)
    delta=extract_delta(vectors)
    evidence={
      "healthy_v11":raw,"healthy_v11_named":named,
      "healthy_feats":feats,"healthy_delta":delta
    }
    validate_trace(evidence,"healthy",require_truth=False)
    return evidence

def inject_digital(cov,site,port,scenario):
    """Apply D R Dᴴ to ONE antenna. This is digital, not physical hardware."""
    if not(0<=site<7 and 0<=port<8):raise ValueError("Bad digital fault target")
    cfg=SCENARIOS[scenario]
    result=np.array(cov,copy=True)
    for t in range(ONSET,len(result)):
        ramp=1.0 if not cfg["ramp_frames"] else min(1.,(t-ONSET)/cfg["ramp_frames"])
        factor=10.**(cfg["gain_db"]*ramp/20.)*np.exp(1j*np.deg2rad(cfg["phase_degrees"]*ramp))
        # Gain-phase modification on antenna port only.
        result[t,site,port,:]*=factor
        result[t,site,:,port]*=np.conjugate(factor)
    return result

def infer(model,cov):
    t=time.perf_counter()
    p=model.predict_trace(source_evidence(cov),"healthy")
    if len(p["alerts"])!=7 or len(p["probabilities"])!=7:
        raise ValueError("Frozen RC3 prediction shape invalid")
    if p["threshold"]!=.50:raise ValueError("Frozen threshold changed")
    return p,float(time.perf_counter()-t)

def run(output_dir,blocks=1,scenarios=("small_ramp","large_ramp","port_loss")):
    if blocks not in (1,2,3) or any(s not in SCENARIOS for s in scenarios):
        raise ValueError("Unapproved benchmark parameters")
    dest=Path(output_dir);dest.mkdir(parents=True,exist_ok=True)
    session,files,spec,spec_sha=get_metadata()
    if spec_sha!=SPEC_SHA256:raise ValueError("Measured RF physical channel map SHA mismatch")
    with tempfile.TemporaryDirectory() as folder:
        input_file=Path(folder)/SOURCE_NAME
        source=download_source(input_file,files[SOURCE_NAME],session)
        cov,ts,mapping=measure_covariances(input_file,spec)
    model=FrozenRC3()  # immutable original trained artifact with internal SHA guards
    clean=[]
    injected=[]
    runtimes=[]
    for b,start in enumerate(WINDOW_STARTS[:blocks]):
        original=cov[start:start+300]
        if original.shape!=(300,7,8,8):raise ValueError("Insufficient native contiguous source data")
        before,secs=infer(model,original)
        runtimes.append(secs)
        clean.append({
          "window":b,
          "real_native_start_time_s":float(ts[start]),
          "real_native_end_time_s":float(ts[start+299]),
          "real_native_elapsed_seconds":float(ts[start+299]-ts[start]),
          "no_injection_predicted_alert_count":int(sum(before["alerts"])),
          "no_injection_probs":before["probabilities"],
          "no_injection_alerts":before["alerts"],
          "actual_hardware_health_verified":False,
        })
        print(f"MEASURED_CSI_NO_INJECTION window={b} alerts={sum(before['alerts'])}/7",flush=True)
        for name in scenarios:
            for site in range(7):
                # Prespecified independent of CSI values and RC3 predictions.
                ant=(site+2)%8
                artificial=inject_digital(original,site,ant,name)
                p,seconds=infer(model,artificial);runtimes.append(seconds)
                detected=bool(p["alerts"][site])
                neighbor_count=int(sum(p["alerts"])-int(detected))
                injected.append({
                  "window":b,
                  "synthetic_fault_family":name,
                  "digital_target_group_zero_based":site,
                  "digital_target_port_one_based":ant+1,
                  "native_onset_frame":ONSET,
                  "native_onset_time_s":float(ts[start+ONSET]),
                  "native_window_elapsed_seconds":float(ts[start+299]-ts[start]),
                  "digital_gain_db":SCENARIOS[name]["gain_db"],
                  "digital_phase_degrees":SCENARIOS[name]["phase_degrees"],
                  "digital_ramp_frames":SCENARIOS[name]["ramp_frames"],
                  "frozen_rc3_alert_at_target":int(detected),
                  "frozen_rc3_alerts_in_other_groups":neighbor_count,
                  "frozen_rc3_target_probability":float(p["probabilities"][site]),
                  "paired_clean_target_probability":float(before["probabilities"][site]),
                  "paired_target_probability_delta":float(p["probabilities"][site]-before["probabilities"][site]),
                  "paired_clean_target_alert":int(before["alerts"][site]),
                  "label":"DIGITAL_FAULT_ON_MEASURED_CSI_NOT_PHYSICAL_HARDWARE"
                })
            n=[x for x in injected if x["window"]==b and x["synthetic_fault_family"]==name]
            print(f"SANAD_FROZEN_MODEL_SHADOW window={b} scenario={name} digitally_injected_target_alerts={sum(x['frozen_rc3_alert_at_target'] for x in n)}/{len(n)}",flush=True)
    breakdown={}
    for name in scenarios:
        cases=[x for x in injected if x["synthetic_fault_family"]==name]
        n=len(cases)
        breakdown[name]={
          "digitally_injected_test_cases":n,
          "target_alerts":sum(x["frozen_rc3_alert_at_target"] for x in cases),
          "snapshot_window_digital_fault_detection_rate":sum(x["frozen_rc3_alert_at_target"] for x in cases)/n,
          "alerts_on_untargeted_subarray_groups":sum(x["frozen_rc3_alerts_in_other_groups"] for x in cases),
          "mean_paired_target_probability_delta":float(np.mean([x["paired_target_probability_delta"] for x in cases])),
        }
    output={
       "test_name":"SANAD_FROZEN_RC3_ON_AUTHENTIC_DICHASUS_DIGITAL_FAULT_SHADOW",
       "source_doi":"10.18419/DARUS-4062",
       "authentic_real_measured_rf":True,
       "source_metadata":source,
       "source_channel_mapping":mapping,
       "original_rc3_weights_modified":False,
       "original_rc3_detection_threshold":0.50,
       "reference_RC3_7x8_bitwise_equivalence":"CHECK_SEPARATELY_WITH_EXISTING_PARITY_SUITE",
       "real_physical_arrays":4,
       "numerical_subarray_groups":7,
       "subarrays_are_independent_physical_sites":False,
       "source_slices_are_genuine_75_hour_15minute_reports":False,
       "native_real_source_windows":blocks,
       "actual_hardware_fault_truth_available":False,
       "fault_origin":"SYNTHETIC_DIGITAL_INJECTIONS_ON_REAL_RF_NOT_REAL_HARDWARE",
       "performance_basis":"NONQUALIFIED_ENDPOINT_SHADOW",
       "healthy_control_label":"UNMODIFIED_UNLABELED_RF_NOT_CERTIFIED_HEALTHY",
       "total_injected_synthetic_fault_trials":len(injected),
       "observed_clean_endpoint_alerts":sum(x["no_injection_predicted_alert_count"] for x in clean),
       "observed_clean_endpoint_decisions":7*len(clean),
       "observed_clean_alert_rate_NOT_GROUND_TRUTH_FAR":sum(x["no_injection_predicted_alert_count"] for x in clean)/(7*len(clean)),
       "test_breakdown":breakdown,
       "hardware_field_recall":None,
       "hardware_field_false_alarm_rate":None,
       "hardware_field_time_to_detect":None,
       "hardware_field_antenna_localization":None,
       "endpoint_average_seconds_of_local_CSI_processing_and_inference":float(np.mean(runtimes)),
       "naked_RC3_final_endpoint_only_no_true_online_TTD":True,
       "real_RF_source_redistributed":False,
       "clean_windows":clean,
       "injected_trials":injected,
    }
    (dest/"sanad_rc3_dichasus_shadow_results.json").write_text(json.dumps(output,indent=2,allow_nan=False))
    with (dest/"sanad_rc3_dichasus_shadow_trials.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(injected[0]))
        writer.writeheader();writer.writerows(injected)
    print("SANAD_RC3_MEASURED_CSI_DIGITAL_FAULT_SHADOW="+json.dumps({k:v for k,v in output.items() if k not in ("clean_windows","injected_trials","source_channel_mapping")},sort_keys=True,allow_nan=False),flush=True)
    print("RESULT_PATH="+str(dest.resolve()),flush=True)
    return output

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--out-dir",default="generalized/reports/real_csi_rc3_shadow_v1")
    ap.add_argument("--blocks",type=int,choices=(1,2,3),default=1)
    ap.add_argument("--scenarios",nargs="+",choices=tuple(SCENARIOS),default=list(SCENARIOS))
    a=ap.parse_args()
    run(a.out_dir,a.blocks,tuple(a.scenarios))
