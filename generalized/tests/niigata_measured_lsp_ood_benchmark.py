"""Independent measured urban propagation OOD, NOT frozen RC3 fault recall/FAR.

Use publisher's full checksum-verified 4.85 GHz Yokohama ZIP; read real
field-measured CSV path loss/delay/angles and selection .mat. No simulated
channels or per-port gains/phases; no pseudo RF stations. No raw data upload.

Explicitly report that this release lacks route R5 and usable native
8x8 complex CSI arrays to feed frozen RC3.
"""
from __future__ import annotations
import csv,hashlib,io,json,zipfile
from collections import Counter
from pathlib import Path
import numpy as np
import scipy.io
from scipy.stats import wasserstein_distance
from generalized.tests.independent_ood_source_inventory import (
    ZENODO_API, get_json, fetch_checked, OFFICIAL_ARCHIVE_MD5
)

ROUTES={
  "R3":{"csv":"R3_measured_valid.csv","select":"SelInd_Yuraku_R3_2.mat","prefix":"Yuraku_R3_2_4_8GHz_all_Param.mat",
         "claimed_available_valid":915,"environment":"UMa"},
  "R4":{"csv":"R4_measured_valid.csv","select":"SelInd_NTT_R4.mat","prefix":"NTT_R4_4_8GHz_all_Param.mat",
         "claimed_available_valid":634,"environment":"UMa"},
  "R6":{"csv":"R6_measured_valid.csv","select":"SelInd_WP_R6.mat","prefix":"WP_R6_4_8GHz_all_Param.mat",
         "claimed_available_valid":1735,"environment":"UMi"},
}
METRICS=["PL","DS","ASA","ASD"]
def extract_data(source):
    meta=get_json(ZENODO_API)
    f=next((x for x in meta["files"] if x["key"]=="4.85GHz Dataset.zip"),None)
    if f is None:raise RuntimeError("Publisher source file missing")
    if f.get("checksum")!="md5:"+OFFICIAL_ARCHIVE_MD5:
        raise RuntimeError("Publisher source MD5 changed")
    target=Path("/tmp/sanad-original-niigata-channel-lsp.zip")
    try:
        dl=fetch_checked(f["links"].get("self") or f["links"]["content"],target,OFFICIAL_ARCHIVE_MD5)
        with zipfile.ZipFile(target) as z:
            files=z.namelist()
            output={}
            for route,defn in ROUTES.items():
                c=next((x for x in files if x.endswith("/"+defn["csv"])),None)
                s=next((x for x in files if x.endswith("/"+defn["select"])),None)
                p=next((x for x in files if x.endswith("/"+defn["prefix"])),None)
                if not (c and s and p):
                    raise RuntimeError(f"Original route {route} measurement subset not fully available")
                raw=z.read(c).decode("utf-8-sig")
                rows=list(csv.DictReader(io.StringIO(raw)))
                if len(rows)!=defn["claimed_available_valid"]:
                    raise RuntimeError(f"{route} CSV actual rows differ from publisher metadata")
                values={key:np.asarray([float(r[key]) for r in rows],dtype=np.float64)
                        for key in METRICS+["RxLat","RxLon"]}
                indices=np.asarray([int(float(r["RxIdx"])) for r in rows],dtype=int)
                los=np.asarray([int(float(r["LoS_Tag"])) for r in rows],dtype=int)
                if (len(np.unique(indices))!=len(indices) or not np.all(np.isin(los,[0,1]))
                    or not np.all(np.isfinite(np.concatenate(list(values.values()))))):
                    raise ValueError(f"Malformed measured route values {route}")
                if not (np.all(np.abs(values["RxLat"])<=90) and np.all(np.abs(values["RxLon"])<=180)):
                    raise ValueError("Bad GPS coordinates in measured route")
                sel=scipy.io.loadmat(io.BytesIO(z.read(s)))
                valid=np.asarray(sel["ValidPos"]).reshape(-1).astype(int)
                validlos=np.asarray(sel["ValidPos_LoS"]).reshape(-1).astype(int)
                validnlos=np.asarray(sel["ValidPos_NLoS"]).reshape(-1).astype(int)
                if len(valid)!=len(rows) or len(validlos)+len(validnlos)!=len(valid):
                    raise ValueError("MATLAB selection mask vs source measured route mismatch")
                # Exact same MATLAB source acquisition indices. Some CSVs use
                # consistent one-based source indices; do not invent an offset.
                source_indices_equal=bool(set(indices)==set(valid))
                has_no_nan=all(np.isfinite(v).all() for v in values.values())
                if not has_no_nan:raise ValueError("Nonfinite measured propagation feature")
                binary_los_counts={"LoS":int(sum(los==1)),"NLoS":int(sum(los==0))}
                vals_out={}
                for key in METRICS:
                    v=values[key]
                    vals_out[key]={
                        "median":float(np.median(v)),
                        "p10":float(np.percentile(v,10)),
                        "p90":float(np.percentile(v,90)),
                        "std":float(np.std(v)),
                        "min":float(np.min(v)),
                        "max":float(np.max(v)),
                        "LoS_median":float(np.median(v[los==1])) if any(los==1) else None,
                        "NLoS_median":float(np.median(v[los==0])) if any(los==0) else None,
                    }
                # Explain why all_Param is NOT direct 8x8 complex CSI.
                pmat=scipy.io.loadmat(io.BytesIO(z.read(p)),variable_names=["MPC","NoPoints"])
                mpc=np.asarray(pmat["MPC"],dtype=object)
                source_mpc_slots=int(mpc.size)
                if source_mpc_slots!=3000:raise ValueError("Publisher source MPC slot count unexpected")
                valid_count_actual=int(len(rows))
                output[route]={
                  "environment":defn["environment"],"real_measured_route_records":valid_count_actual,
                  "LoS_labels":binary_los_counts,
                  "publisher_MAT_validpos_count":len(valid),
                  "publisher_MAT_valid_indices_match_csv":source_indices_equal,
                  "publisher_MAT_pos_tags_counts":{"LoS":len(validlos),"NLoS":len(validnlos)},
                  "publisher_MPC_slots":source_mpc_slots,
                  "MPC_is_derived_multipath_parameters_not_native_8x8_CSI":True,
                  "metric_summary":vals_out,
                  "csv_member_sha256":hashlib.sha256(z.read(c)).hexdigest(),
                  "MAT_selection_sha256":hashlib.sha256(z.read(s)).hexdigest(),
                }
            # Statistics across routes from actual observed CSVs: no fault labels.
            out={
              "source":"Zenodo DOI 10.5281/zenodo.21213882, publisher 4.85GHz Dataset.zip",
              "source_archive_sha256":dl["sha256"],
              "source_archive_original_md5_verified":dl["publisher_md5_validated"],
              "physical_RF_measurements":True,
              "measurement_kind":"PUBLISHED_PROCESSED_LSP_FROM_MEASURED_RF",
              "native_8x8_complex_CSI_in_issued_archive":False,
              "published_full_dataset_valid_count_3386_not_all_in_this_release":True,
              "source_routes_in_archive":sorted(output),
              "missing_route_R5":True,
              "processed_urban_RF_samples_tested":sum(x["real_measured_route_records"] for x in output.values()),
              "fault_origin":"NONE_KNOWN_HARDWARE_GROUND_TRUTH_ABSENT",
              "hardware_fault_labels":None,
              "original_RC3_field_predictions":0,
              "original_RC3_full_protocol_eligible":False,
              "synthetic_faults_added":False,
              "recorded_data_redistributed":False,
              "per_route":output,
              "descriptive_path_loss_shifts_dB":{
                "R3_vs_R4":float(output["R3"]["metric_summary"]["PL"]["median"]-output["R4"]["metric_summary"]["PL"]["median"]),
                "R6_vs_R3":float(output["R6"]["metric_summary"]["PL"]["median"]-output["R3"]["metric_summary"]["PL"]["median"]),
              },
              "result_kind":"INDEPENDENT_FIELD_MEASURED_PROPAGATION_OOD_NOT_RC3_INFERENCE"
            }
            print("NIIGATA_ORIGINAL_MEASURED_PROPAGATION_OOD="+json.dumps(out,sort_keys=True,allow_nan=False),flush=True)
            return out
    finally:
        target.unlink(missing_ok=True)
if __name__=="__main__":extract_data(None)
