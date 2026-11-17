"""Publisher-direct, reproducible real Rice LensFD processed-CSI reanalysis.

Tier: measured complex CSI / descriptive port anomalies; NOT RC3 field inference.
188? NO: 112 native static A2C + 80 native self-interference MAT = 192.
32-port FD matrix axes are NOT assumed to mean physical TX or RX.
Source usage: Rice University dataset; DO NOT upload or redistribute source files.
Citation: Chen et al., IEEE TVT (2023), DOI 10.1109/TVT.2023.3240558.
"""
from __future__ import annotations
import argparse, hashlib, itertools, json, math, os, time
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import urlopen, Request
import numpy as np

ROOT="https://renew.rice.edu/datasets/ashu/traces/LensFD/data_LensFD/"
# Exact SHA-256 from previously independently downloaded official Rice MAT source files.
# These are source-integrity controls, NOT known failure labels or inference labels.
KNOWN={
"indoor/indoor-a2c-static/processed/no-indoor-pro/no-indoor-spot-1.mat":"d60f4726e334bb7dd284cf686fd71330ea545b66a0db1752056e8605679e024c",
"indoor/indoor-a2c-static/processed/sm-indoor-pro/sm-indoor-spot-1.mat":"4a16686207d0ff3f7d038cfdaf2182ad33be9bad9a408d217efd861051367dd3",
"indoor/indoor-a2c-static/processed/md-indoor-pro/md-indoor-spot-1.mat":"55aea29ed5be740fcb3c78c84148d957d96cb7c40d7507a314d944a268372bed",
"indoor/indoor-a2c-static/processed/lg-indoor-pro/lg-indoor-spot-1.mat":"77ba1f51f8be40f8c31c4e15237b8591ed2ab112d70e8bacce5f07ed8b138a11",
"outdoor/outdoor-a2c-static/processed/no-stadium-pro/no-stadium-spot-1.mat":"0191f90a4ee40889dad04cb4af9cf30578409a544bedfaf84d7d1bd9beaefbf0",
"outdoor/outdoor-a2c-static/processed/sm-stadium-pro/sm-stadium-spot-1.mat":"ca56c5ffcf2b7a42318e35a5720b56e70dc79efc0674aaad3c848cd7f2d1342f",
"indoor/indoor-fd/processed/no-indoor-int-pro/no-indoor-Ch12.mat":"63b4b7e44d7b75f38c76543c1079d72c1bcb92fa032572b2340769b60b14a7dc",
"indoor/indoor-fd/processed/no-indoor-int-pro/no-indoor-Ch15.mat":"f9e7e9c04d38a7da524d15c53497b6d8d1a168a563c2ae947bc11874a4cf1d42",
"indoor/indoor-fd/processed/lg-indoor-int-pro/lg-indoor-Ch45.mat":"a504be56bbc2af4461c4903cccc65276bcd50445384eaeccc26f6af042aebb17",
"indoor/indoor-fd/processed/lg-indoor-int-pro/lg-indoor-Ch12.mat":"44b8e6c44782b5bf910e0140cde278d51d5177c18e7762755e0b009937cba059",
}
LENSES=("no","sm","md","lg")
ENVS=(("indoor","indoor"),("outdoor","stadium"))
CHAINS=tuple(itertools.combinations("12345",2))

def expected_cases():
    cases=[]
    for folder,env in ENVS:
        for lens in LENSES:
            for loc in range(1,15):
                name=f"{lens}-{env}-spot-{loc}.mat"
                relative=f"{folder}/{folder}-a2c-static/processed/{lens}-{env}-pro/{name}"
                cases.append({"path":relative,"family":"a2c-static","lens":lens,"env":env,"location":loc})
            for a,b in CHAINS:
                name=f"{lens}-{env}-Ch{a}{b}.mat"
                relative=f"{folder}/{folder}-fd/processed/{lens}-{env}-int-pro/{name}"
                cases.append({"path":relative,"family":"fd","lens":lens,"env":env,"chain_pair":a+b})
    if len(cases)!=192 or len(set(x["path"] for x in cases))!=192:
        raise AssertionError("LensFD publisher's exact 192 MAT case contract changed")
    return cases

def fetch_verified(relative, work, *, max_bytes=16_000_000):
    """One official publisher HTTPS location, finite bounded bytes, integrity checksum."""
    if relative not in {v["path"] for v in expected_cases()}:
        raise ValueError("Unmanifested publisher relative path")
    dest=Path(work);dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():dest.unlink()  # no stale unverified cached source
    url=ROOT+relative
    for retry in range(3):
        try:
            req=Request(url,headers={"User-Agent":"SANAD-Network-RC3-LensFD-Research/1.0"})
            with urlopen(req,timeout=90) as response:
                dest_url=urlsplit(response.geturl())
                if response.status!=200 or dest_url.hostname!="renew.rice.edu" or dest_url.scheme!="https":
                    raise ValueError("Unexpected publisher response or redirect host")
                h=hashlib.sha256()
                size=0
                with dest.open("wb") as out:
                    while True:
                        block=response.read(1024*1024)
                        if not block:break
                        size+=len(block)
                        if size>max_bytes:
                            raise ValueError("Unexpected oversized publisher MAT file")
                        h.update(block);out.write(block)
            digest=h.hexdigest()
            if relative in KNOWN and digest!=KNOWN[relative]:
                raise ValueError(f"Official publisher SHA256 changed for {relative}")
            return {"sha256":digest,"bytes":size,"publisher_url":url}
        except Exception:
            dest.unlink(missing_ok=True)
            if retry==2:raise
            time.sleep(1+retry*2)

def per_port_rms(channel, axis):
    a=np.asarray(channel)
    if not np.iscomplexobj(a) or not np.isfinite(a).all():
        raise ValueError("Require measured finite complex channel, not magnitude-only data")
    x=np.moveaxis(a,axis,0).reshape(a.shape[axis],-1)
    return np.sqrt(np.mean(np.abs(x)**2,axis=1))

def relative_rms_db(rms,index):
    positive=np.asarray(rms)[np.asarray(rms)>0]
    if not len(positive):raise ValueError("No positive physical CSI ports")
    value=float(rms[index])
    return None if value==0 else 20*math.log10(value/float(np.median(positive)))

def examine_native_mat(path,meta):
    import scipy.io
    raw=scipy.io.loadmat(path,verify_compressed_data_integrity=True)
    if not {"chan_full","chan_sci"}.issubset(raw):
        raise ValueError("Missing official processed-complex CSI variables")
    a,s=np.asarray(raw["chan_full"]),np.asarray(raw["chan_sci"])
    want_full=(2,80,52) if meta["family"]=="a2c-static" else (32,32,52)
    want_sci=(2,80) if meta["family"]=="a2c-static" else (32,32)
    if a.shape!=want_full or s.shape!=want_sci or not np.iscomplexobj(s):
        raise ValueError(f"Publisher CSI shape/types differ: {a.shape}, {s.shape}")
    if not np.isfinite(a).all() or not np.isfinite(s).all():
        raise ValueError(f"Nonfinite publisher CSI {meta['path']}: full={int(np.size(a)-np.isfinite(a).sum())} sci={int(np.size(s)-np.isfinite(s).sum())}")
    row={k:meta[k] for k in meta}
    row["values_scanned"]=int(a.size+s.size)
    if meta["family"]=="a2c-static":
        port=per_port_rms(a,1)
        row.update(port67_exact_zero=bool(port[66]==0),
                   port68_exact_zero=bool(port[67]==0),
                   quietest_port_1based=int(np.argmin(port))+1,
                   port67_relative_db=relative_rms_db(port,66),
                   port68_relative_db=relative_rms_db(port,67))
    else:
        c0=per_port_rms(a,0)
        c1=per_port_rms(a,1)
        db0=relative_rms_db(c0,19);db1=relative_rms_db(c1,19)
        row.update(includes_chain5=meta["chain_pair"].endswith("5"),
                   local_axis0_index19_relative_db=db0,
                   local_axis1_index19_relative_db=db1,
                   local_axis0_index19_exact_zero=bool(c0[19]==0),
                   local_axis1_index19_exact_zero=bool(c1[19]==0),
                   local_axis1_index19_below20db=bool(db1 is None or db1 < -20))
    row["rc3_field_inference"]=False
    return row

def summarize(rows,requested,actual_total_bytes):
    a2c=[x for x in rows if x["family"]=="a2c-static"]
    fd=[x for x in rows if x["family"]=="fd"]
    c5=[x for x in fd if x["includes_chain5"]]
    controls=[x for x in fd if not x["includes_chain5"]]
    return {
        "benchmark":"RICE_LENSFD_NATIVE_MEASURED_COMPLEX_CSI_REANALYSIS",
        "source":"Rice RENEW LensFD, https://renew.rice.edu/dataset-lens.html",
        "citation":"Chen et al. (2023), IEEE Transactions on Vehicular Technology, DOI:10.1109/TVT.2023.3240558",
        "research_license":"Rice data: no redistribution of original measurements without permission",
        "source_is_real_measured_csi":True,
        "measurement_kind":"NATIVE_PROCESSED_CSI_NOT_A_15_MINUTE_TIME_SERIES",
        "input_geometry":"ONE_80_PORT_UNIFORM_PLANAR_ARRAY_NOT_5_INDEPENDENT_SITES",
        "requested_files":requested,
        "native_mat_files_passed":len(rows),
        "native_mat_expected_full_corpus":192,
        "native_csi_values_scanned":sum(x["values_scanned"] for x in rows),
        "publisher_bytes_downloaded":actual_total_bytes,
        "source_sha256_pins_matched":sum(x["path"] in KNOWN for x in rows),
        "static_a2c_files":len(a2c),
        "static_port68_exact_zero":sum(x["port68_exact_zero"] for x in a2c),
        "static_port67_exact_zero":sum(x["port67_exact_zero"] for x in a2c),
        "static_neither_port67_nor_68":sum(not(x["port67_exact_zero"] or x["port68_exact_zero"]) for x in a2c),
        "si_files":len(fd),
        "si_chain5_captures":len(c5),
        "si_no_chain5_controls":len(controls),
        "si_chain5_axis1_index19_below20db":sum(x["local_axis1_index19_below20db"] for x in c5),
        "si_otherchain_axis1_index19_below20db":sum(x["local_axis1_index19_below20db"] for x in controls),
        "si_chain5_axis1_index19_exact_zero":sum(x["local_axis1_index19_exact_zero"] for x in c5),
        "physical_port_fault_truth_verified":False,
        "zero_port_could_be_processing_mask":True,
        "research_score_threshold_minus20_db_is_posthoc":True,
        "rows_are_correlated_repeated_observations":True,
        "frozen_rc3_changed":False,
        "frozen_rc3_threshold":0.50,
        "frozen_rc3_real_field_inferences":0,
        "rc3_hardware_fault_recall":None,
        "rc3_hardware_fault_far":None,
        "rc3_qualification":"OUT_OF_SCOPE_SINGLE_UPA_NON_TEMPORAL_MISSING_PHYSICAL_FAULT_LABELS",
        "raw_mat_measurements_redistributed":False,
    }

def run(out_dir,mode="full"):
    allcases=expected_cases()
    if mode=="smoke":
        selected=[x for x in allcases if x["path"] in KNOWN]
    elif mode=="full":
        selected=allcases
    else:raise ValueError("mode must be smoke/full")
    if mode=="smoke" and len(selected)!=len(KNOWN):
        raise ValueError("Smoke source hash matrix incomplete")
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    cache=out/".temporary_source_file.mat"
    rows=[];provenance=[];bytes_total=0
    try:
        for i,case in enumerate(selected,1):
            try:
                source=fetch_verified(case["path"],cache)
                observed=examine_native_mat(cache,case)
                bytes_total+=source["bytes"]
                rows.append(observed)
                provenance.append({"path":case["path"],"sha256":source["sha256"],"bytes":source["bytes"]})
                print(f"MEASURED_RICE_MAT {i}/{len(selected)} SHA256_VERIFIED {case['path']}",flush=True)
            finally:
                cache.unlink(missing_ok=True)
    finally:
        cache.unlink(missing_ok=True)
    summary=summarize(rows,len(selected),bytes_total)
    if mode=="full":
        if len(rows)!=192 or summary["static_a2c_files"]!=112 or summary["si_files"]!=80:
            raise AssertionError("Incomplete published LensFD processed CSI measurement corpus")
    with (out/"publisher_native_sha256.jsonl").open("w") as f:
        for x in provenance:f.write(json.dumps(x,sort_keys=True)+"\n")
    (out/"research_summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True),encoding="utf-8")
    # Per-measurement diagnostic indicators are derived metadata, never actual CSI.
    with (out/"descriptive_measurement_checks.jsonl").open("w") as f:
        for x in rows:f.write(json.dumps(x,sort_keys=True,allow_nan=False)+"\n")
    print("LENSFD_NATIVE_MEASURED_RESULT="+json.dumps(summary,sort_keys=True),flush=True)
    return summary

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--mode",choices=("smoke","full"),default="full")
    p.add_argument("--out",default="results/lensfd")
    a=p.parse_args()
    run(a.out,a.mode)
