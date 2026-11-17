"""Independent measured RF public-source inventory (no SANAD inference).

Zenodo 10.5281/zenodo.21213882 original 37.8MB file, verified against
publisher's MD5 and RECORD API. Only emit metadata, file structure, safe
sample shapes, not copyrighted measurement arrays. 
DICHASUS DaRUS 10.18419/DARUS-4062 spec.json metadata only; no misclaim
that raw 6.2GB+ TFRecords were tested.
"""
from __future__ import annotations
import argparse,hashlib,io,json,os,sys,zipfile
from pathlib import Path
from urllib.parse import urlsplit
import requests
import numpy as np

ZID="21213882"
DOI="10.5281/zenodo.21213882"
ZENODO_API=f"https://zenodo.org/api/records/{ZID}"
OFFICIAL_ARCHIVE_MD5="a19578d10f2edab687136bee42274bf4"
MAX_BYTES=130_000_000

def get_json(url):
    r=requests.get(url,timeout=70,headers={"User-Agent":"SANAD-RC3-RF-Research/1.0"})
    r.raise_for_status()
    if len(r.content)>3_000_000:raise RuntimeError("Unexpectedly large metadata response")
    return r.json()

def fetch_checked(url,path,expected_md5):
    sha=hashlib.sha256();md5=hashlib.md5()
    total=0
    with requests.get(url,timeout=(30,150),stream=True,allow_redirects=True,headers={"User-Agent":"SANAD-RC3-RF-Research/1.0"}) as r:
        r.raise_for_status()
        host=urlsplit(r.url).hostname or ""
        if host not in {"zenodo.org","www.zenodo.org"} and not host.endswith(".zenodo.org"):
            raise RuntimeError(f"Untrusted redirect host: {host}")
        with open(path,"wb") as out:
            for chunk in r.iter_content(1024*1024):
                if not chunk:continue
                total+=len(chunk)
                if total>MAX_BYTES:raise RuntimeError("Unexpected archive exceeds cap")
                sha.update(chunk);md5.update(chunk);out.write(chunk)
    if md5.hexdigest()!=expected_md5:raise ValueError("Original publisher MD5 mismatch")
    return {"publisher_md5_validated":True,"md5":md5.hexdigest(),"sha256":sha.hexdigest(),"size_bytes":total,"last_url_host":urlsplit(r.url).hostname}

def inspect_bytes(name,raw):
    extension=Path(name).suffix.lower()
    summary={"file":name,"bytes":len(raw),"extension":extension}
    if extension==".mat":
        import scipy.io
        try:
            data=scipy.io.whosmat(io.BytesIO(raw))
            summary["mat_variables"]=[{"name":nm,"shape":list(shape),"class":kind} for nm,shape,kind in data[:70]]
        except Exception as e:
            if b"MATLAB 7.3" in raw[:150]:
                summary["mat_format"]="MATLAB_7_3_HDF5"
            else:
                summary["mat_parse_error"]=str(type(e).__name__)+": "+str(e)[:220]
    elif extension==".npy":
        summary["npy_header"]=(raw[:160].hex())
    elif extension in (".csv",".tsv",".txt",".md"):
        summary["first_lines"]=raw[:1000].decode("utf8","replace").splitlines()[:6]
    elif extension in (".json",):
        try:
            j=json.loads(raw)
            summary["json_top_level_keys"]=list(j.keys())[:30] if isinstance(j,dict) else "ARRAY"
        except Exception:summary["json_parse_error"]=True
    elif extension in (".npz",):
        with np.load(io.BytesIO(raw),allow_pickle=False) as f:
            summary["npz_arrays"]={k:{"shape":list(f[k].shape),"dtype":str(f[k].dtype)} for k in f.files[:25]}
    return summary

def inspect_zip(path):
    z=zipfile.ZipFile(path)
    metas=[]
    total_unc=0
    for m in z.infolist():
        if m.is_dir():continue
        n=m.filename
        if n.startswith("/") or ".." in Path(n).parts:raise ValueError("Zip path traversal")
        total_unc+=m.file_size
        if total_unc>4_000_000_000:raise ValueError("Zip expansion over safety cap")
        metas.append({"path":n,"bytes":m.file_size,"compressed_bytes":m.compress_size,
                      "extension":Path(n).suffix.lower()})
    if len(metas)>3000:raise ValueError("Excessive archive members")
    interesting=[m for m in metas if m["extension"] in {".mat",".npz",".npy",".h5",".hdf5",".csv",".tsv",".txt",".json"}]
    # Inspect small source native files only. Never upload the actual source.
    source_inspection=[]
    for m in interesting[:35]:
        if m["bytes"]<=80_000_000:
            raw=z.read(m["path"])
            source_inspection.append(inspect_bytes(m["path"],raw))
    return {"members_count":len(metas),"total_uncompressed_bytes":total_unc,
            "member_summary":metas[:300],
            "parsed_source_files":source_inspection}

def zenodo():
    meta=get_json(ZENODO_API)
    files=meta.get("files",[])
    choices=[f for f in files if str(f.get("key","")).endswith("4.85GHz Dataset.zip")]
    if len(choices)!=1:raise ValueError("Publisher original RF archive was not uniquely identified")
    f=choices[0]
    uri=f["links"].get("self") or f["links"].get("content")
    checksum=f.get("checksum","")
    if "a19578d10f2edab687136bee42274bf4" not in checksum:
        raise ValueError("Zenodo published checksum has changed")
    out=Path("/tmp/sanad-niigata-original-485.zip")
    try:
        source=fetch_checked(uri,out,OFFICIAL_ARCHIVE_MD5)
        archive=inspect_zip(out)
    finally:
        out.unlink(missing_ok=True)
    return {
       "source":"Zenodo publisher original",
       "source_doi":DOI,
       "dataset_type":"REAL_MEASURED_8x8_MIMO_RF",
       "publisher_archive":f["key"],
       "publisher_file_size_bytes":f.get("size"),
       "publisher_file_checksum":checksum,
       "download":source,"archive":archive,
       "original_source_data_redistributed":False,
       "rc3_full_protocol_eligible":False,
       "rc3_field_inferences":0,
    }

def dichasus_spec():
    pid="doi:10.18419/DARUS-4062"
    url="https://darus.uni-stuttgart.de/api/datasets/:persistentId/"
    r=requests.get(url,params={"persistentId":pid},timeout=80)
    r.raise_for_status()
    obj=r.json()
    files=obj["data"]["latestVersion"]["files"]
    spec=[v["dataFile"] for v in files if v["dataFile"].get("filename")=="spec.json"]
    if len(spec)!=1:raise RuntimeError("Unique publisher spec.json absent")
    sf=spec[0]
    u=f"https://darus.uni-stuttgart.de/api/access/datafile/{sf['id']}"
    q=requests.get(u,timeout=70);q.raise_for_status()
    if len(q.content)>100_000:raise ValueError("Spec unexpectedly huge")
    data=q.json()
    return {
      "dataset":"DICHASUS_ADXX",
      "doi":"10.18419/DARUS-4062",
      "source":"DaRUS dataset publisher JSON spec",
      "dataset_version":obj["data"]["latestVersion"].get("versionNumber"),
      "actual_spec_bytes_read":len(q.content),
      "spec_file_id":sf["id"],
      "spec_sha256":hashlib.sha256(q.content).hexdigest(),
      "publisher_spec":data,
      "actual_tfrecords_read":0,
      "rc3_field_inferences":0,
      "rc3_full_protocol_eligible":False
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",choices=("zenodo","dichasus-spec"),required=True)
    a=p.parse_args()
    obj=zenodo() if a.dataset=="zenodo" else dichasus_spec()
    print("SANAD_INDEPENDENT_RF_SOURCE_INVENTORY="+json.dumps(obj,allow_nan=False,sort_keys=True),flush=True)

if __name__=="__main__":main()
