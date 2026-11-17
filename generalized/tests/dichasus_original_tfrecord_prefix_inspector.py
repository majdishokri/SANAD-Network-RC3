"""Download only a bounded prefix of original Stuttgart DICHASUS TFRecords.

Never claim whole-file integrity from a *partial* download. Validate each
record's native TFRecord CRC32C. Inspect the publisher's actual feature keys
and serialized shapes; avoid treating metadata alone as measured CSI tests.
No synthetic samples/geometry, fault labels or frozen RC3 inferences.
"""
from __future__ import annotations
import hashlib,json,struct
import requests
import google_crc32c
from tfrecord import example_pb2
from google.protobuf.json_format import MessageToDict

DOI="doi:10.18419/DARUS-4062"
ROOT="https://darus.uni-stuttgart.de"
MAX_PREFIX=48*1024*1024

def mask_crc(data):
    crc=google_crc32c.value(data)
    return (((crc>>15)|(crc<<17))+0xa282ead8)&0xffffffff

def inspect():
    s=requests.Session()
    meta=s.get(ROOT+"/api/datasets/:persistentId/",params={"persistentId":DOI},timeout=85)
    meta.raise_for_status()
    obj=meta.json()["data"]["latestVersion"]
    candidates=[x["dataFile"] for x in obj["files"] if x["dataFile"].get("filename","").endswith(".tfrecords")]
    if not candidates:raise RuntimeError("No original data files")
    short=sorted(candidates,key=lambda f:int(f.get("filesize",10**18)))
    selected=short[0]
    fileid=selected["id"]
    url=f"{ROOT}/api/access/datafile/{fileid}"
    h=hashlib.sha256()
    prefix=bytearray()
    with s.get(url,headers={"Range":f"bytes=0-{MAX_PREFIX-1}","User-Agent":"SANAD-RC3-Original-DICHASUS-CSI-Forensics/1.0"},timeout=(35,180),stream=True) as r:
        r.raise_for_status()
        status=r.status_code
        host=requests.utils.urlparse(r.url).hostname
        if host not in ("darus.uni-stuttgart.de",) and not host.endswith(".uni-stuttgart.de"):
            raise RuntimeError("Untrusted publisher redirect")
        for chunk in r.iter_content(chunk_size=1024*1024):
            if not chunk:continue
            room=MAX_PREFIX-len(prefix)
            if room<=0:break
            prefix.extend(chunk[:room]);h.update(chunk[:room])
            if len(prefix)==MAX_PREFIX:break
    if len(prefix)<1024:raise ValueError("No sufficiently large publisher data received")
    n=0;pos=0;bad=[]
    parsed=[]
    # max 150 native records, bounded by downloaded prefix.
    while pos+16<=len(prefix) and n<150:
        length=struct.unpack_from("<Q",prefix,pos)[0]
        if length<20 or length>3_000_000:
            if n==0:raise ValueError(f"Expected uncompressed native TFRecord; got length={length}")
            bad.append("length_or_boundary")
            break
        stop=pos+16+length
        if stop>len(prefix):break
        header=bytes(prefix[pos:pos+8])
        hcrc=struct.unpack_from("<I",prefix,pos+8)[0]
        payload=bytes(prefix[pos+12:pos+12+length])
        pcrc=struct.unpack_from("<I",prefix,pos+12+length)[0]
        if mask_crc(header)!=hcrc or mask_crc(payload)!=pcrc:
            raise ValueError(f"Original TFRecord record {n} failed CRC32C")
        ex=example_pb2.Example();ex.ParseFromString(payload)
        summary={}
        for k,v in ex.features.feature.items():
            field=v.WhichOneof("kind")
            if field=="bytes_list":
                xs=list(v.bytes_list.value)
                summary[k]={"kind":"bytes_list","count":len(xs),"byte_lengths":[len(z) for z in xs[:5]],"first_16_bytes_hex":[z[:16].hex() for z in xs[:1]]}
            elif field=="float_list":
                summary[k]={"kind":field,"count":len(v.float_list.value),"first_values":list(v.float_list.value[:6])}
            elif field=="int64_list":
                summary[k]={"kind":field,"count":len(v.int64_list.value),"first_values":list(v.int64_list.value[:6])}
        if n<3:parsed.append(summary)
        n+=1;pos=stop
    output={
        "source":"DaRUS University of Stuttgart",
        "source_doi":"10.18419/DARUS-4062",
        "dataset_file":selected["filename"],
        "datafile_id":fileid,
        "published_datafile_bytes":selected.get("filesize"),
        "published_md5":selected.get("md5"),
        "range_request_status_code":status,
        "prefix_source_bytes_downloaded":len(prefix),
        "prefix_sha256":h.hexdigest(),
        "full_file_sha256_not_claimed":True,
        "original_tfrecord_crc32c_verified_records":n,
        "partial_final_record_discarded":True,
        "first_three_native_feature_schemas":parsed,
        "initial_problem_if_any":bad,
        "hardware_fault_labels":None,
        "rc3_qualified":False,
        "rc3_field_inferences":0,
        "original_measured_CSI_redistributed":False
    }
    print("DICHASUS_ORIGINAL_TFRECORD_PREFIX="+json.dumps(output,sort_keys=True,allow_nan=False),flush=True)
    if n<3:raise RuntimeError("Insufficient native original records to constitute a measured RF test")
    return output
if __name__=="__main__":inspect()
