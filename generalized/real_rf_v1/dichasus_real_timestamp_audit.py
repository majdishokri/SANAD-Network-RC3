"""Reproducible DICHASUS native TFRecord time audit from REAL bytes.

Public measured source: DaRUS 10.18419/DARUS-4062, version 3.
Unlike prior exploratory bench, this reads the actual source 'time' float32,
which is seconds since campaign start (NOT necessarily monotonically ordered).
No inferred 48 ms timestamps. No RC3 model inference and no fake 15-min reports.
Downloads exactly the validated first 180,000,000 bytes with HTTP Range.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import urllib.request
from urllib.error import HTTPError
import numpy as np

SOURCE={
    "ad25":{"file_id":664466, "total_bytes":596950014,
            "prefix_sha256":"e1f6cfd47af0d73c4d494fd73d676d7517401a8493d5ce5c2aac83abaa1e230a"},
    "ad21":{"file_id":664462, "total_bytes":1553015076,
            "prefix_sha256":"1c35fe576de3b604d45fb65a91a6942d6918cbdd60242cea01f65847fbf1fb60"},
}
PREFIX_BYTES=180_000_000

def download_verified_prefix(dataset, dest):
    if dataset not in SOURCE:raise ValueError("Unapproved DICHASUS recording")
    p=Path(dest)
    conf=SOURCE[dataset]
    if not p.exists():
        url=f"https://darus.uni-stuttgart.de/api/access/datafile/{conf['file_id']}"
        req=urllib.request.Request(url, headers={"Range":f"bytes=0-{PREFIX_BYTES-1}",
                                                  "User-Agent":"SANAD-RC3-Research-CSI-Audit/1"})
        with urllib.request.urlopen(req,timeout=180) as conn:
            if conn.status!=206:raise RuntimeError("Provider did not honor strict HTTP Range")
            wanted=f"bytes 0-{PREFIX_BYTES-1}/{conf['total_bytes']}"
            if conn.headers.get("Content-Range")!=wanted:
                raise ValueError("Wrong Content-Range for provenance-locked measured bytes")
            with p.open("wb") as output:
                n=0
                while True:
                    part=conn.read(1024*1024)
                    if not part:break
                    n+=len(part)
                    if n>PREFIX_BYTES:raise ValueError("Range downloaded too much data")
                    output.write(part)
    if p.stat().st_size!=PREFIX_BYTES:
        raise ValueError("Measured source data prefix has wrong byte length")
    h=hashlib.sha256()
    with p.open("rb") as f:
        for part in iter(lambda:f.read(1024*1024),b""):h.update(part)
    digest=h.hexdigest()
    if digest!=conf["prefix_sha256"]:
        raise ValueError(f"Real RF source prefix SHA256 mismatch: {digest}")
    return digest

def _varint(b,i):
    v=0;shift=0
    while True:
        if i>=len(b):raise ValueError("Truncated protobuf varint")
        x=b[i];i+=1
        v|=(x&127)<<shift
        if not (x&128):return v,i
        shift+=7
        if shift>63:raise ValueError("Invalid protobuf varint")

def _fields(b):
    d={};i=0
    while i<len(b):
        tag,i=_varint(b,i)
        num=tag>>3;kind=tag&7
        if not num:raise ValueError("Invalid protobuf field number")
        if kind==0:val,i=_varint(b,i)
        elif kind==1:
            if i+8>len(b):raise ValueError("Truncated fixed64")
            val=b[i:i+8];i+=8
        elif kind==2:
            n,i=_varint(b,i)
            if n>len(b)-i:raise ValueError("Truncated length-delimited field")
            val=b[i:i+n];i+=n
        elif kind==5:
            if i+4>len(b):raise ValueError("Truncated fixed32")
            val=b[i:i+4];i+=4
        else:raise ValueError("Unsupported protobuf wire type")
        d.setdefault(num,[]).append((kind,val))
    return d

def _one_bytes(node,field=1):
    v=node.get(field,[])
    if len(v)!=1 or v[0][0]!=2:
        raise ValueError("Expected one length-delimited protobuf field")
    return v[0][1]

def extract_real_time_seconds(example):
    # tf.train.Example = features(1) -> repeated Features.feature(1)
    root=_fields(example)
    feats=_fields(_one_bytes(root))
    matches=[]
    for kind,entry in feats.get(1,[]):
        if kind!=2:raise ValueError("Bad TF Example map field")
        kv=_fields(entry)
        key=_one_bytes(kv,1).decode("utf-8")
        if key!="time":continue
        feature=_fields(_one_bytes(kv,2))
        # tf.train.Feature.float_list is field 2.
        inner=_fields(_one_bytes(feature,2))
        vals=inner.get(1,[])
        flattened=[]
        for wire,v in vals:
            if wire==2: # packed float32
                if len(v)%4:raise ValueError("Invalid float list length")
                flattened.extend(struct.unpack("<"+"f"*(len(v)//4),v))
            elif wire==5:
                flattened.append(struct.unpack("<f",v)[0])
            else:raise ValueError("Unsupported float list encoding")
        if len(flattened)!=1:raise ValueError("Expected exactly one scalar source time")
        matches.append(float(flattened[0]))
    if len(matches)!=1 or not np.isfinite(matches[0]):
        raise ValueError("Missing/invalid authentic source timestamp")
    return matches[0]

def first_record_times(path,limit=300):
    times=[]
    with Path(path).open("rb") as f:
        for _ in range(limit):
            head=f.read(12)
            if len(head)!=12:raise ValueError("Truncated TFRecord before requested samples")
            n=struct.unpack("<Q",head[:8])[0]
            if n==0 or n>8_000_000:raise ValueError("Unreasonable TFRecord size")
            payload=f.read(n);tail=f.read(4)
            if len(payload)!=n or len(tail)!=4:raise ValueError("Truncated TFRecord")
            times.append(extract_real_time_seconds(payload))
    return np.asarray(times,dtype=float)

def audit_times(times,source_name="test"):
    t=np.asarray(times,dtype=float)
    if t.ndim!=1 or len(t)<2 or not np.isfinite(t).all():
        raise ValueError("Need finite authentic file timestamps")
    unique=len(np.unique(t))
    s=np.sort(t)
    duration=float(s[-1]-s[0])
    pos=np.diff(s)
    return {
        "source":source_name,
        "timestamp_type":"REAL_NATIVE_SOURCE_FLOAT32_SECONDS_SINCE_CAMPAIGN_START",
        "number_of_source_records":int(len(t)),
        "unique_timestamps":int(unique),
        "first_file_record_time_s":float(t[0]),
        "last_file_record_time_s":float(t[-1]),
        "earliest_record_time_s":float(s[0]),
        "latest_record_time_s":float(s[-1]),
        "observed_time_span_s":duration,
        "observed_time_span_minutes":duration/60.,
        "out_of_order_transitions_in_file":int(np.sum(np.diff(t)<0)),
        "median_sorted_interrecord_seconds":float(np.median(pos)),
        "max_sorted_gap_s":float(pos.max()),
        "needed_to_qualify_elapsed_seconds":300*900,
        "source_physically_distinct_receiver_sites":4,
        "minimum_unchanged_rc3_independent_receiver_sites":5,
        "true_300_x_15min_window_span_available":duration>=300*900,
        "source_hardware_fault_labels_verified":False,
        "RC3_REAL_FIELD_QUALIFICATION":"BLOCKED",
        "blocked_reasons":[
          *([] if duration>=300*900 else ["LESS_THAN_75_HOURS_OF_REAL_TIMESTAMPS"]),
          "FOUR_PHYSICAL_SITES_LESS_THAN_FIVE_STATION_REFERENCE",
          "NO_VERIFIED_INDEPENDENT_PHYSICAL_PORT_FAILURE_TRUTH",
          "NO_ALIGNED_300_CONSECUTIVE_ACCEPTED_15MIN_WINDOWS"
        ],
        "note":"These are genuine source times, NOT inferred 48 ms spacings or timestamps fabricated from record indices."
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset",choices=sorted(SOURCE),required=True)
    ap.add_argument("--out",required=True)
    ap.add_argument("--prefix-path",default=None)
    args=ap.parse_args()
    path=args.prefix_path or f"/tmp/sanad-dichasus-{args.dataset}-prefix.tfrecords"
    digest=download_verified_prefix(args.dataset,path)
    t=first_record_times(path)
    result=audit_times(t,source_name=f"DICHASUS-{args.dataset}")
    result.update(
        source_doi="10.18419/DARUS-4062",
        original_measured_prefix_sha256=digest,
        source_file_id=SOURCE[args.dataset]["file_id"],
        original_source_bytes_processed=PREFIX_BYTES
    )
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")
    print("DICHASUS_REAL_TIMESTAMP_AUDIT="+json.dumps(result,allow_nan=False),flush=True)
    return result

if __name__=="__main__":
    main()
