"""Cross-hub physical SDR inventory overlap from two genuine original Rice HDF5.

Distinguishes nominal HDF5 board serial identity from independent physical
wiring/failure certification. Source files read from Rice, original bytes
SHA-256 validated and deleted. No frozen RC3 inference or fault labels.
"""
from __future__ import annotations
import hashlib,json,tempfile
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request,urlopen
import h5py,numpy as np

ROOT="https://renew.rice.edu/datasets/ashu/traces/LensFD/data_LensFD/"
SOURCES={
  "indoor":{
      "url":"indoor/indoor-a2c-static/raw/no-indoor/no-indoor-spot-1.hdf5",
      "sha256":"a793135d1e653ff9bc301e79febbaebf9fc3eb794bf10d561730a5967a73dd43",
  },
  "stadium":{
      "url":"outdoor/outdoor-a2c-static/raw/no-stadium/no-stadium-spot-1.hdf5",
      "sha256":"9ff76d8cb4f95b04132fa2161ed572df88df45962715e78be612147e9d9d88e6",
  }
}
def source(path,item):
    h=hashlib.sha256()
    n=0
    with urlopen(Request(ROOT+item["url"],headers={"User-Agent":"SANAD-RC3-mapping-audit"}),timeout=150) as r:
        u=urlsplit(r.url)
        if r.status!=200 or u.scheme!="https" or u.hostname!="renew.rice.edu":
            raise RuntimeError("Unverified publisher source redirect")
        with Path(path).open("wb") as f:
            while True:
                b=r.read(1024*1024)
                if not b:break
                n+=len(b)
                if n>700_000_000:raise RuntimeError("Unexpected source size")
                h.update(b);f.write(b)
    if h.hexdigest()!=item["sha256"]:raise ValueError("Mismatch published raw RF hash")
    return n

def parse(file):
    with h5py.File(file,"r") as f:
        a=f["Data"].attrs
        def scalar(name):
            return int(np.asarray(a[name]).reshape(-1)[0])
        def strings(name):
            return [v.decode("utf8").strip("\x00") if isinstance(v,bytes)
                    else str(v) for v in np.asarray(a[name]).reshape(-1)]
        ch=scalar("BS_CH_PER_RADIO")
        ids=strings("BS_SDR_ID")
        hubs=strings("BS_HUB_ID")
        ant_offset=scalar("ANT_OFFSET")
        n_ant=scalar("ANT_NUM")
        if len(ids)!=40 or ch!=2 or n_ant!=80 or ant_offset!=0:
            raise ValueError("No verifiable 80 receiver port / 40 SDR board array")
        return {"hub":hubs,"ids":ids,"channels_per_radio":ch,
                "antenna_offset":ant_offset,"antenna_count":n_ant}

def audit():
    with tempfile.TemporaryDirectory() as d:
        files=[]
        records={}
        for area,item in SOURCES.items():
            p=Path(d)/(area+".hdf5")
            size=source(p,item)
            records[area]={**parse(p),"sha256":item["sha256"],"bytes":size}
            p.unlink(missing_ok=True)
        a,b=records["indoor"],records["stadium"]
        aset,bset=set(a["ids"]),set(b["ids"])
        shared=sorted(aset&bset)
        shared_same_position=[{"radio_index_1based":i+1,"serial":s} for i,(s,t) in enumerate(zip(a["ids"],b["ids"])) if s==t]
        cross=[{"serial":s,"indoor_radio_index_1based":a["ids"].index(s)+1,
                 "stadium_radio_index_1based":b["ids"].index(s)+1} for s in shared]
        p=(67-1)//a["channels_per_radio"]
        report={
          "source":"Genuine Rice RENEW LensFD HDF5 Data attributes, source SHA256 pins",
          "hardware_fault_verified":False,
          "physical_port_wiring_assumptions_are_unproven":True,
          "original_RC3_inferences":0,
          "hubs":{k:{"hub":v["hub"],"source_sha256":v["sha256"],"SDR_boards":len(v["ids"]),
                     "antennas":v["antenna_count"],"channels_per_board":v["channels_per_radio"]}
                  for k,v in records.items()},
          "reported_distinct_hub_ids":len(set(x["hub"][0] for x in records.values())),
          "unique_SDR_serials_indoor":len(aset),
          "unique_SDR_serials_stadium":len(bset),
          "shared_SDR_serials_count":len(shared),
          "shared_SDR_board_positions_count":len(shared_same_position),
          "shared_SDR_serial_and_positions":shared_same_position,
          "shared_SDR_serials_at_different_positions":[x for x in cross if
                       x["indoor_radio_index_1based"]!=x["stadium_radio_index_1based"]],
          "nominal_candidate_radio_1based":p+1,
          "nominal_candidate_one_based_ports":[67,68],
          "candidate_SDR_serial_indoor":a["ids"][p],
          "candidate_SDR_serial_stadium":b["ids"][p],
          "candidate_matches_across_hubs":bool(a["ids"][p]==b["ids"][p]),
          "distinct_SDR_lists_across_hubs":bool(a["ids"]!=b["ids"]),
          "real_RF_data_redistributed":False,
        }
        print("LENSFD_VERIFIED_SDR_OVERLAP="+json.dumps(report,sort_keys=True),flush=True)
        return report
if __name__=="__main__":audit()
