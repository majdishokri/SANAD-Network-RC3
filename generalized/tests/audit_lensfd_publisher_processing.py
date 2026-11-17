"""Inspect ORIGINAL Rice LensFD code_LensFD.zip without executing publisher code.

Checks if zeros or channel permutations could be created in MATLAB/Python
processing. Only prints matched source lines; do not confuse with physical
failure certification. Published source is publicly downloadable.
"""
from __future__ import annotations
from urllib.request import urlopen, Request
import io, re, zipfile, hashlib, json
URL="https://renew.rice.edu/datasets/ashu/traces/LensFD/code_LensFD.zip"
KEYS=[r"\b(?:reshape|transpose|permutation|permute|swapaxes|moveaxis)\b",
      r"\b(?:zeros|zero|nan_to_num|nanmean|isnan|isfinite|NaN)\b",
      r"\b(?:chan_full|chan_sci|estimate_channel|channel_estimate|channel_matrix)\b",
      r"\b(?:fault|fail|dead|disable|mask|invalid|missing|bad|drop)\b",
      r"\b(?:antenna|radio|port|hub|chain|self_interf|pilot)\b",
      r"\b(?:savetomat|savemat|write|hdf5|h5py)\b"]
pattern=re.compile("|".join(KEYS),re.IGNORECASE)

def inspect_zip(raw):
    z=zipfile.ZipFile(io.BytesIO(raw))
    outputs=[]
    for info in z.infolist():
        if info.is_dir():continue
        if info.file_size>1_000_000:continue
        if not info.filename.endswith((".py",".m",".txt",".md",".json",".cfg",".yaml",".yml")):continue
        src=z.read(info).decode("utf-8","replace").splitlines()
        matches=[]
        for i,line in enumerate(src):
            if pattern.search(line):
                matches.append({"line":i+1,"text":line.strip()[:240]})
        outputs.append({"path":info.filename,"line_count":len(src),"matched":matches[:140]})
    return outputs

if __name__=="__main__":
    req=Request(URL,headers={"User-Agent":"SANAD-RC3-Rice-Code-Audit"})
    with urlopen(req,timeout=60) as r:
        if r.status!=200:raise ValueError(f"HTTP {r.status}")
        raw=r.read(1024*1024)
    if not 5_000<len(raw)<1024*1024:raise ValueError("Unexpected publisher archive byte count")
    result={"source":URL,"size_bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest(),
            "files":inspect_zip(raw)}
    print("PUBLISHER_CODE_AUDIT="+json.dumps(result,ensure_ascii=True),flush=True)

    z=zipfile.ZipFile(io.BytesIO(raw))
    targets={
      "code_LensFD/hdf5process.py":[(105,160),(190,255)],
      "code_LensFD/hdf5analysis.py":[(68,105),(150,182)],
      "code_LensFD/channel_analysis.py":[(240,315)]
    }
    for name,ranges in targets.items():
        if name not in z.namelist():continue
        lines=z.read(name).decode("utf-8","replace").splitlines()
        for start,end in ranges:
            print("ORIGINAL_CODE_SECTION="+json.dumps({
                "path":name,"start":start,"end":end,
                "lines":[f"{idx+1}: {lines[idx]}" for idx in range(start-1,min(end,len(lines)))]
            }),flush=True)
