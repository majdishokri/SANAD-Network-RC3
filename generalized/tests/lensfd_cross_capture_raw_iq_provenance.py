"""Cross-capture byte provenance: are LensFD's 'silent' ports copied/masked?

Reads two original 401-frame publisher HDF5 files, never redistributes IQ.
Compares exact SHA256 of each antenna stream across captures and checks
suspect pair 68 (no lens) versus 67 (small lens). Does NOT infer failure truth.
"""
from __future__ import annotations
import hashlib
import json
import tempfile
from pathlib import Path
import h5py
import numpy as np
from generalized.tests.lensfd_raw_iq_port_origin import TRIALS, retrieve

EXPECTED={
  "indoor_no_lens_zero_port68":"a793135d1e653ff9bc301e79febbaebf9fc3eb794bf10d561730a5967a73dd43",
  "indoor_small_lens_zero_port67":"c00b7691ae6055aca4ef5342d1558800d49009629652fe4a45f5bfef6422dddf"
}

def port_stats(source,port):
    # Exact original receiver port bytes; shape [frame,client,I/Q] (one cell).
    d=np.ascontiguousarray(source["Data/Pilot_Samples"][:,0,:,port,:])
    s=np.asarray(d,dtype=np.float64)
    vals=s.reshape(-1)
    return {
      "index_zero_based":int(port),
      "frame_count":int(d.shape[0]),
      "I_Q_samples":int(d.size),
      "source_slice_sha256":hashlib.sha256(d.tobytes()).hexdigest(),
      "zero_fraction":float(np.mean(d==0)),
      "rms_counts":float(np.sqrt(np.mean(s*s))),
      "unique_values":int(np.unique(d).size),
      "min":float(vals.min()),
      "max":float(vals.max()),
    },d

def run():
    with tempfile.TemporaryDirectory() as d:
        handles=[]
        metadata=[]
        try:
            for name,rel in TRIALS.items():
                p=Path(d)/f"{name}.hdf5"
                stat=retrieve(rel,p)
                if stat["sha256"]!=EXPECTED[name]:raise ValueError("Unexpected raw source SHA256")
                metadata.append({"case":name,**stat})
                handles.append(h5py.File(p,"r"))
            x,y=handles
            if x["Data/Pilot_Samples"].shape!=y["Data/Pilot_Samples"].shape:
                raise ValueError("Cannot compare different recording shapes directly")
            # Index 67 (one-based 68) quiet in no-lens, 66 (one-based 67) quiet in sm.
            a,xa=port_stats(x,67)
            b,xb=port_stats(y,66)
            c,xc=port_stats(x,66)
            e,xe=port_stats(y,67)
            def compare(v,w):
                v=v.astype(np.float64).reshape(-1)
                w=w.astype(np.float64).reshape(-1)
                diff=v-w
                v0=v-v.mean();w0=w-w.mean()
                denom=np.linalg.norm(v0)*np.linalg.norm(w0)
                return {"exact_bitwise_identical":bool(np.array_equal(v,w)),
                        "pearson_corr":float(v0@w0/denom) if denom else None,
                        "fraction_equal_IQ_values":float(np.mean(v==w)),
                        "difference_rms_counts":float(np.sqrt(np.mean(diff*diff)))}
            comparisons={
               "silent_no68_vs_silent_sm67":compare(xa,xb),
               "strong_no67_vs_strong_sm68":compare(xc,xe),
               "silent_no68_vs_strong_sm68":compare(xa,xe),
               "strong_no67_vs_silent_sm67":compare(xc,xb),
            }
            # Determine whether same-chain port streams were copied or permuted.
            hashes=[]
            for f in handles:
                hashes.append([port_stats(f,port)[0]["source_slice_sha256"] for port in range(80)])
            duplicates=[]
            for i,h in enumerate(hashes[0]):
                for j,k in enumerate(hashes[1]):
                    if h==k:duplicates.append([i+1,j+1])
            result={
              "stage":"RICE_LENSFD_CROSS_RECORDING_NATIVE_IQ_PROVENANCE",
              "source_captures":metadata,
              "tested_full_native_recording_not_invented_timestamps":True,
              "measured_80port_streams_hash_compared":80,
              "identical_raw_receiver_stream_pairings":duplicates,
              "suspect_ports":{
                "no_lens_port68":a,
                "small_lens_port67":b,
                "no_lens_port67":c,
                "small_lens_port68":e
              },
              "comparisons":comparisons,
              "port_wiring_mapping_verified":False,
              "actual_hardware_failures_verified":False,
              "RC3_field_inferences":0,
              "raw_signal_redistributed":False,
            }
            print("LENSFD_CROSS_CAPTURE_RAW_IQ="+json.dumps(result,allow_nan=False,sort_keys=True),flush=True)
            return result
        finally:
            for h in handles:h.close()

if __name__=="__main__":run()
