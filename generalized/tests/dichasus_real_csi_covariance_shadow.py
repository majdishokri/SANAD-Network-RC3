"""Independent REAL DICHASUS 4x(2x8) measured CSI shadow-domain experiment.

Downloads two original DaRUS TFRecord byte-range prefixes (one with a metal
wall, one without). Validates each native TFRecord CRC32C and exactly decodes
the source TensorProto CSI, time and SNR; computes native physical 4-array
covariance PSD, correlation, power and time. DOES NOT call/alter RC3, create
new sites, convert 48ms to 15m, or claim hardware-fault Recall/FAR.

Optional digitally imposed -3dB/+5deg changes are a strictly separate
math sanity check on measured complex channels (not a detector result).
"""
from __future__ import annotations
import hashlib,json,math,struct
from urllib.parse import urlsplit
import numpy as np,requests,google_crc32c
from tfrecord import example_pb2
from tensorboard.compat.proto import tensor_pb2
from generalized.tests.dichasus_original_tfrecord_prefix_inspector import mask_crc

ROOT="https://darus.uni-stuttgart.de"
DOI="doi:10.18419/DARUS-4062"
EXPECTED_SPEC_SHA="ddc951c7d9f65ae22206eccdce556be1a76877e176853e62a4b620892d4fea50"
CASES=("dichasus-ad00.tfrecords","dichasus-ad25.tfrecords")
MAX_PREFIX=48*1024*1024
MIN_RECORDS=80

def get_metadata():
    session=requests.Session()
    r=session.get(ROOT+"/api/datasets/:persistentId/",params={"persistentId":DOI},timeout=90)
    r.raise_for_status()
    files={f["dataFile"]["filename"]:f["dataFile"] for f in r.json()["data"]["latestVersion"]["files"]}
    if "spec.json" not in files:raise ValueError("Missing publisher original spec.json")
    sf=files["spec.json"]
    src=session.get(ROOT+f"/api/access/datafile/{sf['id']}",timeout=80)
    src.raise_for_status()
    sha=hashlib.sha256(src.content).hexdigest()
    if sha!=EXPECTED_SPEC_SHA:raise ValueError("DICHASUS native antenna mapping spec changed")
    spec=src.json()
    if len(spec["antennas"])!=4:raise ValueError("Expected four true physical arrays")
    idx=[]
    for arr in spec["antennas"]:
        a=np.asarray(arr["assignments"])
        if a.shape!=(2,8):raise ValueError("Expected physical planar 2x8, not ULA")
        idx.extend(a.reshape(-1).tolist())
    if sorted(idx)!=list(range(64)):
        raise ValueError("Real publisher antenna assignments must cover original 64 exactly once")
    for name in CASES:
        if name not in files:raise RuntimeError("Missing requested original measured TFRecord")
    return session,files,spec,sha

def load_prefix(session,file):
    id=file["id"]
    url=ROOT+f"/api/access/datafile/{id}"
    chunks=[]
    total=0
    h=hashlib.sha256()
    with session.get(url,stream=True,timeout=(45,190),
       headers={"Range":f"bytes=0-{MAX_PREFIX-1}","User-Agent":"SANAD-independent-RF-64-CSI-benchmark"},
       allow_redirects=True) as r:
        r.raise_for_status()
        dest=urlsplit(r.url).hostname
        if dest!="darus.uni-stuttgart.de" and not dest.endswith(".uni-stuttgart.de"):
            raise RuntimeError("Unexpected source data redirect")
        for piece in r.iter_content(1<<20):
            if not piece:continue
            cap=MAX_PREFIX-total
            if cap==0:break
            chunk=piece[:cap]
            chunks.append(chunk);h.update(chunk);total+=len(chunk)
            if total==MAX_PREFIX:break
        status=r.status_code
    return b"".join(chunks),{"http_status":status,
         "publisher_file_id":id,"publisher_file_name":file["filename"],
         "published_full_file_bytes":file.get("filesize"),"published_MD5_not_whole_file_verified":file.get("md5"),
         "downloaded_source_prefix_bytes":total,"prefix_sha256":h.hexdigest(),
         "full_file_checksum_verified":False,
         "source_uses_unmodified_original_measured_bytes":True}

def tensor(feature):
    if len(feature.bytes_list.value)!=1:raise ValueError("Need one publisher TensorProto")
    t=tensor_pb2.TensorProto()
    t.ParseFromString(feature.bytes_list.value[0])
    shape=tuple(int(dim.size) for dim in t.tensor_shape.dim)
    # 1 = TensorFlow DT_FLOAT; 2 = DT_DOUBLE; no implicit quantization.
    dt={1:np.dtype("<f4"),2:np.dtype("<f8")}.get(t.dtype)
    if dt is None:raise ValueError(f"Unexpected publisher TensorProto dtype={t.dtype}")
    arr=np.frombuffer(t.tensor_content,dtype=dt)
    if arr.size!=math.prod(shape):
        raise ValueError(f"TensorProto content/shape invalid: dtype {t.dtype} shape {shape} len={arr.size}")
    return arr.reshape(shape)

def extract_records(buf):
    pos=0;count=0
    while pos+16<=len(buf):
        size=struct.unpack_from("<Q",buf,pos)[0]
        if size<100 or size>4_000_000:raise RuntimeError("Malformed original TFRecord length")
        nextpos=pos+size+16
        if nextpos>len(buf):break
        header=buf[pos:pos+8]
        hcrc=struct.unpack_from("<I",buf,pos+8)[0]
        p=buf[pos+12:pos+12+size]
        pcrc=struct.unpack_from("<I",buf,pos+12+size)[0]
        if mask_crc(header)!=hcrc or mask_crc(p)!=pcrc:
            raise RuntimeError("Original DICHASUS record CRC failed")
        e=example_pb2.Example();e.ParseFromString(p)
        f=e.features.feature
        if not {"csi","time","snr"}.issubset(f):
            raise RuntimeError("Original time/csi/snr not present")
        v=tensor(f["csi"])
        if v.shape!=(64,1024,2):raise ValueError(f"Publisher physical CSI tensor unexpected shape {v.shape}")
        H=(v[:,:,0]+1j*v[:,:,1]).astype(np.complex64)
        snr=tensor(f["snr"])
        if snr.shape!=(64,):raise ValueError(f"SNR original shape {snr.shape}")
        t=float(f["time"].float_list.value[0])
        yield count,t,H,snr
        count+=1;pos=nextpos

def evaluate_case(source,metadata,physical):
    time=[]
    arrays=[np.asarray(a["assignments"],dtype=int).reshape(-1) for a in physical]
    per={a["name"]:[] for a in physical}
    count=0;invalid=0;min_psd_ratio=1.0
    digital_gains=[];digital_phase=[]
    for n,tm,H,snr in extract_records(source):
        count+=1;time.append(tm)
        if not np.isfinite(H).all():
            invalid+=int((~np.isfinite(H)).sum())
            # Fail closed to preserve genuine source-value constraints; do not fill missing.
            raise RuntimeError(f"Invalid measured complex CSI at native record {n}")
        for a,idx in zip(physical,arrays):
            sub=H[idx,:]
            R=(sub@sub.conj().T)/sub.shape[1]
            asym=np.linalg.norm(R-R.conj().T)/max(float(np.linalg.norm(R)),1e-30)
            if asym>1e-5:raise ValueError("Invalid Hermitian measured covariance")
            lam=np.linalg.eigvalsh(R)
            maxeig=max(float(np.max(np.abs(lam))),1e-30)
            minrat=float(np.min(lam))/maxeig
            min_psd_ratio=min(min_psd_ratio,minrat)
            if minrat<-1e-5:raise ValueError("Measured covariance violates PSD")
            port_pow=np.mean(np.abs(sub)**2,axis=1,dtype=np.float64)
            medpow=float(np.median(port_pow))
            if medpow<=0 or not math.isfinite(medpow):raise ValueError("Empty measured CSI power")
            norm=R/np.sqrt(np.maximum(np.outer(port_pow,port_pow),1e-30))
            coh=np.abs(norm[np.triu_indices(16,1)])
            ratio=np.max(port_pow)/np.min(port_pow[port_pow>0]) if np.any(port_pow>0) else float("inf")
            per[a["name"]].append({
                "mean_rx_channel_power":float(np.mean(port_pow)),
                "median_rx_channel_power":medpow,
                "worst_best_port_power_spread_db":float(10*np.log10(ratio)) if math.isfinite(ratio) else None,
                "inter_element_correlation_median_abs":float(np.median(coh)),
                "csi_finite_fraction":1.0,
                "native_snr_median":float(np.median(snr[idx]))
            })
            # DELIBERATELY SYNTHETIC gain/phase change on actual observed channel.
            # This is a numerical source-aware transformation test, not detection.
            reference=sub[0,:]
            pert=(10**(-3.0/20))*np.exp(1j*np.deg2rad(5.0))*reference
            if float(np.mean(np.abs(reference)**2))>0:
                gain=10*np.log10(np.mean(np.abs(pert)**2)/np.mean(np.abs(reference)**2))
                deg=float(np.rad2deg(np.angle(np.vdot(reference,pert))))
                digital_gains.append(float(gain));digital_phase.append(deg)
    if count<MIN_RECORDS:raise ValueError(f"Too few real measurement frames: {count}")
    dt=np.diff(np.asarray(time))
    if np.any(dt<=0):raise RuntimeError("Native timestamps not strictly increasing")
    out={
      "source":metadata,
      "original_measured_CSI_records":count,
      "native_elapsed_seconds":float(time[-1]-time[0]),
      "native_median_frame_spacing_seconds":float(np.median(dt)),
      "native_first_time_seconds_since_campaign_start":time[0],
      "native_last_time_seconds_since_campaign_start":time[-1],
      "measured_complex_CSI_values_analyzed":count*64*1024,
      "physical_arrays":4,
      "elements_per_physical_array":16,
      "physical_geometry":"4_REAL_PLANAR_2x8_ARRAYS_NOT_4_ULAS",
      "measured_CSI_nonfinite_values":invalid,
      "measured_covariances_PSD_validated":count*4,
      "worst_covariance_min_eigen_ratio":min_psd_ratio,
      "natural_per_array_descriptive_metrics":{
          k:{"measurements":len(v),
             "native_receiver_power_median":float(np.median([a["median_rx_channel_power"] for a in v])),
             "native_receiver_power_cv":float(np.std([a["median_rx_channel_power"] for a in v])/max(np.mean([a["median_rx_channel_power"] for a in v]),1e-30)),
             "median_inter_element_coherence":float(np.median([a["inter_element_correlation_median_abs"] for a in v])),
             "median_port_power_spread_dB":float(np.median([a["worst_best_port_power_spread_db"] for a in v if a["worst_best_port_power_spread_db"] is not None])),
             "median_source_SNR":float(np.median([a["native_snr_median"] for a in v]))
             } for k,v in per.items()},
      "digital_injection_sanity_only":{
          "input_channel":"ORIGINAL_MEASURED_CSI",
          "fault_origin":"SYNTHETIC_DIGITAL_TRANSFORMATION_NOT_PHYSICAL_HARDWARE",
          "nominal_gain_step_dB":-3.0,
          "observed_mean_gain_step_dB":float(np.mean(digital_gains)),
          "nominal_phase_step_degrees":5.0,
          "observed_mean_phase_step_degrees":float(np.mean(digital_phase)),
          "numerical_tests":len(digital_gains),
          "independent_fault_detection_tests":0,
      },
      "hardware_fault_ground_truth":None,
      "frozen_RC3_inferences":0,
      "frozen_RC3_field_FAR":None,
      "frozen_RC3_field_recall":None,
      "full_RC3_75hour_fivesite_qualification":False,
      "source_status":"REAL_RF_MEASURED_CSI_SHADOW_NOT_FIELD_QUALIFICATION"
    }
    return out

def main():
    s,files,spec,sha=get_metadata()
    results={}
    for name in CASES:
        raw,meta=load_prefix(s,files[name])
        results[name]=evaluate_case(raw,meta,spec["antennas"])
        print(f"DICHASUS_REAL_CSI_VERIFIED {name} records={results[name]['original_measured_CSI_records']} timestamp_span_s={results[name]['native_elapsed_seconds']:.4f} measured_PSD={results[name]['measured_covariances_PSD_validated']}",flush=True)
    allresults={
        "source_doi":"10.18419/DARUS-4062","spec_sha256":sha,
        "native_geometry_from_publisher":spec["antennas"],
        "source_original_measured_rf":True,
        "source_csi_tensor":"64 channels x 1024 OFDM subcarriers x real/imag float32",
        "fault_annotations":"NOT_PROVIDED",
        "total_original_measured_RF_records":sum(x["original_measured_CSI_records"] for x in results.values()),
        "total_original_measured_CSI_coefficients":sum(x["measured_complex_CSI_values_analyzed"] for x in results.values()),
        "total_physical_array_covariance_matrices":sum(x["measured_covariances_PSD_validated"] for x in results.values()),
        "raw_source_data_redistributed":False,
        "original_RC3_weights_or_threshold_modified":False,
        "original_RC3_field_inferences":0,
        "full_field_qualification":False,
        "runs":results,
    }
    print("DICHASUS_INDEPENDENT_REAL_CSI_SHADOW_RESULT="+json.dumps(allresults,sort_keys=True,allow_nan=False),flush=True)
    return allresults
if __name__=="__main__":main()
