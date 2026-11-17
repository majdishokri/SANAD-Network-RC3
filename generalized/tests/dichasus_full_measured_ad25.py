"""Full 596,950,014-byte Rice-independent Stuttgart DICHASUS ad25 real CSI.

Verification: DaRUS actual current file ID/size/MD5, own SHA-256, every
native TFRecord CRC32C and exact EOF; *every* native CSI tensor evaluated.
No redistribution of ~597MB source nor field Recall/FAR claims. Explicitly
separate native ~minute-long measurement session from RC3's 75-hour protocol.
"""
from __future__ import annotations
import hashlib,json,mmap,tempfile
from pathlib import Path
from urllib.parse import urlsplit
import requests
from generalized.tests.dichasus_real_csi_covariance_shadow import get_metadata,evaluate_case

NAME="dichasus-ad25.tfrecords"
EXPECTED_MD5="3e907ad72ae6aab1aa44ef53ba60a3ef"
PUBLISHED_EXPECTED_FILE_BYTES=596950014
ROOT="https://darus.uni-stuttgart.de"

def run():
    session,files,spec,spechash=get_metadata()
    f=files[NAME]
    if f.get("md5")!=EXPECTED_MD5 or int(f.get("filesize",0))!=PUBLISHED_EXPECTED_FILE_BYTES:
        raise ValueError("Dataverse published full source fingerprint changed")
    with tempfile.TemporaryDirectory() as temp:
        fn=Path(temp)/"original_stuttgart_64x1024_CSI.tfrecords"
        md5=hashlib.md5()
        sha=hashlib.sha256()
        count=0
        url=ROOT+f"/api/access/datafile/{f['id']}"
        with session.get(url,stream=True,timeout=(50,190),allow_redirects=True,
            headers={"User-Agent":"SANAD-RC3-Full-Source-Checksum-Research/1.0"}) as r:
            r.raise_for_status()
            host=urlsplit(r.url).hostname
            if host!="darus.uni-stuttgart.de" and not host.endswith(".uni-stuttgart.de"):
                raise RuntimeError("Untrusted full CSI source redirect")
            with fn.open("wb") as out:
                for block in r.iter_content(1<<20):
                    if not block:continue
                    count+=len(block)
                    if count>PUBLISHED_EXPECTED_FILE_BYTES:
                        raise RuntimeError("Publisher original source exceeds expected size")
                    md5.update(block);sha.update(block);out.write(block)
        if count!=PUBLISHED_EXPECTED_FILE_BYTES or md5.hexdigest()!=EXPECTED_MD5:
            raise ValueError("Publisher whole-file MD5 validation failure")
        metadata={
            "publisher":"DaRUS University of Stuttgart",
            "source_doi":"10.18419/DARUS-4062",
            "file_name":NAME,
            "publisher_file_id":f["id"],
            "original_full_source_bytes":count,
            "publisher_MD5_whole_file_verified":True,
            "full_file_MD5":md5.hexdigest(),
            "full_file_SHA256":sha.hexdigest(),
            "native_record_CRC32C_validation":"EVERY_RECORD",
            "original_RF_data_redistributed":False
        }
        with fn.open("rb") as source, mmap.mmap(source.fileno(),0,access=mmap.ACCESS_READ) as buf:
            stats=evaluate_case(buf,metadata,spec["antennas"],require_complete=True)
        if stats["measured_CSI_nonfinite_values"]!=0:
            raise ValueError("Unexpected nonfinite source in full measured trace")
        result={
           "benchmark":"DICHASUS_2024_ADXX_FULL_ORIGINAL_AD25_MEASURED_RF",
           "original_64_channel_1024_subcarrier_CSI":True,
           "publisher_spec_sha256":spechash,
           "dataset_original_file_MD5_verified":True,
           "all_native_records_CRC32C_verified":True,
           "no_raw_source_data_redistributed":True,
           "frozen_SANAD_RC3_model_unchanged":True,
           "frozen_RC3_qualifying_inferences":0,
           "RC3_field_FAR":None,
           "RC3_field_recall":None,
           "raw_recording":stats
        }
        print("DICHASUS_FULL_MEASURED_AD25="+json.dumps(result,sort_keys=True,allow_nan=False),flush=True)
        return result
if __name__=="__main__":run()
