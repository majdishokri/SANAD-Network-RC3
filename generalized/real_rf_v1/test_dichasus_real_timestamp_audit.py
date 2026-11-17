"""Tests of REAL timestamp decoder; fixtures test structure, not measured RF."""
import struct
import numpy as np
import pytest
from generalized.real_rf_v1.dichasus_real_timestamp_audit import (
    extract_real_time_seconds, audit_times,
)

def varint(v):
    out=bytearray()
    while True:
        b=v & 0x7f
        v >>= 7
        if not v:return bytes(out+bytes([b]))
        out.append(b|0x80)

def blob(num,val):
    return varint((num<<3)|2)+varint(len(val))+val

def example_with_float_time(t):
    # Native TensorFlow Example wire schema: Features.feature map[string,Feature].
    float_list=blob(1,struct.pack("<f",t))
    feature=blob(2,float_list)
    entry=blob(1,b"time")+blob(2,feature)
    features=blob(1,entry)
    return blob(1,features)

def test_source_timestamp_is_read_from_wire_not_record_index():
    assert extract_real_time_seconds(example_with_float_time(127.125))==127.125

def test_corrupted_feature_without_time_is_rejected():
    with pytest.raises(ValueError,match="timestamp"):
        extract_real_time_seconds(blob(1,b""))

def test_order_not_implied_from_file_sequence():
    out=audit_times([40.0,10.0,60.0,20.0],"fixture")
    assert out["out_of_order_transitions_in_file"]==2
    assert out["observed_time_span_s"]==50.0
    assert out["RC3_REAL_FIELD_QUALIFICATION"]=="BLOCKED"

def test_sparse_points_do_not_establish_accepted_15min_windows():
    out=audit_times([0,300*900],"fake-long-fixture")
    assert out["total_observed_time_span_ge_75_hours"] is True
    assert "NO_ALIGNED_300_CONSECUTIVE_ACCEPTED_15MIN_WINDOWS" in out["blocked_reasons"]
