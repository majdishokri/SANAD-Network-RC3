"""Fail-closed input-quality evidence for signed 16-bit raw RF interleaved I/Q.

Crucial distinction: these are ANALYZER REJECTION reasons, not verified
antenna hardware-fault labels. Neither RC3 weights nor threshold are modified.

A quantized stream with one/two nearly constant values over many samples is
NOT adequate to construct phase/gain calibration covariance. The criterion
is conservative, designed to ABSTAIN, not classify a broken physical radio.

Authorised input contract: one native physical receiver antenna's interleaved
I,Q int16 stream; samples across >=10000 native I/Q scalars, no rewriting.
"""
from __future__ import annotations

from dataclasses import dataclass,asdict
import hashlib
import numpy as np

class RFInputNotQualified(ValueError):
    """Preprocessing must abstain; never silently replace a dead RF input."""

@dataclass(frozen=True)
class RawIQQuality:
    total_IQ_values: int
    distinct_integer_levels: int
    modal_integer_level: int
    modal_fraction: float
    exact_zero_fraction: float
    min_integer_level: int
    max_integer_level: int
    source_slice_sha256: str
    status: str
    decision_kind: str = "INPUT_QUALITY_GATE_NOT_HARDWARE_FAULT_CLASSIFIER"
    physical_hardware_fault_verified: bool = False

    def as_json(self):
        return asdict(self)

def analyze_receiver_iq(
    iq,
    *,
    min_values: int=10000,
    max_levels: int=2,
    modal_fraction_cutoff: float=0.995,
) -> RawIQQuality:
    a=np.asarray(iq)
    if a.dtype!=np.dtype("int16"):
        raise RFInputNotQualified("Unverified signed native int16 I/Q source")
    if a.ndim<1 or a.shape[-1]<2 or (a.shape[-1]%2)!=0:
        raise RFInputNotQualified("Need intact interleaved receiver I/Q samples")
    if a.size<min_values:
        raise RFInputNotQualified("Insufficient source samples for quantization gate")
    if not (0.5<modal_fraction_cutoff<=1) or max_levels<1:
        raise ValueError("Invalid fixed statistical source-quality contract")
    vals,counts=np.unique(a,return_counts=True)
    dom=int(np.argmax(counts))
    fraction=float(counts[dom]/a.size)
    bad=bool(len(vals)<=max_levels and fraction>=modal_fraction_cutoff)
    return RawIQQuality(
       total_IQ_values=int(a.size),
       distinct_integer_levels=int(len(vals)),
       modal_integer_level=int(vals[dom]),
       modal_fraction=fraction,
       exact_zero_fraction=float(counts[vals==0].sum()/a.size),
       min_integer_level=int(vals[0]),
       max_integer_level=int(vals[-1]),
       source_slice_sha256=hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest(),
       status="ABSTAIN_NEAR_CONSTANT_QUANTIZED_RAW_RF" if bad else "NOT_FLAGGED_BY_THIS_ONE_GATE"
    )

def require_qualified_raw_iq(iq,**kwargs)->RawIQQuality:
    result=analyze_receiver_iq(iq,**kwargs)
    if result.status.startswith("ABSTAIN_"):
        raise RFInputNotQualified(
          f"Near-constant raw RF source; no calibration inference. "
          f"Distinct={result.distinct_integer_levels}, modal={result.modal_integer_level}, "
          f"fraction={result.modal_fraction:.6f} source_sha256={result.source_slice_sha256}"
        )
    return result
