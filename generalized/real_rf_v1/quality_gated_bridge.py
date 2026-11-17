"""Optional pre-RC3 RF acquisition-integrity gate. Does NOT edit RC3 mathematics.

A complete set of original source I/Q receiver-port streams is required.
If ANY observed physical element's native signed interleaved I/Q is near-
constant (-16/0 type), refuse to construct RC3 evidence for that session.
Never silently drop the port, refill with noise, swap it, or duplicate it.
Accepting this single source-quality check is NOT a hardware health label.

Raw source I/Q must be from the same acquisition session with a verified
port/time/hardware binding; code cannot independently attest that binding.
"""
from __future__ import annotations

from collections.abc import Mapping
import numpy as np
from .conditioning import AlignmentPolicy, WindowQuality
from .rc3_bridge import PhysicalArray,generate_rc3_evidence
from .raw_iq_quality import RFInputNotQualified,analyze_receiver_iq


def generate_rc3_evidence_with_raw_iq_preflight(
    reports: list[WindowQuality],
    arrays: Mapping[str,PhysicalArray],
    raw_iq_by_physical_port: Mapping[tuple[str,str],np.ndarray],
    *,
    same_acquisition_binding_verified: bool,
    policy: AlignmentPolicy=AlignmentPolicy(),
    allow_synthetic_test_fixture: bool=False,
):
    """Preflight -> unchanged original measured-covariance converter.

    Keys are (site_id, physical element_id) as declared in PhysicalArray.
    A site with 8 elements MUST supply eight independently attributable native
    receiver IQ streams, and all sites must be covered. No source time rebasing.
    """
    if same_acquisition_binding_verified is not True:
        raise RFInputNotQualified(
            "ABSTAIN: independent port-to-acquisition-session binding not verified"
        )
    expected={(site,port) for site,arr in arrays.items() for port in arr.element_ids}
    provided=set(raw_iq_by_physical_port)
    if not expected or expected!=provided:
        raise RFInputNotQualified(
            f"ABSTAIN: incorrect per-element native RF source coverage; "
            f"expected={len(expected)}, provided={len(provided)}, "
            f"missing={len(expected-provided)}, extra={len(provided-expected)}"
        )
    result={}
    failures=[]
    for key in sorted(expected):
        quality=analyze_receiver_iq(raw_iq_by_physical_port[key])
        result[key]=quality
        if quality.status.startswith("ABSTAIN_"):
            failures.append(f"{key[0]}/{key[1]}:{quality.status} "
                            f"sha256={quality.source_slice_sha256}")
    if failures:
        raise RFInputNotQualified(
            "ABSTAIN: one or more source receiver channels invalid; original "
            "RC3 evidence was NOT generated. " + "; ".join(failures)
        )
    observation,meta=generate_rc3_evidence(
        reports,arrays,policy=policy,
        allow_synthetic_test_fixture=allow_synthetic_test_fixture
    )
    out=dict(meta)
    out.update({
        "raw_iq_preflight":"PASS_THIS_SINGLE_RAW_RF_QUALITY_GATE_ONLY",
        "raw_iq_preflight_does_not_certify_healthy_hardware":True,
        "raw_iq_preflight_infers_no_fault_labels":True,
        "raw_iq_preflight_original_rc3_weights_modified":False,
        "raw_iq_preflight_source_sha256":{
            f"{site}/{port}":q.source_slice_sha256
            for (site,port),q in sorted(result.items())
        },
        "raw_iq_preflight_original_acquisition_binding":
            "OPERATOR_ATTESTED_NOT_INDEPENDENTLY_CERTIFIED"
    })
    return observation,out
