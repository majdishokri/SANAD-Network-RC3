"""Executable pre-field tests, including rejection of short measured CSI sequences."""
import numpy as np
import pytest
from generalized.real_rf_v1.conditioning import (
    Observation, AlignmentPolicy, WindowQuality, WindowAligner,
    checked_covariance, validate_run, BaselineNormalizer, ObservationFilter)
from generalized.real_rf_v1.shadow import OneClassThreshold, ShadowHead

def obs(t, sid="site-A", station="physical-A", session="session-1", m=8):
    return Observation(t,sid,station,session,covariance=np.eye(m,dtype=complex))

def test_csi_covariance_is_measured_physics_not_fake_time():
    rng=np.random.default_rng(101)
    h=rng.normal(size=(8,117))+1j*rng.normal(size=(8,117))
    r=checked_covariance(Observation(0,"s","phys","session",csi=h))
    assert r.shape==(8,8)
    assert np.min(np.linalg.eigvalsh(r)) >= -1e-9
    assert np.allclose(r, h@h.conj().T/117)

def test_actual_14_4_seconds_can_never_be_15_minute_report():
    w=WindowAligner()
    for k in range(300):
        assert not w.push(obs(1+k*0.048))
    reports=w.push(obs(901))
    assert len(reports)==1
    assert not reports[0].accepted
    assert "insufficient_temporal_coverage" in reports[0].reason
    assert reports[0].occupied_slots<=2

def test_real_900_second_bins_accept_complete_coverage():
    w=WindowAligner()
    for k in range(60):
        assert w.push(obs(k*15+2)) == []
    first=w.push(obs(902))
    assert len(first)==1 and first[0].accepted
    assert first[0].occupied_slots==60
    assert first[0].n_acquisitions==60
    assert first[0].start_utc_s==0
    for k in range(1,60):
        w.push(obs(900+k*15+2))
    second=w.flush(1800)
    assert len(second)==1 and second[0].accepted

def test_no_silent_window_padding():
    w=WindowAligner()
    w.push(obs(10))
    with pytest.raises(ValueError,match="missing"):
        w.push(obs(1900))

def test_nonmonotone_or_duplicate_frames_rejected():
    w=WindowAligner()
    w.push(obs(10))
    with pytest.raises(ValueError,match="Non-monotone"):
        w.push(obs(10))

def test_nonphysical_covariance_and_no_coherence_rejected():
    bad=np.diag([-1.,2.,3.,4.]).astype(complex)
    with pytest.raises(ValueError,match="positive-semidefinite"):
        checked_covariance(Observation(0,"s","p","r",covariance=bad))
    with pytest.raises(ValueError,match="phase coherence"):
        checked_covariance(Observation(0,"s","p","r",covariance=np.eye(8,dtype=complex),
                                       phase_coherent_within_array=False))

def mocked_reports(n,m,dup_physical=False):
    r=np.eye(m,dtype=complex)
    return [
       WindowQuality(f"site-{i}",f"phys-{i if not dup_physical or i!=1 else 0}",
                     "session",t*900,True,60,60,15.,"OK",r)
       for i in range(n) for t in range(300)
    ]

@pytest.mark.parametrize("n,m",[(5,4),(7,8),(7,16),(8,32),(16,64)])
def test_geometry_compatible_but_not_field_qualified(n,m):
    a=validate_run(mocked_reports(n,m))
    assert a["sites"]==n and a["elements"]==m and a["reports"]==300
    assert a["duration_hours"]==75
    assert a["qualification"]=="NOT_FIELD_QUALIFIED"

def test_four_sites_rejected_by_unchanged_reference_rule():
    with pytest.raises(ValueError,match="at least 5"):
        validate_run(mocked_reports(4,8))

def test_different_rows_one_site_not_independent_sites():
    with pytest.raises(ValueError,match="NOT independent"):
        validate_run(mocked_reports(7,8,dup_physical=True))

def test_missing_300_windows_rejected():
    with pytest.raises(ValueError,match="300 verified"):
        validate_run(mocked_reports(7,8)[:-1])

def test_missing_verified_healthy_baseline_rejected():
    b=BaselineNormalizer()
    arr=[np.eye(8,dtype=complex)]*120
    with pytest.raises(ValueError,match="External healthy"):
        b.fit("s",arr,session_id="known",externally_verified_healthy=False)
    with pytest.raises(ValueError,match="120"):
        b.fit("s",arr[:119],session_id="known",externally_verified_healthy=True)

def test_site_power_baseline_is_frozen_and_psd_preserving():
    b=BaselineNormalizer()
    ref=np.diag([4.,1.,9.,16.,4.,1.,9.,16.]).astype(complex)
    b.fit("s",[ref]*120,session_id="verified-healthy",externally_verified_healthy=True)
    x=b.apply("s",ref)
    assert np.allclose(x,np.eye(8))
    candidate=ref.copy()
    candidate[3,3]*=10**(0.8/10)
    deviation=b.relative_gain_db("s",candidate)
    assert deviation[3]>0.7
    assert np.isclose(np.median(deviation),0)
    with pytest.raises(ValueError,match="locked"):
        b.fit("s",[ref]*120,session_id="new",externally_verified_healthy=True)

def test_temporal_filter_reviews_spikes_and_passes_persistent_ramp():
    f=ObservationFilter(threshold=0.5,persistence=3,median_window=3)
    x=f.step([0.9,0,0,0])
    assert not x["persistent_change"][0] and x["transient_review"][0]
    for _ in range(3):
        x=f.step([0,0,0,0])
    assert not x["persistent_change"][0]
    for v in [0.2,0.4,0.55,0.65,0.75,0.85]:
        x=f.step([v,0,0,0])
    assert x["persistent_change"][0]
    assert not x["persistent_change"][1]

def test_healthy_only_cutoff_is_not_a_field_false_alarm_guarantee():
    t=OneClassThreshold.fit(np.linspace(0.01,0.5,201),alpha=0.02)
    assert t.calibration_n==201 and t.candidate([0.01,0.99]).tolist()==[False,True]
    with pytest.raises(ValueError,match=">=100"):
        OneClassThreshold.fit([0.2]*20)

def test_new_shadow_head_disjoint_sessions_only():
    rng=np.random.default_rng(41)
    x0=rng.normal(0,0.2,(130,3))
    x1=rng.normal(1,0.2,(130,3))
    x=np.vstack([x0,x1])
    y=np.r_[np.zeros(130,int),np.ones(130,int)]
    groups=np.array(["training-A" if i%2==0 else "training-B" for i in range(130)]*2)
    cx=rng.normal(0,0.2,(160,3))
    cy=np.zeros(160,int)
    cg=np.array(["cal-C"]*80+["cal-D"]*80)
    h=ShadowHead()
    with pytest.raises(ValueError,match="physically measured"):
        h.fit(x,y,groups,calibration_X=cx,calibration_y=cy,
              calibration_groups=cg,source_kind="synthetic",source_time_verified=True)
    h.fit(x,y,groups,calibration_X=cx,calibration_y=cy,
          calibration_groups=cg,source_kind="measured_RF",source_time_verified=True)
    assert h.predict_proba(np.array([[1,1,1]])).item()>h.predict_proba(np.array([[0,0,0]])).item()
    assert h.export_manifest()["field_qualified"] is False
    with pytest.raises(ValueError,match="overlap"):
        h.fit(x,y,groups,calibration_X=cx,calibration_y=cy,
              calibration_groups=np.array(["training-A"]*80+["cal-D"]*80),
              source_kind="measured_RF",source_time_verified=True)
