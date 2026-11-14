"""Property and synthetic detection checks (NOT ESPARGOS field-performance claims)."""
import numpy as np
import pytest
from sanad_adaptive.spectral import (spectral_coherence,transform_recording,
                                     SpectralPhaseMonitor,SpectralConfig)


def csi_fixture(m=8,k=48,t=16,seed=10,channel_perturbation=0.,phase_shift=0.,target=2):
    rng=np.random.default_rng(seed)
    freq=np.linspace(0,2*np.pi,k)
    baseline_rng=np.random.default_rng(987 + m*11 + k)
    hbase=((baseline_rng.normal(size=(m,k))+1j*baseline_rng.normal(size=(m,k)))*.5 +
        np.exp(1j*(.4*np.arange(m)[:,None]+freq[None,:]*(.35+.07*np.arange(m)[:,None]))))
    # packet common phase is unobservable and MUST cancel
    common=np.exp(1j*rng.uniform(-np.pi,np.pi,size=(t,1,1)))
    channel=(rng.normal(size=(t,m,k))+1j*rng.normal(size=(t,m,k)))*channel_perturbation
    noise=(rng.normal(size=(t,m,k))+1j*rng.normal(size=(t,m,k)))*.003
    out=common*(hbase[None]+channel+noise)
    if phase_shift:
        out[:,target,:]*=np.exp(1j*np.deg2rad(phase_shift))
    return out


def make_windows(m=8,k=48,seed=10,n=65,phase_shift=0.,onset=0,channel_change=0.):
    return np.asarray([spectral_coherence(csi_fixture(m,k,16,seed+u,channel_change,
                              phase_shift if u>=onset else 0.)) for u in range(n)])


def test_independent_element_and_subcarrier_sizes():
    mon=SpectralPhaseMonitor()
    arrays={}
    for key,m,k in [('tiny',3,8),('mid',8,32),('large',16,64)]:
        arr=make_windows(m,k,seed=17,n=56)
        mon.register(key,arr[:28],arr[28:])
        arrays[key]=arr[-1]
    out=mon.evaluate(arrays)
    assert out['arrays_observed']==3
    assert [len(out['decisions'][x]['phase_alerts']) for x in arrays]==[3,8,16]


def test_phase_only_five_degree_persistent_detection():
    cfg=SpectralConfig(phase_window=8,persistence=3)
    mon=SpectralPhaseMonitor(cfg)
    data=make_windows(n=94,seed=33)
    mon.register('test',data[:32],data[32:64])
    mon.reset_history()
    findings=[]
    for j in range(20):
        # Online injection into the already acquired CSI spectral covariance:
        # D S D^H, no mutation of the known-healthy baseline.
        cur=data[64+j].copy()
        if j>=7:
            D=np.ones(8,complex);D[2]=np.exp(1j*np.deg2rad(5))
            cur=D[:,None,None]*cur*D.conj()[None,:,None]
        dec=mon.evaluate({'test':cur})['decisions']['test']
        findings.append(dec)
    assert any(x['phase_alerts'][2] for x in findings[7:])
    assert not any(x['channel_shift'] for x in findings[7:])


def test_noise_and_unobservable_return_review_not_fake_success():
    mon=SpectralPhaseMonitor()
    data=make_windows(n=64,seed=41)
    mon.register('test',data[:32],data[32:])
    blank=data[0].copy()
    blank[0,1,:]=0;blank[1,0,:]=0
    # Spectral matrix remains Hermitian. A disconnected graph cannot identify phase.
    for i in range(1,8):
        blank[0,i,:]=0;blank[i,0,:]=0
    out=mon.evaluate({'test':blank})['decisions']['test']
    assert out['channel_shift'] and not any(out['phase_alerts'])
    with pytest.raises(ValueError,match='recalibrate'):
        mon.evaluate({'test':np.eye(7,dtype=complex)[:,:,None].repeat(48,axis=2)})


def test_common_phase_does_not_change_spectral_observation():
    h=csi_fixture(seed=123)
    phase=np.exp(1j*np.linspace(0,5,len(h)))[:,None,None]
    assert np.allclose(spectral_coherence(h),spectral_coherence(h*phase),atol=1e-8)


def test_calibration_not_implicitly_permitted_on_changed_array():
    mon=SpectralPhaseMonitor();x=make_windows(m=4,k=12,n=54)
    mon.register('X',x[:27],x[27:])
    with pytest.raises(ValueError,match='already'):
        mon.register('X',x[:27],x[27:])


def test_transform_recording_shape():
    h=np.stack([csi_fixture(m=4,k=9,t=16,seed=i) for i in range(2)],axis=1)
    assert transform_recording(h,8).shape==(2,2,4,4,9)
