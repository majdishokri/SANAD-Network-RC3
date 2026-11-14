"""Deterministic shape, gauge, geometry, fail-closed and injection property tests."""
import numpy as np
import pytest
from sanad_adaptive import AdaptiveCore, ArrayGeometry, DetectorConfig, covariance_from_csi, discover_arrays


def make_cov(m, seed):
    rng = np.random.default_rng(seed)
    # Highly coherent multipath + individual spatial differences: stable SPD.
    common = (rng.normal(size=(1, 30)) + 1j*rng.normal(size=(1, 30)))
    private = (rng.normal(size=(m, 30)) + 1j*rng.normal(size=(m, 30)))
    h = (1.2*common + .5*private)/np.sqrt(2)
    return h @ h.conj().T / h.shape[1] + np.eye(m)*0.10


def perturb(c, count, seed, gain_sigma=.012, phase_sigma=.12):
    rng = np.random.default_rng(seed)
    m = len(c)
    out=[]
    for _ in range(count):
        gains = rng.normal(size=m)*gain_sigma
        phase = rng.normal(size=m)*phase_sigma
        d = 10**(gains/20)*np.exp(1j*np.deg2rad(phase))
        out.append(d[:,None] * c * d[None,:].conj())
    return np.array(out)


def inject(c, antenna, gain_db=.8, phase_deg=8):
    d = np.ones(c.shape[0],dtype=complex)
    d[antenna] = 10**(gain_db/20)*np.exp(1j*np.deg2rad(phase_deg))
    return d[:,None]*c*d[None,:].conj()


@pytest.mark.parametrize('arrays,elements',[(1,[3]),(2,[4,8]),(3,[2,8,16]),(4,[4,8,16,32]),(7,[8]*7)])
def test_variable_dimensions_and_gain_phase(arrays,elements):
    conf=DetectorConfig(min_reference_frames=24,min_calibration_frames=16)
    core=AdaptiveCore(conf)
    base={f'a{i}':make_cov(m,i+1) for i,m in enumerate(elements)}
    for i,(name,c) in enumerate(base.items()):
        core.register(name,perturb(c,28,100+i),perturb(c,20,200+i),
                      ArrayGeometry(positions=np.arange(len(c))[:,None]))
    healthy=core.evaluate(base)
    assert healthy['arrays_observed']==arrays
    assert healthy['elements_per_array']==dict(zip(base,elements))
    assert all(not any(v['alerts']) for v in healthy['decisions'].values())
    target=max(base,key=lambda x:len(base[x]))
    if len(base[target])>=3:
        faulty={key:inject(c,1) if key==target else c for key,c in base.items()}
        for i in range(6):
            result=core.evaluate(faulty)
        assert result['decisions'][target]['alerts'][1]
        assert not result['decisions'][target]['channel_shift']
        assert result['decisions'][target]['score'][1]>1.0


def test_heterogeneous_arrays_and_dynamic_availability():
    cov3=make_cov(3,1);cov12=make_cov(12,2)
    core=AdaptiveCore()
    core.register_many({'small':perturb(cov3,28,3),'large':perturb(cov12,28,4)},
                       {'small':perturb(cov3,20,5),'large':perturb(cov12,20,6)})
    assert core.evaluate({'small':cov3})['arrays_observed']==1
    assert core.evaluate({'large':cov12})['arrays_observed']==1
    with pytest.raises(ValueError,match='changed from'):
        core.evaluate({'large':cov3})
    with pytest.raises(ValueError,match='Unregistered'):
        core.evaluate({'new':cov3})


def test_arbitrary_csi_element_and_subcarrier_counts():
    rng=np.random.default_rng(100)
    for m,k in [(3,7),(4,117),(8,117),(16,64),(32,12)]:
        h=rng.standard_normal((4,m,k))+1j*rng.standard_normal((4,m,k))
        cov=covariance_from_csi(h)
        assert cov.shape==(4,m,m)
        assert np.min(np.linalg.eigvalsh(cov))>-1e-8
    with pytest.raises(ValueError, match='complex'):
        covariance_from_csi(np.ones((3,4,8)))


def test_count_discovery_and_bad_schema():
    c=make_cov(8,4)
    n=np.stack([perturb(c,3,1),perturb(c,3,2)],axis=1)
    assert len(discover_arrays(n))==2
    with pytest.raises(ValueError, match='Hermitian'):
        bad=c.copy();bad[0,1] += 2j
        discover_arrays({'a':bad[None,:,:]})
    with pytest.raises(ValueError, match='positions'):
        ArrayGeometry(positions=np.zeros((8,2))).validate(8)


def test_channel_shift_abstention():
    b=make_cov(8,1);c=make_cov(8,25)
    core=AdaptiveCore()
    core.register('x',perturb(b,28,1),perturb(b,20,2))
    r=core.evaluate({'x':c})['decisions']['x']
    assert r['channel_shift'] and r['status']=='CHANNEL_SHIFT_REVIEW'
    assert not any(r['alerts'])


def test_one_and_two_element_nonidentifiability():
    core=AdaptiveCore()
    for m in (1,2):
        b=make_cov(m,m)
        core.register(str(m),perturb(b,28,m+3),perturb(b,20,m+5))
        r=core.evaluate({str(m):inject(b,0)})['decisions'][str(m)]
        assert r['quality']=='relative_calibration_underidentified'
        assert not any(r['alerts'])


def test_geometry_rejects_silent_element_count_reconfiguration():
    b=make_cov(4,1)
    core=AdaptiveCore()
    with pytest.raises(ValueError):
        core.register('bad',perturb(b,28,2),perturb(b,20,3),ArrayGeometry(positions=np.ones((8,2))))


def test_preserve_common_gain_unobservability():
    b=make_cov(8,2)
    core=AdaptiveCore()
    core.register('x',perturb(b,28,3),perturb(b,20,4))
    common=10**(.8/10)*b   # R scaled equally: no relative gain fault
    r=core.evaluate({'x':common})['decisions']['x']
    r0=core.evaluate({'x':b})['decisions']['x']
    assert np.max(np.abs(np.array(r['gain_db'])-np.array(r0['gain_db'])))<1e-9
    assert not any(r['alerts'])

