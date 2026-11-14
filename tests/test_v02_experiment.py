import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from run_espargos_v02 import covariance_windows,inject,run


def mock_csi(nwindows=90,window=16,n_arrays=2,m=4,k=12,seed=9):
    rng=np.random.default_rng(seed)
    base=(rng.normal(size=(n_arrays,m,k))+1j*rng.normal(size=(n_arrays,m,k)))
    phases=np.exp(1j*rng.uniform(-np.pi,np.pi,size=(nwindows*window,n_arrays,1,1)))
    noise=.0004*(rng.normal(size=(nwindows*window,n_arrays,m,k))+
                 1j*rng.normal(size=(nwindows*window,n_arrays,m,k)))
    return (phases*(base[None]+noise)).astype(np.complex64)


def test_covariance_properties_and_injection():
    h=mock_csi()
    c=covariance_windows(h,16)
    assert c.shape==(90,2,4,4)
    assert np.allclose(c,c.conj().swapaxes(-1,-2),atol=1e-4)
    x=c[0,0]
    inj=inject(x,2,0,5)
    assert np.allclose(inj,inj.conj().T)
    assert not np.allclose(x,inj)


def test_e2e_synthetic_does_not_claim_measured(tmp_path):
    h=mock_csi()
    result=run(h,h.copy(),tmp_path,measured_source=False,profiles=((0.,5.),))
    assert result['protocol']['arrays']==2
    assert result['injections']['total_trials']==8
    assert result['measured_source'] is False
    assert (tmp_path/'adaptive_v02_report.json').exists()
    assert (tmp_path/'adaptive_v02_trials.csv').exists()
    assert result['injections']['profiles'][0]['v02_fused_incremental_detected_fraction']>=0
