import pytest
import numpy as np
from sanad_adaptive import AdaptiveCore
from sanad_adaptive.shared_head import FEATURE_NAMES, per_element_features, ExperimentalSharedHead
from test_adaptive_core import make_cov, perturb


def test_same_feature_width_across_mixed_arrays():
    engine=AdaptiveCore()
    bases={'x':make_cov(3,1),'y':make_cov(16,2),'z':make_cov(7,3)}
    for i,(name,c) in enumerate(bases.items()):
        engine.register(name,perturb(c,28,i+1),perturb(c,20,i+11))
    X,ids=per_element_features(engine.evaluate(bases))
    assert X.shape==(26,len(FEATURE_NAMES))
    assert set(k for k,_ in ids)==set(bases)
    with pytest.raises(RuntimeError, match='UNTRAINED'):
        ExperimentalSharedHead().predict_proba(X)


def test_shared_head_can_fit_mixed_config_features():
    rng=np.random.default_rng(3200)
    X=rng.normal(size=(60,len(FEATURE_NAMES))).astype(np.float32)
    y=np.r_[np.zeros(30,dtype=int),np.ones(30,dtype=int)]
    head=ExperimentalSharedHead().fit(X,y)
    p=head.predict_proba(X)
    assert p.shape==(60,) and np.isfinite(p).all() and np.all((0<=p)&(p<=1))
