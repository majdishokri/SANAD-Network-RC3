import gzip
import hashlib
from pathlib import Path
import numpy as np
import pytest
from rc3 import FrozenRC3, make_features, meta_features
from sanad_rc3_engine import SANADRC3Engine
from verify_engine import verify

HERE=Path(__file__).resolve().parent
FIXTURE=HERE/"fixtures"/"s2643.npz"

@pytest.fixture(scope="module")
def trace():
    with np.load(FIXTURE,allow_pickle=False) as z:
        return {k:z[k] for k in z.files}

@pytest.fixture(scope="module")
def engine():
    return SANADRC3Engine()

def test_exact_frozen_source_hashes():
    assert verify()==14

@pytest.mark.parametrize("stream",["healthy","injected"])
def test_thin_wrapper_matches_frozen_entrypoint(engine,trace,stream):
    expected=engine._engine.predict_trace(trace,stream)
    actual=engine.predict(trace,stream)
    assert actual["alerts"] == expected["alerts"]
    np.testing.assert_allclose(actual["probabilities"], expected["probabilities"], rtol=0, atol=1e-12)
    assert len(actual["alerts"])==7
    assert len(actual["probabilities"])==7
    assert actual["threshold"]==0.5

def test_labels_and_onset_are_never_input_features(engine,trace):
    edited={k:v.copy() for k,v in trace.items()}
    for stream in ("healthy","injected"):
        edited[f"{stream}_expected"][:]=0
        edited[f"{stream}_onset"][:]=-1
        expected=engine.predict(trace,stream)
        actual=engine.predict(edited,stream)
        assert actual["alerts"] == expected["alerts"]
        np.testing.assert_allclose(actual["probabilities"],expected["probabilities"],rtol=0,atol=1e-12)

def test_fixed_693_plus_15_feature_contract(trace):
    feat=make_features(trace,"injected")
    assert feat.shape==(7,693)
    meta=meta_features(np.linspace(.01,.99,7),np.array([1,0,1,0,0,1,0],bool))
    assert meta.shape==(7,15)

def test_npz_path(engine,trace):
    old=engine.predict(trace,"healthy")
    new=engine.predict_npz(FIXTURE,"healthy")
    assert old["alerts"]==new["alerts"]
    np.testing.assert_allclose(old["probabilities"],new["probabilities"],rtol=0,atol=1e-12)

def test_invalid_stream_rejected(engine,trace):
    with pytest.raises(ValueError):
        engine.predict(trace,"simulated")

def test_wrong_array_shape_rejected(engine,trace):
    corrupted={k:v.copy() for k,v in trace.items()}
    corrupted["injected_delta"]=corrupted["injected_delta"][:,:,:2]
    with pytest.raises((ValueError,AssertionError)):
        engine.predict(corrupted,"injected")
