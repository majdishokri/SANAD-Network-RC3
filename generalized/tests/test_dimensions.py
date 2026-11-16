"""No-algorithm-change regression and shape tests. Requires root requirements.txt."""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
SNIPPET=r"""
import json,numpy as np
from rc3 import FrozenRC3,make_features,meta_features
from geometry_config import N_ARRAYS,N_ELEMENTS
n=N_ARRAYS
# Missing measurements are valid as long as triplets are all missing.
obs={
    'healthy_feats':np.full((2,300,n,3),np.nan,dtype=np.float32),
    'healthy_delta':np.full((300,n,3),np.nan,dtype=np.float32),
    'healthy_v11':np.zeros((300,n),dtype=np.int8),
    'healthy_v11_named':np.zeros((300,n),dtype=np.int8)
}
f=make_features(obs,'healthy')
scores=np.linspace(.1,.9,n);flags=np.zeros(n,bool)
spatial=meta_features(scores,flags)
out=FrozenRC3().predict_trace(obs,'healthy')
assert f.shape==(n,693) and spatial.shape==(n,15)
assert len(out['alerts'])==len(out['probabilities'])==n
assert out['threshold']==0.50
assert all(np.isfinite(out['probabilities']))
print(json.dumps({'N':n,'M':N_ELEMENTS,'features':f.tolist(),
                  'spatial':spatial.tolist(),'prediction':out},sort_keys=True))
"""
# Legacy module has no geometry_config import; use a separate snippet.
LEGACY=SNIPPET.replace("from geometry_config import N_ARRAYS,N_ELEMENTS","N_ARRAYS,N_ELEMENTS=7,8")
def run(directory,n,m,legacy=False):
    env=os.environ.copy()
    env.update(SANAD_STATIONS=str(n),SANAD_ELEMENTS=str(m))
    proc=subprocess.run([sys.executable,"-c",LEGACY if legacy else SNIPPET],
                        cwd=directory,env=env,check=True,text=True,capture_output=True)
    return json.loads(proc.stdout.strip().splitlines()[-1])
if __name__=="__main__":
    original=run(ROOT,7,8,True)
    clone=run(ROOT/"generalized",7,8)
    assert original==clone,"7x8 frozen inference regression failed"
    print("PASS: bit-for-bit 7x8 feature/meta/prediction equivalence")
    for n,m in ((4,4),(8,16),(16,32),(32,64),(64,8)):
        res=run(ROOT/"generalized",n,m)
        assert len(res["prediction"]["alerts"])==n
        print(f"PASS: variable-shape inference {n} stations x {m} elements (unqualified)")
