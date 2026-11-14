import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from run_espargos_adaptive import run, injection
from test_adaptive_core import make_cov,perturb


def test_reproducible_end_to_end_experiment_on_synthetic_fixture(tmp_path):
    base=[make_cov(8,i+7) for i in range(4)]
    static=np.stack([np.stack([perturb(c,1,10+j*100+i)[0] for j,c in enumerate(base)])
                     for i in range(1536)])
    changed=[make_cov(8,i+90) for i in range(4)]
    walking=np.stack([np.stack([perturb(c,1,30+j*100+i)[0] for j,c in enumerate(changed)])
                     for i in range(1536)])
    result=run(static,walking,tmp_path,measured_source=False)
    assert result['core_supports_variable_shapes'] is True
    assert result['injections']['total_trials']==160
    assert len(result['injections']['profiles'])==5
    assert (tmp_path/'adaptive_espargos_report.json').exists()
    assert (tmp_path/'adaptive_espargos_trials.csv').exists()
    assert result['measured_source'] is False
