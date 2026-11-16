"""Score an existing independent evidence set with unchanged RC3 prediction rule.
Ground truth is accessed only AFTER inference. Does not generate data, tune or retrain.
"""
import argparse,json,sys
from pathlib import Path
import numpy as np
from rc3 import FrozenRC3
from runtime.geometry_config import N_ARRAYS,N_ELEMENTS

def run(files):
    model=FrozenRC3()
    stats=dict(tp=0,fault_n=0,healthy_fp=0,healthy_n=0,neighbor_fp=0,neighbor_n=0)
    for file in files:
        with np.load(file,allow_pickle=False) as z:
            trace={k:z[k] for k in z.files}
        pred={s:np.asarray(model.predict_trace(trace,s)['alerts'],bool)
              for s in ('healthy','injected')}
        # Labels accessed here, only after both predictions.
        true={s:np.asarray(trace[f'{s}_expected'])>=0 for s in ('healthy','injected')}
        if any(len(v)!=N_ARRAYS for v in [*pred.values(),*true.values()]):
            raise ValueError(f"Wrong number of base stations in {file}")
        if true['healthy'].any() or true['injected'].sum()!=2:
            raise ValueError(f"Unexpected fault labels in {file}")
        stats['tp']+=int((pred['injected']&true['injected']).sum())
        stats['fault_n']+=int(true['injected'].sum())
        stats['healthy_fp']+=int(pred['healthy'].sum())
        stats['healthy_n']+=N_ARRAYS
        stats['neighbor_fp']+=int((pred['injected']&~true['injected']).sum())
        stats['neighbor_n']+=N_ARRAYS-int(true['injected'].sum())
    stats['recall']=stats['tp']/stats['fault_n'] if stats['fault_n'] else None
    stats['healthy_far']=stats['healthy_fp']/stats['healthy_n'] if stats['healthy_n'] else None
    stats['geometry']={"stations":N_ARRAYS,"elements":N_ELEMENTS}
    stats['files']=len(files)
    stats['qualified']=False
    return stats
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("files",nargs="+")
    args=parser.parse_args()
    print(json.dumps(run([Path(x) for x in args.files]),indent=2))
