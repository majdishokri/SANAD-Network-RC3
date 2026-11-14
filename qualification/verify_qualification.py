"""Check the frozen RC3 prospective Sionna result against the pre-registered lock."""
from pathlib import Path
import json, hashlib, math
D=Path(__file__).resolve().parent


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def wilson_upper(k,n,z=1.959963984540054):
    p=k/n;den=1+z*z/n
    return (p+z*z/(2*n)+z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)))/den

def check():
    lock=json.loads((D/'QUALIFICATION_LOCK.json').read_text())
    summary_file=D/'SUMMARY_FROZEN.json'
    manifest_file=D/'MANIFEST_FROZEN.json'
    assert sha(summary_file)==lock['summary_sha256']
    assert sha(manifest_file)==lock['pre_registration_sha256']
    s=json.loads(summary_file.read_text())
    m=json.loads(manifest_file.read_text())
    assert s['study']==m['study']==lock['study']
    assert s['model_sha256']==m['rc3_candidate_sha256']==lock['model_sha256']
    assert s['source_zip_sha256']==m['source_zip_sha256']==lock['original_source_zip_sha256']
    assert s['frozen_threshold']==m['decision_threshold']==lock['decision_threshold']==.5
    assert s['completed_seeds']==s['required_seeds']==m['required_seeds']==144
    assert m['seed_range']==[3000,3143]
    assert m['status']=='FROZEN_BEFORE_PROSPECTIVE_EVALUATION'
    assert m['all_training_and_tuning_complete_before_seed_3000'] is True
    assert s['training_used_new_seed_labels'] is False
    rc=s['methods']['RC3'];old=s['methods']['RC0']
    assert rc['tp']==274 and rc['fault_n']==288
    assert old['tp']==253 and old['fault_n']==288
    assert rc['healthy_fp']==0 and rc['healthy_n']==1008
    assert old['healthy_fp']==7 and old['healthy_n']==1008
    assert rc['neighbor_fp']==13 and rc['neighbor_n']==720
    assert old['neighbor_fp']==6 and old['neighbor_n']==720
    assert math.isclose(rc['recall'],rc['tp']/rc['fault_n'])
    assert math.isclose(rc['far'],rc['healthy_fp']/rc['healthy_n'])
    assert math.isclose(rc['wilson_far_upper_95'],wilson_upper(rc['healthy_fp'],rc['healthy_n']))
    checks={'recall_strictly_greater_than_95pct':rc['recall']>.95,
            'wilson_far_upper_strictly_below_2pct':rc['wilson_far_upper_95']<.02,
            'seed_cluster_far_upper_strictly_below_2pct':rc['seed_cluster_far_ci95'][1]<.02}
    assert all(checks.values())
    assert checks=={k:v for k,v in s['qualification'].items() if k!='all_pass'}
    assert s['qualification']['all_pass'] is True
    assert s['methods']['RC3']==lock['result_rc3'] and s['methods']['RC0']==lock['result_rc0']
    print('PASS: RC3 independent Sionna prospective frozen result 144/144; 274/288 recall, 0/1008 healthy FP')
if __name__=='__main__':check()
