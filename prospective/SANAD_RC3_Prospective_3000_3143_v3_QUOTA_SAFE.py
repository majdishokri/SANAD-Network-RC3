"""SANAD RC3 Spatial Meta independent prospective validation on fresh NVIDIA Sionna seeds.

Research-only. No model training, selection or threshold search is permitted here.
Pre-committed fixed threshold: 0.50, NOT the post-hoc threshold stored in the
exploratory development bundle. Do not modify after looking at any new outcome.

Results and immutable per-seed evidence are uploaded to a NEW Google Drive folder.
Runner v3 adds rate-limited Google Drive calls, exponential backoff and cached folder
inventories; it preserves the v2 verified SHA and nonnegative fault label sentinel.
This is a transport-only correction: no simulation, model, threshold, labels,
scoring, acceptance rule, or frozen study identifier has changed.
This is a GPU Sionna experiment; not a simulation of field RF hardware.
"""
from __future__ import annotations
import sys, subprocess, importlib.metadata as im

def require_package(name, version):
    try: found = im.version(name)
    except im.PackageNotFoundError: found = None
    if found != version:
        if name == 'scikit-learn' and any(k=='sklearn' or k.startswith('sklearn.') for k in sys.modules):
            raise RuntimeError('scikit-learn already imported; restart with fresh runtime')
        print(f'Installing {name}=={version} (found {found})',flush=True)
        subprocess.check_call([sys.executable,'-m','pip','install','-q',f'{name}=={version}'])

require_package('scikit-learn','1.8.0')
require_package('sionna','2.2.0')

import io, os, re, json, hashlib, zipfile, warnings, datetime, time, random
from pathlib import Path
import numpy as np
import torch
import sklearn
import joblib
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
from googleapiclient.errors import HttpError
from googleapiclient.discovery import build
from google.colab import auth
from google.auth import default

STUDY='SANAD_RC3_SPATIAL_META_FROZEN_0p50_GPU_PROSPECTIVE_3000_3143'
SEEDS=tuple(range(3000,3144))
BASE_SOURCE_ID='1rmFdmdoVh3cek0VBW1B1HkHlhzMZ-ml5'
RC3_MODEL_ID='1Krcz_w32nhhegSqLyFO5QAGCR9OcnBx0'
PARENT_FOLDER_ID='1N-O7jSgZVmV8-QAiBk8dmsAzJvXKfE6X'
STUDY_FOLDER_NAME='RC3_FROZEN_0p50_PROSPECTIVE_3000_3143'
SOURCE_SHA256='121ce9aba417fb27b07d7f7f6e52633a8ef29a2c99aefddc12a10428d02ddafb'
MODEL_SHA256='0a90835bb57a56437fc09211902b581bb22377676f91dce68f309c92e3a95bf4'
DECISION_THRESHOLD=0.50  # frozen BEFORE generating any new-seed labels
MAX_NEW_SEEDS_THIS_SESSION=144  # set lower to 8/16 if GPU runtime is limited; re-run to resume
N=144
ROOT=Path('/content/SANAD_RC3_FROZEN_PROSPECTIVE')
ROOT.mkdir(parents=True,exist_ok=True)
SRC=ROOT/'rc0_source'
SRC.mkdir(exist_ok=True)

if not torch.cuda.is_available():
    raise RuntimeError('A CUDA GPU is required: Colab > Runtime > Change runtime type > T4 GPU')
print('GPU:',torch.cuda.get_device_name(0),'| scikit-learn:',sklearn.__version__,flush=True)

auth.authenticate_user()
creds,_=default()
service=build('drive','v3',credentials=creds,cache_discovery=False)

# RC3 v3 TRANSPORT PATCH ONLY. Google Drive shared-project requests/minute may
# trigger 403 rateLimitExceeded. Keep every request below a conservative pace.
# Retry only explicitly rate-limited errors, without changing any research data.
_DRIVE_MIN_INTERVAL_SECONDS=2.25
_DRIVE_MAX_RETRIES=12
_drive_last_call=[0.0]
_rng_retry=random.Random(30003143)  # affects retry delays only; never Sionna seeds

def _rate_limited(err):
    status=getattr(getattr(err,'resp',None),'status',None)
    if status==429:return True
    if status!=403:return False
    try:
        body=getattr(err,'content',b'').decode('utf8','replace')
    except AttributeError:
        body=str(err)
    return ('rateLimitExceeded' in body or 'userRateLimitExceeded' in body
            or 'Requests per minute' in body)

def drive_call(operation, *, description):
    """Retry Google quota errors with jitter; throttle all Drive HTTP calls."""
    for attempt in range(_DRIVE_MAX_RETRIES+1):
        elapsed=time.monotonic()-_drive_last_call[0]
        if elapsed<_DRIVE_MIN_INTERVAL_SECONDS:
            time.sleep(_DRIVE_MIN_INTERVAL_SECONDS-elapsed)
        _drive_last_call[0]=time.monotonic()
        try:
            return operation()
        except HttpError as exc:
            if not _rate_limited(exc):
                raise  # Do not swallow auth, permission or permanent errors.
            if attempt==_DRIVE_MAX_RETRIES:
                raise RuntimeError(
                    'Drive rate limit persists; checkpoints are safe. Rerun this v3 '
                    'notebook when quota is available. Last action: '+description
                ) from exc
            delay=min(100.0, max(3.0,2.0**(attempt+2)))+_rng_retry.uniform(0.,1.5)
            print(f'Drive quota 403/429 during {description}; retry {attempt+1}/'
                  f'{_DRIVE_MAX_RETRIES}, backoff {delay:.1f}s',flush=True)
            time.sleep(delay)
    raise AssertionError('unreachable retry state')

def digest(path):
    sha=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''): sha.update(chunk)
    return sha.hexdigest()

# Freeze lock: audited against original Google Drive source ZIP and model bytes.
assert SOURCE_SHA256 == '121ce9aba417fb27b07d7f7f6e52633a8ef29a2c99aefddc12a10428d02ddafb'
assert MODEL_SHA256 == '0a90835bb57a56437fc09211902b581bb22377676f91dce68f309c92e3a95bf4'
assert DECISION_THRESHOLD == 0.50 and SEEDS == tuple(range(3000, 3144))
print('FREEZE LOCK — source sha256:', SOURCE_SHA256, flush=True)
print('FREEZE LOCK — model sha256: ', MODEL_SHA256, flush=True)

def fetch_file(file_id,path,expected_sha):
    path=Path(path)
    if path.exists() and digest(path)==expected_sha:return path
    if path.exists():path.unlink()
    with path.open('wb') as out:
        down=MediaIoBaseDownload(out,service.files().get_media(fileId=file_id),chunksize=1048576)
        done=False
        while not done: _,done=drive_call(lambda: down.next_chunk(), description='fetch pinned artifact')
    actual=digest(path)
    print(f'PINNED ARTIFACT {file_id}\n  expected SHA-256: {expected_sha}\n  observed SHA-256: {actual}\n  matching: {actual == expected_sha}', flush=True)
    if actual!=expected_sha:
        path.unlink(missing_ok=True)
        raise RuntimeError(f'SHA mismatch for pinned artifact {file_id}; expected={expected_sha}; observed={actual}')
    return path

zip_file=fetch_file(BASE_SOURCE_ID,ROOT/'frozen_source.zip',SOURCE_SHA256)
model_file=fetch_file(RC3_MODEL_ID,ROOT/'rc3_candidate.joblib',MODEL_SHA256)
with zipfile.ZipFile(zip_file) as z:
    if any(n.startswith('/') or '..' in Path(n).parts for n in z.namelist()):
        raise RuntimeError('Unexpected archived source paths')
    z.extractall(SRC)
if str(SRC) not in sys.path:sys.path.insert(0,str(SRC))

# Only load pinned, hash-verified joblib files. Never load untrusted pickle input.
from inference_v35_gate import FrozenV35LearnedGate
from prospective_gpu_runner import create_covariance
from v35_replay import extract_seed_v35
rc0=FrozenV35LearnedGate(SRC/'frozen_candidate'/'selected_model.joblib')
bundle=joblib.load(model_file)
assert bundle['train_seeds']==[2500,2643]
assert bundle['model_kind']=='spatial_meta_two_stage'
assert bundle['feature_count']==693 and bundle['meta_feature_count']==15
assert bundle['rc0_source_sha256']==SOURCE_SHA256
base_models=bundle['rc1_models_full']
meta_model=bundle['rc3_meta_model']
assert len(base_models)==2
print('Pinned model SHA:',MODEL_SHA256)
print('Threshold fixed at:',DECISION_THRESHOLD,'(ignoring tuned development cutoff in bundle)')

# The feature transformations below are extracted unchanged from the RC3 training
# source. They read ONLY evidence available by report 300; no labels or seed IDs.
def stats(x):
    x=np.asarray(x,dtype=np.float64).ravel()
    x=x[np.isfinite(x)]
    if not len(x):return [0.0]*9
    q=np.quantile(x,[.1,.25,.5,.75,.9])
    return [float(x.size),float(np.mean(x)),float(np.std(x)),*map(float,q),float(np.max(x))]

def make_features(z,prefix):
    f=np.asarray(z[prefix+'_feats'],dtype=float)
    d=np.asarray(z[prefix+'_delta'],dtype=float)
    v=np.asarray(z[prefix+'_v11'],dtype=float)
    if f.shape!=(2,300,7,3) or d.shape!=(300,7,3) or v.shape!=(300,7):
        raise ValueError('Unexpected source trace schema')
    raw=np.concatenate([f.transpose(1,2,0,3).reshape(300,7,6),d],axis=-1)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        baseline=np.nanmedian(raw,axis=1,keepdims=True)
    rel=raw-baseline
    rows=[]
    for j in range(7):
        feats=[]
        for a in (raw[:,j,:],rel[:,j,:]):
            for c in range(a.shape[-1]):
                trace=a[:,c]
                for l,r in ((180,220),(220,260),(260,300),(280,300)):
                    feats.extend(stats(trace[l:r]))
                recent=np.asarray(trace[260:300],dtype=float)
                former=np.asarray(trace[220:260],dtype=float)
                recent=recent[np.isfinite(recent)]
                former=former[np.isfinite(former)]
                feats.append(float(np.median(recent)-np.median(former)) if len(recent) and len(former) else 0.)
        for l,r in ((180,220),(220,260),(260,300)):
            feats.extend(stats(v[l:r,j]))
        rows.append(feats)
    a=np.asarray(rows,dtype=np.float32)
    a=np.nan_to_num(a,nan=0,posinf=1e6,neginf=-1e6)
    assert a.shape==(7,693) and np.isfinite(a).all()
    return a

def meta_features(scores,flags):
    scores=np.asarray(scores,float);flags=np.asarray(flags,dtype=bool)
    assert scores.shape==flags.shape==(7,) and np.isfinite(scores).all()
    assert scores.min()>=0 and scores.max()<=1
    a=scores.reshape(-1,7);b=flags.reshape(-1,7)
    rank=(np.argsort(np.argsort(-a,axis=1),axis=1)+1).reshape(-1)
    descending=np.sort(a,axis=1)[:,::-1]
    top1=np.repeat(descending[:,0],7)
    top2=np.repeat(descending[:,1],7)
    top3=np.repeat(descending[:,2],7)
    mean=np.repeat(a.mean(axis=1),7)
    std=np.repeat(a.std(axis=1),7)
    rc0count=np.repeat(b.sum(axis=1),7)
    count90=np.repeat((a>.9).sum(axis=1),7)
    count40=np.repeat((a>.4).sum(axis=1),7)
    eps=1e-5
    v=np.column_stack([scores,np.log((scores+eps)/(1-scores+eps)),
      flags.astype(float),rank,top1,top2,top3,mean,std,top1-scores,
      scores/(top1+eps),scores/(top2+eps),count90,count40,rc0count])
    assert v.shape==(7,15)
    return np.asarray(v,dtype=np.float32)

def predict_unlabeled(trace,stream):
    # IMPORTANT: This function accesses no *_expected, *_onset or stream truth.
    x=make_features(trace,stream)
    flags=np.asarray([v.learned_confirmed_alert for v in rc0.predict_trace(trace,stream)],dtype=bool)
    probs=.5*(base_models[0].predict_proba(x)[:,1]+base_models[1].predict_proba(x)[:,1])
    spatial=meta_features(probs,flags)
    p=np.asarray(meta_model.predict_proba(spatial)[:,1],dtype=float)
    assert p.shape==(7,) and np.isfinite(p).all()
    return p>=DECISION_THRESHOLD,flags,p

# Each folder is listed once at startup, then indexed in-memory. This avoids
# one list() request for every seed/checkpoint upload. Integrity still checked
# against immutable SHA-256 properties and stored bytes, not trusted cache alone.
_drive_index={}

def _folder_index(parent):
    if parent in _drive_index:
        return _drive_index[parent]
    result={}
    page_token=None
    while True:
        req=lambda: service.files().list(
            q=f"'{parent}' in parents and trashed = false",
            fields='nextPageToken,files(id,name,mimeType,appProperties)',
            pageSize=1000,pageToken=page_token).execute()
        data=drive_call(req,description='inventory folder '+parent)
        for f in data.get('files',[]):
            if f['name'] in result:
                raise RuntimeError('Ambiguous duplicate Drive item: '+f['name'])
            result[f['name']]=f
        page_token=data.get('nextPageToken')
        if not page_token:break
    _drive_index[parent]=result
    return result

def find_one(parent,name):
    return _folder_index(parent).get(name)

def mkdir_drive(parent,name):
    obj=find_one(parent,name)
    if obj:
        if obj['mimeType']!='application/vnd.google-apps.folder':
            raise RuntimeError('Folder name collision')
        return obj['id']
    obj=drive_call(lambda: service.files().create(
        body={'name':name,'parents':[parent],
              'mimeType':'application/vnd.google-apps.folder'},
        fields='id,name,mimeType,appProperties').execute(),
        description='create folder '+name)
    _folder_index(parent)[name]=obj
    return obj['id']

STUDY_ID=mkdir_drive(PARENT_FOLDER_ID,STUDY_FOLDER_NAME)
EVIDENCE_ID=mkdir_drive(STUDY_ID,'evidence_npz')
CHECKPOINT_ID=mkdir_drive(STUDY_ID,'seed_checkpoints')
for _folder_id in (EVIDENCE_ID,CHECKPOINT_ID):
    _folder_index(_folder_id)
print('RC3 resume inventory:',
      sum(k.startswith('s') and k.endswith('.json') for k in _folder_index(CHECKPOINT_ID)),
      'checkpoints,',
      sum(k.startswith('s') and k.endswith('.npz') for k in _folder_index(EVIDENCE_ID)),
      'evidence files',flush=True)
print('Prospective Drive folder:',f'https://drive.google.com/drive/folders/{STUDY_ID}',flush=True)

def upload_once(parent,name,path,mime,extra_properties=None):
    sha=digest(path)
    prior=find_one(parent,name)
    if prior:
        if prior.get('appProperties',{}).get('sha256')!=sha:
            raise RuntimeError('Immutable output mismatch: '+name)
        return prior['id']
    props={'sha256':sha,'study':STUDY[:95]}
    if extra_properties: props.update(extra_properties)
    obj=drive_call(lambda: service.files().create(
        body={'name':name,'parents':[parent],'appProperties':props},
        media_body=MediaFileUpload(str(path),mimetype=mime,resumable=True),
        fields='id,name,mimeType,appProperties').execute(),
        description='immutable upload '+name)
    _folder_index(parent)[name]={'id':obj['id'],'name':name,
                                  'mimeType':mime,'appProperties':props}
    return obj['id']

def load_checkpoint(name,dst):
    obj=find_one(CHECKPOINT_ID,name)
    if not obj:return False
    with open(dst,'wb') as f:
        d=MediaIoBaseDownload(f,service.files().get_media(fileId=obj['id']),chunksize=1048576)
        done=False
        while not done:_,done=drive_call(lambda: d.next_chunk(), description='download checkpoint '+name)
    if obj.get('appProperties',{}).get('sha256')!=digest(dst):
        raise RuntimeError('Corrupt checkpoint '+name)
    return True

manifest={
 'status':'FROZEN_BEFORE_PROSPECTIVE_EVALUATION','study':STUDY,
 'seed_range':[3000,3143],'required_seeds':N,
 'simulation':'NVIDIA Sionna 2.2.0 / 3GPP TR 38.901 UMa / GPU covariance / 300 15-min records',
 'development_seeds':[2500,2643],'feature_count':693,'spatial_feature_count':15,
 'source_zip_sha256':SOURCE_SHA256,'rc3_candidate_sha256':MODEL_SHA256,
 'decision_threshold':DECISION_THRESHOLD,
 'primary_metric':'fraction of injected faulty arrays detected in final 40/300 reports',
 'acceptance':{'recall_strictly_greater_than_95pct':True,
               'wilson_far_upper_strictly_below_2pct':True,
               'seed_cluster_far_upper_strictly_below_2pct':True},
 'all_training_and_tuning_complete_before_seed_3000':True,'research_only':True,
 'no_rf_hardware_actuation':True,
}
manifest_path=ROOT/'MANIFEST_FROZEN.json'
manifest_path.write_text(json.dumps(manifest,indent=2,sort_keys=True),encoding='utf-8')
upload_once(STUDY_ID,'MANIFEST_FROZEN.json',manifest_path,'application/json')

def score_one(seed,evidence,cov_sha):
    with np.load(evidence,allow_pickle=False) as f:
        z={k:f[k] for k in f.files}
    # Model inference FIRST; injected truth accessed only after inference is complete.
    pred={s:predict_unlabeled(z,s) for s in ('healthy','injected')}
    lab={s:np.asarray(z[f'{s}_expected'])>=0 for s in ('healthy','injected')}
    assert lab['healthy'].sum()==0 and lab['injected'].sum()==2
    methods={}
    for name,i in [('RC3',0),('RC0',1)]:
        h=pred['healthy'][i];inj=pred['injected'][i]
        methods[name]={'tp':int((inj&lab['injected']).sum()),
            'healthy_fp':int(h.sum()),'neighbor_fp':int((inj&~lab['injected']).sum())}
    return {'seed':int(seed),'study':STUDY,'methods':methods,
            'model_sha256':MODEL_SHA256,'source_sha256':SOURCE_SHA256,
            'decision_threshold':DECISION_THRESHOLD,
            'evidence_sha256':digest(evidence),'covariance_sha256':cov_sha,
            'research_only':True,'no_rf_hardware_actuation':True}

def wilson_upper(fp,n,z=1.959963984540054):
    p=fp/n;d=1+z*z/n
    return float((p+z*z/(2*n)+z*np.sqrt(p*(1-p)/n+z*z/(4*n*n)))/d)

def summarize(rows):
    rows=sorted(rows,key=lambda r:r['seed']);n=len(rows)
    assert len({r['seed'] for r in rows})==n
    results={'status':'COMPLETE' if n==N else 'INCOMPLETE',
      'study':STUDY,'completed_seeds':n,'required_seeds':N,
      'source_zip_sha256':SOURCE_SHA256,'model_sha256':MODEL_SHA256,
      'frozen_threshold':DECISION_THRESHOLD,'methods':{},'qualification':None,
      'research_only':True,'training_used_new_seed_labels':False}
    rng=np.random.default_rng(30103143)
    draws=rng.integers(0,n,size=(5000,n)) if n>1 else None
    for method in ('RC3','RC0'):
        tp=np.asarray([r['methods'][method]['tp'] for r in rows]);fp=np.asarray([r['methods'][method]['healthy_fp'] for r in rows]);nf=np.asarray([r['methods'][method]['neighbor_fp'] for r in rows]);
        obj={'tp':int(tp.sum()),'fault_n':int(2*n),'recall':float(tp.sum()/(2*n)) if n else None,
             'healthy_fp':int(fp.sum()),'healthy_n':int(7*n),
             'far':float(fp.sum()/(7*n)) if n else None,
             'neighbor_fp':int(nf.sum()),'neighbor_n':int(5*n),
             'wilson_far_upper_95':wilson_upper(int(fp.sum()),7*n) if n else None}
        if draws is not None:
            obj['seed_cluster_far_ci95']=np.quantile(fp[draws].sum(axis=1)/(7*n),[.025,.975]).tolist()
            obj['seed_cluster_recall_ci95']=np.quantile(tp[draws].sum(axis=1)/(2*n),[.025,.975]).tolist()
        results['methods'][method]=obj
    if draws is not None:
        gains=np.asarray([r['methods']['RC3']['tp']-r['methods']['RC0']['tp'] for r in rows]);
        results['paired_recall_gain_rc3_vs_rc0_ci95']=np.quantile(gains[draws].sum(axis=1)/(2*n),[.025,.975]).tolist()
    if n==N:
        rc=results['methods']['RC3']
        results['qualification']={
          'recall_strictly_greater_than_95pct':bool(rc['recall']>.95),
          'wilson_far_upper_strictly_below_2pct':bool(rc['wilson_far_upper_95']<.02),
          'seed_cluster_far_upper_strictly_below_2pct':bool(rc['seed_cluster_far_ci95'][1]<.02)}
        results['qualification']['all_pass']=bool(all(results['qualification'].values()))
    return results

rows=[];newly=0
for idx,seed in enumerate(SEEDS,1):
    checkpoint=ROOT/f's{seed}.json'
    evidence=ROOT/f's{seed}.npz'
    covariance=ROOT/f'cov_s{seed}.npz'
    if load_checkpoint(f's{seed}.json',checkpoint):
        row=json.loads(checkpoint.read_text())
        if (row['seed']!=seed or row['study']!=STUDY or
            row['model_sha256']!=MODEL_SHA256 or row['source_sha256']!=SOURCE_SHA256 or
            row['decision_threshold']!=DECISION_THRESHOLD):
            raise RuntimeError('Checkpoint provenance mismatch')
        rows.append(row)
        checkpoint.unlink(missing_ok=True)
        continue
    if newly>=MAX_NEW_SEEDS_THIS_SESSION:
        continue
    # Strict new-seed generation, or resume pre-existing, hash-verified evidence.
    saved_evidence=find_one(EVIDENCE_ID,f's{seed}.npz')
    if saved_evidence:
        with evidence.open('wb') as f:
            d=MediaIoBaseDownload(f,service.files().get_media(fileId=saved_evidence['id']),chunksize=1048576)
            done=False
            while not done: _,done=drive_call(lambda: d.next_chunk(), description='download evidence '+str(seed))
        props=saved_evidence.get('appProperties',{})
        cov_sha=props.get('covariance_sha256')
        if digest(evidence)!=props.get('sha256') or not cov_sha or len(cov_sha)!=64:
            raise RuntimeError('Existing evidence lacks valid SHA / covariance provenance')
    else:
        cov_sha=create_covariance(seed,SRC,covariance)
        extract_seed_v35(covariance,SRC,seed,evidence)
        upload_once(EVIDENCE_ID,f's{seed}.npz',evidence,'application/octet-stream',
                    {'covariance_sha256':cov_sha})
    row=score_one(seed,evidence,cov_sha)
    checkpoint.write_text(json.dumps(row,indent=2,sort_keys=True),encoding='utf-8')
    upload_once(CHECKPOINT_ID,f's{seed}.json',checkpoint,'application/json')
    rows.append(row)
    newly+=1
    covariance.unlink(missing_ok=True);evidence.unlink(missing_ok=True);checkpoint.unlink(missing_ok=True)
    if newly%8==0:
        print(f'RC3 NEW PROSPECTIVE: {len(rows)}/{N}; generated this session: {newly}',flush=True)

summary=summarize(rows)
if len(rows)==N:
    summary_path=ROOT/'SUMMARY_FROZEN.json'
    summary_path.write_text(json.dumps(summary,indent=2,sort_keys=True),encoding='utf-8')
    upload_once(STUDY_ID,'SUMMARY_FROZEN.json',summary_path,'application/json')
print('\n===== RC3 INDEPENDENT PROSPECTIVE RESULTS =====')
print(json.dumps(summary,indent=2))
print('Drive:',f'https://drive.google.com/drive/folders/{STUDY_ID}')
print('DO NOT retrain or adjust cutoff based on these outcomes; doing so consumes this holdout.')
print('RC3 transport runner v3 used read-only frozen model and threshold 0.50; previously uploaded checkpoints reused.')
