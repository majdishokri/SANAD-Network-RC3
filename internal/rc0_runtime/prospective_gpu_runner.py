"""GPU Sionna new-seed evaluator for frozen SANAD v35 learned-gate candidate.

Run only from the accompanying Google Colab notebook on actual NVIDIA GPU.
Sionna channel generation runs on CUDA; detector replay, learned scoring, and
statistical inference run on CPU. Reruns restore immutable Google Drive
checkpoints. NEVER tunes a model or threshold from new evaluation labels.
"""
from __future__ import annotations
import hashlib,json,os,sys,time,importlib,io,tempfile
from pathlib import Path
import numpy as np
import torch
from qualification import wilson
from inference_v35_gate import FrozenV35LearnedGate

NEW_SEEDS=tuple(range(2500,2644))
STUDY='SANAD_V35_LEARNED_GATE_RC0_SIONNA_GPU_NEW_SEEDS_2500_2643'
HISTORICAL_TRAINING_SEEDS='1264..1407'
PERIODS=300
N_ARRAYS=7
REQUIRED=144

def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()

class DriveCheckpoint:
    def __init__(self,service,root_folder_id):
        self.d=service
        self.root=root_folder_id
        self.evidence=self.ensure_folder('evidence_npz',root_folder_id)
        self.checkpoints=self.ensure_folder('seed_checkpoints',root_folder_id)
    def _find(self,parent,name):
        query=f"'{parent}' in parents and name = '{name}' and trashed = false"
        result=self.d.files().list(q=query,fields='nextPageToken,files(id,name,mimeType,appProperties)',pageSize=100).execute()
        entries=result.get('files',[])
        if len(entries)>1:raise RuntimeError('Ambiguous checkpoint file '+name)
        return entries[0] if entries else None
    def ensure_folder(self,name,parent):
        f=self._find(parent,name)
        if f:
            if f['mimeType']!='application/vnd.google-apps.folder':raise RuntimeError('Name collision')
            return f['id']
        data={'name':name,'parents':[parent],'mimeType':'application/vnd.google-apps.folder'}
        return self.d.files().create(body=data,fields='id').execute()['id']
    def load(self,parent,name,dst):
        f=self._find(parent,name)
        if f is None:return False
        dst=Path(dst);dst.parent.mkdir(parents=True,exist_ok=True)
        from googleapiclient.http import MediaIoBaseDownload
        with dst.open('wb') as out:
            down=MediaIoBaseDownload(out,self.d.files().get_media(fileId=f['id']),chunksize=1024*1024)
            done=False
            while not done:_,done=down.next_chunk()
        expected=f.get('appProperties',{}).get('sha256')
        if not expected or digest(dst)!=expected:raise RuntimeError('Checkpoint checksum mismatch '+name)
        return True
    def upload_once(self,parent,name,src,mimetype,extra_properties=None):
        src=Path(src)
        sha=digest(src)
        old=self._find(parent,name)
        if old:
            if old.get('appProperties',{}).get('sha256')!=sha:
                raise RuntimeError('Existing immutable file differs: '+name)
            return old['id']
        props={'sha256':sha,'study':STUDY[:96]}
        if extra_properties:props.update(extra_properties)
        body={'name':name,'parents':[parent], 'appProperties':props}
        from googleapiclient.http import MediaFileUpload
        media=MediaFileUpload(str(src),mimetype=mimetype,resumable=True)
        return self.d.files().create(body=body,media_body=media,fields='id').execute()['id']


def create_covariance(seed,base,output):
    from sionna.phy import config as phy_config
    if not torch.cuda.is_available() or 'cuda:0' not in phy_config.available_devices:
        raise RuntimeError('CUDA GPU not detected by Sionna. No CPU fallback.')
    phy_config.device='cuda:0'
    if 'sionna_cuda_v34' not in sys.modules:
        from sionna_cuda_v34 import covariances
    else:
        from sionna_cuda_v34 import covariances
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    rng=np.random.default_rng(seed)
    slices=[]
    for index in range(6):
        block=covariances(rng,50)
        if block.shape!=(50,7,8,8) or not np.isfinite(block).all():raise RuntimeError('GPU covariance validation failed')
        slices.append(block)
    arr=np.concatenate(slices,axis=0)
    if arr.shape!=(300,7,8,8):raise RuntimeError('Covariance shape mismatch')
    np.savez_compressed(output,**{f's{seed}':arr})
    from qualification import validate_covariance
    validate_covariance(arr)
    return digest(output)


def score_seed(seed,trace_path,model,package_sha,model_sha,covariance_sha):
    with np.load(trace_path,allow_pickle=False) as f:trace={k:f[k] for k in f.files}
    outcome={}
    for stream in ('healthy','injected'):
        decisions=model.predict_trace(trace,stream) # NOT reading truth until AFTER scores
        injected=np.asarray(trace[f'{stream}_expected'])>0
        learned=np.array([x.learned_confirmed_alert for x in decisions],bool)
        sensitive=np.array([x.v35_candidate for x in decisions],bool)
        conservative=np.array([x.v33_conservative_alert for x in decisions],bool)
        v11=np.asarray(trace[f'{stream}_v11'][260:300],bool).any(axis=0)
        for key,flag in [('learned',learned),('v35',sensitive),('v33',conservative),('v11',v11)]:
            row=outcome.setdefault(key,dict(tp=0,fp=0,neighbor_fp=0))
            if stream=='healthy':row['fp']=int(flag.sum())
            else:
                row['tp']=int((flag&injected).sum())
                row['neighbor_fp']=int((flag&~injected).sum())
    return {'seed':seed,'protocol':STUDY,'source_zip_sha256':package_sha,'model_sha256':model_sha,
            'evidence_sha256':digest(trace_path),'covariance_sha256':covariance_sha,
            'methods':outcome,'research_only':True,'train_seeds':HISTORICAL_TRAINING_SEEDS,
            'cuda':torch.cuda.get_device_name(0),'no_hardware_actuation':True}


def summary(rows):
    n=len(rows);out={'status':'COMPLETE' if n==REQUIRED else 'INCOMPLETE',
                      'protocol':STUDY,'new_seed_count':n,'required_seeds':REQUIRED,
                      'methods':{},'qualification':None,'training_used_prospective_labels':False,
                      'research_only':True}
    if n==0:return out
    if len(set(x['seed'] for x in rows))!=n:raise RuntimeError('Duplicate seed')
    for m in ('learned','v35','v33','v11'):
        tp=sum(r['methods'][m]['tp'] for r in rows)
        fp=sum(r['methods'][m]['fp'] for r in rows)
        neighbor=sum(r['methods'][m]['neighbor_fp'] for r in rows)
        out['methods'][m]={'tp':tp,'fault_n':2*n,'recall':tp/(2*n),'fp':fp,'healthy_n':7*n,
                            'far':fp/(7*n),'neighbor_fp':neighbor,'wilson_far_ci95':wilson(fp,7*n)}
    if n>=2:
        rng=np.random.default_rng(351335);draws=rng.integers(0,n,size=(5000,n))
        for m in out['methods']:
            f=np.array([r['methods'][m]['fp'] for r in rows]);t=np.array([r['methods'][m]['tp'] for r in rows])
            out['methods'][m]['cluster_far_ci95']=np.quantile(f[draws].sum(axis=1)/(7*n),[.025,.975]).tolist()
            out['methods'][m]['cluster_recall_ci95']=np.quantile(t[draws].sum(axis=1)/(2*n),[.025,.975]).tolist()
        dx=np.array([r['methods']['learned']['tp']-r['methods']['v11']['tp'] for r in rows])
        out['paired_recall_gain_learned_vs_v11_ci95']=np.quantile(dx[draws].sum(axis=1)/(2*n),[.025,.975]).tolist()
    if n==REQUIRED:
        l=out['methods']['learned'];wf=l['wilson_far_ci95'][1];cf=l['cluster_far_ci95'][1]
        boost=out['paired_recall_gain_learned_vs_v11_ci95'][0]
        out['qualification']={'all_gates_pass':bool(l['recall']>=.90 and wf<.02 and cf<.02 and boost>0),
           'recall_at_least_90pct':l['recall']>=.90,'wilson_far_upper_below_2pct':wf<.02,
           'cluster_far_upper_below_2pct':cf<.02,'paired_recall_lower_above_zero':boost>0,
           'aspirational_recall_at_least_95_14pct_and_observed_far_le_1_69pct':bool(
              l['recall']>=274/288 and l['far']<=17/1008)}
    return out


def evaluate_new_seeds(service,root_folder_id,base,source_zip_sha,local_root='/content/SANAD_V35_Learned_Gate_New_Seeds',max_seeds=None):
    base=Path(base)
    loc=Path(local_root);loc.mkdir(exist_ok=True,parents=True)
    root=DriveCheckpoint(service,root_folder_id)
    model=FrozenV35LearnedGate(base/'frozen_candidate'/'selected_model.joblib')
    model_sha=digest(base/'frozen_candidate'/'selected_model.joblib')
    rows=[]
    if max_seeds is None:max_seeds=len(NEW_SEEDS)
    max_seeds=int(max_seeds)
    if not (0<=max_seeds<=len(NEW_SEEDS)):raise ValueError('max_seeds must be 0..144')
    newly_processed=0
    for number,seed in enumerate(NEW_SEEDS,1):
        # If result already exists, restore and validate immutable provenance.
        checkpoint=loc/f's{seed}.json';evidence=loc/f's{seed}.npz';cov=loc/f'cov_s{seed}.npz'
        if root.load(root.checkpoints,f's{seed}.json',checkpoint):
            r=json.loads(checkpoint.read_text())
            if (r['seed']!=seed or r['protocol']!=STUDY or r['model_sha256']!=model_sha or
                r['source_zip_sha256']!=source_zip_sha):raise RuntimeError('Checkpoint provenance mismatch')
            rows.append(r);continue
        if newly_processed>=max_seeds:
            continue  # Resume next not-yet-completed seed on future notebook run.
        if root.load(root.evidence,f's{seed}.npz',evidence):
            saved=root._find(root.evidence,f's{seed}.npz')
            properties=saved.get('appProperties',{})
            covsha=properties.get('cov_sha256')
            if not covsha or len(covsha)!=64:
                raise RuntimeError('Recovered evidence lacks immutable covariance provenance')
        else:
            covsha=create_covariance(seed,base,cov)
            from v35_replay import extract_seed_v35
            extract_seed_v35(cov,base,seed,evidence)
            root.upload_once(root.evidence,f's{seed}.npz',evidence,'application/octet-stream',
                             extra_properties={'cov_sha256':covsha,'source_sha256':source_zip_sha})
            cov.unlink(missing_ok=True)
        result=score_seed(seed,evidence,model,source_zip_sha,model_sha,covsha)
        checkpoint.write_text(json.dumps(result,sort_keys=True,indent=2))
        root.upload_once(root.checkpoints,f's{seed}.json',checkpoint,'application/json')
        rows.append(result)
        newly_processed+=1
        evidence.unlink(missing_ok=True)
        if newly_processed%8==0:
            print('NEW PROSPECTIVE SEEDS:',len(rows),'/',REQUIRED,'— generated this session:',newly_processed,flush=True)
    rows.sort(key=lambda d:d['seed'])
    final=summary(rows)
    if len(rows)==REQUIRED:
        outpath=loc/'SUMMARY_FROZEN.json';outpath.write_text(json.dumps(final,sort_keys=True,indent=2))
        root.upload_once(root.root,'SUMMARY_FROZEN.json',outpath,'application/json')
    print('STUDY STATUS:',final['status'],'| seeds',len(rows),'/',REQUIRED,flush=True)
    return final
