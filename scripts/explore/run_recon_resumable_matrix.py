#!/usr/bin/env python3
"""Durable continuation: 20 validated reused trials and 20 cache-backed trials."""
import argparse,concurrent.futures,csv,fcntl,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from covor.explore.runner import write_json,csv_write
from covor.explore.execution_state import atomic_json
p=argparse.ArgumentParser();p.add_argument('--root',default='results/fire_recon_20261108_resume');p.add_argument('--previous',default='results/fire_recon_20261107');p.add_argument('--workers',type=int,default=20);a=p.parse_args();root=Path(a.root);prev=Path(a.previous);main=root/'main';main.mkdir(parents=True,exist_ok=True);(root/'audit').mkdir(exist_ok=True);lock=(root/'matrix.lock').open('a+')
try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError:raise RuntimeError('resumable matrix already running')
cfg=json.load(open(prev/'main/fixed_config.json'));fixed=main/'fixed_config.json'
if fixed.exists():assert json.load(open(fixed))==cfg
else:write_json(fixed,cfg)
for name in ['evaluation_cases.json','reference_refinements.json']:
 if not (root/name).exists():
  if (prev/name).exists():shutil.copy2(prev/name,root/name)
  elif name=='reference_refinements.json':write_json(root/name,[])
  else:raise FileNotFoundError(prev/name)
for name in ['gates_complete','assets','intake']:
 if not (root/name).exists():(root/name).symlink_to((prev/name).resolve(),target_is_directory=True)
if not (main/'scenes').exists():(main/'scenes').symlink_to((prev/'main/scenes').resolve(),target_is_directory=True)
for source in (prev/'main').glob('scene*'):
 if not source.is_dir() or not (source/'reuse_provenance.json').exists():continue
 target=main/source.name
 if (target/'summary.json').exists():continue
 target.mkdir(exist_ok=True)
 for item in source.iterdir():
  dest=target/item.name
  if item.name=='replay_manifest.json':continue
  if item.is_dir():
   if not dest.exists():dest.symlink_to(item.resolve(),target_is_directory=True)
  else:shutil.copy2(item,dest)
for name in ['reused_runs.csv','reuse_observation_replays.csv','reuse_checks.json']:
 if not (main/name).exists():shutil.copy2(prev/'main'/name,main/name)
sources=set(Path('covor').rglob('*.py'))|set(Path('scripts/explore').glob('*.py'));manifest={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(sources)}
if (main/'source_manifest.json').exists():assert json.load(open(main/'source_manifest.json'))==manifest,'source changed since resumable trials started'
else:
 write_json(main/'source_manifest.json',manifest)
 for f in sources:
  target=main/'source_snapshot'/f;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,target)
write_json(root/'execution.json',dict(started=time.time(),pid=os.getpid(),command=sys.argv,previous_interrupted_root=str(prev),numeric_config_unchanged=True,execution_only_change='checkpoint/recovery and causal cached depth replay',verification='validation/execution_v1/verification.json',checkpoint_seconds=4.,cache_read_contract='current time/drone/pose/teammates must match; old maps are evaluation-only exact-prefix checks'))
items=[(i,s,c,p0) for i in cfg['scene_seeds'] for s in cfg['run_seeds'] for c in cfg['conditions'] for p0 in cfg['policies']];rows=[];errors=[]
def status(phase,**kw):atomic_json(root/'pipeline_status.json',dict(phase=phase,**kw))
status('main_matrix_running',expected_runs=len(items),reused_runs=20,fresh_runs=20,durable_checkpoints=True)
def one(item):
 i,s,c,policy=item;name=f'scene{i}_seed{s}_{c}_{policy}';folder=main/name
 if (folder/'summary.json').exists() and (folder/'maps/final.npz').exists():return name,json.load(open(folder/'summary.json')),None
 command=[sys.executable,'scripts/explore/run_fire_recon_resumable.py','--config',str(fixed),'--out',str(folder),'--scene',str(i),'--seed',str(s),'--condition',c,'--policy',policy,'--cache',str(prev/'main'/name)]
 with (main/(name+'.log')).open('a') as f:result=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT)
 if result.returncode:return name,None,dict(run=name,returncode=result.returncode,command=command)
 return name,json.load(open(folder/'summary.json')),None
with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
 for future in concurrent.futures.as_completed([pool.submit(one,item) for item in items]):
  name,row,error=future.result()
  if error:errors.append(error)
  else:rows.append(dict(run=name,**row))
  csv_write(main/'aggregate.csv',sorted(rows,key=lambda q:q['run']));write_json(root/'execution_failures.json',errors);print('FINISHED',name,len(rows),'errors',len(errors),flush=True)
if errors:status('execution_failed',errors=errors);raise RuntimeError(errors)
assert len(rows)==len(items);assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in manifest.items());write_json(main/'complete.json',dict(runs=len(rows),all_expected_runs_retained=True,source_unchanged=True));status('main_complete',runs=len(rows),reused_runs=20,cached_prefix_trials=20);print('ALL DURABLE RUNS COMPLETE',flush=True)
