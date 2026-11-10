#!/usr/bin/env python3
"""Run a predeclared set of utility variants from immutable Python snapshots."""
import argparse,concurrent.futures,fcntl,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from covor.explore.execution_state import atomic_json
p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--out',required=True);p.add_argument('--workers',type=int,default=16);p.add_argument('--global-limit',type=int,default=0);p.add_argument('--source-from');a=p.parse_args();root=Path(a.out).resolve();root.mkdir(parents=True,exist_ok=True)
lock=(root/'batch.lock').open('a+')
try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError:sys.exit('batch already active')
spec=json.load(open(a.spec));fixed=root/'fixed_spec.json'
controller=root/'execution_controller.py'
if not controller.exists():shutil.copy2(Path(__file__),controller)
if (root/'execution_control.json').exists():
 history=root/'execution_control_history';history.mkdir(exist_ok=True)
 shutil.copy2(root/'execution_control.json',history/(str(time.time_ns())+'.json'))
atomic_json(root/'execution_control.json',dict(command=sys.argv,controller_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),pid=os.getpid(),workers=a.workers,global_limit=a.global_limit,mathematical_source_override=a.source_from))
if fixed.exists():assert json.load(open(fixed))==spec,'batch specification changed'
else:atomic_json(fixed,spec)
snap=root/'source_snapshot';manifest=root/'source_manifest.json'
if not manifest.exists():
 origin=Path(a.source_from).resolve() if a.source_from else Path.cwd()
 paths=sorted(set((origin/'covor').rglob('*.py'))|set((origin/'scripts/explore').glob('*.py')))
 hashes={}
 for source in paths:
  relative=source.relative_to(origin);dest=snap/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest);hashes[str(relative)]=hashlib.sha256(dest.read_bytes()).hexdigest()
 overlays=[]
 if a.source_from:
  # Keep the selected mathematical implementation byte-for-byte. Only the
  # validated execution CLI and exact native memoizer may be newer.
  for name in ['scripts/explore/run_entry_trial.py','covor/explore/execution_acceleration.py']:
   source=Path(name);dest=snap/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest);hashes[name]=hashlib.sha256(dest.read_bytes()).hexdigest();overlays.append(name)
 atomic_json(root/'source_origin.json',dict(origin=str(origin),execution_only_overlays=overlays))
 atomic_json(manifest,hashes)
else:
 hashes=json.load(open(manifest));assert all(hashlib.sha256((snap/name).read_bytes()).hexdigest()==h for name,h in hashes.items())
items=[]
for variant,path in spec['variants'].items():
 cfgpath=root/'configs'/f'{variant}.json';cfgpath.parent.mkdir(exist_ok=True)
 cfg=json.load(open(path))
 if cfgpath.exists():assert json.load(open(cfgpath))==cfg
 else:atomic_json(cfgpath,cfg)
 for scene in spec['scenes']:
  for seed in spec['seeds']:
   for condition in spec['conditions']:items.append((variant,cfgpath,scene,seed,condition))
atomic_json(root/'status.json',dict(phase='running',expected=len(items),pid=os.getpid(),started=time.time(),workers=a.workers,global_limit=a.global_limit,spec=spec))
def start_trial(command,log):
 if not a.global_limit:return subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
 while True:
  with (root.parent/'compute_resources.lock').open('a+') as resource:
   fcntl.flock(resource,fcntl.LOCK_EX);active=0
   for process in Path('/proc').iterdir():
    if not process.name.isdigit():continue
    try:
     raw=(process/'cmdline').read_bytes()
     active+=b'run_entry_trial.py' in raw and b'covor_slam' in raw
    except (OSError,PermissionError):pass
   if active<a.global_limit:return subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
  time.sleep(2)
def trial(item):
 variant,config,scene,seed,condition=item;policy=spec.get('variant_policies',{}).get(variant,'entry');name=f'scene{scene}_seed{seed}_{condition}_{policy}';folder=root/variant/name;folder.parent.mkdir(exist_ok=True)
 # A controller can be restarted or resize its pool without killing existing
 # simulation workers. Wait for their advisory lock before reading a possibly
 # still-being-written summary; never start a duplicate or count it as failure.
 workerlock=folder/'.execution/worker.lock'
 while workerlock.exists():
  with workerlock.open('a+') as probe:
   try:fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB)
   except BlockingIOError:time.sleep(2);continue
   fcntl.flock(probe,fcntl.LOCK_UN);break
 if (folder/'summary.json').exists():return dict(variant=variant,run=name,status='complete',summary=json.load(open(folder/'summary.json')))
 command=[sys.executable,str(snap/'scripts/explore/run_entry_trial.py'),'--config',str(config),'--out',str(folder),'--scene',str(scene),'--seed',str(seed),'--condition',condition,'--policy',policy]
 if spec.get('duration') is not None:command+=['--duration',str(spec['duration'])]
 if spec.get('stop_after') is not None:command+=['--stop-after',str(spec['stop_after'])]
 if spec.get('memoize_motion'):command+=['--memoize-motion']
 if spec.get('defer_dense'):command+=['--defer-dense']
 with (folder.parent/(name+'.log')).open('a') as log:returncode=start_trial(command,log).wait()
 final=folder/'summary.json';screen=folder/'screen_complete.json';status='complete' if final.exists() else 'screened' if screen.exists() else 'failed'
 return dict(variant=variant,run=name,status=status,returncode=returncode,summary=json.load(open(final)) if final.exists() else None)
rows=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
 submitted={pool.submit(trial,item):item for item in items}
 for future in concurrent.futures.as_completed(submitted):
  try:row=future.result()
  except Exception as exc:
   variant,_,scene,seed,condition=submitted[future];policy=spec.get('variant_policies',{}).get(variant,'entry')
   row=dict(variant=variant,run=f'scene{scene}_seed{seed}_{condition}_{policy}',status='failed',error=repr(exc),summary=None)
  rows.append(row);atomic_json(root/'trial_results.json',rows);print('FINISHED',row['variant'],row['run'],row['status'],len(rows),'/',len(items),flush=True)
failed=[q for q in rows if q['status']=='failed'];atomic_json(root/'status.json',dict(phase='failed' if failed else 'complete',expected=len(items),finished=len(rows),failures=failed,completed=time.time()))
assert not failed,failed
assert all(hashlib.sha256((snap/name).read_bytes()).hexdigest()==h for name,h in hashes.items())
