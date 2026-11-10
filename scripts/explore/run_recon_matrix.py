#!/usr/bin/env python3
"""Run paired policies with a fixed, already validated scene revision.

No benchmark defaults are guessed here: config and per-scene complete-map gates
are required. Sources/config are snapshotted before any policy is launched.
"""
import argparse,concurrent.futures,csv,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from covor.explore.fire_recon_scene import ReconFireWorld
from covor.explore.runner import write_json,csv_write
p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--gates',required=True);p.add_argument('--out',required=True);p.add_argument('--workers',type=int,default=8);a=p.parse_args();cfgpath=Path(a.config);cfg=json.load(open(cfgpath));out=Path(a.out);out.mkdir(parents=True,exist_ok=True);gates=Path(a.gates)
assert set(cfg['policies'])=={'unknown','entry'};assert cfg['scene']['repair_bottom_landings'];assert cfg['scene']['type']=='multistorey_fire'
fixed=out/'fixed_config.json'
if fixed.exists():assert json.load(open(fixed))==cfg,'configuration changed; use a new output path'
else:write_json(fixed,cfg)
(out/'scenes').mkdir(exist_ok=True)
for index in cfg['scene_seeds']:
 b=ReconFireWorld(index,cfg);gatecfg=json.load(open(gates/f'scene{index}'/'config.json'));assert gatecfg['human']==cfg['human'] and gatecfg['map_resolution']==cfg['map_resolution'];gate=json.load(open(gates/f'scene{index}'/'summary.json'));assert gate['geometry_sha256']==b.geometry_hash();assert gate['invalid_paths']==0 and gate['false_blocked']==0 and gate['valid_confirmed']==gate['reachable_denominator'];assert json.load(open(gates/f'scene{index}'/'prior.json'))==b.prior()
 write_json(out/'scenes'/f'scene{index}.json',b.metadata())
sources=set(Path('covor').rglob('*.py'))|set(Path('scripts/explore').glob('*.py'))|{cfgpath,Path('tests/test_fire_recon_scene.py')};manifest={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(sources)}
if (out/'source_manifest.json').exists():assert json.load(open(out/'source_manifest.json'))==manifest,'source changed; use a new output path'
else:
 write_json(out/'source_manifest.json',manifest)
 for f in sources:
  relative=f if not f.is_absolute() else Path('_external')/f.name
  target=out/'source_snapshot'/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,target)
items=[(i,s,c,p0) for i in cfg['scene_seeds'] for s in cfg['run_seeds'] for c in cfg['conditions'] for p0 in cfg['policies']]
write_json(out/'execution.json',dict(started_unix=time.time(),command=sys.argv,expected_runs=len(items),base_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),world_factory='covor.explore.fire_recon_scene.ReconFireWorld',cli='scripts/explore/run_fire_recon.py',main_difference='observation utility only',complete_geometry_gates=str(gates)))
def one(item):
 i,s,c,policy=item;name=f'scene{i}_seed{s}_{c}_{policy}';folder=out/name
 if (folder/'summary.json').exists():return name,json.load(open(folder/'summary.json')),None
 command=[sys.executable,'scripts/explore/run_fire_recon.py','--config',str(fixed),'--scene',str(i),'--seed',str(s),'--condition',c,'--policy',policy,'--out',str(folder)]
 with (out/(name+'.log')).open('w') as f:result=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT)
 if result.returncode:return name,None,dict(run=name,returncode=result.returncode,command=command,log=str(out/(name+'.log')))
 return name,json.load(open(folder/'summary.json')),None
rows=[];failures=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
 for future in concurrent.futures.as_completed([pool.submit(one,item) for item in items]):
  name,result,error=future.result()
  if error:failures.append(error)
  else:rows.append(dict(run=name,**result))
  csv_write(out/'aggregate.csv',sorted(rows,key=lambda q:q['run']));write_json(out/'execution_failures.json',failures);print('FINISHED',name,'completed',len(rows),'execution errors',len(failures),flush=True)
assert not failures,failures;assert len(rows)==len(items);assert all(hashlib.sha256(Path(f).read_bytes()).hexdigest()==h for f,h in manifest.items()),'run source changed'
write_json(out/'complete.json',dict(runs=len(rows),all_expected_runs_retained=True,source_unchanged=True));print('ALL RUNS COMPLETE',flush=True)
