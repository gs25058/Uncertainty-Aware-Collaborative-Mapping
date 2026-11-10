#!/usr/bin/env python3
"""Reuse unchanged validated geometry; validate the remaining repaired scenes.

All goal labels/regions are fixed before the new policy matrix. No policy result
is used for these gates, and old caches/results are never overwritten.
"""
import concurrent.futures,hashlib,json,pickle,shutil,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from covor.explore.fire_world import FireWorld
from covor.explore.fire_recon_scene import ReconFireWorld
from covor.explore.runner import write_json,csv_write
root=Path('results/fire_recon_20261107');old=Path('results/fire_closed_loop_20261107');out=root/'gates_complete';out.mkdir(parents=True,exist_ok=True);cfg=json.load(open('configs/fire_recon_20261107.json'));oldcfg=json.load(open(old/'main/fixed_config.json'));write_json(out/'fixed_config.json',cfg)
protected=json.load(open(root/'intake/common_pipeline_unchanged.json'));oldmanifest=json.load(open(old/'main/source_manifest.json'))
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==oldmanifest[p] for p in protected['source_hashes_unchanged'])
asset_checks={q['scene']:q for q in __import__('csv').DictReader(open(root/'intake/asset_geometry_checks.csv'))}
fresh=[];provenance=[]
for i in cfg['scene_seeds']:
 b=ReconFireWorld(i,cfg);target=out/f'scene{i}'
 if not b.geometry_repairs:
  assert b.geometry_hash()==asset_checks[str(i)]['base_geometry_sha256']
  source=old/'gates_complete'/f'scene{i}';prior=json.load(open(old/'reference_checks'/f'scene{i}'/'prior.json'));assert prior==b.prior();s=json.load(open(source/'summary.json'));assert s['valid_confirmed']==s['reachable_denominator'] and s['invalid_paths']==0 and s['false_blocked']==0
  target.mkdir(exist_ok=True);shutil.copy2(source/'targets.csv',target/'targets.csv');write_json(target/'prior.json',prior);write_json(target/'geometry.json',b.metadata());write_json(target/'config.json',oldcfg)
  reference=FireWorld(i,oldcfg).reference(cfg['map_resolution']);write_json(target/'reference_targets.json',reference[0]);cache=b.reference_cache_path(cfg['map_resolution']);cache.parent.mkdir(parents=True,exist_ok=True)
  if not cache.exists():
   with cache.open('wb') as f:pickle.dump(reference,f)
  write_json(cache.with_suffix('.provenance.json'),dict(reused_reference_from=str(old/'reference'),same_solid_geometry_sha256=b.geometry_hash(),same_mission_regions=True,reference_source_unchanged=True))
  s.update(geometry_sha256=b.geometry_hash(),third_floor_valid=0,reference_reachable_third_floor=0,reused_from=str(source),reuse_basis='exactly unchanged solids, regions and protected human/path code');write_json(target/'summary.json',s);provenance.append(dict(scene=i,reused=True,source=str(source)));print('REUSED',i,flush=True)
 elif i==1:
  source=root/'preflight/scene1_complete';s=json.load(open(source/'summary.json'));assert s['geometry_sha256']==b.geometry_hash();assert json.load(open(source/'prior.json'))==b.prior();target.mkdir(exist_ok=True)
  for name in ['summary.json','targets.csv','positive_controls.csv','reference_targets.json','prior.json','geometry.json','config.json']:shutil.copy2(source/name,target/name)
  provenance.append(dict(scene=i,reused=True,source=str(source)));print('REUSED',i,'corrected school gate',flush=True)
 else:fresh.append(i)
write_json(out/'reuse_provenance.json',provenance)
sources=set(Path('covor').rglob('*.py'))|set(Path('scripts/explore').glob('*.py'))|{Path('configs/fire_recon_20261107.json'),Path('tests/test_fire_recon_scene.py')};manifest={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(sources)};write_json(out/'source_manifest.json',manifest)
for p in sources:
 target=out/'source_snapshot'/p;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
write_json(out/'status.json',dict(phase='fresh_geometry_gates_running',fresh=fresh,reused=len(provenance)))
def one(i):
 log=root/'intake'/f'gate_scene{i}.log'
 with log.open('w') as f:subprocess.run([sys.executable,'scripts/explore/validate_fire_recon.py','--config',str(out/'fixed_config.json'),'--scene',str(i),'--out',str(out/f'scene{i}')],stdout=f,stderr=subprocess.STDOUT,check=True)
 print('GATE PASS',i,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(one,fresh))
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in manifest.items())
cases=[];totals={k:0 for k in ['reachable','blocked','valid_complete','correct_blocked_complete','third_floor_valid']}
for i in cfg['scene_seeds']:
 folder=out/f'scene{i}';s=json.load(open(folder/'summary.json'));ref=json.load(open(folder/'reference_targets.json'));prior=json.load(open(folder/'prior.json'));assert [r['id'] for r in ref]==[g['id'] for g in prior['goals']]
 cases.append(dict(scene=i,prior=prior,reference_targets=ref,geometry_sha256=s['geometry_sha256'],complete_map_summary=s));totals['reachable']+=s['reachable_denominator'];totals['blocked']+=s['blocked_denominator'];totals['valid_complete']+=s['valid_confirmed'];totals['correct_blocked_complete']+=s['correct_blocked'];totals['third_floor_valid']+=s['third_floor_valid']
assert totals['third_floor_valid']>0 and totals['reachable'] and totals['blocked'];write_json(root/'evaluation_cases.json',cases);write_json(out/'status.json',dict(phase='complete',scenes=len(cases),reused=len(provenance),fresh=len(fresh),**totals));print('ALL GEOMETRY GATES COMPLETE',totals,flush=True)
