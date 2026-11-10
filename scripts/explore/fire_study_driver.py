#!/usr/bin/env python3
"""Gate the fixed 40-case study; freeze source/config before paired decisions."""
import argparse,csv,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from covor.explore.runner import write_json,csv_write
csv.field_size_limit(sys.maxsize)
p=argparse.ArgumentParser();p.add_argument('--root',default='results/fire_closed_loop_20261107');p.add_argument('--config',default='configs/fire_closed_loop_20261107.json');p.add_argument('--report',default='reports/fire_closed_loop_20261107');p.add_argument('--workers',type=int,default=40);a=p.parse_args();root=Path(a.root);report=Path(a.report);report.mkdir(parents=True,exist_ok=True)
def status(phase,**kw):write_json(root/'pipeline_status.json',dict(phase=phase,**kw))
status('waiting_for_complete_geometry_and_four_drone_gates')
deadline=time.monotonic()+5400
while True:
    gates=list((root/'gates_complete').glob('scene*/summary.json'));pilots=list((root/'pilot').glob('four16_*/summary.json'))
    for f in (root/'audit').glob('gate_complete_*.log'):
        if 'Traceback (most recent call last)' in f.read_text():status('gate_failed',log=str(f));raise RuntimeError(f)
    for policy in ('unknown','entry'):
        f=root/'audit'/f'pilot_four16_{policy}.log'
        if f.exists() and 'Traceback (most recent call last)' in f.read_text():status('pilot_failed',log=str(f));raise RuntimeError(f)
    if len(gates)==10 and len(pilots)==2:break
    if time.monotonic()>deadline:raise TimeoutError('gates did not complete')
    time.sleep(15)
cfg=json.loads(Path(a.config).read_text());cases=[];promotions=[];certified=root/'reference_final';certified.mkdir(exist_ok=True)
for i in cfg['scene_seeds']:
    folder=root/'gates_complete'/f'scene{i}';summary=json.load(open(folder/'summary.json'));assert summary['invalid_paths']==0 and summary['valid_confirmed']>0
    details=list(csv.DictReader(open(folder/'targets.csv')));truth=json.load(open(root/'reference_checks'/f'scene{i}'/'truth.json'));prior=json.load(open(root/'reference_checks'/f'scene{i}'/'prior.json'))
    # A separately mesh-validated existence witness can repair a finite-grid
    # reference miss BEFORE policy runs. No policy observation is consulted.
    for t,d in zip(truth,details):
        if not t['reachable'] and d['path_status']=='valid':
            t.update(reachable=True,path=json.loads(d['path']),reference_method='pre-comparison complete-geometry witness independently mesh-validated')
            promotions.append(dict(scene=i,goal=t['id'],reason='independent mesh-validated existence witness'))
    upper_ref=any(t['reachable'] and not t['id'].startswith('L1_') for t in truth)
    upper_valid=any(d['path_status']=='valid' and not d['goal'].startswith('L1_') for d in details)
    if upper_ref and not upper_valid:status('upper_floor_gate_failed',scene=i);raise RuntimeError(f'no complete-map upper path in scene {i}')
    write_json(certified/f'scene{i}.json',truth);cases.append(dict(scene=i,prior=prior,reference_targets=truth,complete_map_summary=summary))
cfg['scene']['certified_reference_dir']=str(certified);write_json(Path(a.config),cfg);write_json(root/'evaluation_cases.json',cases);write_json(root/'reference_refinements.json',promotions)
main=root/'main';main.mkdir(exist_ok=True);snapshot=root/'source_snapshot';sources=sorted(set(Path('covor').rglob('*.py'))|set(Path('scripts/explore').glob('*.py'))|{Path(a.config),Path('tests/test_fire_multistorey.py')})
manifest={}
for f in sources:
    manifest[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest();target=snapshot/f;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,target)
write_json(main/'source_manifest.json',manifest);write_json(root/'execution.json',dict(command=sys.argv,workers=a.workers,expected_runs=40,base_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),reference_promotions=len(promotions),conditions=cfg['conditions'],duration=cfg['duration'],seed_policy='scene/run/drone/time/actual pose; no policy key'))
status('main_matrix_running',expected_runs=40,source_files=len(manifest))
with (root/'audit/matrix.log').open('w') as f:subprocess.run([sys.executable,'scripts/explore/run_matrix.py','--config',a.config,'--out',str(main),'--workers',str(a.workers)],stdout=f,stderr=subprocess.STDOUT,check=True)
assert len(list(main.glob('scene*/summary.json')))==40
assert all(hashlib.sha256(Path(f).read_bytes()).hexdigest()==h for f,h in manifest.items()),'main-run source changed'
status('main_complete_postprocessing',completed_runs=40)
with (root/'audit/summarize.log').open('w') as f:subprocess.run([sys.executable,'scripts/explore/summarize.py','--root',str(main),'--out',str(report/'data')],stdout=f,stderr=subprocess.STDOUT,check=True)
status('complete',completed_runs=40);print('ALL 40 RUNS AND TABLES COMPLETE',flush=True)
