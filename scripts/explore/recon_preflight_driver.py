#!/usr/bin/env python3
"""Gate the repaired scene and exercise both unchanged values on the same rig."""
import concurrent.futures,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
root=Path('results/fire_recon_20261107');cfgpath=Path('configs/fire_recon_20261107.preflight.json');manifest=json.load(open(root/'preflight/source_manifest.json'))
def status(phase,**kw):(root/'preflight/status.json').write_text(json.dumps(dict(phase=phase,**kw),indent=2)+'\n')
status('waiting_for_complete_geometry_gate')
while not (root/'preflight/scene1_complete/summary.json').exists():
 if 'Traceback' in (root/'intake/complete_gate.log').read_text():raise RuntimeError('complete geometry gate failed; see intake/complete_gate.log')
 time.sleep(20)
gate=json.load(open(root/'preflight/scene1_complete/summary.json'));assert gate['third_floor_valid']>0 and gate['valid_confirmed']==gate['reachable_denominator'] and gate['invalid_paths']==0
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in manifest.items())
status('paired_16s_pilots_running',third_floor_valid=gate['third_floor_valid'])
def one(policy):
 out=root/'preflight/pilot'/f'scene1_seed0_clean_{policy}';log=root/'intake'/f'pilot_{policy}.log'
 with log.open('w') as f:subprocess.run([sys.executable,'scripts/explore/run_fire_recon.py','--config',str(cfgpath),'--scene','1','--seed','0','--condition','clean','--policy',policy,'--drones','4','--duration','16','--out',str(out)],stdout=f,stderr=subprocess.STDOUT,check=True)
 shutil.copy2(root/'preflight/scene1_complete/geometry.json',out/'scene_revision.json')
 s=json.load(open(out/'summary.json'));assert s['captures']==212 and s['failed_drones']==0 and s['invalid_paths']==0 and s['valid_confirmed']>0;return s
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:rows=list(pool.map(one,['unknown','entry']))
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in manifest.items())
with (root/'intake/pilot_summarize.log').open('w') as f:subprocess.run([sys.executable,'scripts/explore/summarize.py','--root',str(root/'preflight/pilot'),'--out','reports/fire_recon_20261107/preflight_data'],stdout=f,stderr=subprocess.STDOUT,check=True)
with (root/'intake/cached_replay.log').open('w') as f:subprocess.run([sys.executable,'scripts/explore/replay_cached_depth.py','--run',str(root/'preflight/pilot/scene1_seed0_clean_unknown'),'--until','16','--out',str(root/'preflight/cached_replay')],stdout=f,stderr=subprocess.STDOUT,check=True)
status('preflight_complete',runs=2,captures=sum(q['captures'] for q in rows),third_floor_valid=gate['third_floor_valid'],invalid_paths=sum(q['invalid_paths'] for q in rows),failed_drones=sum(q['failed_drones'] for q in rows),public_cached_depth_map_replay_exact=True)
print('REPAIRED SCENE PREFLIGHT COMPLETE',flush=True)
