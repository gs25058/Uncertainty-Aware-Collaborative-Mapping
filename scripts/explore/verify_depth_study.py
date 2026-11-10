#!/usr/bin/env python
"""Post-selection backend replay, privacy, timestamp, and source audit."""
import sys,json,csv,hashlib,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from covor.explore.scene import Building
from covor.explore.depth_observation import RayDepthSensor
from covor.explore.runner import write_json
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);p.add_argument('--expected',type=int,required=True);a=p.parse_args();root=Path(a.root);folders=sorted(q.parent for q in root.glob('scene*/summary.json'));assert len(folders)==a.expected
replays=[];graph_rows=0;pool_rows=0;total_frames=0;faults=0
for folder in folders:
 cc=json.load(open(folder/'config.json'));cfg=cc['configuration'];summary=json.load(open(folder/'summary.json'));frames=list(csv.DictReader(open(folder/'frames.csv')))
 assert summary['reachable_denominator']==4 and summary['blocked_denominator']==3
 assert len(frames)==summary['captures'];assert len(set(f['frame_id'] for f in frames))==len(frames);assert [int(f['map_version']) for f in frames]==list(range(1,len(frames)+1));assert not list((folder/'frames').glob('*.png'))
 sensor=RayDepthSensor(Building(cc['scene_seed']),cfg,cc['scene_seed']*1000+cc['run_seed'],cc['condition'])
 for f in (frames[0],frames[len(frames)//2]):
  q=np.load(folder/'frames'/(f['frame_id']+'.npz'));assert not set(q.files)&{'true_z','outlier','drop','transmission','sigma_true','normal_error'}
  z,sigma,v=sensor.measure(q['T'],float(q['time']),int(q['drone_id']),q['teammates']);assert np.array_equal(z,q['Z'],equal_nan=True);assert np.array_equal(sigma,q['sigma'],equal_nan=True);assert np.array_equal(v,q['valid']);assert np.isnan(z[~v]).all()
  replays.append(dict(run=folder.name,frame=f['frame_id'],exact_depth=True,exact_sigma=True,exact_valid=True))
 period=cfg['sensor_period']
 for f in frames:assert abs(float(f['time'])/period-round(float(f['time'])/period))<1e-8
 pool={}
 for line in open(folder/'candidates.jsonl'):
  q=json.loads(line);assert q['entry_graph_version']==q['map_version'];graph_rows+=1;pool.setdefault(q['time'],set()).add(q['id'])
 for e in csv.DictReader(open(folder/'events.csv')):
  if e['event']=='route_pool_all_candidates':assert set(json.loads(e['candidate_ids']))==pool[float(e['time'])];pool_rows+=1
 trajectory=list(csv.DictReader(open(folder/'trajectories.csv')))
 for drone in range(4):
  failed=[r for r in trajectory if int(r['drone'])==drone and r['failed']=='True']
  if failed:
   points=np.array([[float(r[k]) for k in ('x','y','z')] for r in failed]);assert np.allclose(points,points[0],atol=1e-12,rtol=0);faults+=1
 total_frames+=len(frames)
manifest=json.load(open(root/'source_manifest.json'));unchanged={p:hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in manifest.items()};assert all(unchanged.values()),[p for p,b in unchanged.items() if not b]
write_json(a.out,dict(runs=len(folders),actual_frames=total_frames,replays=replays,fresh_graph_candidate_rows=graph_rows,complete_route_pools=pool_rows,persistent_faults=faults,no_stereo_PNG=True,no_truth_fields_in_inputs=True,source_unchanged=unchanged))
print('PASS',len(folders),'runs,',len(replays),'exact observation replays,',total_frames,'actual inputs,',faults,'persistent faults')
