#!/usr/bin/env python
"""Reintegrate cached direct-depth observations, without mesh/SGBM/GT."""
import sys,argparse,csv,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from covor.explore.depth_observation import camera_for
from covor.explore.sensing import SharedMap
from covor.explore.runner import write_json
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--out',required=True);p.add_argument('--until',type=float,default=48.);a=p.parse_args();root=Path(a.run);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
if (out/'verification.json').exists():raise FileExistsError('use a new output directory')
cfg=json.load(open(root/'config.json'))['configuration'];prior=json.load(open(root/'prior.json'));sd=camera_for(cfg);m=SharedMap(prior,cfg);poses={}
for r in csv.DictReader(open(root/'trajectories.csv')):poses.setdefault(float(r['time']),{})[int(r['drone'])]=np.array([float(r[k]) for k in ('estimated_x','estimated_y','estimated_z')])
last=None
for f in csv.DictReader(open(root/'frames.csv')):
 t=float(f['time'])
 if t>a.until:break
 if last is not None and last!=t:m.dense()
 q=np.load(root/'frames'/(f['frame_id']+'.npz'));mates=[p for i,p in sorted(poses[t].items()) if i!=int(f['drone'])];m.integrate(sd,np.array(json.loads(f['camera_pose'])),q['Z'],q['sigma'],q['valid'],mates);assert m.version==int(f['map_version']);last=t
lab,layer=m.dense();m.save(out/'replayed.npz',lab,layer);name='final.npz' if a.until==cfg['duration'] else f't{int(round(a.until/cfg["dt"])):05d}.npz';original=np.load(root/'maps'/name);checks=dict(labels=np.array_equal(lab,original['labels']))
for k,v in layer.items():checks[k]=np.array_equal(v,original[k],equal_nan=True)
assert all(checks.values()),checks
write_json(out/'verification.json',dict(source=str(root),until=a.until,actual_frames=m.version,exact=checks,no_ground_truth_or_stereo_read=True));print(m.version,'frames, exact map/support/height/votes/evidence PASS')
