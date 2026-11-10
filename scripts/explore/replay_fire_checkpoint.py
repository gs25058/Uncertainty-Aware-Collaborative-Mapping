#!/usr/bin/env python3
"""Reintegrate a completed snapshot from public cached depth, even during a run.

Inputs: public frame NPZ files, prior, configuration, checkpoint. No mesh or
truth is read. Requires perfect odometry, explicitly checked in configuration.
"""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from covor.explore.depth_observation import camera_for
from covor.explore.sensing import SharedMap
from covor.explore.runner import write_json
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--until',type=float,default=20);p.add_argument('--out',required=True);a=p.parse_args();run=Path(a.run);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
if (out/'verification.json').exists():raise FileExistsError(out)
cfg=json.load(open(run/'config.json'))['configuration'];assert cfg['pose_mode']=='perfect_odometry','use frames.csv estimated poses for other modes';prior=json.load(open(run/'prior.json'));sd=camera_for(cfg);m=SharedMap(prior,cfg)
tick=int(round(a.until/cfg['dt']));source=run/'maps'/f't{tick:05d}.npz';assert source.exists();paths=[]
for f in (run/'frames').glob('*.npz'):
 d,t=f.stem.split('_t');t=int(t)
 if t<=tick:paths.append((t,int(d[1:]),f))
last=None
for t,d,f in sorted(paths):
 if last is not None and last!=t:m.dense()
 with np.load(f) as q:
  assert float(q['time'])==t*cfg['dt'];m.integrate(sd,q['T'],q['Z'],q['sigma'],q['valid'],q['teammates'])
 last=t
lab,layer=m.dense();checks={}
with np.load(source) as q:
 checks['version']=m.version==int(q['version']);checks['labels']=np.array_equal(lab,q['labels'])
 for k,v in layer.items():checks[k]=np.array_equal(v,q[k],equal_nan=True)
assert all(checks.values()),checks
write_json(out/'verification.json',dict(source_run=str(run),checkpoint=str(source),actual_frames=m.version,exact=checks,no_ground_truth_or_mesh_read=True,no_observations_generated=True,estimated_poses_equal_public_T_per_perfect_odometry=True));print(m.version,'cached frames, bit-exact checkpoint PASS',flush=True)
