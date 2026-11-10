#!/usr/bin/env python3
"""Map-only candidate capacity audit: same positions, existing vs aimed yaw.

No sensor frames/meshes/reference labels are opened. This does not change any
running trial and is not an oracle of actual measurement or task success.
"""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from covor.explore.entry_objective import TaskContext
from covor.explore.human import Human
from covor.explore.sensing import camera_pose
from covor.explore.depth_observation import camera_for
from covor.explore.runner import write_json
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--config',required=True);p.add_argument('--time',type=float,default=20.);p.add_argument('--out',required=True);a=p.parse_args();run=Path(a.run);cfg=json.load(open(a.config));prior=json.load(open(run/'prior.json'))
tick=round(a.time/cfg['dt'])
with np.load(run/'maps'/f't{tick:05d}.npz') as q:lab=q['labels'];ijk=q['ijk'];layer={k:q[k] for k in q.files if k not in ('labels','ijk','res','version')}
analysis=Human(prior,cfg).analyze(lab,layer,ijk,False);context=TaskContext(analysis,lab,layer,ijk,cfg['map_resolution'],cfg,prior);sd=camera_for(cfg)
pool=[json.loads(x) for x in open(run/'candidates.jsonl') if json.loads(x)['time']==a.time];rows=[]
for query in context.queries:
 if query.kind not in ('boundary','entry'):continue
 known=[];aimed=[];target=query.body.mean(0)
 for c in pool:
  pos=np.asarray(c['position']);yaw=float(np.arctan2(target[1]-pos[1],target[0]-pos[0]))
  for heading,output in [(c['yaw'],known),(yaw,aimed)]:
   views=[camera_pose(pos,heading,pitch,cfg['sensor']['baseline']) for pitch in cfg['candidates']['pitches']]
   value,detail=context.observation_value(query,views,sd);output.append(dict(id=c['id'],yaw=heading,value=value,**detail))
 rows.append(dict(query=query.key,feet_bounds=[query.feet.min(0).tolist(),query.feet.max(0).tolist()],unknown_body_bounds=[query.body.min(0).tolist(),query.body.max(0).tolist()],goals=query.goals,existing=max(known,key=lambda q:q['value']),aimed=max(aimed,key=lambda q:q['value'])))
write_json(a.out,dict(time=a.time,map_only=True,unchanged_positions=True,sensor_observations_read=False,queries=rows))
for q in rows:print(q['query'],q['existing']['value'],q['aimed']['value'],flush=True)
