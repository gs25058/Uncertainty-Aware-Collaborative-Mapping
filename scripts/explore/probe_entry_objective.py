#!/usr/bin/env python3
"""Map-only utility probe: never opens depth frames, references, or scene meshes."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from covor.explore.human import Human
from covor.explore.policy import score_pool
from covor.explore.depth_observation import camera_for
from covor.explore.runner import write_json
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--config',required=True);p.add_argument('--checkpoint',default='t00080.npz');p.add_argument('--time',type=float,default=20.);p.add_argument('--out',required=True);a=p.parse_args();run=Path(a.run);cfg=json.load(open(a.config));prior=json.load(open(run/'prior.json'))
with np.load(run/'maps'/a.checkpoint) as q:lab=q['labels'];ijk=q['ijk'];layer={k:q[k] for k in q.files if k not in ('labels','ijk','res','version')}
pool=[]
for line in open(run/'candidates.jsonl'):
 q=json.loads(line)
 if q['time']==a.time:pool.append({k:q[k] for k in ('id','position','yaw','pitch','kind')})
start=time.monotonic();analysis=Human(prior,cfg).analyze(lab,layer,ijk,True);human_wall=time.monotonic()-start
start=time.monotonic();items=score_pool(pool,analysis,lab,layer,ijk,cfg['map_resolution'],camera_for(cfg),cfg,prior);wall=time.monotonic()-start
out=dict(map_only_inputs=True,queries=items[0]['objective_diagnostics']['queries'] if items else 0,human_wall=human_wall,objective_wall=wall,candidates=[dict(id=q['candidate']['id'],position=q['candidate']['position'],unknown=len(q['unknown']),score=sum(q['targets'].values()),goals=q['targets'],diagnostics=q.get('objective_diagnostics')) for q in items]);write_json(a.out,out);print('queries',out['queries'],'human_s',human_wall,'objective_s',wall,'positive_candidates',sum(q['score']>0 for q in out['candidates']),flush=True)
