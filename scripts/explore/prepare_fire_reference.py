#!/usr/bin/env python3
import sys,json,argparse,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from covor.explore.fire_world import FireWorld
from covor.explore.runner import write_json,csv_write
from covor.entry.path_eval import ReferenceValidator,PathEvalCfg,motion_path
p=argparse.ArgumentParser();p.add_argument('--config',default='configs/fire_closed_loop_20261107.json');p.add_argument('--scene',type=int,required=True);p.add_argument('--out',required=True);a=p.parse_args();c=json.load(open(a.config));out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
t=time.monotonic();b=FireWorld(a.scene,c);truth,nodes,modes=b.reference(c['map_resolution']);lab,ijk=b.labels(c['map_resolution']);b.render_scene(False)
validator=ReferenceValidator(lab,ijk,c['map_resolution'],b.scene,cfg=PathEvalCfg(step_limit_m=.18,contact_epsilon_m=.001));rows=[]
for ref,goal in zip(truth,b.goals):
 v=validator.validate(motion_path(ref['path']),b.entry,goal) if ref['reachable'] else dict(status='not_proposed',reason='no_path_in_independent_mesh_roadmap')
 rows.append(dict(goal=ref['id'],reference_reachable=ref['reachable'],**v))
 if ref['reachable']:assert v['status']=='valid',(ref['id'],v)
write_json(out/'truth.json',truth);write_json(out/'prior.json',b.prior());csv_write(out/'reference_path_validation.csv',rows)
summary=dict(scene=a.scene,id=b.id,reachable=sum(r['reachable'] for r in truth),blocked=sum(not r['reachable'] for r in truth),total=len(truth),valid_nodes=len(nodes),wall_seconds=time.monotonic()-t,independent_paths_validated=True)
write_json(out/'summary.json',summary);print(summary,flush=True)
