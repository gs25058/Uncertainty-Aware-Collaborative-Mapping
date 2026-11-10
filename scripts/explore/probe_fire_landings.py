#!/usr/bin/env python3
"""Causal geometry control, separate from the frozen policy comparison.

Add only the missing bottom landings in an in-memory copy. Do not edit original
assets, current FireBuilding, or the frozen 40-run scene/reference inputs.
"""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from covor.explore.fire_world import FireWorld
from covor.explore.fire_reference import MeshHumanReference
from covor.entry.path_eval import ReferenceValidator,PathEvalCfg,motion_path
from covor.explore.runner import write_json,csv_write
p=argparse.ArgumentParser();p.add_argument('--scene',type=int,default=1);p.add_argument('--config',default='configs/fire_closed_loop_20261107.json');p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
if (out/'summary.json').exists():raise FileExistsError(out)
start=time.monotonic();cfg=json.load(open(a.config));b=FireWorld(a.scene,cfg);old,_,_=b.reference(cfg['map_resolution']);added=[]
for s in b.stairs:
 if s['floor']==0:continue
 x,y,f=s['x'],s['y'],s['floor'];center=[x+.625,y-.29,f*b.storey-.10];size=[1.25,.58,.20]
 b.box('diagnostic_missing_bottom_landing',center,size,f,'stair','stair');added.append(dict(center=center,size=size,floor=f))
b.scene=None;truth,_,_=MeshHumanReference(b,cfg['map_resolution']).solve();b.render_scene(False);lab,ijk=b.labels(cfg['map_resolution']);validator=ReferenceValidator(lab,ijk,cfg['map_resolution'],b.scene,cfg=PathEvalCfg(step_limit_m=.18,contact_epsilon_m=.001));rows=[]
for before,after,goal in zip(old,truth,b.goals):
 result=validator.validate(motion_path(after['path']),b.entry,goal) if after['reachable'] else dict(status='not_proposed')
 rows.append(dict(goal=goal['id'],before_reachable=before['reachable'],after_reachable=after['reachable'],**result))
 if after['reachable']:assert result['status']=='valid',(goal['id'],result)
write_json(out/'reference_targets.json',truth);write_json(out/'added_geometry.json',added);csv_write(out/'comparison.csv',rows);summary=dict(scene=a.scene,before_reachable=sum(q['reachable'] for q in old),after_reachable=sum(q['reachable'] for q in truth),before_third_floor_reachable=sum(q['reachable'] and q['id'].startswith('L3_') for q in old),after_third_floor_reachable=sum(q['reachable'] and q['id'].startswith('L3_') for q in truth),added_landings=len(added),all_positive_paths_independently_mesh_validated=True,original_geometry_and_running_experiments_unchanged=True,wall_seconds=time.monotonic()-start);write_json(out/'summary.json',summary);print(summary,flush=True)
