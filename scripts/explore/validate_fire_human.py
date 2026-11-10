#!/usr/bin/env python3
import sys,argparse,json,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from covor.explore.fire_world import FireWorld
from covor.explore.human import Human,complete_support
from covor.explore.runner import evaluate,csv_write,write_json
from covor.entry.path_eval import ReferenceValidator,PathEvalCfg
p=argparse.ArgumentParser();p.add_argument('--config',default='configs/fire_closed_loop_20261107.json');p.add_argument('--scene',type=int,required=True);p.add_argument('--out',required=True);a=p.parse_args();c=json.load(open(a.config));out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
start=time.monotonic();b=FireWorld(a.scene,c);truth,_,_=b.reference(c['map_resolution']);lab,ijk=b.labels(c['map_resolution']);b.render_scene(False);layer=complete_support(b,ijk,lab.shape,c['map_resolution'])
h=Human(b.prior(),c);analysis=h.analyze(lab,layer,ijk,False);v=ReferenceValidator(lab,ijk,c['map_resolution'],b.scene,cfg=PathEvalCfg(step_limit_m=.18,contact_epsilon_m=.001));rows,summary=evaluate(analysis,truth,v,b.prior(),0,0)
csv_write(out/'targets.csv',rows);write_json(out/'summary.json',dict(**summary,wall_seconds=time.monotonic()-start));print(summary,flush=True)
fail=[];g=analysis['graph']
for ref,row in zip(truth,rows):
    if not ref['reachable'] or row['path_status']=='valid':continue
    first=None;previous=None
    for p in ref['path']:
        x,y,z=p['xyz'];q=np.floor(np.array([x,y,z-.025])/c['map_resolution']).astype(int)-ijk
        if not ((q>=0)&(q<lab.shape)).all():first=dict(stage='outside_map',xyz=p['xyz']);break
        i,j,k=q;ks=np.flatnonzero(layer['support'][i,j]);nearest=min(ks,key=lambda kk:abs(layer['height'][i,j,kk]-(z-.025))) if len(ks) else None
        if nearest is None:first=dict(stage='support',xyz=p['xyz']);break
        q=(i,j,nearest);node=int(g.ids[q])
        if not g.confirmed[q]:first=dict(stage='body',xyz=p['xyz'],voxel=q,foot_z=float(g.foot_z[q]));break
        if previous is not None and node not in [v for v,_,_ in g.edges(previous,True)]:first=dict(stage='connection',xyz=p['xyz'],node=node,previous=previous);break
        previous=node
    fail.append(dict(goal=ref['id'],verdict=row['verdict'],first_break=first,path_status=row['path_status'],path_reason=row['path_reason']))
write_json(out/'first_breaks.json',fail)
assert summary['invalid_paths']==0,'complete-observation proposed an invalid path'
assert summary['valid_confirmed']>0,'even complete observation yields no valid routes'
