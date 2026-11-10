#!/usr/bin/env python3
"""Post-hoc first-edge diagnosis on a fixed development reference route.

Never changes maps, decisions, utility, or the route supplied by the system.
"""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from covor.explore.human import Human
from covor.explore.runner import write_json
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--out',required=True);a=p.parse_args();run=Path(a.run)
cfg=json.load(open(run/'config.json'))['configuration'];prior=json.load(open(run/'prior.json'))
with np.load(run/'maps/final.npz') as q:lab=q['labels'];ijk=q['ijk'];layer={k:q[k] for k in q.files if k not in ('labels','ijk','res','version')}
analysis=Human(prior,cfg).analyze(lab,layer,ijk,False);g=analysis['graph'];off=g.map_offset;ref=json.load(open(run/'reference_targets.json'));results=[]
def info(u):
 q=tuple(g.xyz[u]);full=tuple(g.xyz[u]+off)
 return dict(id=int(u),ijk=(g.xyz[u]+g.ijk).tolist(),xy=g.node_xy[u].tolist(),foot=float(g.node_height[u]),confirmed=bool(g.node_known[u]),reachable_from_entry=int(u) in analysis['confirmed_search']['best'],possible=bool(g.possible[q]),posture=int(g.pp[q]),votes=int(layer['votes'][full]),std=float(layer['height_std'][full]),hit_z_max=float(layer['hit_z_max'][full]),region=int(g.node_region[u]))
for reference,decision in zip(ref,analysis['decisions']):
 if not reference['reachable'] or decision['verdict']=='passable':continue
 previous=None
 for index,p in enumerate(reference['path']):
  xyz=np.array(p['xyz']);xy=np.floor(xyz[:2]/g.res).astype(int)-ijk[:2]
  if (xy<0).any() or (xy>=lab.shape[:2]).any():break
  finite=np.flatnonzero(np.isfinite(layer['height'][tuple(xy)])&(abs(layer['height'][tuple(xy)]+.025-xyz[2])<=.075))
  if not len(finite):break
  k=min(finite,key=lambda kk:abs(layer['height'][tuple(xy)][kk]+.025-xyz[2]));local=np.r_[xy,k]-off
  if (local<0).any() or (local>=g.lab.shape).any():break
  u=int(g.ids[tuple(local)])
  if u<0 or not g.node_known[u]:break
  if previous is not None and previous!=u and u not in [v for v,_,_ in g.edges(previous,True)]:
   before=g.xyz[previous];after=g.xyz[u];st=int(np.ceil(g.body.step/g.res));corners=[]
   if before[0]!=after[0] and before[1]!=after[1]:
    for i,j in ((before[0],after[1]),(after[0],before[1])):
     ids=g.ids[i,j,max(0,after[2]-st):min(g.lab.shape[2],after[2]+st+1)];ids=ids[ids>=0]
     corners.append([dict(**info(v),step_ok=abs(g.node_height[v]-g.node_height[previous])<=g.body.step+1e-9 and abs(g.node_height[v]-g.node_height[u])<=g.body.step+1e-9,below_endpoint_foot=bool(g.node_height[v]<=max(g.node_height[previous],g.node_height[u])+1e-9)) for v in ids])
   results.append(dict(goal=reference['id'],reference_index=index,before=info(previous),after=info(u),raw_neighbor=u in [v for v,_,_ in g._raw_neighbors(previous)],possible_edge=u in [v for v,_,_ in g.edges(previous)],corners=corners));break
  previous=u
write_json(a.out,dict(run=run.name,scope='post-hoc reference-route edge only; endpoint reachability includes actual alternate paths; not a proof about every route',edges=results));print('diagnosed',len(results),'first edges',flush=True)
