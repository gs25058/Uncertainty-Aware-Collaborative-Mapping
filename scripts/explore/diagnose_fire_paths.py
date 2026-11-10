#!/usr/bin/env python3
"""Post-run reference-route probe. Evaluation only; no planning feedback.

A failure on this particular independent reference route is diagnostic, not a
proof that every alternative route is impossible. No truth is fused into maps.
"""
import argparse,csv,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from covor.explore.human import Human
from covor.explore.runner import csv_write,write_json
csv.field_size_limit(sys.maxsize)
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--out',required=True);p.add_argument('--checkpoint',default='final.npz');a=p.parse_args();run=Path(a.run);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
cc=json.load(open(run/'config.json'));cfg=cc['configuration'];prior=json.load(open(run/'prior.json'));truth=json.load(open(run/'reference_targets.json'))
with np.load(run/'maps'/a.checkpoint) as q:lab=q['labels'];ijk=q['ijk'];layer={k:q[k] for k in q.files if k not in ('labels','ijk','res','version')}
analysis=Human(prior,cfg).analyze(lab,layer,ijk,False);g=analysis['graph'];rows=[];off=g.map_offset
for ref,d in zip(truth,analysis['decisions']):
 if not ref['reachable'] or d['verdict']=='passable':continue
 first=None;previous=None;reached=0
 for index,p in enumerate(ref['path']):
  xyz=np.array(p['xyz']);xy=np.floor(xyz[:2]/g.res).astype(int)-ijk[:2];first=dict(reference_path_index=index,xyz=xyz.tolist());reached=index
  if (xy<0).any() or (xy>=lab.shape[:2]).any():first.update(stage='outside_map');break
  col=tuple(xy);finite=np.flatnonzero(np.isfinite(layer['height'][col]) & (abs(layer['height'][col]+.025-xyz[2])<=.075))
  if not len(finite):first.update(stage='support_evidence',reason='no_observed_height_near_reference_foot');break
  k=min(finite,key=lambda kk:abs(layer['height'][col][kk]+.025-xyz[2]));q=(*xy,int(k));first.update(votes=int(layer['votes'][q]),evidence=float(layer['evidence'][q]),height_std=float(layer['height_std'][q]),measured_foot_z=float(layer['height'][q]+.025),max_occupied_hit=float(layer['hit_z_max'][q]))
  if not layer['support'][q]:first.update(stage='support_evidence',reason='insufficient_votes_evidence_or_height_stability');break
  local=np.array(q)-off
  if (local<0).any() or (local>=g.lab.shape).any():first.update(stage='outside_graph_roi');break
  local=tuple(local);node=int(g.ids[local]);first['confirmed_posture']=int(g.cp[local]);first['possible_posture']=int(g.pp[local])
  if not g.confirmed[local]:
   upper=max(float(layer['height'][q]+3*layer['height_std'][q]+.005),float(layer['hit_z_max'][q]+.005) if lab[q]!=1 else -np.inf)
   first.update(stage='foot_contact_or_body',reason='foot_contact_upper_bound' if upper>float(g.foot_z[local])+1e-9 else 'observed_obstacle_or_unknown_body_volume',height_upper=upper)
   break
  if previous is not None and previous!=node and node not in [v for v,_,_ in g.edges(previous,True)]:first.update(stage='next_support_connection',reason='step_corner_or_current_representative_xy_connection');break
  previous=node;first=None
 rows.append(dict(run=run.name,goal=ref['id'],system_verdict=d['verdict'],reference_route_nodes=len(ref['path']),reference_nodes_examined=reached+1,first_failure=first,diagnostic_scope='this_reference_route_only_not_all_routes'))
csv_write(out/'reference_route_first_break.csv',rows);write_json(out/'summary.json',dict(run=run.name,reachable_but_not_passable=len(rows),stages={s:sum(r['first_failure'] is not None and r['first_failure']['stage']==s for r in rows) for s in sorted({r['first_failure']['stage'] for r in rows if r['first_failure']})},uses_truth_for_postrun_diagnosis_only=True));print(out,flush=True)
