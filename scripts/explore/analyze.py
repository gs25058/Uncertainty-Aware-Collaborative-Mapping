#!/usr/bin/env python
"""Post-selection audit only: raw snapshots are never changed."""
import sys,argparse,json,csv,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from covor.explore.scene import Building
from covor.explore.human import Human
from covor.explore.runner import csv_write,write_json
from covor.entry import traverse as TV
from scipy import ndimage


def read_csv(p):return list(csv.DictReader(open(p)))
def lines(p):return [json.loads(s) for s in open(p) if s.strip()]

def body_counts(g,node,gt):
 i,j,k=node;z=g.foot_z[i,j,k];rows=[]
 for posture,height in enumerate((g.body.h_stand,g.body.h_stoop,g.body.h_crawl)):
  for width,radius in enumerate((g.body.w/2,g.body.w_side/2)):
   r=int(np.ceil(radius/g.res+.5));top=int(np.ceil((z+height+.02)/g.res-1e-9))-g.ijk[2];low=max(k+1,int(np.floor((z+g.body.step)/g.res+1e-9))-g.ijk[2]);cells=set((i,j,kk) for kk in range(k+1,min(top,g.lab.shape[2])))
   for di in range(-r,r+1):
    for dj in range(-r,r+1):
     if np.hypot(max(abs(di)*g.res-g.res/2,0),max(abs(dj)*g.res-g.res/2,0))<=radius+1e-9:
      for kk in range(low,min(top,g.lab.shape[2])):cells.add((i+di,j+dj,kk))
   q=np.array(sorted(cells));inside=((q>=0)&(q<g.lab.shape)).all(1);v=np.zeros(len(q),np.uint8);v[inside]=g.lab[tuple(q[inside].T)];ref=np.zeros(len(q),np.uint8);ref[inside]=gt[tuple(q[inside].T)]
   rows.append(dict(posture=posture,width=width,n_unknown=int((v==0).sum()),n_occupied=int((v==2).sum()),n_false_occupied_reference_free=int(((v==2)&(ref==1)).sum()),n_actual_obstacle=int(((v==2)&(ref==2)).sum())))
 return rows


def analyze(root,out):
 root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=True);rows=[];diagnoses=[];legacy=[];temporal=[];candidate_audit=[];first_differences=[];truth_cache={}
 for folder in sorted(root.glob('scene*')):
  if not (folder/'summary.json').exists():continue
  s=json.load(open(folder/'summary.json'));cfg=json.load(open(folder/'config.json'))['configuration'];name=folder.name;prior=json.load(open(folder/'prior.json'));data=np.load(folder/'maps/final.npz');lab=data['labels'];ijk=data['ijk'];res=float(data['res']);layer={k:data[k] for k in ('support','height','votes','evidence')}
  b=Building(s['scene_seed']);gt,_=b.labels(res);a=Human(prior,cfg).analyze(lab,layer,ijk,False);g=a['graph']
  assignments=lines(folder/'assignments.jsonl');frames=read_csv(folder/'frames.csv');trajectory=read_csv(folder/'trajectories.csv');events=read_csv(folder/'events.csv');decisions=lines(folder/'decisions.jsonl');pool=lines(folder/'candidates.jsonl')
  per_drone=[]
  for i in range(s['n_drones']):
   p=[r for r in trajectory if int(r['drone'])==i]
   per_drone.append(dict(drone=i,start=[float(p[0][k]) for k in ('x','y','z')],end=[float(p[-1][k]) for k in ('x','y','z')],distance=sum(np.linalg.norm(np.array([float(a[k])-float(b[k]) for k in ('x','y','z')])) for a,b in zip(p,p[1:])),frames=sum(int(r['drone'])==i for r in frames)))
  wait=sum(min(max(s['duration']-r['time'],0),r['trajectory']['wait']) for r in assignments)
  trials=sum(bool(c['entry_target_scores']) and max(c['entry_target_scores'].values())>0 for c in pool)
  stats=dict(run=name,**s,actual_reservation_wait=wait,mean_valid_depth_ratio=float(np.mean([float(f['valid_ratio']) for f in frames])),total_support_points=sum(int(f['support_points']) for f in frames),surface_confirmed=int(layer['support'].sum()),map_occupied=int((lab==2).sum()),map_free=int((lab==1).sum()),map_unknown=int((lab==0).sum()),n_assignment_actions=len(assignments),positive_entry_candidate_rows=trials,entry_fallback_fraction=s['fallback_seconds']/s['duration'] if s['policy']=='entry' else None,stops=sum(e['event']=='stopped_at_rest_for_replan' for e in events),sensor_wall=sum(float(f['sensor_wall']) for f in frames),mapping_wall=sum(float(f['mapping_wall']) for f in frames))
  rows.append(stats);write_json(out/(name+'_drones.json'),per_drone)
  # First broken contract at the fixed entry: evidence -> height -> posture/body.
  x0,y0,x1,y1=prior['entry']['xy_bounds'];lo,hi=prior['entry']['floor_z_bounds'];x=(np.arange(lab.shape[0])+ijk[0]+.5)*res;y=(np.arange(lab.shape[1])+ijk[1]+.5)*res
  valid_xy=(x[:,None]>=x0)&(x[:,None]<=x1)&(y[None,:]>=y0)&(y[None,:]<=y1)
  observed=np.argwhere((layer['votes']>0)&valid_xy[:,:,None]&np.isfinite(layer['height'])&(layer['height']>=lo)&(layer['height']<=hi))
  for node in observed:
   q=tuple(node);details=body_counts(g,node,gt)
   reason='confirmed_entry' if g.confirmed[q] else ('insufficient_surface_votes_or_consistency' if not g.support[q] else 'body_space_occupied_or_unknown')
   diagnoses.append(dict(run=name,location='entry',voxel=(node+ijk).tolist(),surface_height=float(layer['height'][q]),foot_height=float(g.foot_z[q]),reference_surface_error=float(layer['height'][q]-.05),votes=int(layer['votes'][q]),evidence=float(layer['evidence'][q]),confirmed=bool(g.confirmed[q]),first_break=reason,modes=details))
  if not len(observed):diagnoses.append(dict(run=name,location='entry',first_break='no_observed_support_at_fixed_entry'))
  for t in decisions:
   for d in t['decisions']:
    candidate_audit.append(dict(run=name,time=t['time'],goal=d['goal'],verdict=d['verdict'],possible_route=bool(d['possible_path']),dependent_segments=len(d['uncertain_segments']),positive_segments=sum(d['goal'] in v for v in t['impact'].values())))
  # Archived area definitions, including snap/gap rules, are run unmodified and
  # saved separately. These do NOT supply target truth or replace proposed paths.
  xy=[(x0+x1)/2,(y0+y1)/2]
  if s['scene_seed'] not in truth_cache:truth_cache[s['scene_seed']]=TV.traverse(gt,TV.TravCfg(res=res),xy,ijk)
  tm=TV.traverse(lab,TV.TravCfg(res=res),xy,ijk);area=TV.score(tm,truth_cache[s['scene_seed']]);legacy.append(dict(run=name,**area,legacy_gap_m=.5,legacy_snap_limit_m=1.5))
  by={}
  for r in read_csv(folder/'target_paths.csv'):by.setdefault(r['goal'],[]).append(r)
  for goal,seq in by.items():
   for old,new in zip(seq,seq[1:]):
    temporal.append(dict(run=name,time=float(new['time']),goal=goal,unknown_to_valid=old['verdict']=='unverified' and new['path_status']=='valid',unknown_to_correct_blocked=old['verdict']=='unverified' and new['verdict']=='blocked' and new['reference_reachable']=='False',new_invalid=new['path_status']=='invalid' and old['path_status']!='invalid',revoked_passable=old['verdict']=='passable' and new['verdict']!='passable',revoked_invalid=old['path_status']=='invalid' and new['verdict']!='passable',lost_valid=old['path_status']=='valid' and new['path_status']!='valid'))
 # Paired assignment comparisons. Different maps after divergence are expected.
 for a in rows:
  if a['policy']!='unknown':continue
  folder=root/a['run'];other=root/a['run'].replace('_unknown','_entry')
  if not (other/'summary.json').exists():continue
  aa={(r['time'],r['drone']):r for r in lines(folder/'assignments.jsonl')};bb={(r['time'],r['drone']):r for r in lines(other/'assignments.jsonl')}
  keys=sorted(set(aa)|set(bb));diff=[]
  for key in keys:
   ca=aa.get(key,{}).get('candidate');cb=bb.get(key,{}).get('candidate')
   if ca!=cb:diff.append(dict(time=key[0],drone=key[1],unknown_candidate=ca,entry_candidate=cb))
  first_differences.append(dict(scene_seed=a['scene_seed'],run_seed=a['run_seed'],condition=a['condition'],different_assignment_events=len(diff),first_difference=diff[0] if diff else None))
 csv_write(out/'run_metrics.csv',rows);csv_write(out/'entry_failure_diagnosis.csv',diagnoses);csv_write(out/'legacy_area_metrics.csv',legacy);csv_write(out/'temporal_transitions.csv',temporal);csv_write(out/'decision_signal.csv',candidate_audit);write_json(out/'paired_assignment_changes.json',first_differences)
 grouped=[]
 for condition in sorted({r['condition'] for r in rows}):
  for policy in ('unknown','entry'):
   group=[r for r in rows if r['condition']==condition and r['policy']==policy]
   if not group:continue
   n=len(group);V=sum(r['valid_confirmed'] for r in group);R=sum(r['reachable_denominator'] for r in group);C=sum(r['correct_blocked'] for r in group);B=sum(r['blocked_denominator'] for r in group);I=sum(r['invalid_paths'] for r in group);P=sum(r['verified_path_denominator'] for r in group)
   grouped.append(dict(condition=condition,policy=policy,runs=n,valid_paths=V,reachable_targets=R,valid_rate=V/R,correct_blocked=C,blocked_targets=B,blocked_rate=C/B,balanced_score=.5*(V/R+C/B),invalid_paths=I,checked_proposals=P,wrong_path_rate=I/P if P else None,all_proposals=sum(r['proposed_paths'] for r in group),unverifiable=sum(r['unverifiable_paths'] for r in group),unknown=sum(r['unknown'] for r in group),decided=sum(r['decided'] for r in group),all_targets=sum(r['total_goals'] for r in group),mean_distance=float(np.mean([r['travel_distance'] for r in group])),captures=sum(r['captures'] for r in group),failed_drones=sum(r['failed_drones'] for r in group),fallback_seconds=sum(r['fallback_seconds'] for r in group),total_sim_seconds=sum(r['duration'] for r in group),valid_depth_ratio=float(np.mean([r['mean_valid_depth_ratio'] for r in group]))))
 csv_write(out/'method_aggregate.csv',grouped);print(json.dumps(grouped,indent=2),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);a=p.parse_args();analyze(a.root,a.out)
