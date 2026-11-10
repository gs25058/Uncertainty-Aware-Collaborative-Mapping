#!/usr/bin/env python3
"""Post-hoc floor-stratified goal/route and actual observation coverage.

Goals/denominators are taken from the pre-fixed cases. Drone altitude is never
used to declare a human route valid. No data is fed back into the policy.
"""
import argparse,csv,json,sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from covor.explore.runner import csv_write
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--report',required=True);a=p.parse_args();root=Path(a.root);out=Path(a.report)/'data';cfg=json.load(open(root/'main/fixed_config.json'));cases={q['scene']:q for q in json.load(open(root/'evaluation_cases.json'))};read=lambda path:list(csv.DictReader(open(path)));paths=read(out/'paths_final_compact.csv');summary=read(out/'runs_final.csv');groups={};flight=[]
for s in summary:
 run=s['run'];i=int(s['scene_seed']);c=s['condition'];policy=s['policy'];case=cases[i];prior=case['prior'];lo,hi=prior['exploration_box'];folder=root/'main'/run;traj=read(folder/'trajectories.csv');lookup={(float(q['time']),int(q['drone'])):q for q in traj};audit={q['frame_id']:int(q['n_valid']) for q in read(folder/'simulator_audit.csv')};captures=defaultdict(int);valid_captures=defaultdict(int);active=defaultdict(int);drones=defaultdict(set);first={}
 def floor_of(q):
  x,y,z=[float(q[k]) for k in ('x','y','z')]
  if not (lo[0]<=x<=hi[0] and lo[1]<=y<=hi[1] and 0<=z<=hi[2]):return None
  return int(z//3.3)+1
 for q in traj:
  f=floor_of(q);t=float(q['time']);d=int(q['drone'])
  if f is not None and q['active']=='True' and q['failed']=='False':
   drones[f].add(d);first[f]=min(first.get(f,t),t)
   if t<float(s['duration']):active[f]+=1
 for q in read(folder/'frames.csv'):
  f=floor_of(lookup[float(q['time']),int(q['drone'])])
  if f is not None:captures[f]+=1;valid_captures[f]+=audit[q['frame_id']]>0
 local=[q for q in paths if q['run']==run]
 for f in sorted({int(t['id'].split('_')[0][1:]) for t in case['reference_targets']}):
  refs=[t for t in case['reference_targets'] if t['id'].startswith(f'L{f}_')];rows=[q for q in local if int(q['floor'])==f];R=sum(t['reachable'] for t in refs);B=len(refs)-R;V=sum(q['path_status']=='valid' and q['reference_reachable']=='True' for q in rows);C=sum(q['verdict']=='blocked' and q['reference_reachable']=='False' for q in rows);I=sum(q['path_status']=='invalid' for q in rows);P=sum(q['path_status'] in ('valid','invalid') for q in rows)
  key=(c,policy,f);g=groups.setdefault(key,dict(condition=c,policy=policy,floor=f,reachable=0,blocked=0,total=0,valid_routes=0,correct_blocked=0,invalid_routes=0,checked_routes=0,unknown=0,captures=0,captures_with_valid_depth=0,active_drone_seconds=0.,runs_visiting_floor=0))
  for k,v in dict(reachable=R,blocked=B,total=len(refs),valid_routes=V,correct_blocked=C,invalid_routes=I,checked_routes=P,unknown=sum(q['verdict']=='unverified' for q in rows),captures=captures[f],captures_with_valid_depth=valid_captures[f],active_drone_seconds=active[f]*cfg['dt'],runs_visiting_floor=bool(drones[f])).items():g[k]+=v
  flight.append(dict(run=run,scene=i,condition=c,policy=policy,floor=f,first_active_inside_time=first.get(f,''),visiting_drones=sorted(drones[f]),captures=captures[f],captures_with_valid_depth=valid_captures[f],active_drone_seconds=active[f]*cfg['dt'],valid_goal_routes=V,reference_reachable=R,valid_paths_in_run_touching_upper_floor=sum(q['path_status']=='valid' and q['path_max_z'] and float(q['path_max_z'])>=3.3 for q in local)))
rows=[]
for key,g in sorted(groups.items()):
 g['valid_confirmation_rate']=g['valid_routes']/g['reachable'] if g['reachable'] else 'N/A';g['correct_blocked_rate']=g['correct_blocked']/g['blocked'] if g['blocked'] else 'N/A';g['wrong_route_ratio']=g['invalid_routes']/g['checked_routes'] if g['checked_routes'] else 'N/A';rows.append(g)
csv_write(out/'floor_goal_routes.csv',rows);csv_write(out/'floor_observation_coverage.csv',flight);print('floor groups',len(rows),'per-run floor records',len(flight),flush=True)
