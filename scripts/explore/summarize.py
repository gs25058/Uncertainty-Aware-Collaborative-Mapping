#!/usr/bin/env python
"""Final goal-level denominators, paired deltas and deterministic replay lists."""
import sys,csv,json,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from covor.explore.runner import write_json,csv_write

def table(root,out):
 root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=True);runs=[];details=[];times=[];operations=[];temporal=[]
 for path in sorted(root.glob('scene*/summary.json')):
  folder=path.parent;r=json.load(open(path));r['run']=folder.name;runs.append(r);events=list(csv.DictReader(open(folder/'events.csv')));assign=[json.loads(l) for l in open(folder/'assignments.jsonl')];frames=list(csv.DictReader(open(folder/'frames.csv')));cfg=json.load(open(folder/'config.json'))['configuration']
  by={}
  for row in csv.DictReader(open(folder/'target_paths.csv')):
   by.setdefault(row['goal'],[]).append(row)
   if float(row['time'])==r['duration']:details.append(dict(run=folder.name,scene_seed=r['scene_seed'],seed=r['run_seed'],condition=r['condition'],policy=r['policy'],captures=r['captures'],travel_distance=r['travel_distance'],**row))
  for goal,seq in by.items():
   seq=sorted(seq,key=lambda q:float(q['time']));first=seq[0];last=seq[-1];ever_valid=any(q['path_status']=='valid' for q in seq);ever_invalid=any(q['path_status']=='invalid' for q in seq)
   temporal.append(dict(run=folder.name,goal=goal,ever_valid=ever_valid,final_valid=last['path_status']=='valid',lost_earlier_valid_at_final=ever_valid and last['path_status']!='valid',ever_invalid=ever_invalid,final_invalid=last['path_status']=='invalid',unknown_to_valid_events=sum(a['verdict']=='unverified' and b['path_status']=='valid' for a,b in zip(seq,seq[1:])),unknown_to_correct_blocked_events=sum(a['verdict']=='unverified' and b['verdict']=='blocked' and b['reference_reachable']=='False' for a,b in zip(seq,seq[1:])),new_wrong_route_events=sum(b['path_status']=='invalid' and a['path_status']!='invalid' for a,b in zip(seq,seq[1:])),revoked_passable_events=sum(a['verdict']=='passable' and b['verdict']!='passable' for a,b in zip(seq,seq[1:])),revoked_invalid_events=sum(a['path_status']=='invalid' and b['verdict']!='passable' for a,b in zip(seq,seq[1:])),invalid_to_valid_path_events=sum(a['path_status']=='invalid' and b['path_status']=='valid' for a,b in zip(seq,seq[1:])),lost_valid_events=sum(a['path_status']=='valid' and b['path_status']!='valid' for a,b in zip(seq,seq[1:]))))
  operations.append(dict(run=folder.name,travel_distance=r['travel_distance'],captures=r['captures'],entry_wait_drone_seconds=r['entry_wait'],reservation_wait_planned=r['reservation_wait'],reservation_wait_executed=sum(min(max(r['duration']-a['time'],0),a['trajectory']['wait']) for a in assign),failed_drones=r['failed_drones'],geometry_collisions=sum(e['event']=='geometry_collision_frozen' for e in events),teammate_collision_drone_events=sum(e['event']=='teammate_collision_frozen' for e in events),planning_failures=r['planning_failures'],safe_hover_events=r['safe_hover_events'],completed_replan_stops=sum(e['event']=='stopped_at_rest_for_replan' for e in events),joint_stop_requests=sum(e['event']=='map_change_joint_safe_stop' for e in events),no_joint_safe_stop=sum(e['event']=='no_joint_safe_rest_stop' for e in events),fallback_seconds=r['fallback_seconds'],wall_seconds=r['wall_seconds'],render_wall=sum(float(f.get('render_wall') or 0) for f in frames),smoke_wall=sum(float(f.get('smoke_wall') or 0) for f in frames),sgbm_wall=sum(float(f.get('stereo_wall') or 0) for f in frames),mapping_wall=sum(float(f['mapping_wall']) for f in frames)))
  for row in csv.DictReader(open(folder/'metrics_time.csv')):times.append(dict(run=folder.name,condition=r['condition'],policy=r['policy'],**row))
  maps=[]
  for q in sorted((folder/'maps').glob('*.npz')):
   z=np.load(q);maps.append(dict(file=str(q.relative_to(folder)),time=r['duration'] if q.stem=='final' else int(q.stem[1:])*cfg['dt'],map_version=int(z['version'])))
  write_json(folder/'replay_manifest.json',dict(pose_mode=cfg['pose_mode'],frame_order='frames.csv sorted by map_version; integrate each frame once',actual_trajectory='trajectories.csv',assignment_trajectories='assignments.jsonl',map_checkpoints=maps,reference='reference_targets.json; evaluation only',inputs=[dict(frame_id=f['frame_id'],time=float(f['time']),drone=int(f['drone']),map_version=int(f['map_version']),npz='frames/'+f['frame_id']+'.npz',left='frames/'+f['frame_id']+'_left.png',right='frames/'+f['frame_id']+'_right.png',integration_camera_pose=json.loads(f['camera_pose'])) for f in frames]))
 groups=[]
 for condition in sorted({r['condition'] for r in runs}):
  for policy in ('unknown','entry'):
   a=[r for r in runs if r['condition']==condition and r['policy']==policy];n=len(a)
   if not n:continue
   V=sum(r['valid_confirmed'] for r in a);R=sum(r['reachable_denominator'] for r in a);B=sum(r['blocked_denominator'] for r in a);C=sum(r['correct_blocked'] for r in a);I=sum(r['invalid_paths'] for r in a);P=sum(r['verified_path_denominator'] for r in a);D=sum(r['decided'] for r in a);G=sum(r['total_goals'] for r in a)
   groups.append(dict(condition=condition,policy=policy,n_runs=n,valid_paths=V,reachable_denominator=R,valid_path_confirmation_rate=V/R,correct_blocked=C,blocked_denominator=B,correct_blocked_rate=C/B,balanced_score=.5*(V/R+C/B),invalid_paths=I,checked_path_denominator=P,wrong_path_ratio=I/P if P else 'N/A',proposed_paths=sum(r['proposed_paths'] for r in a),unverifiable_paths=sum(r['unverifiable_paths'] for r in a),false_blocked=sum(r['false_blocked'] for r in a),decided=D,total_goal_denominator=G,decision_rate=D/G,unknown=G-D,reference_undecidable=0,mean_travel_distance=float(np.mean([r['travel_distance'] for r in a])),actual_captures=sum(r['captures'] for r in a),failed_drones=sum(r['failed_drones'] for r in a),entry_fallback_seconds=sum(r['fallback_seconds'] for r in a),total_sim_seconds=sum(r['duration'] for r in a)))
 csv_write(out/'summary_by_method.csv',groups);csv_write(out/'targets_final.csv',details);csv_write(out/'operations.csv',operations);csv_write(out/'metrics_time_all.csv',times);csv_write(out/'temporal_goal_summary.csv',temporal)
 pairs=[]
 for a in runs:
  if a['policy']!='unknown':continue
  b=next(q for q in runs if q['scene_seed']==a['scene_seed'] and q['run_seed']==a['run_seed'] and q['condition']==a['condition'] and q['policy']=='entry')
  pairs.append(dict(scene=a['scene_seed'],seed=a['run_seed'],condition=a['condition'],entry_minus_unknown_valid=b['valid_confirmed']-a['valid_confirmed'],entry_minus_unknown_correct_blocked=b['correct_blocked']-a['correct_blocked'],entry_minus_unknown_invalid=b['invalid_paths']-a['invalid_paths'],entry_minus_unknown_false_blocked=b['false_blocked']-a['false_blocked'],entry_minus_unknown_distance=b['travel_distance']-a['travel_distance'],entry_minus_unknown_failed_drones=b['failed_drones']-a['failed_drones']))
 csv_write(out/'paired_comparison.csv',pairs);print(json.dumps(groups,indent=2),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);a=p.parse_args();table(a.root,a.out)
