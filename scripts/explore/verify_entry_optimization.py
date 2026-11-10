#!/usr/bin/env python3
"""Independent final validation audit; never feeds observations/labels to policy."""
import argparse,csv,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from covor.explore.fire_recon_scene import ReconFireWorld
from covor.explore.sensing import Sensor
from covor.explore.runner import write_json,csv_write
from covor.entry.path_eval import ReferenceValidator,PathEvalCfg
csv.field_size_limit(sys.maxsize)
p=argparse.ArgumentParser();p.add_argument('--root',default='results/entry_optimization_20261109');p.add_argument('--report',default='reports/entry_optimization_20261109');a=p.parse_args();root=Path(a.root);report=Path(a.report);audit=root/'audit';audit.mkdir(exist_ok=True);(report/'data').mkdir(exist_ok=True)
winner=json.load(open(root/'winner_frozen.json'));assert hashlib.sha256(Path(winner['config']).read_bytes()).hexdigest()==winner['config_sha256']
base_cfg=json.load(open(root/'intake/source_snapshot/configs/fire_recon_20261107.json'))
protected=json.load(open(root/'validation/shared_stack_unchanged.json'));original=json.load(open(root/'intake/source_manifest.json'));assert all(hashlib.sha256(Path(f).read_bytes()).hexdigest()==original[f] for f in protected)
worlds={};validators={};runs=[];targets=[];frames_total=0;replays=[];faults=[];versions=0;contradictions=[];first={};times=[];transitions=[]
for batch_name in ['validation_seed2','validation_seed3']:
 batch=root/batch_name;assert json.load(open(batch/'status.json'))['phase']=='complete';spec=json.load(open(batch/'fixed_spec.json'));manifest=json.load(open(batch/'source_manifest.json'))
 assert all(hashlib.sha256((batch/'source_snapshot'/f).read_bytes()).hexdigest()==h for f,h in manifest.items())
 for label in spec['variants']:
  for folder in sorted((batch/label).glob('scene*')):
   if not folder.is_dir():continue
   summary=json.load(open(folder/'summary.json'));cc=json.load(open(folder/'config.json'));cfg=cc['configuration'];scene=summary['scene_seed'];seed=summary['run_seed'];condition=summary['condition'];name=batch_name+'/'+label+'/'+folder.name
   assert all(cfg[k]==v for k,v in base_cfg.items() if k not in ('utility','protocol_version')),'non-utility experimental setting changed'
   if spec.get('defer_dense'):
    mode=json.load(open(folder/'.execution/dense_read_mode.json'))
    assert mode['defer_dense'] and mode['helper_sha256']==manifest['covor/explore/execution_acceleration.py']
    dense_log=json.load(open(folder/'dense_read_acceleration.json'))
    assert dense_log['inner_occupancy_updates_preserved'] and dense_log['all_sensor_integrations_preserved']
   if scene not in worlds:
    b=ReconFireWorld(scene,cfg);lab,ijk=b.labels(cfg['map_resolution']);b.render_scene(False);worlds[scene]=b
    validators[scene]=ReferenceValidator(lab,ijk,cfg['map_resolution'],b.scene,cfg=PathEvalCfg(step_limit_m=cfg['human']['exact_step_limit'],contact_epsilon_m=cfg['human']['contact_epsilon_m']))
   b=worlds[scene];prior=json.load(open(folder/'prior.json'));assert prior==b.prior();reference=json.load(open(folder/'reference_targets.json'));assert sum(q['reachable'] for q in reference)==summary['reachable_denominator'];ref={q['id']:q for q in reference}
   expected=b.reference(cfg['map_resolution'])[0];assert {q['id']:q['reachable'] for q in expected}=={q['id']:q['reachable'] for q in reference}
   data=[];previous={};history={}
   for row in csv.DictReader(open(folder/'target_paths.csv')):
    goal=row['goal'];h=history.setdefault(goal,dict(unknown_to_valid=0,unknown_to_correct_blocked=0,new_wrong_route=0,revoked_passable=0,revoked_invalid=0,lost_valid=0,ever_valid=False));old=previous.get(goal)
    if old:
     h['unknown_to_valid']+=old['verdict']=='unverified' and row['path_status']=='valid'
     h['unknown_to_correct_blocked']+=old['verdict']=='unverified' and row['verdict']=='blocked' and row['reference_reachable']=='False'
     h['new_wrong_route']+=row['path_status']=='invalid' and old['path_status']!='invalid'
     h['revoked_passable']+=old['verdict']=='passable' and row['verdict']!='passable'
     h['revoked_invalid']+=old['path_status']=='invalid' and row['verdict']!='passable'
     h['lost_valid']+=old['path_status']=='valid' and row['path_status']!='valid'
    h['ever_valid']|=row['path_status']=='valid';previous[goal]=row
    if float(row['time'])==summary['duration']:data.append(row)
   assert len(data)==summary['total_goals']
   for goal,h in history.items():transitions.append(dict(batch=batch_name,label=label,run=folder.name,scene=scene,seed=seed,condition=condition,goal=goal,final_lost_earlier_valid=h['ever_valid'] and previous[goal]['path_status']!='valid',**h))
   for row in csv.DictReader(open(folder/'metrics_time.csv')):times.append(dict(batch=batch_name,label=label,run=folder.name,seed=seed,scene=scene,condition=condition,**row))
   valid=invalid=unverifiable=0
   for row in data:
    assert (row['reference_reachable']=='True')==ref[row['goal']]['reachable']
    path_nodes=0;path_length=0.
    if row['verdict']=='passable':
     path=json.loads(row['path']);goal=next(q for q in prior['goals'] if q['id']==row['goal']);checked=validators[scene].validate(path,prior['entry'],goal);assert checked['status']==row['path_status'],(name,row['goal'],checked,row['path_status'])
     path_nodes=len(path);path_length=float(np.linalg.norm(np.diff(np.asarray([p['xyz'] for p in path]),axis=0),axis=1).sum()) if len(path)>1 else 0.
     valid+=checked['status']=='valid' and ref[row['goal']]['reachable'];invalid+=checked['status']=='invalid';unverifiable+=checked['status']=='unverifiable'
     if checked['status']=='valid' and not ref[row['goal']]['reachable']:contradictions.append(dict(run=name,goal=row['goal']))
    targets.append(dict(batch=batch_name,label=label,run=folder.name,seed=seed,scene=scene,condition=condition,split='reserved_geometry' if scene in spec['reserved_scenes'] else 'development_geometry_new_seed',proposed_path_nodes=path_nodes,proposed_path_length_m=path_length,valid_over_1m=row['path_status']=='valid' and path_length>1.,valid_single_node=row['path_status']=='valid' and path_nodes==1,**row))
   assert valid==summary['valid_confirmed'] and invalid==summary['invalid_paths'] and unverifiable==summary['unverifiable_paths']
   frame_rows=list(csv.DictReader(open(folder/'frames.csv')));assert len(frame_rows)==summary['captures'];assert len({q['frame_id'] for q in frame_rows})==len(frame_rows);frames_total+=len(frame_rows)
   trace={(float(q['time']),int(q['drone'])):q for q in csv.DictReader(open(folder/'trajectories.csv'))}
   sensor=Sensor(b,cfg,scene*1000+seed,condition)
   samples=[frame_rows[0],frame_rows[len(frame_rows)//2]]
   for row in samples:
    fid=row['frame_id'];t=float(row['time']);drone=int(row['drone']);pose=trace[t,drone];position=np.array([float(pose[k]) for k in ('x','y','z')])
    with np.load(folder/'frames'/f'{fid}.npz') as q:
     assert set(q.files)=={'Z','sigma','valid','T','time','drone_id','teammates'},q.files
     saved={k:q[k].copy() for k in q.files}
    destination=audit/'sensor_replays'/batch_name/label/folder.name/'frames'
    _,z,sz,mask,_=sensor.capture(position,float(pose['yaw']),float(pose['gimbal_pitch']),t,saved['teammates'],destination,fid,False,drone_id=drone)
    assert np.array_equal(z,saved['Z'],equal_nan=True) and np.array_equal(sz,saved['sigma'],equal_nan=True) and np.array_equal(mask,saved['valid'])
    replays.append(dict(run=name,frame=fid,depth_exact=True,sigma_exact=True,valid_exact=True))
    if row is frame_rows[0]:
     key=(scene,seed,condition);signature={k:hashlib.sha256(saved[k].tobytes()).hexdigest() for k in ['Z','sigma','valid','T']}
     if key in first:assert signature==first[key]
     else:first[key]=signature
   failed={int(q['drone']):q for q in trace.values() if q['failed']=='True'}
   for drone in failed:
    seq=sorted((q for q in trace.values() if int(q['drone'])==drone),key=lambda q:float(q['time']));tail=seq[next(i for i,q in enumerate(seq) if q['failed']=='True'):]
    assert all(q['failed']=='True' and all(q[k]==tail[0][k] for k in ['x','y','z']) for q in tail);faults.append(dict(run=name,drone=drone,persistent=True))
   pools={}
   for line in open(folder/'candidates.jsonl'):
    q=json.loads(line);assert q['map_version']==q['entry_graph_version'];pools.setdefault(q['time'],set()).add(q['id']);versions+=1
   assert all(len(ids)<=cfg['candidates']['max_total'] for ids in pools.values())
   for line in open(folder/'assignments.jsonl'):
    q=json.loads(line);assert q['candidate']['id'] in pools[q['time']] and q['selection_before_sensor']
   write_json(folder/'replay_manifest.json',dict(source=str(batch/'source_snapshot'),configuration='config.json',trajectory='trajectories.csv',frame_order='frames.csv by map_version; each physical observation once',depth_directory='frames; NPZ only',maps='maps/*.npz',assignments='assignments.jsonl',events='events.csv',reference_evaluation_only='reference_targets.json',captures=len(frame_rows)))
   runs.append(dict(batch=batch_name,label=label,run=folder.name,split='reserved_geometry' if scene in spec['reserved_scenes'] else 'development_geometry_new_seed',**summary));print('AUDITED',name,flush=True)
csv_write(report/'data/validation_runs.csv',runs);csv_write(report/'data/validation_targets.csv',targets);csv_write(report/'data/sensor_replay_audit.csv',replays);csv_write(report/'data/fault_audit.csv',faults)
csv_write(report/'data/validation_time.csv',times);csv_write(report/'data/validation_transitions.csv',transitions)
write_json(audit/'validation_checks.json',dict(runs=len(runs),logical_captures=frames_total,exact_sensor_replays=len(replays),independently_revalidated_final_paths=sum(q['verdict']=='passable' for q in targets),reference_contradictions=contradictions,candidate_rows_with_matching_map_versions=versions,persistent_failed_drones=len(faults),winner_config_hash_unchanged=True,frozen_sources_unchanged=True,all_goal_regions_unchanged=True,public_depth_fields_only=True,initial_measurements_identical_all_policy_pairs=True,validation_did_not_select_parameters=True))
# Persist disputed cases and raw tables before failing the audit. A contradiction
# must be investigated against geometry, never hidden by discarding its run.
assert len(runs)==100 and not contradictions,(len(runs),contradictions)
print('FINAL VALIDATION AUDIT COMPLETE',flush=True)
