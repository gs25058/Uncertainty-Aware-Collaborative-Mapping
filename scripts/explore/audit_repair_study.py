#!/usr/bin/env python3
"""Post-selection source, observation replay, failure and actual-path audit."""
import argparse,csv,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from covor.explore.fire_recon_scene import ReconFireWorld
from covor.explore.sensing import Sensor,camera_pose
from covor.explore.runner import write_json,csv_write
from covor.entry.path_eval import ReferenceValidator,PathEvalCfg


def records(path):
    csv.field_size_limit(sys.maxsize)
    return list(csv.DictReader(path.open()))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('results/method_refinement_20261110'));a=p.parse_args()
    batch=a.root/'paired32';spec=json.loads((batch/'fixed_spec.json').read_text())
    hashes=json.loads((batch/'source_manifest.json').read_text())
    assert all(hashlib.sha256((batch/'source_snapshot'/f).read_bytes()).hexdigest()==h for f,h in hashes.items())
    out=a.root/'audit';out.mkdir(exist_ok=True);worlds={};validators={};replays=[];checks=[];faults=[]
    for job in spec['jobs']:
        folder=Path(job['out']);summary=json.loads((folder/'summary.json').read_text());cfg=json.loads((folder/'config.json').read_text())['configuration']
        assert not cfg['human'].get('full_goal_cut_certificate') and not cfg['human'].get('observed_route_quality_weight')
        scene=job['scene']
        if scene not in worlds:
            b=ReconFireWorld(scene,cfg);gt,ijk=b.labels(cfg['map_resolution']);b.render_scene(False);worlds[scene]=b
            validators[scene]=ReferenceValidator(gt,ijk,cfg['map_resolution'],b.scene,cfg=PathEvalCfg(step_limit_m=cfg['human']['exact_step_limit'],contact_epsilon_m=cfg['human']['contact_epsilon_m']))
        b=worlds[scene];prior=json.loads((folder/'prior.json').read_text());assert b.prior()==prior
        truth=json.loads((folder/'reference_targets.json').read_text());expected=b.reference(cfg['map_resolution'])[0]
        assert {q['id']:q['reachable'] for q in truth}=={q['id']:q['reachable'] for q in expected}
        frames=records(folder/'frames.csv');assert len(frames)==summary['captures']==summary['map_version']
        assert len({q['frame_id'] for q in frames})==len(frames)
        assert [int(q['map_version']) for q in frames]==list(range(1,len(frames)+1))
        new=[q for q in frames if float(q['time'])>180]
        trace=records(folder/'trajectories.csv');poses={(float(q['time']),int(q['drone'])):q for q in trace}
        for q in new:
            r=poses[float(q['time']),int(q['drone'])]
            body=np.array([float(r[k]) for k in ['x','y','z']]);T=camera_pose(body,float(r['yaw']),float(r['gimbal_pitch']),cfg['sensor']['baseline'])
            np.testing.assert_allclose(T,np.array(json.loads(q['camera_pose_truth'])),rtol=0,atol=1e-12)
        sensor=Sensor(b,cfg,scene*1000,job['condition'])
        for q in [new[0],new[len(new)//2]]:
            r=poses[float(q['time']),int(q['drone'])];fid=q['frame_id']
            with np.load(folder/'frames'/(fid+'.npz')) as saved:
                assert set(saved.files)=={'Z','sigma','valid','T','time','drone_id','teammates'}
                position=np.array([float(r[k]) for k in ['x','y','z']])
                _,z,sz,valid,_=sensor.capture(position,float(r['yaw']),float(r['gimbal_pitch']),float(q['time']),saved['teammates'],out/'sensor_replays'/job['variant']/folder.name/'frames',fid,drone_id=int(q['drone']))
                assert np.array_equal(z,saved['Z'],equal_nan=True)
                assert np.array_equal(sz,saved['sigma'],equal_nan=True)
                assert np.array_equal(valid,saved['valid'])
            replays.append(dict(variant=job['variant'],scene=scene,condition=job['condition'],frame=fid,depth_sigma_valid_exact=True))
        for drone in range(cfg['n_drones']):
            seq=sorted((q for q in trace if int(q['drone'])==drone),key=lambda q:float(q['time']))
            first=next((i for i,q in enumerate(seq) if q['failed']=='True'),None)
            if first is not None:
                tail=seq[first:]
                assert all(q['failed']=='True' and all(q[k]==tail[0][k] for k in ['x','y','z']) for q in tail)
                faults.append(dict(variant=job['variant'],scene=scene,condition=job['condition'],drone=drone,first_failure=float(tail[0]['time']),inherited=float(tail[0]['time'])<=180,not_respawned=True))
        final=[q for q in records(folder/'target_paths.csv') if float(q['time'])==212]
        valid_count=invalid=unverifiable=0
        for q in final:
            if q['verdict']!='passable':continue
            goal=next(g for g in prior['goals'] if g['id']==q['goal'])
            result=validators[scene].validate(json.loads(q['path']),prior['entry'],goal)
            assert result['status']==q['path_status']
            valid_count+=result['status']=='valid';invalid+=result['status']=='invalid';unverifiable+=result['status']=='unverifiable'
        assert valid_count==summary['valid_confirmed'] and invalid==summary['invalid_paths'] and unverifiable==summary['unverifiable_paths']
        pools={}
        for line in (folder/'candidates.jsonl').open():
            q=json.loads(line)
            if q['time']<=180:continue
            assert q['map_version']==q['entry_graph_version'];pools.setdefault(q['time'],set()).add(q['id'])
        assert all(len(v)<=cfg['candidates']['max_total'] for v in pools.values())
        for line in (folder/'assignments.jsonl').open():
            q=json.loads(line)
            if q['time']>180:assert q['selection_before_sensor'] and q['candidate']['id'] in pools[q['time']]
        checks.append(dict(variant=job['variant'],scene=scene,condition=job['condition'],new_frames=len(new),revalidated_paths=valid_count+invalid+unverifiable,failed_drones=sum(q['failed']=='True' for q in trace if float(q['time'])==212),map_versions_match=True))
        print('AUDITED',job['variant'],scene,job['condition'],flush=True)
    csv_write(out/'sensor_replays.csv',replays);csv_write(out/'runs.csv',checks);csv_write(out/'faults.csv',faults)
    write_json(out/'checks.json',dict(runs=len(checks),new_observations=sum(q['new_frames'] for q in checks),exact_new_sensor_replays=len(replays),independently_revalidated_final_paths=sum(q['revalidated_paths'] for q in checks),persistent_failed_drones=len(faults),frozen_source_unchanged=True,reference_and_targets_unchanged=True,public_depth_fields_only=True,observation_poses_and_versions_match=True,selection_before_observation=True))


if __name__=='__main__':main()
