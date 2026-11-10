#!/usr/bin/env python3
"""Audit and explain the frozen multistorey study; never alter policy inputs.

Usage: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/explore/post_fire_study.py --wait
Only postprocessing outputs/replay manifests are written; no run is discarded.
"""
import argparse,csv,hashlib,json,sys,time
from collections import Counter,defaultdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
csv.field_size_limit(sys.maxsize)
from covor.explore.runner import write_json,csv_write

def readcsv(p):
    with Path(p).open() as f:return list(csv.DictReader(f))
def readjson(p):return json.loads(Path(p).read_text())
def jsonlines(p):
    with Path(p).open() as f:return [json.loads(s) for s in f if s.strip()]
def truth(v):return v is True or v=='True'
def length(path):
    xyz=np.array([p['xyz'] for p in path]);return float(np.linalg.norm(np.diff(xyz,axis=0),axis=1).sum()) if len(xyz)>1 else 0.

def analyze(root,report):
    main=root/'main';out=report/'data';out.mkdir(parents=True,exist_ok=True)
    cfg=readjson(main/'fixed_config.json');folders=sorted(p.parent for p in main.glob('scene*/summary.json'))
    expected={f'scene{i}_seed{s}_{c}_{p}' for i in cfg['scene_seeds'] for s in cfg['run_seeds'] for c in cfg['conditions'] for p in cfg['policies']}
    assert {p.name for p in folders}==expected
    manifest=readjson(main/'source_manifest.json');changed=[p for p,h in manifest.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h];assert not changed,changed
    from covor.explore.fire_world import FireWorld
    from covor.explore.depth_observation import RayDepthSensor
    activity=[];paths=[];sensor_rows=[];replays=[];map_rows=[];faults=[];timings=[];pool_audits=0;frame_total=0;initial={};summaries=[];assignments={}
    for folder in folders:
        run=folder.name;cc=readjson(folder/'config.json');s=readjson(folder/'summary.json');summaries.append(dict(run=run,**s));frames=readcsv(folder/'frames.csv');trajectory=readcsv(folder/'trajectories.csv');assign=jsonlines(folder/'assignments.jsonl');assignments[run]=assign
        refs=readjson(folder/'reference_targets.json');R=sum(q['reachable'] for q in refs);assert R==s['reachable_denominator'];assert len(refs)-R==s['blocked_denominator'];assert len(refs)==s['total_goals'];assert s['duration']==cfg['duration']
        assert len(frames)==s['captures'];assert [int(f['map_version']) for f in frames]==list(range(1,len(frames)+1));assert len(set(f['frame_id'] for f in frames))==len(frames)
        assert not list((folder/'frames').glob('*.png'));assert all(f['backend']=='depth_error_model' and float(f['stereo_wall'])==0 for f in frames)
        for f in frames:assert abs(float(f['time'])/cfg['sensor_period']-round(float(f['time'])/cfg['sensor_period']))<1e-8
        inputs=[]
        for f in frames:
            inputs.append(dict(frame_id=f['frame_id'],time=float(f['time']),drone=int(f['drone']),map_version=int(f['map_version']),npz='frames/'+f['frame_id']+'.npz',integration_camera_pose=json.loads(f['camera_pose']),true_camera_pose=json.loads(f['camera_pose_truth'])))
        rm=readjson(folder/'replay_manifest.json');rm.update(backend='depth_error_model',camera=cfg['sensor'],sensor_period=cfg['sensor_period'],depth_convention='metres, optical camera Z; T maps optical camera coordinates to world',inputs=inputs,truth_pose_role='simulator/replay audit only; mapping uses integration_camera_pose')
        write_json(folder/'replay_manifest.json',rm)
        pool={};positive_epochs=set();candidate_count=0;positive_candidates=0;contrib=0
        for q in jsonlines(folder/'candidates.jsonl'):
            assert q['entry_graph_version']==q['map_version'];assert q['planning_inputs']=='observed_map_and_current_poses_only'
            pool.setdefault(q['time'],set()).add(q['id']);candidate_count+=1
            if any(v>0 for v in q['entry_target_scores'].values()):positive_candidates+=1;positive_epochs.add(q['time'])
            contrib+=len(q['contributions'])
        for e in readcsv(folder/'events.csv'):
            if e['event']=='route_pool_all_candidates':assert set(json.loads(e['candidate_ids']))==pool[float(e['time'])];pool_audits+=1
        for q in assign:
            assert q['selection_before_sensor'];assert q['candidate']['id'] in pool[q['time']]
        decisions=jsonlines(folder/'decisions.jsonl');impacts=sum(bool(q['impact']) for q in decisions)
        max_z=max(float(q['z']) for q in trajectory);inside=[q for q in trajectory if float(q['y'])>=.2];upper=sum(float(q['z'])>=3.3 for q in inside)
        activity.append(dict(run=run,scene=s['scene_seed'],condition=s['condition'],policy=s['policy'],assignments=len(assign),fallback_assignments=sum(q['fallback'] for q in assign),fallback_seconds=s['fallback_seconds'],mission_seconds=s['duration'],candidate_rows=candidate_count,positive_entry_candidates=positive_candidates,positive_entry_epochs=len(positive_epochs),epochs=len(pool),impact_analysis_states=impacts,analysis_states=len(decisions),contribution_rows=contrib,max_drone_z=max_z,upper_floor_trajectory_samples=upper,inside_trajectory_samples=len(inside)))
        for d in range(cfg['n_drones']):
            seq=[q for q in trajectory if int(q['drone'])==d and truth(q['failed'])]
            if seq:
                points=np.array([[float(q[k]) for k in ('x','y','z')] for q in seq]);assert np.allclose(points,points[0],atol=1e-12,rtol=0)
                faults.append(dict(run=run,drone=d,first_failed_time=seq[0]['time'],frozen=True))
        for q in readcsv(folder/'target_paths.csv'):
            if float(q['time'])!=s['duration']:continue
            p=json.loads(q['path']);geo_len=length(p);valid=q['path_status']=='valid';ref=truth(q['reference_reachable'])
            paths.append(dict(run=run,scene=s['scene_seed'],condition=s['condition'],policy=s['policy'],goal=q['goal'],floor=int(q['goal'].split('_')[0][1:]),reference_reachable=ref,verdict=q['verdict'],path_status=q['path_status'],reason=q['reason'],path_reason=q['path_reason'],first_failure=q['first_failure'],path_nodes=len(p),path_length_m=geo_len,path_max_z=max((v['xyz'][2] for v in p),default=''),gap_dependency=any(v.get('gap_dependency',False) for v in p),valid_over_1m=valid and geo_len>1.,reference_contradiction=valid and not ref))
        audit=readcsv(folder/'simulator_audit.csv');ng=sum(int(q['n_geometry']) for q in audit);nv=sum(int(q['n_valid']) for q in audit);nn=sum(int(q['n_normal_geometry']) for q in audit);nnv=sum(int(q['n_normal_valid']) for q in audit)
        weighted=sum(int(q['n_valid'])*float(q['mean_error']) for q in audit)/max(nv,1);rms=np.sqrt(sum(int(q['n_valid'])*float(q['rms_error'])**2 for q in audit)/max(nv,1))
        sensor_rows.append(dict(run=run,scene=s['scene_seed'],condition=s['condition'],policy=s['policy'],frames=len(frames),geometric_pixels=ng,valid_pixels=nv,conditional_valid_ratio=nv/ng if ng else '',normal_geometry_pixels=nn,normal_valid_pixels=nnv,normal_conditional_valid_ratio=nnv/nn if nn else '',mean_error_m=weighted,rms_error_m=rms,outliers=sum(int(q['n_outlier']) for q in audit),random_drops=sum(int(q['n_random_drop']) for q in audit)))
        bycomp=defaultdict(float)
        for q in readcsv(folder/'timing.csv'):bycomp[q['component']]+=float(q['seconds'])
        for comp,seconds in sorted(bycomp.items()):timings.append(dict(run=run,component=comp,wall_seconds=seconds))
        sensor=RayDepthSensor(FireWorld(s['scene_seed'],cfg),cfg,s['scene_seed']*1000+s['run_seed'],s['condition'])
        for f in [frames[0],frames[len(frames)//2]]:
            with np.load(folder/'frames'/(f['frame_id']+'.npz')) as q:
                assert set(q.files)=={'Z','sigma','valid','T','time','drone_id','teammates'}
                z,sz,v=sensor.measure(q['T'],float(q['time']),int(q['drone_id']),q['teammates'])
                assert np.array_equal(z,q['Z'],equal_nan=True);assert np.array_equal(sz,q['sigma'],equal_nan=True);assert np.array_equal(v,q['valid']);assert np.isnan(z[~v]).all()
                replays.append(dict(run=run,frame=f['frame_id'],depth_exact=True,sigma_exact=True,valid_exact=True))
        with np.load(folder/'frames'/(frames[0]['frame_id']+'.npz')) as q:initial[run]=tuple(q[k].copy() for k in ['Z','sigma','valid','T'])
        with np.load(folder/'maps/final.npz') as q:
            assert int(q['version'])==len(frames);a=q['labels'];map_rows.append(dict(run=run,unknown_voxels=int((a==0).sum()),free_voxels=int((a==1).sum()),occupied_voxels=int((a==2).sum()),confirmed_supports=int(q['support'].sum()),observed_support_candidates=int((q['votes']>0).sum()),map_version=int(q['version'])))
        frame_total+=len(frames);print('AUDITED',run,flush=True)
    pairs=[]
    for s in summaries:
        if s['policy']!='unknown':continue
        u=s['run'];e=u.removesuffix('unknown')+'entry';assert all(np.array_equal(x,y,equal_nan=True) for x,y in zip(initial[u],initial[e]))
        au,ae=assignments[u],assignments[e];divergence=None
        for x,y in zip(au,ae):
            if (x['time'],x['drone'],x['candidate'])!=(y['time'],y['drone'],y['candidate']):divergence=min(x['time'],y['time']);break
        if divergence is None and len(au)!=len(ae):divergence=(au if len(au)>len(ae) else ae)[min(len(au),len(ae))]['time']
        pairs.append(dict(scene=s['scene_seed'],condition=s['condition'],same_initial_measurement=True,unknown_assignments=len(au),entry_assignments=len(ae),first_assignment_divergence_time=divergence if divergence is not None else 'identical'))
    csv_write(out/'runs_final.csv',summaries);csv_write(out/'policy_activity.csv',activity);csv_write(out/'paths_final_compact.csv',paths);csv_write(out/'sensor_observations.csv',sensor_rows);csv_write(out/'map_counts.csv',map_rows);csv_write(out/'component_wall_times.csv',timings);csv_write(out/'paired_assignment_audit.csv',pairs)
    csv_write(out/'persistent_faults.csv',faults)
    qc=dict(runs=len(folders),actual_frames=frame_total,expected_runs_retained=True,reference_promotions=readjson(root/'reference_refinements.json'),sources_unchanged=len(manifest),exact_sensor_replays=replays,pool_audits=pool_audits,persistent_faults=faults,public_depth_fields_only=True,no_stereo_PNG=True,same_initial_measurements_all_pairs=True,reference_contradictions=[q for q in paths if q['reference_contradiction']])
    write_json(root/'audit/main_checks.json',qc);plot(report,root,summaries,paths);write_json(root/'post_complete.json',dict(runs=len(folders),frames=frame_total,audit_passed=True,reference_contradictions=len(qc['reference_contradictions'])));print('POSTPROCESSING COMPLETE',flush=True)

def plot(report,root,summaries,paths):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd
    out=report/'figures';out.mkdir(exist_ok=True);df=pd.read_csv(report/'data/metrics_time_all.csv')
    fig,axs=plt.subplots(2,3,figsize=(13,7))
    for i,c in enumerate(['clean','strong']):
        for p,col in [('unknown','#377eb8'),('entry','#e6550d')]:
            q=df[(df.condition==c)&(df.policy==p)].groupby('time').sum(numeric_only=True)
            axs[i,0].plot(q.index,q.valid_confirmed/q.reachable_denominator,label=p,color=col)
            axs[i,1].plot(q.index,q.correct_blocked/q.blocked_denominator,label=p,color=col)
            axs[i,2].plot(q.index,q.invalid_paths,label=p,color=col)
        for j,title in enumerate(['Valid route confirmation / reachable','Correct blockage / blocked','Invalid proposed paths (count)']):axs[i,j].set(title=c+': '+title,xlabel='Simulation time (s)');axs[i,j].grid(alpha=.2);axs[i,j].legend()
    fig.tight_layout();fig.savefig(out/'time_metrics.png',dpi=160);plt.close(fig)
    # Same-height, same-scene map views; no flattening of separate floors.
    from covor.explore.fire_world import FireWorld
    cfg=readjson(root/'main/fixed_config.json');b=FireWorld(1,cfg);gt,ijk=b.labels(cfg['map_resolution']);res=cfg['map_resolution'];extent=[ijk[0]*res,(ijk[0]+gt.shape[0])*res,ijk[1]*res,(ijk[1]+gt.shape[1])*res]
    from matplotlib.colors import ListedColormap
    cmap=ListedColormap(['#b7b7b7','#fafafa','#333333'])
    fig,axs=plt.subplots(2,3,figsize=(13,8))
    for i,z in enumerate([.8,4.1]):
        k=int(np.floor(z/res))-ijk[2];axs[i,0].imshow(gt[:,:,k].T,origin='lower',extent=extent,cmap=cmap,vmin=0,vmax=2);axs[i,0].set_title(f'Reference geometry, z={z} m')
        for j,p in enumerate(['unknown','entry'],1):
            run=root/'main'/f'scene1_seed0_strong_{p}'
            with np.load(run/'maps/final.npz') as q:axs[i,j].imshow(q['labels'][:,:,k].T,origin='lower',extent=extent,cmap=cmap,vmin=0,vmax=2)
            tr=pd.read_csv(run/'trajectories.csv');floor=tr[(tr.z>=i*3.3)&(tr.z<(i+1)*3.3)]
            for d,g in floor.groupby('drone'):axs[i,j].plot(g.x,g.y,lw=.7,label=f'D{d}')
            axs[i,j].set_title(f'{p}, strong, t=180 s, z={z} m')
        for ax in axs[i]:ax.set(xlabel='x (m)',ylabel='y (m)');ax.set_aspect('equal')
    fig.tight_layout();fig.savefig(out/'school_map_floor_comparison.png',dpi=150);plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='results/fire_closed_loop_20261107');p.add_argument('--report',default='reports/fire_closed_loop_20261107');p.add_argument('--wait',action='store_true');a=p.parse_args();root=Path(a.root)
    while a.wait:
        s=readjson(root/'pipeline_status.json')
        if s['phase']=='complete':break
        if 'failed' in s['phase']:raise RuntimeError(s)
        time.sleep(30)
    analyze(root,Path(a.report))
