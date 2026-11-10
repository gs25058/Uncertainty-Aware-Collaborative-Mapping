#!/usr/bin/env python3
"""Audit frozen control/utility factorial and report every final path and fault."""
import argparse,copy,csv,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from covor.explore.execution_state import atomic_json
from covor.explore.fire_recon_scene import ReconFireWorld
from covor.explore.sensing import Sensor,camera_pose
from covor.explore.runner import csv_write
from covor.explore.motion import Trajectory
from covor.explore.flight_control import BrakingTrajectory
from covor.entry.path_eval import ReferenceValidator,PathEvalCfg
from analyze_entry_outcomes import counts,statistics
csv.field_size_limit(sys.maxsize)


def restore_trajectory(record,cfg):
    if record.get('type')=='continuous_state_quintic_brake':
        c=np.array(record['coefficients']);tr=BrakingTrajectory(c[:,0],c[:,1],2*c[:,2],record['points'][-1],record['durations'][0],record['start'],record['yaw'],record['pitch'],cfg)
        np.testing.assert_allclose(tr.coefficients,c,rtol=0,atol=1e-10)
        return tr
    tr=object.__new__(Trajectory)
    for k in ('start','wait','end','yaw','pitch','dy','dp','durations'):setattr(tr,k,record[k])
    tr.points=np.array(record['points']);tr.cfg=cfg;tr.move=sum(tr.durations)
    tr.rotation=max(1.875*abs(tr.dy)/cfg['yaw_rate'],1.875*abs(tr.dp)/cfg['pitch_rate'],.1)
    tr.total=tr.wait+max(tr.move,tr.rotation);tr.stop_at=None
    return tr


def audit_motion(folder,cfg):
    events=list(csv.DictReader(open(folder/'events.csv')));actions={};brakes=0
    for line in open(folder/'assignments.jsonl'):
        q=json.loads(line);actions.setdefault(int(q['drone']),[]).append((float(q['time']),1,'assign',q['trajectory']))
    for q in events:
        event=q['event'];drone=int(q['drone']);t=float(q['time'])
        if event=='observed_map_guard_stop':
            record=json.loads(q['trajectory']);tr=restore_trajectory(record,cfg['motion'])
            if q['kind']=='continuous_brake':
                assert tr.within_limits();brakes+=1
                for derivative,key in enumerate(['initial_position','initial_velocity','initial_acceleration']):
                    np.testing.assert_allclose(tr.sample(t,derivative),json.loads(q[key]),rtol=0,atol=1e-9)
            actions.setdefault(drone,[]).append((t,0,'replace',(record,float(q['stop_time']))))
        elif event=='map_change_joint_safe_stop':
            actions.setdefault(drone,[]).append((t,0,'legacy_stop',float(q['stop_time'])))
    trace=list(csv.DictReader(open(folder/'trajectories.csv')));by_drone={};position_checks=0;faults=0
    for q in trace:by_drone.setdefault(int(q['drone']),[]).append(q)
    for drone,rows in by_drone.items():
        ledger=sorted(actions.get(drone,[]),key=lambda x:(x[0],x[1]));cursor=0;active=None;frozen=None;last=None
        for q in rows:
            t=float(q['time']);position=np.array([float(q[k]) for k in ['estimated_x','estimated_y','estimated_z']])
            while cursor<len(ledger) and ledger[cursor][0]<=t+1e-9:
                _,_,kind,value=ledger[cursor];cursor+=1
                if kind=='legacy_stop':
                    assert active is not None;active=copy.copy(active);active.stop_at=value;active.end=value
                else:
                    record,stop=(value if kind=='replace' else (value,None));active=restore_trajectory(record,cfg['motion'])
                    if stop is not None:active.stop_at=stop;active.end=stop
            if q['failed']=='True':
                if frozen is None:frozen=position.copy();faults+=1
                np.testing.assert_allclose(position,frozen,rtol=0,atol=1e-12)
            else:
                assert frozen is None,'failed drone respawned'
                if active is not None:
                    np.testing.assert_allclose(position,active.sample(t),rtol=0,atol=1e-9);position_checks+=1
            if last:
                dt=t-float(last['time']);dy=abs((float(q['yaw'])-float(last['yaw'])+np.pi)%(2*np.pi)-np.pi)
                dp=abs(float(q['gimbal_pitch'])-float(last['gimbal_pitch']))
                assert dy<=cfg['motion']['yaw_rate']*dt+1e-7 and dp<=cfg['motion']['pitch_rate']*dt+1e-7
                previous=np.array([float(last[k]) for k in ['estimated_x','estimated_y','estimated_z']])
                assert np.linalg.norm(position-previous)<=cfg['motion']['vmax']*dt+1e-7
            last=q
    return trace,dict(trajectory_position_checks=position_checks,continuous_brakes=brakes,persistent_failed_drones=faults,
        guard_stop_events=sum(q['event']=='observed_map_guard_stop' for q in events),guard_failures=sum(q['event']=='observed_map_guard_no_safe_brake' for q in events))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--report',required=True);args=p.parse_args()
    root=Path(args.root);report=Path(args.report);data=report/'data';data.mkdir(exist_ok=True);audit=root/'audit';audit.mkdir(exist_ok=True)
    batch=root/'validation';assert json.load(open(batch/'status.json'))['phase']=='complete'
    manifest=json.load(open(batch/'source_manifest.json'));assert all(hashlib.sha256((batch/'source_snapshot'/f).read_bytes()).hexdigest()==h for f,h in manifest.items())
    original=json.load(open('results/entry_optimization_20261109/intake/source_manifest.json'))
    protected=json.load(open('results/entry_optimization_20261109/validation/shared_stack_unchanged.json'))
    for f in protected:assert manifest[f]==original[f],('shared stack modified',f)
    frozen=json.load(open(root/'winner_frozen.json'))
    for q in frozen['configurations'].values():assert hashlib.sha256(Path(q['path']).read_bytes()).hexdigest()==q['sha256']
    worlds={};validators={};runs=[];targets=[];times=[];replays=[];motion=[];first={};contradictions=[];transitions=[];logical_captures=0
    spec=json.load(open(batch/'fixed_spec.json'))
    for label in spec['variants']:
        for folder in sorted((batch/label).glob('scene*')):
            if not folder.is_dir():continue
            s=json.load(open(folder/'summary.json'));cfg=json.load(open(folder/'config.json'))['configuration']
            scene=s['scene_seed'];seed=s['run_seed'];condition=s['condition'];name=label+'/'+folder.name
            assert seed==4 and s['duration']==180. and s['n_drones']==4
            if scene not in worlds:
                b=ReconFireWorld(scene,cfg);lab,ijk=b.labels(cfg['map_resolution']);b.render_scene(False);worlds[scene]=b
                validators[scene]=ReferenceValidator(lab,ijk,cfg['map_resolution'],b.scene,cfg=PathEvalCfg(step_limit_m=cfg['human']['exact_step_limit'],contact_epsilon_m=cfg['human']['contact_epsilon_m']))
            b=worlds[scene];prior=json.load(open(folder/'prior.json'));assert prior==b.prior()
            ref={q['id']:q['reachable'] for q in b.reference(cfg['map_resolution'])[0]}
            final=[];previous={};history={}
            for q in csv.DictReader(open(folder/'target_paths.csv')):
                goal=q['goal'];old=previous.get(goal);h=history.setdefault(goal,dict(unknown_to_valid=0,unknown_to_correct_blocked=0,new_invalid=0,revoked_passable=0,revoked_invalid=0,lost_valid=0,ever_valid=False))
                if old:
                    h['unknown_to_valid']+=old['verdict']=='unverified' and q['path_status']=='valid'
                    h['unknown_to_correct_blocked']+=old['verdict']=='unverified' and q['verdict']=='blocked' and not ref[goal]
                    h['new_invalid']+=q['path_status']=='invalid' and old['path_status']!='invalid'
                    h['revoked_passable']+=old['verdict']=='passable' and q['verdict']!='passable'
                    h['revoked_invalid']+=old['path_status']=='invalid' and q['verdict']!='passable'
                    h['lost_valid']+=old['path_status']=='valid' and q['path_status']!='valid'
                h['ever_valid']|=q['path_status']=='valid';previous[goal]=q
                if float(q['time'])==180.:final.append(q)
            for goal,h in history.items():transitions.append(dict(label=label,scene=scene,seed=seed,condition=condition,goal=goal,final_lost_earlier_valid=h['ever_valid'] and previous[goal]['path_status']!='valid',**h))
            assert len(final)==len(ref)==s['total_goals']
            valid=invalid=blocked=0
            for q in final:
                assert (q['reference_reachable']=='True')==ref[q['goal']]
                length=0.;nodes=0
                if q['verdict']=='passable':
                    path=json.loads(q['path']);goal=next(g for g in prior['goals'] if g['id']==q['goal'])
                    result=validators[scene].validate(path,prior['entry'],goal);assert result['status']==q['path_status'],(name,q['goal'],result,q['path_status'])
                    length=float(np.linalg.norm(np.diff(np.array([v['xyz'] for v in path]),axis=0),axis=1).sum()) if len(path)>1 else 0.;nodes=len(path)
                    valid+=result['status']=='valid' and ref[q['goal']];invalid+=result['status']=='invalid'
                    if result['status']=='valid' and not ref[q['goal']]:contradictions.append(dict(run=name,goal=q['goal']))
                blocked+=q['verdict']=='blocked' and not ref[q['goal']]
                targets.append(dict(label=label,run=folder.name,scene=scene,seed=seed,condition=condition,path_length_m=length,path_nodes=nodes,**q))
            assert valid==s['valid_confirmed'] and invalid==s['invalid_paths'] and blocked==s['correct_blocked']
            trace,checks=audit_motion(folder,cfg);assert checks['persistent_failed_drones']==s['failed_drones'];motion.append(dict(run=name,**checks))
            trace_index={(float(q['time']),int(q['drone'])):q for q in trace}
            frames=list(csv.DictReader(open(folder/'frames.csv')));assert len(frames)==s['captures'] and len({q['frame_id'] for q in frames})==len(frames);logical_captures+=len(frames)
            sensor=Sensor(b,cfg,scene*1000+seed,condition)
            for row in [frames[0],frames[len(frames)//2]]:
                fid=row['frame_id'];t=float(row['time']);drone=int(row['drone']);q=trace_index[t,drone]
                position=np.array([float(q[k]) for k in ['x','y','z']])
                with np.load(folder/'frames'/(fid+'.npz')) as f:
                    assert set(f.files)=={'Z','sigma','valid','T','time','drone_id','teammates'};saved={k:f[k].copy() for k in f.files}
                expected=camera_pose(position,float(q['yaw']),float(q['gimbal_pitch']),cfg['sensor']['baseline'])
                np.testing.assert_allclose(saved['T'],expected,rtol=0,atol=1e-12)
                _,z,sigma,mask,_=sensor.capture(position,float(q['yaw']),float(q['gimbal_pitch']),t,saved['teammates'],audit/'sensor_replays'/label/folder.name/'frames',fid,False,drone_id=drone)
                assert np.array_equal(z,saved['Z'],equal_nan=True) and np.array_equal(sigma,saved['sigma'],equal_nan=True) and np.array_equal(mask,saved['valid'])
                replays.append(dict(run=name,frame=fid,depth_exact=True,sigma_exact=True,valid_exact=True))
                if row is frames[0]:
                    key=(label.startswith('new_'),scene,condition);signature=[hashlib.sha256(saved[k].tobytes()).hexdigest() for k in ['Z','sigma','valid','T']]
                    if key in first:assert first[key]==signature
                    else:first[key]=signature
            for q in csv.DictReader(open(folder/'metrics_time.csv')):times.append(dict(label=label,run=folder.name,scene=scene,seed=seed,condition=condition,**q))
            runs.append(dict(label=label,run=folder.name,**s,**checks));print('AUDITED',name,flush=True)
            atomic_json(folder/'replay_manifest.json',dict(trajectory='trajectories.csv',frames='frames.csv, ascending map_version; NPZ directory frames',assignments='assignments.jsonl',control_changes='events.csv: observed_map_guard_stop contains actual continuous-state polynomial and stop time',maps='maps/*.npz',source=str(batch/'source_snapshot'),reference_evaluation_only='reference_targets.json'))
    csv_write(data/'runs.csv',runs);csv_write(data/'targets.csv',targets);csv_write(data/'time.csv',times);csv_write(data/'motion_audit.csv',motion);csv_write(data/'sensor_replay.csv',replays)
    csv_write(data/'transitions.csv',transitions)
    atomic_json(audit/'checks.json',dict(runs=len(runs),logical_captures=logical_captures,exact_sensor_replays=len(replays),final_paths_revalidated=sum(q['verdict']=='passable' for q in targets),reference_contradictions=contradictions,frozen_sources_unchanged=True,same_initial_measurements_per_backend_policy_pair=True,p_v_a_continuous_brakes=sum(q['continuous_brakes'] for q in motion),trajectory_position_checks=sum(q['trajectory_position_checks'] for q in motion),persistent_failed_drones=sum(q['persistent_failed_drones'] for q in motion)))
    assert len(runs)==80 and not contradictions,(len(runs),contradictions)
    groups=[]
    for condition in ['clean','strong']:
        for label in spec['variants']:
            for split in ['all','reserved_geometry','development_geometry_new_seed']:
                rr=[q for q in runs if q['condition']==condition and q['label']==label and (split=='all' or ((q['scene_seed'] not in spec['development_scenes'])==(split=='reserved_geometry')))]
                x=counts(rr);stats=statistics(x);v,r,c,b,i,p,fb,failed=x
                groups.append(dict(condition=condition,label=label,split=split,runs=len(rr),valid=int(v),reachable=int(r),correct_blocked=int(c),blocked=int(b),balanced=float(stats[2]),invalid=int(i),checked_paths=int(p),wrong_ratio=float(stats[4]) if p else 'N/A',proposed_paths=sum(q['proposed_paths'] for q in rr),unverifiable_paths=sum(q['unverifiable_paths'] for q in rr),false_blocked=int(fb),failed_drones=int(failed),unknown=sum(q['unknown'] for q in rr),decided=sum(q['decided'] for q in rr),total_goals=sum(q['total_goals'] for q in rr),valid_rate=float(stats[0]),blocked_rate=float(stats[1]),decision_rate=sum(q['decided'] for q in rr)/sum(q['total_goals'] for q in rr),captures=sum(q['captures'] for q in rr),travel_m=sum(q['travel_distance'] for q in rr),braking_plans=sum(q['continuous_brakes'] for q in rr),no_safe_brake=sum(q['guard_failures'] for q in rr)))
    csv_write(data/'summary.csv',groups)
    comparisons=[]
    for condition in ['clean','strong']:
        for name,a,b in [('value_original','original_entry_selected','original_unknown'),('value_new','new_entry_selected','new_unknown'),('control_unknown','new_unknown','original_unknown'),('control_entry','new_entry_selected','original_entry_selected')]:
            aa=next(q for q in groups if q['condition']==condition and q['label']==a and q['split']=='all');bb=next(q for q in groups if q['condition']==condition and q['label']==b and q['split']=='all')
            comparisons.append(dict(condition=condition,comparison=name,a=a,b=b,**{k+'_delta':aa[k]-bb[k] for k in ['valid','correct_blocked','balanced','invalid','false_blocked','failed_drones','unknown','decided','travel_m','captures']}))
    csv_write(data/'factorial_comparisons.csv',comparisons)
    index={(q['scene'],q['condition'],q['label'],q['goal']):q for q in targets};changes=[];floor=[];intervals=[]
    for backend in ['original','new']:
        for q in targets:
            if q['label']!=backend+'_entry_selected':continue
            b=index[q['scene'],q['condition'],backend+'_unknown',q['goal']]
            changes.append(dict(backend=backend,scene=q['scene'],condition=q['condition'],goal=q['goal'],reference_reachable=q['reference_reachable'],baseline_verdict=b['verdict'],entry_verdict=q['verdict'],baseline_path_status=b['path_status'],entry_path_status=q['path_status'],gained_valid=q['path_status']=='valid' and b['path_status']!='valid',lost_valid=b['path_status']=='valid' and q['path_status']!='valid',gained_correct_blocked=q['reference_reachable']=='False' and q['verdict']=='blocked' and b['verdict']!='blocked',new_invalid=q['path_status']=='invalid' and b['path_status']!='invalid',removed_invalid=b['path_status']=='invalid' and q['path_status']!='invalid',entry_path=q['path'],baseline_path=b['path'],entry_failure=q['first_failure'],baseline_failure=b['first_failure']))
    csv_write(data/'goal_changes.csv',changes)
    for condition in ['clean','strong']:
        for label in spec['variants']:
            for level in [1,2,3]:
                rr=[q for q in targets if q['condition']==condition and q['label']==label and q['goal'].startswith(f'L{level}_')]
                floor.append(dict(condition=condition,label=label,floor=level,reachable=sum(q['reference_reachable']=='True' for q in rr),blocked=sum(q['reference_reachable']=='False' for q in rr),valid=sum(q['path_status']=='valid' for q in rr),correct_blocked=sum(q['reference_reachable']=='False' and q['verdict']=='blocked' for q in rr),invalid=sum(q['path_status']=='invalid' for q in rr),unknown=sum(q['verdict']=='unverified' for q in rr),valid_over_1m=sum(q['path_status']=='valid' and q['path_length_m']>1. for q in rr),valid_single_node=sum(q['path_status']=='valid' and q['path_nodes']==1 for q in rr)))
        for comparison in [q for q in comparisons if q['condition']==condition]:
            clusters=[]
            for scene in range(10):
                clusters.append(np.stack([counts([q for q in runs if q['scene_seed']==scene and q['condition']==condition and q['label']==comparison[k]]) for k in ['b','a']]))
            clusters=np.stack(clusters);rng=np.random.default_rng(20261109);draws=rng.integers(10,size=(10000,10));boot=statistics(clusters[draws].sum(axis=1));delta=boot[:,1]-boot[:,0]
            for column,metric in enumerate(['valid_rate','correct_blocked_rate','balanced','invalid_count','wrong_path_rate','false_blocked_count','failed_drone_count']):
                values=delta[:,column];values=values[np.isfinite(values)]
                lo,hi=np.quantile(values,[.025,.975]) if len(values) else (np.nan,np.nan)
                intervals.append(dict(condition=condition,comparison=comparison['comparison'],metric=metric,clusters=10,draws=10000,finite_draws=len(values),lower=float(lo),upper=float(hi),seed=20261109,scope='paired building clusters of known simulator library, not real-fire generalization'))
    csv_write(data/'floors.csv',floor);csv_write(data/'paired_cluster_intervals.csv',intervals)
    lines=['# 관측 가치 × 드론 제어의 실제 비교','','선택 설정: **'+frozen['selected']['variant']+'**. 개발96회 뒤 동결하고 새 시드4에서80회 검증했다. 기존 제어/새 제어 각각에 동일 unknown과 선택entry를 적용했다. 기본 센서 오류·인체·기준 형상·목표는 유지했다. 원래 제어와 새 제어의 pitch 범위가 달라질 수 있으므로 목적함수만의 효과라고 합산하지 않는다.','']
    if not frozen['eligible']:lines+=['모든 새 설정이 개발의 무효 안내/차단 오판 제한을 만족하지 못했다. 아래 선택은 특성을 확인하는 평가이며 추천 설정이나 안전성 향상으로 부르지 않는다.','']
    lines+=['| 조건 | 제어/정책 | 유효/R | 정차단/B | 균형 | 무효/검증 제안 | 미확인/G | 판정/G | 차단 오판 | 실패 기체 |','|---|---|---|---|---|---|---|---|---|---|']
    for q in groups:
        if q['split']=='all':lines.append('| '+' | '.join(map(str,[q['condition'],q['label'],f"{q['valid']}/{q['reachable']}",f"{q['correct_blocked']}/{q['blocked']}",f"{100*q['balanced']:.2f}%",f"{q['invalid']}/{q['checked_paths']}" if q['checked_paths'] else 'N/A (0 proposals checked)',f"{q['unknown']}/{q['total_goals']}",f"{q['decided']}/{q['total_goals']}",q['false_blocked'],q['failed_drones']]))+' |')
    lines+=['','균형은 유효/R와 정차단/B의 평균이다. 무효 비율은 무효/(유효+무효로 검증된 제안)이며 미확인·고장 실행을 제외하지 않았다. 유효는 시스템이 실제 제시한 경로의 기준 메시 검증이다. 독립 평가기가 더 좋은 경로로 대체하지 않았다.','',f"논리 촬영{logical_captures:,}개, 정확 센서 재생{len(replays)}개, 실제 위치와 기록 궤적의 대조{sum(q['trajectory_position_checks'] for q in motion):,}개, p/v/a 연속 감속{sum(q['continuous_brakes'] for q in motion)}개를 감사했다. 정답과 유효 제안의 모순{len(contradictions)}건이다.",'','[실험 정의와 명령](METHOD.md) · [실행별](data/runs.csv) · [목표/실제 경로별](data/targets.csv) · [집계](data/summary.csv) · [제어와 가치 효과 분리](data/factorial_comparisons.csv) · [시간별](data/time.csv) · [동역학/실패 감사](data/motion_audit.csv).','', '원시 NPZ·지도·실제/추정 궤적·감속 다항식·배정·촬영 순서는 results/entry_control_20261109/validation 아래 실행별 replay_manifest.json을 따른다. 가상 짐벌·오프라인 동기 시뮬레이션이며 실제 소방 현장의 열·가스·구조 안정성이나 실제 비행 제어 성능을 검증한 결과가 아니다.','']
    (report/'RESULTS.md').write_text('\n'.join(lines));print('CONTROL FACTORIAL AUDIT AND REPORT COMPLETE',flush=True)


if __name__=='__main__':main()
