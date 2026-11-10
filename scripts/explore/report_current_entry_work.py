#!/usr/bin/env python3
"""Audit/report only batches started before the user's finish-and-push request.

No experiments or candidate observations are selected here. All observations,
maps, paths and outcomes already exist. Sensor generation below is exact replay
for verification, after completed decisions, and cannot affect policy outcomes.
"""
import argparse,copy,csv,gzip,hashlib,io,json,shutil,sys,tarfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from covor.explore.execution_state import atomic_json
from covor.explore.fire_recon_scene import ReconFireWorld
from covor.explore.fire_world import FireWorld
from covor.explore.sensing import Sensor
from covor.explore.runner import csv_write
from covor.entry.path_eval import ReferenceValidator,PathEvalCfg
from covor.explore.execution_acceleration import enable_exact_motion_cache
from report_entry_control import audit_motion
csv.field_size_limit(sys.maxsize)


def aggregate(rows):
    keys=['valid_confirmed','reachable_denominator','correct_blocked','blocked_denominator','invalid_paths','verified_path_denominator','proposed_paths','unverifiable_paths','false_blocked','unknown','decided','total_goals','captures','failed_drones','collision_events','planning_failures','travel_distance','wall_seconds']
    q={k:sum(float(r[k]) for r in rows) for k in keys}
    for k in keys:
        if k not in ['travel_distance','wall_seconds']:q[k]=int(q[k])
    q['valid_rate']=q['valid_confirmed']/q['reachable_denominator'];q['blocked_rate']=q['correct_blocked']/q['blocked_denominator'];q['balanced_score']=.5*(q['valid_rate']+q['blocked_rate'])
    q['wrong_path_rate']=q['invalid_paths']/q['verified_path_denominator'] if q['verified_path_denominator'] else 'N/A'
    q['decision_rate']=q['decided']/q['total_goals'];return q


def archive_source(source,dest):
    if not dest.exists():
        files=sorted((source/'covor').rglob('*.py'))+[source/'scripts/explore/run_entry_trial.py']
        with dest.open('wb') as raw,gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as compressed,tarfile.open(fileobj=compressed,mode='w') as archive:
            for path in files:
                content=path.read_bytes();info=tarfile.TarInfo(str(path.relative_to(source)));info.size=len(content);info.mode=0o644;info.mtime=0;archive.addfile(info,io.BytesIO(content))
    return hashlib.sha256(dest.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',default='reports/entry_optimization_20261109/current');a=p.parse_args()
    report=Path(a.out);data=report/'data';data.mkdir(parents=True,exist_ok=True);(report/'sources').mkdir(exist_ok=True)
    primary=Path('results/entry_optimization_20261109');control=Path('results/entry_control_20261109');audit=primary/'current_scope_audit';audit.mkdir(exist_ok=True)
    base_cfg=json.load(open(primary/'intake/source_snapshot/configs/fire_recon_20261107.json'))
    declared=[('round01',primary/'round01',48,'pure'),('round02',primary/'round02',24,'pure'),('round03',primary/'round03',56,'pure'),('control',control/'development',72,'control')]
    sources={};case_list=[];not_started=[]
    protected=json.load(open(primary/'validation/shared_stack_unchanged.json'));original=json.load(open(primary/'intake/source_manifest.json'))
    for name,batch,expected,family in declared:
        state=json.load(open(batch/'status.json'));assert state['phase'] in ['complete','user_scope_closed'],(name,state)
        skipped=state.get('not_started',[]);assert state['finished']+len(skipped)==expected,(name,state)
        not_started.extend(dict(batch=name,**q) for q in skipped)
        manifest=json.load(open(batch/'source_manifest.json'));source=batch/'source_snapshot'
        assert all(hashlib.sha256((source/f).read_bytes()).hexdigest()==h for f,h in manifest.items())
        assert all(manifest[f]==original[f] for f in protected),('body/sensor/integrator changed',name)
        archive=report/'sources'/(name+'.tar.gz');sources[name]=dict(snapshot=str(source),archive=str(archive),archive_sha256=archive_source(source,archive),source_manifest=manifest)
        spec=json.load(open(batch/'fixed_spec.json'));folders=list(batch.glob('*/scene*/summary.json'));assert len(folders)==state['finished']
        for path in sorted(folders):case_list.append((family,path.parent.parent.name,path.parent,name))
    baseline=Path('results/fire_recon_20261108_resume/main')
    baseline_cases=[]
    for label in ['unknown','entry']:
        for scene in [0,1,4,8]:
            for condition in ['clean','strong']:baseline_cases.append(('baseline','original_'+label,baseline/f'scene{scene}_seed0_{condition}_{label}','previous_completed'))
    case_list=baseline_cases+case_list
    enable_exact_motion_cache();worlds={};validators={};runs=[];targets=[];time_rows=[];replay_rows=[];motion_rows=[];contradictions=[];first={};configs={}
    for family,label,folder,batch_name in case_list:
        s=json.load(open(folder/'summary.json'));meta=json.load(open(folder/'config.json'));cfg=meta['configuration'];scene=s['scene_seed'];condition=s['condition'];name=family+'/'+label+'/'+folder.name
        assert s['duration']==180. and s['n_drones']==4 and s['run_seed']==0
        assert cfg['human']==base_cfg['human']
        if family!='baseline':assert cfg['scene']==base_cfg['scene']
        allowed=('utility','protocol_version') if family=='pure' else ('utility','protocol_version','motion','sensor','candidates') if family=='control' else ('utility','protocol_version','sensor','scene','input_provenance')
        assert all(cfg[k]==v for k,v in base_cfg.items() if k not in allowed),('non-shared setting',name)
        if family=='baseline':
            a_sensor=copy.deepcopy(cfg['sensor']);b_sensor=copy.deepcopy(base_cfg['sensor'])
            a_sensor['fire_smoke'].pop('cache',None);b_sensor['fire_smoke'].pop('cache',None)
            assert a_sensor==b_sensor,'only storage locations may differ for reused baseline'
            assert all(cfg['scene'][k]==base_cfg['scene'][k] for k in ['type','fire_scale'])
        if family=='control':
            assert all(cfg['sensor'][k]==v for k,v in base_cfg['sensor'].items() if k!='gimbal_scan')
            assert all(cfg['candidates'][k]==v for k,v in base_cfg['candidates'].items() if k!='pitches')
            assert all(cfg['motion'][k]==base_cfg['motion'][k] for k in ['radius','margin','vmax','amax','yaw_rate','pitch_rate'])
        if scene not in worlds:
            b=ReconFireWorld(scene,base_cfg);gt,ijk=b.labels(cfg['map_resolution']);b.render_scene(False);worlds[scene]=b
            validators[scene]=ReferenceValidator(gt,ijk,cfg['map_resolution'],b.scene,cfg=PathEvalCfg(step_limit_m=cfg['human']['exact_step_limit'],contact_epsilon_m=cfg['human']['contact_epsilon_m']))
        b=worlds[scene];prior=json.load(open(folder/'prior.json'));assert prior==b.prior();ref={q['id']:q['reachable'] for q in b.reference(cfg['map_resolution'])[0]}
        if family=='baseline' and cfg['scene']!=base_cfg['scene']:
            # Previous two-floor runs were reused because no landing repair
            # changed their actual geometry. Do not demand identical cache-path
            # metadata; verify every component vertex/face and the fire sources.
            old=(ReconFireWorld if cfg['scene'].get('repair_bottom_landings') else FireWorld)(scene,cfg)
            assert len(old.components)==len(b.components) and old.fires==b.fires
            for x,y in zip(old.components,b.components):
                assert x['name']==y['name'];np.testing.assert_array_equal(x['vertices'],y['vertices']);np.testing.assert_array_equal(x['faces'],y['faces'])
        final=[q for q in csv.DictReader(open(folder/'target_paths.csv')) if float(q['time'])==180.];assert len(final)==s['total_goals']==len(ref)
        valid=invalid=correct=0
        for q in final:
            assert (q['reference_reachable']=='True')==ref[q['goal']]
            path=json.loads(q['path']);length=0.;nodes=len(path)
            if q['verdict']=='passable':
                goal=next(g for g in prior['goals'] if g['id']==q['goal']);v=validators[scene].validate(path,prior['entry'],goal)
                assert v['status']==q['path_status'],(name,q['goal'],v,q['path_status'])
                valid+=v['status']=='valid' and ref[q['goal']];invalid+=v['status']=='invalid'
                if v['status']=='valid' and not ref[q['goal']]:contradictions.append(dict(run=name,goal=q['goal']))
                if nodes>1:length=float(np.linalg.norm(np.diff(np.array([n['xyz'] for n in path]),axis=0),axis=1).sum())
            correct+=q['verdict']=='blocked' and not ref[q['goal']]
            targets.append(dict(family=family,label=label,source=str(folder),scene=scene,condition=condition,proposed_path_nodes=nodes,proposed_path_length_m=length,**q))
        assert (valid,invalid,correct)==(s['valid_confirmed'],s['invalid_paths'],s['correct_blocked'])
        trace,motion=audit_motion(folder,cfg);assert motion['persistent_failed_drones']==s['failed_drones'];motion_rows.append(dict(run=name,**motion))
        trace_index={(float(q['time']),int(q['drone'])):q for q in trace}
        frames=list(csv.DictReader(open(folder/'frames.csv')));assert len(frames)==s['captures'] and len({q['frame_id'] for q in frames})==len(frames)
        sensor=Sensor(b,cfg,scene*1000,condition)
        for row in [frames[0],frames[len(frames)//2]]:
            fid=row['frame_id'];t=float(row['time']);drone=int(row['drone']);pose=trace_index[t,drone]
            with np.load(folder/'frames'/(fid+'.npz')) as f:
                assert set(f.files)=={'Z','sigma','valid','T','time','drone_id','teammates'};saved={k:f[k].copy() for k in f.files}
            position=np.array([float(pose[k]) for k in ['x','y','z']]);dest=audit/'sensor_replays'/family/label/folder.name/'frames'
            _,z,sigma,mask,_=sensor.capture(position,float(pose['yaw']),float(pose['gimbal_pitch']),t,saved['teammates'],dest,fid,False,drone_id=drone)
            assert np.array_equal(z,saved['Z'],equal_nan=True) and np.array_equal(sigma,saved['sigma'],equal_nan=True) and np.array_equal(mask,saved['valid'])
            replay_rows.append(dict(run=name,frame=fid,depth_sigma_valid_exact=True))
            if row is frames[0]:
                profile=label.rsplit('_',1)[0] if family=='control' else 'original';key=(profile,scene,condition)
                signature=[hashlib.sha256(saved[k].tobytes()).hexdigest() for k in ['Z','sigma','valid','T']]
                if key in first:assert signature==first[key],('initial measurement differs',name)
                else:first[key]=signature
        for q in csv.DictReader(open(folder/'metrics_time.csv')):time_rows.append(dict(family=family,label=label,scene=scene,condition=condition,**q))
        runs.append(dict(family=family,label=label,source=str(folder),batch=batch_name,**s,**motion))
        if family!='baseline':
            cfg_path=Path('configs/entry_current_20261109')/family/(label+'.json');cfg_path.parent.mkdir(parents=True,exist_ok=True)
            if cfg_path.exists():assert json.load(open(cfg_path))==cfg
            else:atomic_json(cfg_path,cfg)
            configs[family+'/'+label]=dict(path=str(cfg_path),sha256=hashlib.sha256(cfg_path.read_bytes()).hexdigest(),source_archive=sources[batch_name]['archive'])
        print('AUDITED',name,flush=True)
    csv_write(data/'runs.csv',runs);csv_write(data/'targets.csv',targets);csv_write(data/'time.csv',time_rows);csv_write(data/'motion_audit.csv',motion_rows);csv_write(data/'sensor_replays.csv',replay_rows)
    csv_write(data/'not_started.csv',not_started)
    atomic_json(report/'source_manifest.json',dict(sources=sources,configurations=configs))
    checks=dict(new_full_missions=sum(q['family']!='baseline' for q in runs),previous_baseline_cases=16,logical_captures=sum(q['captures'] for q in runs),exact_sensor_replays=len(replay_rows),independently_revalidated_final_paths=sum(q['verdict']=='passable' for q in targets),reference_contradictions=contradictions,trajectory_position_checks=sum(q['trajectory_position_checks'] for q in motion_rows),persistent_failed_drones=sum(q['persistent_failed_drones'] for q in motion_rows),continuous_brake_plans_checked=sum(q['continuous_brakes'] for q in motion_rows),frozen_sources_unchanged=True,shared_physical_sensor_human_reference_rules_unchanged=True,all_failures_and_unknowns_retained=True,no_heldout_validation_executed=True)
    checks['not_started']=not_started
    atomic_json(report/'audit.json',checks);assert checks['new_full_missions']+len(not_started)==200 and not contradictions,checks
    groups=[]
    for family in ['baseline','pure','control']:
        for label in sorted({q['label'] for q in runs if q['family']==family}):
            for condition in ['clean','strong','pooled']:
                rr=[q for q in runs if q['family']==family and q['label']==label and (condition=='pooled' or q['condition']==condition)]
                groups.append(dict(family=family,label=label,condition=condition,actual_cases=len(rr),expected_cases=8 if condition=='pooled' else 4,**aggregate(rr)))
    index={(q['family'],q['label'],q['condition']):q for q in groups};rank=[]
    for q in groups:
        if q['family']=='baseline' or q['condition']!='pooled' or q['label'].endswith('_unknown'):continue
        comparator=q['label'].rsplit('_',1)[0]+'_unknown' if q['family']=='control' else 'original_unknown';comp_family='control' if q['family']=='control' else 'baseline';eligible=True
        own_runs=[r for r in runs if r['family']==q['family'] and r['label']==q['label']]
        base_runs=[r for r in runs if r['family']==comp_family and r['label']==comparator]
        paired={(r['scene_seed'],r['condition']) for r in own_runs}&{(r['scene_seed'],r['condition']) for r in base_runs}
        for condition in ['clean','strong']:
            own=aggregate([r for r in own_runs if r['condition']==condition and (r['scene_seed'],condition) in paired]);base=aggregate([r for r in base_runs if r['condition']==condition and (r['scene_seed'],condition) in paired])
            eligible&=own['invalid_paths']<=base['invalid_paths'] and own['false_blocked']<=base['false_blocked']
        paired_own=aggregate([r for r in own_runs if (r['scene_seed'],r['condition']) in paired]);paired_base=aggregate([r for r in base_runs if (r['scene_seed'],r['condition']) in paired])
        rank.append(dict(family=q['family'],label=q['label'],eligible=eligible,complete_comparison_cohort=len(paired)==8,paired_cases=len(paired),balanced_score=paired_own['balanced_score'],valid=paired_own['valid_confirmed'],correct_blocked=paired_own['correct_blocked'],invalid=paired_own['invalid_paths'],failed_drones=paired_own['failed_drones'],comparison=comparator,comparator_balanced=paired_base['balanced_score']))
    csv_write(data/'summary.csv',groups);csv_write(data/'development_rank.csv',rank)
    floors=[]
    for group in groups:
        if group['condition']=='pooled':continue
        for level in [1,2,3]:
            rr=[q for q in targets if q['family']==group['family'] and q['label']==group['label'] and q['condition']==group['condition'] and q['goal'].startswith(f'L{level}_')]
            floors.append(dict(family=group['family'],label=group['label'],condition=group['condition'],floor=level,reachable=sum(q['reference_reachable']=='True' for q in rr),blocked=sum(q['reference_reachable']=='False' for q in rr),valid=sum(q['path_status']=='valid' for q in rr),correct_blocked=sum(q['reference_reachable']=='False' and q['verdict']=='blocked' for q in rr),invalid=sum(q['path_status']=='invalid' for q in rr),unknown=sum(q['verdict']=='unverified' for q in rr),valid_over_1m=sum(q['path_status']=='valid' and q['proposed_path_length_m']>1. for q in rr),valid_single_node=sum(q['path_status']=='valid' and q['proposed_path_nodes']==1 for q in rr)))
    csv_write(data/'floors.csv',floors)
    best={}
    for family in ['pure','control']:
        pool=[q for q in rank if q['family']==family and q['complete_comparison_cohort']];eligible=[q for q in pool if q['eligible']]
        best[family]=sorted(eligible or pool,key=lambda q:(-q['balanced_score'],q['invalid'],q['failed_drones'],q['label']))[0]
    atomic_json(report/'development_selection.json',dict(selection=best,independent_validation_not_run=True,only_best_observed_development_configuration=True))
    lines=['# 현재 범위에서 완료한 진입 관측·조종 실험','', f"사용자 요청에 따라 시작한 범위까지만 마쳤다. **순수 목적함수128회와 조종/가치{checks['new_full_missions']-128}회, 총{checks['new_full_missions']}회의4대·180초 폐루프**를 실제 실행했다. 학교16초 연결 검사6회도 완료했다. 조종 조건1개는 /proc 조회 경쟁 오류로 촬영 전에 시작되지 않았으며 재실행하지 않았다. 이후 예정한 새 시드 검증100회, 결합 개발24회, 새 조종 검증80회는 실행하지 않았다. 아래 수치는 개발 장면0/1/4/8·실행 seed0의 결과이며 독립 검증이나 최종 최적 성능을 뜻하지 않는다.",'', '완전한4개 장면 조건의 분모는 도달 가능35개·도달 불가37개, 전체72개 목표다. guarded_unknown 강한 조건은3/4사례만 완료했으므로 실제 분모를 별도로 표시한다. guarded의 가치 비교는 두 방식 모두 완료한7/8사례로만 계산하며 완전한8사례 설정과 통합 순위를 매기지 않는다. 촬영 전 미실행과 실행 중 드론 실패는 구분했다. 무연기와 강한 통제 저하를 구분한다. 기준은 지정한 인체와 자세·18cm 걸음의 기하학적 통과 가능성이다. 실제 화재의 열·가스·구조 안정성을 평가하지 않았다.','']
    def table(title,rows):
        out=['## '+title,'','| 조건 | 방식 | 사례 | 유효/R | 정차단/B | 균형 | 무효/검증 제안 | 미확인/G | 판정/G | 오차단 | 실패 기체 |','|---|---|---|---|---|---|---|---|---|---|---|']
        for q in rows:out.append('| '+' | '.join(map(str,[q['condition'],q['label'],f"{q['actual_cases']}/{q['expected_cases']}",f"{q['valid_confirmed']}/{q['reachable_denominator']}",f"{q['correct_blocked']}/{q['blocked_denominator']}",f"{100*q['balanced_score']:.2f}%",f"{q['invalid_paths']}/{q['verified_path_denominator']}" if q['verified_path_denominator'] else 'N/A (0)',f"{q['unknown']}/{q['total_goals']}",f"{q['decided']}/{q['total_goals']}",q['false_blocked'],q['failed_drones']]))+' |')
        return out+['']
    lines+=['## 먼저 볼 결과','']
    for family,selected in best.items():
        delta=100*(selected['balanced_score']-selected['comparator_balanced'])
        lines.append(f"- {family}: 개발에서 선택한 {selected['label']}의 pooled 균형 점수는{100*selected['balanced_score']:.2f}%, 같은 비교의{selected['comparison']}은{100*selected['comparator_balanced']:.2f}%다(차이{delta:+.2f}%p). 무효/오차단 수 제한 충족 여부는{selected['eligible']}다. 이는 개발 선택 결과다.")
    lines+=['','아래에서 무연기/강한 저하와 비용을 따로 확인한다. pooled 향상만으로 두 조건의 일관된 우위라고 부르지 않는다.','']
    for family,title in [('baseline','기존 실제 실행 기준선'),('pure','공통 탐색 파이프라인의 목적함수 비교'),('control','공통 조종 조건 안에서의 가치 비교')]:lines+=table(title,[q for q in groups if q['family']==family and q['condition']!='pooled'])
    lines+=['## 해석과 제한','', '균형 점수는 유효/R와 정차단/B의 평균이다. 무효 비율은 무효/(유효+무효로 검증된 실제 제안)이다. 분모0은 N/A이며 전체 제안·검증 불가는 CSV에 있다. 미확인과 실패 드론을 제외하지 않았다. 입구와 겹친 한 점 경로도 분모에서 사후 제거하지 않았으며 경로 길이를 목표 CSV에 남겼다.','', '순수 비교는 센서·후보 생성·배정·이동·지도·사람 판정을 고정했다. 조종 비교의 heading/guarded/look_step은 별도 공통 백엔드다. 각 백엔드의 unknown/progress/rate를 비교해야 가치 함수 효과를 판단할 수 있다. look_step은 더 넓은 가상 짐벌 범위도 포함하므로 기존 시스템과의 차이를 목적함수만의 효과로 부르지 않는다.','', '전체를 본 뒤 개발 설정을 고른 것이므로 일반화 성능이나 통계적으로 독립된 우위로 해석하지 않는다. 최종 제시 경로를 기준 메시에서 다시 검증했으며 더 좋은 참 경로로 바꾸지 않았다. 보고에 없는 예정 실험은 미실행이다.','', '## 감사와 재현','',f"이 보고의{len(runs)}개 실행(새{checks['new_full_missions']}개+기존 기준16개)에서 논리 촬영{checks['logical_captures']:,}개를 확인했다. 센서 재생{checks['exact_sensor_replays']}개, 최종 실제 제시 경로{checks['independently_revalidated_final_paths']}개, 실제 위치/저장 궤적{checks['trajectory_position_checks']:,}개를 대조했다. 참 판정과 검증된 경로의 모순{len(contradictions)}건이다. 고장 기체는 복구/재생성하지 않았다.",'','[실행별](data/runs.csv) · [목표·실제 경로별](data/targets.csv) · [집계](data/summary.csv) · [개발 순위/제약](data/development_rank.csv) · [시간별](data/time.csv) · [동역학/실패 감사](data/motion_audit.csv) · [정확 센서 재생](data/sensor_replays.csv).','', '수학 소스와 실제 설정은 source_manifest.json 및 sources/*.tar.gz, configs/entry_current_20261109에 있다. 원본 깊이·지도·프레임 순서·포즈·배정·실패 기록은 서버 results 아래에 보존하며 대용량 원시 자료는 Git에 넣지 않는다. [순수 목적함수 방법](../METHOD.md), [새 조종 방법](../../entry_control_20261109/METHOD.md).','']
    diagnostic=control/'diagnostics';sampling=list(csv.DictReader(open(diagnostic/'school_support_sampling/comparison.csv')));ideal=json.load(open(diagnostic/'school_fixed_trajectory_ideal/ideal_depth/summary.json'));recorded=json.load(open(baseline/'scene1_seed0_clean_unknown/summary.json'))
    diagnostic_rows=[dict(label='recorded_noisy_stride4',**recorded),dict(label='fixed_trajectory_ideal_depth',**ideal)]+[dict(label='recorded_depth_stride'+q['surface_stride'],**q) for q in sampling]
    csv_write(data/'diagnostics.csv',diagnostic_rows)
    shutil.copy2(diagnostic/'school_support_sampling/baseline_verification.json',data/'support_replay_verification.json')
    lines+=['## 같은 기록 입력의 진단','','학교scene1·seed0·clean의 같은2836개 촬영 시각·궤적에서 비교했다. 아래는 새 정책의 폐루프 성능이 아니라 관측/지도 처리 진단이다.','',f"- 기록된 noisy 깊이: 유효{recorded['valid_confirmed']}/{recorded['reachable_denominator']}, 무효{recorded['invalid_paths']}. 같은 위치의 가림·거리 제한을 유지한 이상 깊이: 유효{ideal['valid_confirmed']}/{ideal['reachable_denominator']}, 무효{ideal['invalid_paths']}. 이상 깊이에서도 추가된 유효 경로는1층이며2/3층 연결은 확인하지 못했다."]
    for q in sampling:lines.append(f"- 같은 실제 noisy 깊이의 발판 샘플링 stride{q['surface_stride']}: 유효{q['valid_confirmed']}/{q['reachable_denominator']}, 정차단{q['correct_blocked']}/{q['blocked_denominator']}, 무효{q['invalid_paths']}, 확인 노드{q['n_confirmed']}. 점유/빈 공간과 all-hit 높이 최대값은 바꾸지 않았다.")
    lines+=['','stride4 재적분의 지원·높이·증거·투표·산포·XY7개 배열이 기존 결과와 정확히 같았다. 이 자료에서는 센서 오차 제거 또는 발판 샘플링 밀도 증가만으로 대다수 경로 실패가 해결되지 않았다. 관측 범위와 발판/몸 공간 연결 확인의 문제를 함께 남겨 둔다.','']
    checks_folder=report/'checks';checks_folder.mkdir(exist_ok=True)
    for path in (control/'checks').glob('*'):
        if path.is_file():shutil.copy2(path,checks_folder/path.name)
    lines+=['핵심 단위 검사39개와 깊이/장면 회귀21개, 총60개 및 학교16초 연결6회가 통과했다. 추가 검증100회·결합24회·조종 검증80회는 사용자 요청으로 미실행이며, 과거 계획서의 예정 수를 완료 수로 사용하지 않았다.','']
    (report/'RESULTS.md').write_text('\n'.join(lines))
    reproduce=['# 현재 완료 범위의 재현','','대용량 입력·깊이·지도는 서버 results에 보존한다. 외부 FUEL native library와 환경은 기존 [외부 코드 감사](../../closed_loop_20261105/EXTERNAL_AUDIT.md) 및 [지도 입력 문서](../../../OCCUPANCY_PIPELINE.md)를 따른다. 실제 수학 소스는 동결 아카이브, 실행 설정은 configs/entry_current_20261109다.','', '```bash','export COVOR_PY=/src/gs25058/miniconda3/envs/covor/bin/python','export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1','mkdir -p results/current_source_replay/control','tar -xzf reports/entry_optimization_20261109/current/sources/control.tar.gz -C results/current_source_replay/control','$COVOR_PY results/current_source_replay/control/scripts/explore/run_entry_trial.py \\', '  --config configs/entry_current_20261109/control/heading_rate.json \\', '  --out results/current_reproduction/heading_rate_scene1_clean_16s \\', '  --scene 1 --seed 0 --condition clean --policy entry --duration 16 \\', '  --memoize-motion --defer-dense','```','','180초 재현은 새 출력 경로와 `--duration 180`을 사용한다. 같은 source archive를 유지해야 체크포인트를 재개할 수 있다. 현재 사용자 종료 범위 뒤의 전체 driver/validation을 자동 실행하지 않는다.','', '보고·감사는 새 관측 배정을 실행하지 않는다:','', '```bash','$COVOR_PY scripts/explore/report_current_entry_work.py','```','', '실행별 frames.csv가 촬영 순서·시각·지도 버전·실제/추정 포즈를, trajectories.csv가 실제 이동과 짐벌 방향을, assignments.jsonl/events.csv가 계획·감속·정지·실패를 보존한다.','']
    (report/'REPRODUCE.md').write_text('\n'.join(reproduce))
    atomic_json(primary/'current_scope_audit_complete.json',dict(report=str(report/'RESULTS.md'),audit=str(report/'audit.json'),new_full_missions=checks['new_full_missions']))
    print('CURRENT USER SCOPE REPORT COMPLETE; NO NEW EXPERIMENT WAS LAUNCHED',flush=True)


if __name__=='__main__':main()
