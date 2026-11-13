#!/usr/bin/env python3
"""Finish reviewed human-mapping experiments without changing archived results."""
import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
for p in (ROOT,ROOT/'scripts/reobserve',ROOT/'scripts/smoke'):sys.path.insert(0,str(p))
import gtsam  # before open3d
import cv2
import numpy as np
from covor.occupancy import StereoDepth,load_stereo_calib
from covor.entry import traverse as TV
from covor.entry.path_eval import ReferenceValidator
from common import make_builder,sparse_map,dense_map,write_json,write_csv,digest,MAPPING_ARMS
import repair as R

NAMES=('clean','heterogeneous_weak')
VARIANTS=('raw','support','surface_geometry','registered','geometry')


def weights(out):
    """Optional volume-weight ablation; exactly the same 300 views per arm."""
    from run_corridor import teammates,traversal
    from covor.synth import metrics as ME
    old=json.load(open(R.SOURCE/'design.json'))
    poses={r:dict(np.load(R.SOURCE/'poses'/f'{r}.npz')) for r in old['initial']}
    gt=np.load(ROOT/'results/synth_corridor915f/gt_voxel_fs.npz')
    target=traversal(gt['labels'],gt['ijk_min'],float(gt['res']))
    rows=[];frames=[]
    for name in NAMES:
        builders={arm:make_builder(arm) for arm in MAPPING_ARMS}
        for r,indices in old['initial'].items():
            sd=StereoDepth(load_stereo_calib(r))
            for n,i in enumerate(indices):
                z,sz,v=R.depth(out,name,r,i,poses,sd)
                pc,sp=sd.backproject(z,sz,v,downsample=4)
                T=poses[r]['T'][i];tr=float(poses[r]['tr'][i]);t=poses[r]['t'][i]
                mates=teammates(poses,r,t)
                for arm,b in builders.items():
                    if v.sum()>=100:b.integrate_frame(T,tr,pc,sp,teammates=mates)
                    w=b.weight(tr,sp)
                    frames.append(dict(condition=name,arm=arm,robot=r,index=i,
                        valid_depth_fraction=float(v.mean()),pose_trace=tr,n_points=len(pc),
                        mean_ray_weight=float(w.mean()) if len(w) else None))
                if (n+1)%25==0:print('WEIGHTS',name,r,n+1,flush=True)
        for arm,b in builders.items():
            lab=dense_map(sparse_map(b),gt['ijk_min'],gt['labels'].shape)
            if arm=='depth_only':
                assert np.array_equal(lab,np.load(R.SOURCE/f'map_{name}_initial.npz')['labels']), 'corrected observation replay differs from archived baseline'
            path=out/'weight_maps'/f'{name}_{arm}.npz';path.parent.mkdir(exist_ok=True)
            np.savez_compressed(path,labels=lab,ijk_min=gt['ijk_min'],res=gt['res'])
            score=ME.score(lab==2,lab==1,gt['labels'],gt['ijk_min'],float(gt['res']),np.ones(lab.shape,bool))
            known=gt['labels']!=0
            row=dict(condition=name,arm=arm,n_observations=sum(map(len,old['initial'].values())),
                n_integrated_frames=b.n_frames,
                n_map_known_gt_domain=int(((lab!=0)&known).sum()),n_gt_known=int(known.sum()),
                known_completeness=float(((lab!=0)&known).sum()/known.sum()),
                **{'voxel_'+k:v for k,v in score.items()},
                **{'area_'+k:v for k,v in TV.score(traversal(lab,gt['ijk_min'],float(gt['res'])),target).items()})
            rows.append(row);print('WEIGHT_RESULT',name,arm,'iou',score['iou'],'false_free',score['false_free_rate'],'known',row['known_completeness'],flush=True)
            write_csv(out/'weighting_summary.csv',rows);write_csv(out/'weighting_frames.csv',frames)
    write_json(out/'weighting_complete.json',dict(complete=True,depth_only_labels_exact_against_archive=True,
        arms=MAPPING_ARMS,conditions=NAMES,domain='all fixed reference-known voxels; no method-dependent visibility mask',
        goal_path_validation='not repeated for volume-weight component; separate representation and policy experiments validate paths'))


def support_check(validator,points):
    """Batch version of ReferenceValidator.support; same tolerance and unknowns."""
    if not len(points):return np.empty(0,dtype='U12'),np.empty(0)
    tol=validator.cfg.support_tolerance_m
    starts=points+np.array([0,0,tol]);directions=np.tile([0.,0.,-1.],(len(points),1))
    hit=validator._cast(starts,directions)
    ind=np.floor((points-[0,0,validator.res*.5])/validator.res).astype(int)-validator.ijk
    inside=((ind>=0)&(ind<validator.lab.shape)).all(1)
    vals=np.zeros(len(points),np.uint8);vals[inside]=validator.lab[tuple(ind[inside].T)]
    state=np.full(len(points),'unverifiable',dtype='U12');state[vals==1]='invalid'
    valid=np.isfinite(hit)&(hit<=2*tol+1e-6);state[valid]='valid'
    return state,starts[:,2]-hit


def local_geometry(out):
    """Independent surface checks and fixed local human probes; no route replanning."""
    from run import reference_scene
    gt=np.load(ROOT/'results/synth_corridor915f/gt_voxel_fs.npz');res=float(gt['res']);ijk=gt['ijk_min']
    scene,_=reference_scene();validator=ReferenceValidator(gt['labels'],ijk,res,scene)
    # Freeze reference candidates BEFORE accessing any evaluated map graph.
    gt_nodes=np.argwhere(TV.supports(gt['labels']))
    pts=(gt_nodes+ijk+.5)*res;pts[:,2]=(gt_nodes[:,2]+ijk[2]+1)*res
    state,heights=support_check(validator,pts)
    candidates=gt_nodes[state=='valid'];ground=heights[state=='valid']
    # This scan contains one evaluated floor; probes span its observed support
    # surfaces. Vertical tolerance fixes correspondence, not a target snap.
    indices=np.linspace(0,len(candidates)-1,min(128,len(candidates))).astype(int)
    probes=[]
    for ident,n in enumerate(indices):
        node=candidates[n];p=(node+ijk+.5)*res;p[2]=ground[n]+.005
        body_result=[]
        for posture,width in ((p,w) for p in (0,1,2) for w in (0,1)):
            body_result.append(validator.body_clear(p,posture,width)[0])
        reference='feasible' if 'valid' in body_result else ('undetermined' if 'unverifiable' in body_result else 'not_feasible')
        probes.append(dict(id=ident,xy_ij=node[:2].tolist(),reference_xyz=p.tolist(),reference=reference,
            reference_modes=body_result,floor_bounds=[float(p[2]-.15),float(p[2]+.15)]))
    write_json(out/'local_cases.json',dict(probes=probes,n_gt_surface_candidates=len(gt_nodes),
        n_mesh_verified_supports=len(candidates),n_excluded_reference_supports=int((state!='valid').sum()),
        scope='fixed local configurations, not global entry reachability; no route replacement',
        deterministic='linspace of lexicographically ordered reference support candidates',
        support_tolerance_m=validator.cfg.support_tolerance_m))
    rows=[];summaries=[]
    for name in NAMES:
        for variant in VARIANTS:
            q=np.load(out/'evaluated_graphs'/f'{name}_{variant}.npz')
            nodes=np.argwhere(q['support']);points=(nodes+ijk+.5)*res
            if variant in ('geometry','surface_geometry'):
                # Strip the body's 5 mm contact offset for surface accuracy.
                points[:,2]=q['foot_z'][tuple(nodes.T)]-.005
            elif variant=='raw':points[:,2]=(nodes[:,2]+ijk[2]+1)*res
            else:
                _,surf,_,_=R.variant_inputs(out,name,variant)
                layer=R.SupportSurfaces.load(surf).dense(ijk,gt['labels'].shape)
                points[:,2]=layer['height'][tuple(nodes.T)]
            states,_=support_check(validator,points)
            lookup={}
            for node,p in zip(nodes,points):lookup.setdefault(tuple(node[:2]),[]).append(p[2])
            recovered=sum(any(abs(z-h)<=validator.cfg.support_tolerance_m for z in lookup.get(tuple(node[:2]),[]))
                          for node,h in zip(candidates,ground))
            detail=[]
            for probe in probes:
                i,j=probe['xy_ij'];lo,hi=probe['floor_bounds'];post=q['posture'][i,j];z=q['foot_z'][i,j]
                matches=np.flatnonzero((post>=0)&(z>=lo)&(z<=hi))
                check=dict(status='not_proposed',reason='observed_support_or_body_missing')
                point=None
                if len(matches):
                    k=min(matches,key=lambda k:(TV._factor(post[k],q['width'][i,j,k],TV.TravCfg()),z[k],int(k)))
                    xyz=[float((i+ijk[0]+.5)*res),float((j+ijk[1]+.5)*res),float(z[k])]
                    point=dict(xyz=xyz,posture=int(post[k]),width=int(q['width'][i,j,k]),observed_support=True)
                    region=dict(xy_bounds=[xyz[0]-.04,xyz[1]-.04,xyz[0]+.04,xyz[1]+.04],floor_z_bounds=[lo,hi])
                    check=validator.validate([point],region,region)
                detail.append(dict(condition=name,variant=variant,probe_id=probe['id'],reference=probe['reference'],
                    status=check['status'],reason=check['reason'],point=json.dumps(point),first_failure=json.dumps(check.get('first_failure'))))
            checked=int(np.isin(states,['valid','invalid']).sum());valid_ref=sum(p['reference']=='feasible' for p in probes)
            hit=sum(r['reference']=='feasible' and r['status']=='valid' for r in detail)
            verified=sum(r['status'] in ('valid','invalid') for r in detail);bad=sum(r['status']=='invalid' for r in detail)
            summaries.append(dict(condition=name,variant=variant,n_surface_claims=len(points),
                surface_valid=int((states=='valid').sum()),surface_invalid=int((states=='invalid').sum()),
                surface_unverifiable=int((states=='unverifiable').sum()),surface_checked=checked,
                surface_precision=float((states=='valid').sum()/checked) if checked else 'N/A',
                reference_mesh_supports=len(candidates),recovered_mesh_supports=int(recovered),surface_recall=recovered/max(len(candidates),1),
                n_local_probes=len(probes),n_reference_local_feasible=valid_ref,n_valid_local=hit,
                local_confirmation_rate=hit/valid_ref if valid_ref else 'N/A',n_local_proposed=sum(r['status']!='not_proposed' for r in detail),
                n_invalid_local=bad,n_unverifiable_local=sum(r['status']=='unverifiable' for r in detail),
                n_checked_local=verified,wrong_local_rate=bad/verified if verified else 'N/A'))
            rows+=detail
            print('LOCAL',name,variant,'surface recovered',recovered,'local valid',hit,'invalid',bad,flush=True)
            write_csv(out/'local_geometry_summary.csv',summaries);write_csv(out/'local_geometry_details.csv',rows)
    paired=[];examples=[]
    lookup={(r['condition'],r['variant'],r['probe_id']):r for r in rows}
    for name in NAMES:
        for a,b in (('raw','support'),('support','surface_geometry'),('surface_geometry','geometry')):
            pairs=[(lookup[name,a,p['id']],lookup[name,b,p['id']]) for p in probes]
            paired.append(dict(condition=name,before=a,after=b,n_probes=len(pairs),
                gained_valid=sum(x['status']!='valid' and y['status']=='valid' for x,y in pairs),
                lost_valid=sum(x['status']=='valid' and y['status']!='valid' for x,y in pairs),
                invalid_to_valid=sum(x['status']=='invalid' and y['status']=='valid' for x,y in pairs),
                invalid_to_no_proposal=sum(x['status']=='invalid' and y['status']=='not_proposed' for x,y in pairs),
                added_invalid=sum(x['status']!='invalid' and y['status']=='invalid' for x,y in pairs)))
            for kind,predicate in (
                ('invalid_to_valid',lambda x,y:x['status']=='invalid' and y['status']=='valid'),
                ('valid_lost',lambda x,y:x['status']=='valid' and y['status']!='valid'),
                ('added_invalid',lambda x,y:x['status']!='invalid' and y['status']=='invalid')):
                first=next(((x,y) for x,y in pairs if predicate(x,y)),None)
                if first:
                    x,y=first;examples.append(dict(kind=kind,condition=name,before_variant=a,after_variant=b,
                        probe=probes[x['probe_id']],before=x,after=y))
    write_csv(out/'local_geometry_changes.csv',paired);write_json(out/'representative_cases.json',examples)


def table(rows,keys):
    lines=['| '+' | '.join(keys)+' |','|'+'|'.join(['---']*len(keys))+'|']
    for row in rows:
        values=[]
        for k in keys:
            v=row.get(k,'')
            try:
                f=float(v)
                if np.isfinite(f) and not f.is_integer():v=f'{f:.4f}'
            except (ValueError,TypeError):pass
            values.append(str(v))
        lines.append('| '+' | '.join(values)+' |')
    return '\n'.join(lines)


def report(out):
    read=lambda name:list(csv.DictReader(open(out/name)))
    ablation=read('ablation_summary.csv');local=read('local_geometry_summary.csv')
    policies=read('policy_surface_geometry_summary.csv')+read('policy_summary.csv')
    weighting=read('weighting_summary.csv');paired=read('local_geometry_changes.csv')
    assert len(ablation)==len(NAMES)*len(VARIANTS)
    assert len(local)==len(ablation) and len(policies)==len(NAMES)*2*5 and len(weighting)==len(NAMES)*3
    assert json.load(open(out/'weighting_complete.json'))['depth_only_labels_exact_against_archive']
    dest=ROOT/'reports'/out.name;dest.mkdir(exist_ok=True,parents=True)
    review=json.load(open(out/'experiment_review.json'))
    configs=dict(review=review,source_design_sha256=digest(R.SOURCE/'design.json'),
        cases_sha256=digest(R.STUDY/'cases_geometry.json'),references_sha256=digest(R.STUDY/'references.json'),
        body=R.frozen()['body'],surface=R.asdict(R.SurfaceCfg()),
        geometry=dict(foot_contact_offset_m=.005,head_margin_m=.02,exact_step_limit_m=.18),
        local_probe_cases_sha256=digest(out/'local_cases.json'),
        mapping_arms={arm:R.asdict(make_builder(arm).cfg) for arm in MAPPING_ARMS})
    write_json(out/'configuration.json',configs)
    for source,name in ((R.SOURCE/'design.json','input_design.json'),(R.STUDY/'cases_geometry.json','frozen_cases.json'),
                        (R.STUDY/'references.json','references.json')):shutil.copy2(source,out/name)
    decisions=[
        dict(component='surface_support',action='keep_experimental',reason='Small local support recovery gain; no entry-to-goal valid paths; false and unverifiable surface claims remain.'),
        dict(component='surface_body_geometry',action='keep_validation_option',reason='Exact geometry exposes bad estimated foot heights; local successes and failures must both be retained; no uniform net improvement.'),
        dict(component='relative_z_registration',action='diagnostic_only',reason='Large area recall gain accompanies increased false reach and many invalid local configurations; common-floor assumption.'),
        dict(component='one_frame_readiness_score',action='do_not_promote',reason='Both conditions fall back to nearest; no verified target improvement demonstrated.'),
        dict(component='uncertainty_weighting',action='optional_ablation',reason='Full pose-depth weighting lowers voxel IoU/completeness and raises exact false-free in this scene; no universal best method established.')]
    write_csv(out/'adoption_decisions.csv',decisions)
    verified=0
    for folder in ('smoke_20261103','decision_paths_20261103','diagnosis_20261104'):
        root=ROOT/'reports'/folder
        for name,expected in json.load(open(root/'artifacts.json')).items():
            assert digest(root/name)==expected['sha256'],str(root/name);verified+=1
    previous=json.load(open(out/'previous_interrupted_artifacts.json')) if (out/'previous_interrupted_artifacts.json').exists() else {}
    for name,expected in previous.items():assert digest(name)==expected['sha256'],name
    source_checks=0
    for path,expected in json.load(open(R.SOURCE/'design.json'))['inputs'].items():
        assert digest(path)==expected,path;source_checks+=1
    map_checks=0
    for path in (R.STUDY/'selection').glob('*.json'):
        selection=json.load(open(path))
        assert digest(R.SOURCE/f"map_{selection['condition']}_initial.npz")==selection['current_map_sha256']
        map_checks+=1
    files=[ROOT/p for p in ('covor/entry/surfaces.py','covor/entry/registration.py','covor/entry/decision.py',
        'covor/entry/assignment.py','scripts/reobserve/repair.py','scripts/reobserve/reviewed.py',
        'scripts/smoke/common.py','tests/test_support_surfaces.py')]
    write_json(out/'integrity.json',dict(prior_report_artifacts_unchanged=verified,interrupted_artifacts_unchanged=len(previous),
        original_map_hashes_checked=map_checks,original_source_input_hashes_checked=source_checks,
        code_sha256={str(p.relative_to(ROOT)):digest(p) for p in files}))
    for name in NAMES:
        for folder in ('selection','selection_surface_geometry'):
            s=json.load(open(out/folder/f'{name}.json'))
            assert not s['actual_candidate_data_access'] and not s['reference_access']
        for root in ('policy_goals_paths.csv','policy_surface_geometry_goals_paths.csv'):
            rr=read(root)
            assert all(int(r['additional_observations'])==int(r['policy']!='initial') for r in rr)
    write_json(out/'complete.json',dict(complete=True,conditions=NAMES,
        completed_experiments=[e['id'] for e in review['experiments']],
        n_ablation_rows=len(ablation),n_policy_summary_rows=len(policies),n_weight_rows=len(weighting),
        n_goal_path_rows=len(read('ablation_goals_paths.csv'))+len(read('policy_goals_paths.csv'))+len(read('policy_surface_geometry_goals_paths.csv')),
        n_local_rows=len(read('local_geometry_details.csv')),all_appropriate_experiments_finished=True,
        no_new_main_method_adopted=True))
    examples=json.load(open(out/'representative_cases.json'))
    example_text='국소 구성에서 무효→유효 사례가 없었다.'
    if examples:
        e=examples[0]
        example_text=f"첫 대표 사례는 {e['condition']}의 probe {e['probe']['id']} ({e['kind']})이다. {e['before_variant']}의 {e['before']['point']} / {e['before']['reason']}에서 {e['after_variant']}의 {e['after']['point']} / {e['after']['reason']}로 바뀌었다. 상세 좌표와 첫 실패 위치는 `representative_cases.json`에 있다."
    case_info=json.load(open(out/'local_cases.json'))
    text=f'''# 사람용 3D 지도 목적에 맞춘 중단 실험 재검토·재개 결과

**적합하다고 판단한 실험을 끝까지 실행했다. 그러나 이번 설정에서 유효한 입구→목표 경로는
모든 방식이 0/10개였고, 주 방법으로 새 구성을 채택할 근거는 확보하지 못했다.**
원래 포즈의 표면 높이·몸 검사 조건은 두 조건 모두 국소 유효 구성 1/86개를 확인했지만,
제시한 국소 구성 6개 중 5개는 무효였다. 이는 경로 성공이 아니다.
상대 높이 등록은 면적 도달을 크게 늘리는 대신 오도달도 늘렸다.
포즈+깊이 가중도 이 장면에서는 무가중보다 3D IoU와 완성도가 낮고 exact false-free가 높았다.

## 판단과 실제 실행

목표는 모의 화재 센서 조건에서 여러 드론으로 **사람이 활용할 수 있는 3D 지도**를 만드는 것이다.
가중치나 특정 재관측 점수의 우수성을 전제하지 않았다. 동일한 `corridor915f`, seed 0,
3대×100 초기 스테레오 쌍, 고정 입구·22개 목표를 사용했다.

| 중단/미완료 실험 | 판단 | 이번 실행 |
|---|---|---|
| 부피 점유와 발판 표면 증거 분리 | 사람용 지도에 직접 관련 | 원래 포즈에서 기존 판정과 비교 완료 |
| 관측 발 높이·몸 폭·머리 공간 검증 | 사람의 기하학적 이동 판단에 직접 관련 | 높이 등록 없이 분리 검증 완료 |
| 겹치는 바닥을 이용한 상대 Z 등록 | 공통 바닥 가정이 있는 선택적 정합 대조 | 별도 조건으로만 완료 |
| 한 번의 추가 관측 배정 | 지도 개선 수단으로 관련 | 원래 포즈/등록 포즈 각각 동일 후보·예산으로 완료 |
| 무가중/깊이/포즈+깊이 가중 | 효과가 있을 때 채택할 구성 요소 | 같은 300쌍 입력으로 별도 대조 완료 |

무처리와 **약한 비균일 모의 연기 beta=0.15**를 실행했다. 나머지 연기 강도·MILUV·VIO 재실행,
추가 장면·시드·동시 배정·신경망은 이번 재개 범위에 넣지 않았다.
이번 실험은 같은 진단 장면에서 한 탐색적 비교이며 독립 장면의 일반화 검증은 아니다.

## 재개 전에 바로잡은 입력 오류

중단된 `repair.py`는 이미 정렬된 연기 영상을 `StereoDepth.depth`로 다시 정렬하고 있었다.
기존 정식 `run_corridor.py`/`run_miluv.py`의 `_depth_rectified` 처리에는 이 오류가 없다.
드론별 첫 프레임 감사에서 유효 마스크가 1109/533/9788픽셀 달랐다.
해당 중단 실험의 연기 깊이·표면층·등록·지도·선택은 재사용하지 않고 새로 만들었다.
깨끗한 입력과 검증된 깨끗한 표면/등록 결과는 해시와 함께 재사용했다.
수정 캐시는 입력·설정 해시와 `rectified_sgbm_v1` 표시를 요구한다. 회귀 테스트도 추가했다.
새 depth-only 재생의 두 조건 지도는 **기존 정식 지도와 분류 배열이 정확히 일치**했다.
따라서 기존 결과는 보존되며, 중단된 연기 높이 보정 거절 결과는 유효한 결론으로 사용하지 않는다.

## 1. 지도 표현과 목표 도달 — 분리된 비교

`raw`: 기존 점유 복셀 발판. `support`: 독립 표면 증거 + 기존 몸/격자 높이 판정.
`surface_geometry`: 원래 포즈 + 표면 높이/정확한 몸 원판-복셀 모서리 검사.
`registered`: 상대 Z 등록 + 표면 증거 + 기존 몸 판정.
`geometry`: 상대 Z 등록 + 표면 높이/몸 검사. 모든 조건의 진입점·목표·사람 크기는 같다.

{table(ablation,['condition','variant','support_cells','entry_confirmed_nodes','entry_component_columns','valid_path_confirmed','n_proposed','n_invalid_paths','reach_recall','reach_iou','false_reach_rate'])}

목표 유효 경로 확인율의 분모는 기준 메시에서 독립적으로 도달 경로를 확인한 10개다.
22개 중 12개는 기준 판정 불가, 기준 차단은 0개다. 차단 정확도는 이 표본에서 검증할 수 없다.
기존 면적 재현율의 분모는 기준 도달 열 8045개, 면적 false-reach의 분모는 기준에서 국소 몸이
들어가지 못하는 열 39582개다. 기존 cleanup/gap 규약의 면적 지표를 보존했다.
이 면적 증가를 유효한 제시 경로 증가로 해석하지 않는다.

## 2. 사람용 지도의 국소 유효성 — 경로 지표와 구분

고정된 메시 검증 지지점 {case_info['n_mesh_verified_supports']}개에 대해 같은 XY 열과 높이 오차
7.5 cm 이내에서 지지 회복률을 평가했다. 기준 복셀 후보 {case_info['n_gt_surface_candidates']}개 중
메시에서 지지를 검증하지 못한 {case_info['n_excluded_reference_supports']}개는 분모에서 제외했다.
예측 지지점 자체도 기준 메시의 수직 광선으로 유효/무효/검증 불가를 구분했다.
표면 precision의 분모는 valid+invalid이며 검증 불가 수를 함께 제공한다.

방법별 결과를 읽기 전에 같은 기준 지지점에서 128개 국소 검사 위치를 고정했다.
각 위치의 기준 몸 공간은 메시에서 3자세×2폭을 별도 검사했다. 평가 대상은 각 방법이
실제로 선언한 발 높이·자세·폭이며, 기준의 더 좋은 자세로 바꿔 검증하지 않는다.
XY는 같은 열이며 층 대응 높이 범위 ±0.15 m를 고정했다. 전역 목표 스냅은 바꾸지 않았다.
**이 표는 국소적으로 사람이 설 수 있는 구성의 유효성이고, 입구에서 도달한 경로 수가 아니다.**

{table(local,['condition','variant','n_surface_claims','surface_valid','surface_invalid','surface_unverifiable','surface_recall','n_reference_local_feasible','n_local_proposed','n_valid_local','n_invalid_local','n_unverifiable_local','local_confirmation_rate'])}

같은 검사 위치의 변화:

{table(paired,['condition','before','after','gained_valid','lost_valid','invalid_to_valid','invalid_to_no_proposal','added_invalid'])}

{example_text}

## 3. 추가 관측 한 번 — 실제 적분과 경로 재검사

같은 24개 후보와 드론별 마지막 초기 위치에서의 직선 이동 비용 ≤6 m를 사용했다.
실제 비행 경로 계획이나 온라인 실험이 아닌 **기록 시점 재생**이다. 후보 포즈는 기존 전체
궤적 최적화 결과이며 온라인 의사결정 성능으로 해석하지 않는다.
기존 융합은 전 궤적 GT 정렬 초기화도 포함한다. 이번 결과는 이 포즈 조건에서의 지도·판정
구성 요소 비교이며, 사전 인프라 없는 현장 위치추정 성능의 검증이 아니다.
두 포즈 조건의 모든 선택 로그를 먼저 확정하고, 선택한 영상·깊이와 기준 메시를 평가에 읽었다.
각 방식은 동일 초기 전체 log-odds/표면층의 독립 복제본에 관측 한 번만 더했다.

`nearest`: 최근접, `unknown`: 미관측 공간, `decision_impact`: 기존 구간 중요도/가시성,
`decision`: 단일 프레임 증거와 직렬 구간 가시성을 요구하는 보수적 대조다.
후자를 필수 제안 방법으로 채택하지 않았다. 최소 2프레임·순 표면 증거 1.2 규칙 때문에
전혀 처음 보는 발판은 한 번으로 확인되지 않는다는 구현상의 상한이 있다.

{table(policies,['condition','variant','policy','candidate_id','move_cost','additional_observations','valid_path_confirmed','n_proposed','n_invalid_paths','n_unverified','wrong_path_rate','unverified_to_valid_path','new_invalid_guidance'])}

잘못 안내한 경로 비율은 invalid/(valid+invalid), 분모 0은 N/A다.
제안 0건에서 오류 0건은 성능 개선이 아니다. 판정률은 (통과 가능+차단)/22이며 CSV에 있다.
보류→유효 경로, 보류→올바른 차단, 새 오안내, 기존 통과 취소, 오안내 취소와 유효 안내 손실도
집계 CSV에 저장했다. 이 결과로 재관측 우위를 주장할 수 있는지는 실제 목표·경로 수로 판단해야 한다.

## 4. 선택적 가중치 비교

{table(weighting,['condition','arm','n_observations','voxel_iou','voxel_iou_1vox','voxel_n_false_free','voxel_n_gt_occ_domain','voxel_false_free_rate','known_completeness','area_reach_recall','area_false_reach_rate'])}

같은 초기 300쌍·원래 포즈·지도 해상도·깊이·동료 제외를 사용했다. 지지층과 후보 선택을 바꾸는
실험이 아닌 **부피 점유 갱신 가중치의 구성 요소 대조**다. 포즈+깊이 가중의 이득을 가정하지 않았다.
3D IoU/false-free는 고정된 전체 기준 known 복셀 영역에서 계산했다. false-free 분모는
`voxel_n_gt_occ_domain`, 관측 완성도 분모는 `n_gt_known`이다. 미관측을 늘려 얻은 낮은 오류율도
같이 보이도록 완성도를 보고했다. 1복셀 허용 IoU도 따로 표시했다.
이 대조에서 목표 경로 검증을 다시 수행한 것은 아니며, 면적 지표만으로 사람 경로 유효성을 주장하지 않는다.

## 해석과 채택 범위

연구 목적에 맞는 실험은 완료했지만 모든 구성 요소를 주 방법으로 채택한 것은 아니다.
표면 지지와 몸 공간 검증은 계속 검토할 직접 관련 구성 요소다. 두 프레임 규칙은 검증 중인
보수적 설정이고 필수 안전 정의가 아니다. 상대 Z 등록은 한 층의 겹치는 바닥이라는 가정과
보정 공분산 미반영 문제가 있으므로 면적 재현율만 보고 채택하지 않는다.
불확실성 가중과 새 점수도 실제 지도 활용도·오안내·완성도·비용의 이득으로 판단한다.
임계값, 목표와 기준 상태를 성능에 맞춰 조정하지 않았다.

실험은 단일 스캔 장면·한 시드·약한 연기만 다룬다. 이번 재개에서는 드론 1/2/3대 비교나
동시 배정을 새로 검증하지 않았으므로 협업 규모에 따른 효율 개선을 추가로 주장하지 않는다.
현재 연기 모델은 적외선 영상의 거리 의존 감쇠·대비 저하 근사이며, 실제 화재·온도·가스 위험
검증이나 실제 연기 농도로의 환산이 아니다.

## 산출물과 재실행

판단 근거: `experiment_review.json`, 설정·입력 해시: `configuration.json`.
목표/제시 경로: `ablation_goals_paths.csv`, `policy*_goals_paths.csv`, `paths/`, `policy_paths*/`.
국소 지도: `local_geometry_details.csv`, `local_cases.json`, `representative_cases.json`.
선택: `selection*/`, 프레임 ID·드론 포즈·입력 순서·전후 지도: `replay*.json`, `maps*/`.
가중치: `weighting_summary.csv`, `weighting_frames.csv`, `weight_maps/`.
큰 지도와 경로는 `{out}`에 있고, 집계·설정·선택 로그는 이 보고서 옆에도 복사했다.
기존 보고서 {verified}개와 중단 산출물 {len(previous)}개의 해시 보존을 확인했다.
기존 지도 7개와 원본 설정/포즈/기준 입력 5개의 저장된 해시도 확인했다.
새 테스트 10건·기존 목표/경로 테스트 15건·기존 면적 탐색 테스트 11건이 통과했다.

이번 재개에서 수정/추가한 코드는 `scripts/reobserve/repair.py`(정렬 오류 수정·독립 대조·선택과
평가 분리·같은 후보 결과 재사용), `scripts/reobserve/reviewed.py`(가중치·국소 형상 평가·보고),
`tests/test_support_surfaces.py`(정렬 회귀), `configs/fire_mapping_resume.json`이다.
원래 융합·점유 갱신·면적 탐색의 기본 동작은 변경하지 않았다.

```bash
export MPLCONFIGDIR=/tmp/covor_mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
PY=python
O=results/fire_mapping_resume_20261104_replay
$PY scripts/reobserve/reviewed.py --out $O --stage prepare
$PY scripts/reobserve/repair.py --out $O --stage surfaces
$PY scripts/reobserve/repair.py --out $O --stage register
$PY scripts/reobserve/repair.py --out $O --stage integrate
for V in surface_geometry geometry; do
  $PY scripts/reobserve/repair.py --out $O --stage select --policy-variant $V
done
for V in surface_geometry geometry; do
  $PY scripts/reobserve/repair.py --out $O --stage policy-evaluate --policy-variant $V
done
$PY scripts/reobserve/repair.py --out $O --stage assess --variants raw,support,surface_geometry,registered,geometry
$PY scripts/reobserve/reviewed.py --out $O --stage weights
$PY scripts/reobserve/reviewed.py --out $O --stage local
$PY scripts/reobserve/reviewed.py --out $O --stage report
$PY tests/test_support_surfaces.py
$PY tests/test_decision_paths.py
$PY tests/test_entry_traverse.py
```
'''
    (dest/'RESULTS.md').write_text(text)
    for pattern in ('*.csv','*.json'):
        for p in out.glob(pattern):shutil.copy2(p,dest/p.name)
    for folder in ('selection','selection_surface_geometry'):
        shutil.copytree(out/folder,dest/folder,dirs_exist_ok=True)
    for name in ('support','decision','traverse'):
        p=Path(f'/tmp/fire_resume_{name}_tests.log')
        if p.exists():shutil.copy2(p,dest/p.name)
    manifest={str(p.relative_to(dest)):dict(bytes=p.stat().st_size,sha256=digest(p))
              for p in sorted(dest.rglob('*')) if p.is_file() and p.name!='artifacts.json'}
    write_json(dest/'artifacts.json',manifest)
    print('REPORT',dest/'RESULTS.md',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True)
    p.add_argument('--stage',choices=('prepare','weights','local','report'),required=True)
    args=p.parse_args();out=Path(args.out).resolve();cv2.setNumThreads(1)
    if args.stage=='prepare':
        out.mkdir(exist_ok=True,parents=True)
        shutil.copy2(ROOT/'configs/fire_mapping_resume.json',out/'experiment_review.json')
    elif args.stage=='weights':weights(out)
    elif args.stage=='local':local_geometry(out)
    else:report(out)


if __name__=='__main__':main()
