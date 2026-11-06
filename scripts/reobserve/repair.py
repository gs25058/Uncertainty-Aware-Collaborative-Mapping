#!/usr/bin/env python3
"""Ordered repair ablations: support evidence, relative height, body geometry, view selection.

Frozen original cases/frames; source artifacts read-only. Reference geometry is
accessed only in evaluation; future candidate observations only after selection.
"""
import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
import shutil

ROOT=Path(__file__).resolve().parents[2]
for p in (ROOT,ROOT/'scripts/smoke',ROOT/'scripts/reobserve',ROOT/'scripts/synth',ROOT/'scripts/entry'):
    sys.path.insert(0,str(p))
import gtsam  # before open3d
import cv2
import numpy as np
from covor.occupancy import StereoDepth,load_stereo_calib
from covor.entry.surfaces import SupportSurfaces,SurfaceCfg
from covor.entry.decision import DecisionGraph
from covor.entry.path_eval import ReferenceValidator,motion_path
from common import write_json,write_csv,make_builder,sparse_map,dense_map,digest
from evaluation import metrics

SOURCE=ROOT/'results/smoke_20261103/corridor'
STUDY=ROOT/'results/decision_paths_20261103'


def frozen():return json.load(open(STUDY/'cases_geometry.json'))


def depth(out,name,r,i,poses,sd):
    """Initial observations only; cache identical smoke/SGBM recomputation."""
    path=out/'depth_cache'/f'{name}_{r}_{i:05d}.npz'
    if path.exists():
        with np.load(path) as q:
            if 'pipeline' not in q or str(q['pipeline'])!='rectified_sgbm_v1':
                raise ValueError(f'{path}: legacy/unverified depth cache; use a new output directory')
            if str(q['input_sha256'])!=digest(SOURCE/'input_cache'/f'{r}_{i:05d}.npz'):
                raise ValueError('depth cache input hash differs')
            if str(q['design_sha256'])!=digest(SOURCE/'design.json'):
                raise ValueError('depth cache smoke/design settings differ')
            return q['z'],q['sz'],q['valid']
    with np.load(SOURCE/'input_cache'/f'{r}_{i:05d}.npz') as q:
        if name=='clean':return q['z'],q['sz'],q['valid']
        from covor.smoke import SmokeCfg,stereo_smoke
        cfg=json.load(open(SOURCE/'design.json'))['conditions'][name]
        l,rr,_=stereo_smoke(sd,q['left'],q['right'],q['z'],q['zr'],poses[r]['T_truth'][i],
                           float(poses[r]['t'][i]),SmokeCfg(**cfg))
    # Rendered and smoke-synthesized pairs are ALREADY rectified. Calling
    # sd.depth would remap them a second time and change the stereo experiment.
    from covor.synth.render import _depth_rectified
    z,sz,v=_depth_rectified(sd,l,rr)
    path.parent.mkdir(exist_ok=True)
    np.savez_compressed(path,z=z,sz=sz,valid=v,pipeline='rectified_sgbm_v1',
        input_sha256=digest(SOURCE/'input_cache'/f'{r}_{i:05d}.npz'),design_sha256=digest(SOURCE/'design.json'))
    return z,sz,v


def merged(surfaces):
    result=SupportSurfaces()
    for s in surfaces:
        for key,value in s.cells.items():result.cells.setdefault(key,np.zeros(5))[:]+=value
        result.n_frames+=s.n_frames
    return result


def surfaces(out,names):
    from run_corridor import teammates
    old=json.load(open(SOURCE/'design.json'))
    poses={r:dict(np.load(SOURCE/'poses'/f'{r}.npz')) for r in old['initial']}
    for name in names:
        common=SupportSurfaces()
        for r,indices in old['initial'].items():
            sd=StereoDepth(load_stereo_calib(r));s=SupportSurfaces()
            for n,i in enumerate(indices):
                z,sz,v=depth(out,name,r,i,poses,sd)
                mates=teammates(poses,r,poses[r]['t'][i]);T=poses[r]['T'][i]
                s.integrate(sd,z,sz,v,T,mates)
                common.integrate(sd,z,sz,v,T,mates)
                if (n+1)%25==0:print('surface',name,r,n+1,flush=True)
            p=out/'surfaces';p.mkdir(exist_ok=True)
            s.save(p/f'{name}_{r}.npz')
        common.save(out/'surfaces'/f'{name}_merged.npz')
        q=np.load(SOURCE/f'map_{name}_initial.npz')
        np.savez_compressed(out/f'map_{name}_support.npz',**dict(q))
    write_json(out/'surface_config.json',asdict(SurfaceCfg()))


def graph(q,layer=None,geometry=False):
    kwargs=dict(support_layer=layer) if layer is not None else {}
    if geometry:kwargs['surface_geometry']=True
    return DecisionGraph(q['labels'],q['ijk_min'],float(q['res']),**kwargs)


def variant_inputs(out,name,variant):
    if variant not in ('raw','support','surface_geometry','registered','geometry'):raise ValueError(variant)
    registered=variant in ('registered','geometry')
    path=(SOURCE/f'map_{name}_initial.npz' if variant=='raw' else
          out/f'map_{name}_{"registered" if registered else "support"}.npz')
    surface=None if variant=='raw' else out/'surfaces'/f'{name}_{"registered" if registered else "merged"}.npz'
    tree=out/f'initial_{name}_registered.ot' if registered else SOURCE/f'initial_{name}.ot'
    poses=out/'poses'/name if registered else SOURCE/'poses'
    return path,surface,tree,poses


def register(out,names):
    from covor.entry.registration import relative_heights,RegistrationCfg
    old=json.load(open(SOURCE/'design.json'))
    for name in names:
        inputs={r:SupportSurfaces.load(out/'surfaces'/f'{name}_{r}.npz') for r in old['initial']}
        offsets,pairs,status=relative_heights(inputs)
        # Corrections derive solely from the SAME condition's initial depth.
        write_json(out/f'registration_{name}.json',dict(offsets=offsets,pairs=pairs,status=status,
            cfg=asdict(RegistrationCfg()),truth_access=False,initial_frames_only=True,
            model='constant relative Z surface registration; no fusion reoptimization'))
        directory=out/'poses'/name;directory.mkdir(exist_ok=True,parents=True)
        for r,dz in offsets.items():
            with np.load(SOURCE/'poses'/f'{r}.npz') as p:
                data={k:p[k].copy() for k in p.files if k!='T_truth'}
            data['T'][:,2,3]+=dz;data['p_body'][:,2]+=dz
            np.savez_compressed(directory/f'{r}.npz',**data)
        print('REGISTER',name,offsets,status,flush=True)


def integrate(out,names):
    from run_corridor import teammates
    old=json.load(open(SOURCE/'design.json'))
    original={r:dict(np.load(SOURCE/'poses'/f'{r}.npz')) for r in old['initial']}
    for name in names:
        poses={r:dict(np.load(out/'poses'/name/f'{r}.npz')) for r in old['initial']}
        builder=make_builder();surfaces=SupportSurfaces()
        for r,indices in old['initial'].items():
            sd=StereoDepth(load_stereo_calib(r))
            for n,i in enumerate(indices):
                z,sz,v=depth(out,name,r,i,original,sd)
                pc,sp=sd.backproject(z,sz,v,downsample=4)
                mates=teammates(poses,r,poses[r]['t'][i]);T=poses[r]['T'][i]
                if v.sum()>=100:builder.integrate_frame(T,float(poses[r]['tr'][i]),pc,sp,teammates=mates)
                surfaces.integrate(sd,z,sz,v,T,mates)
                if (n+1)%25==0:print('integrate',name,r,n+1,flush=True)
        builder.finalize();builder.tree.write(str(out/f'initial_{name}_registered.ot').encode())
        q=np.load(SOURCE/f'map_{name}_initial.npz')
        lab=dense_map(sparse_map(builder),q['ijk_min'],q['labels'].shape)
        np.savez_compressed(out/f'map_{name}_registered.npz',labels=lab,ijk_min=q['ijk_min'],res=q['res'])
        surfaces.save(out/'surfaces'/f'{name}_registered.npz')


def assess(out,names,variants):
    from run import reference_scene
    from run_corridor import traversal
    from covor.entry import traverse as TV
    cfg=frozen();refs={r['case_id']:r for r in json.load(open(STUDY/'references.json'))}
    gt=np.load(ROOT/'results/synth_corridor915f/gt_voxel_fs.npz')
    scene,_=reference_scene();validator=ReferenceValidator(gt['labels'],gt['ijk_min'],.1,scene)
    target=traversal(gt['labels'],gt['ijk_min'],.1)
    rows=[];summaries=[];entry_rows=[];pose_rows=[]
    for name in names:
        for variant in variants:
            map_file,surface_file,_,_=variant_inputs(out,name,variant)
            q=np.load(map_file)
            layer=None
            if variant!='raw':
                s=SupportSurfaces.load(surface_file)
                layer=s.dense(q['ijk_min'],q['labels'].shape)
            g=graph(q,layer,variant in ('geometry','surface_geometry'));result=g.decide(cfg['entry'],cfg['goals'])
            directory=out/'evaluated_graphs';directory.mkdir(exist_ok=True)
            np.savez_compressed(directory/f'{name}_{variant}.npz',support=g.support,posture=g.cp,width=g.cw,
                foot_z=g.foot_z,ijk_min=q['ijk_min'],res=q['res'])
            if layer is not None:
                xx=(np.arange(q['labels'].shape[0])+q['ijk_min'][0]+.5)*.1
                yy=(np.arange(q['labels'].shape[1])+q['ijk_min'][1]+.5)*.1
                x0,y0,x1,y1=cfg['entry']['xy_bounds']
                for i in np.flatnonzero((xx>=x0)&(xx<=x1)):
                    for j in np.flatnonzero((yy>=y0)&(yy<=y1)):
                        for k in np.flatnonzero(layer['votes'][i,j]):
                            entry_rows.append(dict(condition=name,variant=variant,world_ijk=(np.array([i,j,k])+q['ijk_min']).tolist(),
                                height_m=float(layer['height'][i,j,k]),votes=int(layer['votes'][i,j,k]),
                                evidence=float(layer['evidence'][i,j,k]),support=bool(g.support[i,j,k]),
                                body_mode=int(g.cp[i,j,k]),volume_label=int(g.lab[i,j,k])))
            detail=[]
            for goal,d in zip(cfg['goals'],result['decisions']):
                path=motion_path(d['path']);check=validator.validate(path,cfg['entry'],goal) if path else dict(status='not_proposed',reason=d['reason'])
                f=out/'paths'/f'{name}_{variant}_{goal["old_id"]}.json'
                write_json(f,dict(decision=d,proposed_path=path,validation=check))
                detail.append(dict(condition=name,variant=variant,policy='initial',case_id=goal['id'],
                    reference_state=refs[goal['id']]['reference_state'],verdict=d['verdict'],reason=d['reason'],
                    path_status=check['status'],path_reason=check['reason'],path_file=str(f),
                    first_failure=json.dumps(check.get('first_failure')),additional_observations=0,move_cost=0))
            starts=g.mission_nodes(cfg['entry'],True);search=g.search(starts,True)
            summary=dict(condition=name,variant=variant,policy='initial',**metrics(detail),
                entry_confirmed_nodes=len(starts),entry_component_columns=int(g.area_result(search).sum()),
                support_cells=int(g.support.sum()),area_false_reach_denominator=int((~target['feasible']).sum()),
                **TV.score(traversal(q['labels'],q['ijk_min'],.1),target))
            rows+=detail;summaries.append(summary)
            print('ASSESS',name,variant,summary,flush=True)
        for r in json.load(open(SOURCE/'design.json'))['initial']:
            if not (out/'poses'/name/f'{r}.npz').exists():continue
            old=np.load(SOURCE/'poses'/f'{r}.npz');new=np.load(out/'poses'/name/f'{r}.npz')
            for arm,pose in (('original',old['T']),('registered',new['T'])):
                diff=pose[:,:3,3]-old['T_truth'][:,:3,3]
                pose_rows.append(dict(condition=name,robot=r,arm=arm,mean_dz_m=float(diff[:,2].mean()),
                    common_frame_position_rmse_m=float(np.sqrt(np.mean(np.sum(diff**2,axis=1))))))
    write_csv(out/'ablation_goals_paths.csv',rows);write_csv(out/'ablation_summary.csv',summaries)
    write_csv(out/'entry_evidence.csv',entry_rows);write_csv(out/'pose_evaluation.csv',pose_rows)


def policies(out,names,variant='geometry',evaluate=True):
    from run import proposals
    from covor.entry.assignment import choose
    old=json.load(open(SOURCE/'design.json'));cfg=frozen()
    calibrations={r:StereoDepth(load_stereo_calib(r)) for r in old['initial']}
    for name in names:
        corrections=(json.load(open(out/f'registration_{name}.json'))['offsets'] if variant=='geometry' else {r:0. for r in old['initial']})
        candidates=proposals(SOURCE,old)
        for c in candidates:
            c['T']=np.array(c['T']);c['T'][2,3]+=corrections[c['robot']]
        map_file,surface_file,_,_=variant_inputs(out,name,variant)
        q=np.load(map_file)
        s=SupportSurfaces.load(surface_file)
        print('select graph',name,flush=True)
        g=graph(q,s.dense(q['ijk_min'],q['labels'].shape),True)
        print('select influence',name,len(g.xyz),'nodes',len(g.segments),'segments',flush=True)
        count=[0]
        def progress(sid):
            count[0]+=1
            if count[0]%10==0:print('select cut',name,count[0],sid,flush=True)
        analysis=g.decide(cfg['entry'],cfg['goals'],with_influence=True,progress=progress)
        print('select scoring',name,flush=True)
        original,old_scores,old_reason=choose(g,analysis,candidates,calibrations)
        picks,scores,reason=choose(g,analysis,candidates,calibrations,single_view=True,surface_cfg=s.cfg)
        picks['decision_impact']=original['decision']
        tag='' if variant=='geometry' else '_'+variant
        write_json(out/('selection'+tag)/f'{name}.json',dict(picks=picks,scores=scores,reason=reason,variant=variant,
            original_scores=old_scores,original_reason=old_reason,candidates=candidates,
            segments=analysis['impact'],initial_decisions=analysis['decisions'],
            map_sha256=digest(map_file),surfaces_sha256=digest(surface_file),
            actual_candidate_data_access=False,reference_access=False,
            pose_provenance='retrospective batch-fused'+(' + initial-current-condition surface Z registration' if variant=='geometry' else '; original fixed poses'),
            min_evidence_is_not_probability=True))
        print('SELECT',name,picks,reason,flush=True)
    # All selections are now committed; evaluation may load real observations.
    if evaluate:policy_evaluate(out,names,variant)


def policy_evaluate(out,names,variant='geometry'):
    import octomap
    from run import reference_scene,observation
    from run_corridor import teammates,traversal
    from covor.smoke import SmokeCfg
    from covor.entry import traverse as TV
    cfg=frozen();old=json.load(open(SOURCE/'design.json'))
    refs={r['case_id']:r for r in json.load(open(STUDY/'references.json'))}
    gt=np.load(ROOT/'results/synth_corridor915f/gt_voxel_fs.npz');scene,_=reference_scene()
    validator=ReferenceValidator(gt['labels'],gt['ijk_min'],.1,scene)
    target=traversal(gt['labels'],gt['ijk_min'],.1)
    rows=[];summary=[];replay=[];renderers={}
    tag='' if variant=='geometry' else '_'+variant
    for name in names:
        log=json.load(open(out/('selection'+tag)/f'{name}.json'))
        map_file,surface_file,tree_file,pose_dir=variant_inputs(out,name,variant)
        assert log['map_sha256']==digest(map_file)
        assert log['surfaces_sha256']==digest(surface_file)
        candidates={c['id']:c for c in log['candidates']}
        initial=dict(np.load(map_file))
        q=initial;layer=SupportSurfaces.load(surface_file)
        variants={'initial':(q,layer,None,None)};outcomes={}
        poses={r:dict(np.load(pose_dir/f'{r}.npz')) for r in old['initial']}
        for policy,cid in log['picks'].items():
            c=candidates[cid];r=c['robot'];i=c['index']
            if cid not in outcomes:
                prior=STUDY/'observations'/f'{name}_candidate{cid}.npz'
                if not prior.exists():prior=out/'observations'/f'{name}_candidate{cid}.npz'
                if prior.exists():obs=dict(np.load(prior));actual_file=str(prior)
                else:
                    observation(SOURCE,out,name,c,SmokeCfg(**old['conditions'][name]),renderers)
                    prior=out/'observations'/f'{name}_candidate{cid}.npz'
                    obs=dict(np.load(prior));actual_file=str(prior)
                T=poses[r]['T'][i];t=poses[r]['t'][i];mates=teammates(poses,r,t)
                b=make_builder();b.tree=octomap.OcTree.read(str(tree_file).encode())
                assert np.array_equal(dense_map(sparse_map(b),q['ijk_min'],q['labels'].shape),initial['labels'])
                if obs['valid'].sum()>=100:b.integrate_frame(T,float(poses[r]['tr'][i]),obs['P_cam'],obs['sZ'],teammates=mates)
                lab=dense_map(sparse_map(b),q['ijk_min'],q['labels'].shape)
                s=SupportSurfaces.load(surface_file)
                sd=StereoDepth(load_stereo_calib(r))
                s.integrate(sd,obs['Z'],obs['sigma_Z'],obs['valid'],T,mates)
                outcomes[cid]=(dict(labels=lab,ijk_min=q['ijk_min'],res=q['res']),s,actual_file)
            post,s,actual_file=outcomes[cid];variants[policy]=(post,s,c,actual_file)
        before=None
        decision_cache={None:dict(decisions=log['initial_decisions'])};area_cache={}
        for policy,(q,s,c,actual_file) in variants.items():
            count=int(c is not None);key=None if c is None else c['id']
            if key not in decision_cache:
                print('policy graph',name,variant,key,flush=True)
                g=graph(q,s.dense(q['ijk_min'],q['labels'].shape),True)
                decision_cache[key]=g.decide(cfg['entry'],cfg['goals'])
            result=decision_cache[key];detail=[]
            snapshot=out/('maps'+tag)/f'{name}_{policy}.npz';snapshot.parent.mkdir(exist_ok=True)
            np.savez_compressed(snapshot,**q);s.save(snapshot.with_name(snapshot.stem+'_surfaces.npz'))
            for goal,d in zip(cfg['goals'],result['decisions']):
                path=motion_path(d['path']);check=validator.validate(path,cfg['entry'],goal) if path else dict(status='not_proposed',reason=d['reason'])
                path_file=out/('policy_paths'+tag)/f'{name}_{policy}_{goal["old_id"]}.json'
                write_json(path_file,dict(proposed_path=path,decision=d,validation=check))
                detail.append(dict(condition=name,variant=variant,policy=policy,case_id=goal['id'],
                    reference_state=refs[goal['id']]['reference_state'],verdict=d['verdict'],reason=d['reason'],
                    path_status=check['status'],path_reason=check['reason'],path_file=str(path_file),
                    first_failure=json.dumps(check.get('first_failure')),additional_observations=count,
                    move_cost=0 if c is None else c['move_cost'],candidate_id='' if c is None else c['id']))
            if policy=='initial':before=detail
            if key not in area_cache:area_cache[key]=TV.score(traversal(q['labels'],q['ijk_min'],.1),target)
            summary.append(dict(condition=name,variant=variant,policy=policy,additional_observations=count,
                candidate_id='' if c is None else c['id'],move_cost=0 if c is None else c['move_cost'],
                fallback=log['reason']['fallback'] if policy=='decision' else '',
                **metrics(detail,None if policy=='initial' else before),area_false_reach_denominator=int((~target['feasible']).sum()),
                **area_cache[key]))
            rows+=detail
            replay.append(dict(condition=name,policy=policy,initial_frame_order=old['initial'],
                initial_poses=str(pose_dir),candidate=c,actual_observation=actual_file,
                before_map=str(map_file),after_map=str(snapshot),
                additional_observations=count))
            print('POLICY',name,policy,'valid',summary[-1]['n_valid_paths'],'invalid',summary[-1]['n_invalid_paths'],flush=True)
            write_csv(out/f'policy{tag}_goals_paths.csv',rows);write_csv(out/f'policy{tag}_summary.csv',summary)
            write_json(out/f'replay{tag}.json',replay)


def report(out,names):
    def read(name):return list(csv.DictReader(open(out/name)))
    def table(rows,keys):
        result=['| '+' | '.join(keys)+' |','|'+'|'.join(['---']*len(keys))+'|']
        for row in rows:
            values=[]
            for key in keys:
                v=row.get(key,'')
                try:
                    x=float(v)
                    if np.isfinite(x) and not x.is_integer():v=f'{x:.4f}'
                except (ValueError,TypeError):pass
                values.append(str(v))
            result.append('| '+' | '.join(values)+' |')
        return '\n'.join(result)
    dest=ROOT/'reports'/out.name;dest.mkdir(exist_ok=True,parents=True)
    ablation=read('ablation_summary.csv');policy=read('policy_summary.csv')
    registrations={name:json.load(open(out/f'registration_{name}.json')) for name in names}
    old=json.load(open(SOURCE/'design.json'));cfg=frozen()
    write_json(out/'configuration.json',dict(scene='corridor915f',sequence='synth_corridor915f_0',seed=0,
        conditions={name:old['conditions'][name] for name in names},initial=old['initial'],
        cases=cfg,occupancy_cfg=old['occ_cfg'],surfaces=asdict(SurfaceCfg()),
        stage_order=['support','relative_z_registration','surface_body_geometry','single_view_selection'],
        registered_pose_scope='constant Z registration only; fusion backend unchanged',
        validation_scope='exploratory repair on diagnosed scene; not held-out evaluation'))
    count=0
    for folder in ('smoke_20261103','decision_paths_20261103','diagnosis_20261104'):
        root=ROOT/'reports'/folder
        for name,expected in json.load(open(root/'artifacts.json')).items():
            assert digest(root/name)==expected['sha256'],str(root/name)
            count+=1
    write_json(out/'integrity.json',dict(prior_report_artifacts_unchanged=count,
        code_sha256={str(p.relative_to(ROOT)):digest(p) for p in (
            ROOT/'covor/entry/surfaces.py',ROOT/'covor/entry/registration.py',ROOT/'covor/entry/decision.py',
            ROOT/'covor/entry/assignment.py',ROOT/'scripts/reobserve/repair.py',ROOT/'tests/test_support_surfaces.py')}))
    lines=f'''# 2 → 1 → 3 → 4 순서의 1차 수정과 실행 결과

같은 `corridor915f`, seed 0, 3대×100 초기 프레임, 고정 입구·22개 목표를 사용했다.
무처리와 기존 약한 비균일 모의 연기(`beta=0.15`, 실제 농도 아님)를 비교했다.
**발판 증거 분리·상대 높이 등록·연속 발 높이/자세·한 프레임 점수 옵션을 구현했다.
그러나 이 실행에서 유효 경로 확인 개선은 달성하지 못했다. 기본 실행 동작에는 채택하지 않았다.**
기준 도달 가능은 이전 메시 검증으로 고정한 10개, 기준 판정 불가는 12개, 기준 차단은 0개다.
성능 향상을 위해 목표, 입구, 기준 상태, 허용 범위를 변경하지 않았다.

## 2. 부피 점유와 발판 지지 증거 분리

`covor/entry/surfaces.py`는 정렬된 깊이의 이웃 평면으로 지지 표면을 추정한다.
세계 수평과 30도 이내인 표면만 위에서 본 관측으로 인정하며, 2.5 cm 평면 잔차,
5 m 거리 제한, 프레임당 셀별 1표 상한을 둔다. 깊이 가중치는 기존 depth-only 식이다.
최소 2프레임·순 증거 1.2·높이 표준편차 5 cm 이하를 확인 조건으로 고정했다.
이 값은 검증된 구조 안전 문턱이 아니라 첫 구현의 재현 가능한 설정이다.
표면 위 공기 광선은 지지를 취소하지 않는다. 이후 영상에서 해당 표면보다 5 cm 아래까지
측정 광선이 이어지고 예상 표면 깊이보다 max(10 cm, 3 sigma_Z) 길면 반대 증거를 반영한다.
같은 프레임 내부의 높이 분산도 보존하며, 여러 드론은 같은 누적층에 차례로 반영한다.
점유/빈 공간 복셀을 덮어쓰거나 발판 구멍을 보간해 확인으로 바꾸지 않는다.

첫 단계에서 지원 셀 수는 늘었으나 고정된 입구의 반복 관측 증거가 부족해 확인 경로는 0개였다.
이는 발판 표현의 단위 검증 통과와 임무 성능 달성이 다르다는 결과다.
`entry_evidence.csv`에 입구의 실제 표 수·증거·높이·몸 모드를 저장했다.
이번 구현의 최소 2프레임 규칙은 중요한 제약이다. 완전히 처음 보는 발판은 관측 한 번으로
확인될 수 없고, 기존 약한 표면 표가 남은 곳만 한 번 더 관측해 확인될 수 있다.
이 제약 때문에 생긴 성능 상한을 데이터 자체의 한계로만 돌리지 않는다.

## 1. 참 높이 없이 관측 표면으로 상대 Z 등록

`covor/entry/registration.py`는 같은 조건의 초기 관측에서 얻은 겹치는 XY 바닥 표면 차이로
드론별 상수 Z 보정만 추정한다. 첫 드론이 게이지이며 참 지도/참 포즈는 읽지 않는다.
40개 이상 겹침·차이 MAD 12 cm 이하·최대 보정 2 m·고리 잔차 15 cm 이하를 요구한다.
전체 연결이 확인되지 않으면 모든 보정을 거절한다. 각 조건의 설정과 거절 이유는
`registration_*.json`에 있다.

```json
{json.dumps({k:dict(offsets=v['offsets'],status=v['status']) for k,v in registrations.items()},ensure_ascii=False,indent=2)}
```

이것은 **지도 적분 전의 상대 높이 등록 옵션**이며 원래 융합 그래프를 고친 것은 아니다.
원래 포즈의 yaw/XY/시간 변화 오차, 첫 드론의 절대 높이 오차와 공분산은 남는다.
보정 불확실성을 융합 공분산에 반영하지 않았으므로 신뢰구간 개선도 주장하지 않는다.
연기에서 거절된 보정을 무처리 관측이나 참 높이로 대신하지 않았다.
기존 면적 재현율이 개선되더라도 오도달 증가를 함께 봐야 한다.

## 3. 발 높이·몸 높이·자세 일관성

`DecisionGraph(..., support_layer=..., surface_geometry=True)`에서 발 XYZ는 관측 표면 높이 +5 mm다.
몸의 상단이 걸치는 복셀까지 검사하고 머리 여유 2 cm를 요구한다. 최대 실제 높이 변화는
인체 설정의 18 cm이며, 기존 격자 반올림의 20 cm 허용과 구분한다.
몸 원판과 복셀의 사각 모서리가 실제로 겹치는지 검사하고 지도 밖은 관측된 빈 공간으로 세지 않는다.
미확인 발판 후보 높이는 인접 관측 표면의 높이로 제한하고, 임의의 ±2행 높이를 모두 세지 않는다.
단일 표면 표가 남아 있는 mixed free 복셀은 확인되지 않은 표면 후보로만 다룬다.
긍정적 표면 증거가 전혀 없는 관측된 빈 발판을 새 통로로 열지 않는다.
자세 전환을 포함한 제시 경로를 같은 독립 메시 평가기로 검증한다.
이 실행은 입구부터 확인된 경로가 없어 큰 장면에서 자세 수정의 경로 성공 효과까지 입증하지 못했다.

## 단계별 비교 — 기존 면적 지표와 새 목표 지표

{table(ablation,['condition','variant','support_cells','entry_confirmed_nodes','entry_component_columns','valid_path_confirmed','n_proposed','n_invalid_paths','reach_recall','reach_iou','false_reach_rate'])}

`raw`는 이전 확인 그래프, `support`는 지지 증거만 분리, `registered`는 상대 Z 등록까지,
`geometry`는 연속 발 높이·자세까지 적용한 누적 조건이다.
면적 지표는 기존 cleanup/gap 규약과 `TV.score`를 그대로 사용했다.
면적 재현율 분모는 기준 도달 열 수 8045, 면적 false-reach 분모는 기준에서 국소 몸이 들어가지
못하는 열 수 {ablation[0]['area_false_reach_denominator']}이다(목표의 잘못 안내한 경로 비율과 다르다).
목표 유효 경로 확인율 분모는 독립 검증 도달 목표 10개이고 판정률 분모는 22개다.
수치가 증가한 면적을 확인 그래프가 확보한 안전 경로처럼 해석하지 않는다.

## 4. 실제 추가 관측 한 번

같은 24개 후보·6 m 이동 제한을 사용한다. 최근접·미관측 공간·기존 구간 중요도 점수
(`decision_impact`)·새 단일 관측 점수(`decision`)를 함께 보존했다.
새 점수는 현재 표면 표 수와 증거에 한 프레임을 더해 확인 문턱에 도달할 수 있는지를
가중치로 사용하며, 같은 경로의 직렬 미확인 구간은 모두 관측할 수 있어야 높은 목표 점수를 준다.
참 결과를 예측 정답으로 넣지 않으며 점수는 확률로 보정된 값도 아니다.
양의 점수가 없으면 최근접으로 대체하고 로그에 남긴다.
현재 가능한 경로 하나의 구간을 사용한 규칙이므로, 다른 경로의 더 좋은 관측 기회를
찾는 최적화나 실제 스테레오 성공 확률 추정을 완료한 것은 아니다.

선택 로그를 먼저 저장하고 이후 선택한 관측 한 번만 초기 전체 log-odds 지도와 지지층의
복제본에 적분했다. 후보의 실제 깊이와 참 지도는 점수 계산 인터페이스에 없다.
포즈 카탈로그는 기존 전체 궤적 최적화 결과이므로 **오프라인 기록 시점 재생**이다.
초기 상대 높이 등록에도 초기 관측 전체를 사용했으며 온라인 비행 성능으로 해석하지 않는다.

{table(policy,['condition','policy','candidate_id','move_cost','additional_observations','valid_path_confirmed','n_proposed','n_invalid_paths','n_unverified','wrong_path_rate','decision_rate','unverified_to_valid_path','new_invalid_guidance'])}

잘못 안내한 경로 비율은 invalid/(valid+invalid)이고 분모 0은 N/A다.
제안이 0개인 조건의 오류 0개를 성능 개선으로 해석하지 않는다.
`policy_summary.csv`에는 올바른 차단, 오차단, 미확인 전환, 기존 오안내 취소와 유효 안내 손실도 포함했다.
기준 차단 목표가 없어 차단 정확도는 검증되지 않았다.

## 산출물과 재현

재사용: 기존 깨끗한 스테레오/SGBM 캐시, 융합 포즈, 초기 지도, 22개 목표와 메시 검증 기준,
선택 후 기존 후보 관측 캐시. 저장되지 않았던 연기 초기 깊이는 같은 합성 설정·시드로 SGBM을
다시 계산해 `depth_cache/`에 저장했다. 새로운 후보라면 선택 뒤에만 기존 렌더러로 관측을 생성했다.
부피 점유 지도는 동일한 `OccupancyBuilder`를 재사용했다.

수정/추가 코드: `covor/entry/surfaces.py`, `registration.py`, `decision.py`, `assignment.py`,
`scripts/reobserve/repair.py`, `tests/test_support_surfaces.py`.
기존 `traverse.py`의 면적 계산이나 융합 기본값은 이번 수정으로 바꾸지 않았다.
설정은 `configuration.json`, 목표·경로 상세는 `ablation_goals_paths.csv`와 `policy_goals_paths.csv`,
집계는 `ablation_summary.csv`와 `policy_summary.csv`, 선택은 `selection/*.json`이다.
경로 XYZ·자세는 `paths/`, `policy_paths/`, 전후 지도와 지지층은 `maps/`에 있다.
`replay.json`에 초기 입력 순서, 선택 프레임 ID·드론 포즈·관측 파일·지도 경로를 저장했다.

```bash
export MPLCONFIGDIR=/tmp/covor_mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
PY=/src/gs25058/miniconda3/envs/covor/bin/python
O=results/repair_20261104_replay
$PY scripts/reobserve/repair.py --out $O --stage surfaces
$PY scripts/reobserve/repair.py --out $O --stage register
$PY scripts/reobserve/repair.py --out $O --stage integrate
$PY scripts/reobserve/repair.py --out $O --stage policies
$PY scripts/reobserve/repair.py --out $O --stage assess --variants raw,support,registered,geometry
$PY scripts/reobserve/repair.py --out $O --stage report
$PY tests/test_support_surfaces.py
$PY tests/test_decision_paths.py
$PY tests/test_entry_traverse.py
```

같은 진단 장면을 사용한 탐색적 수정이며 별도 장면·시드로 일반화 성능을 확인한 실험은 아니다.
이번에는 MILUV와 나머지 연기 강도를 다시 실행하지 않았다. 모의 연기는 적외선 영상의
거리 의존 감쇠/대비 저하 모델이며 실제 화재 검증이 아니다.
기존 보고서 산출물 {count}개 해시 보존을 확인했다.
현재 결과로는 제안 방식이 미관측 공간 우선을 이긴다고 결론 낼 수 없다.
다음 수정도 정한 순서에 따라 발판 증거의 관측 범위와 고도 정합을 먼저 해결해야 한다.
'''
    (dest/'RESULTS.md').write_text(lines)
    for pattern in ('*.csv','*.json'):
        for p in out.glob(pattern):shutil.copyfile(p,dest/p.name)
    for folder in ('selection',):shutil.copytree(out/folder,dest/folder,dirs_exist_ok=True)
    manifest={str(p.relative_to(dest)):dict(bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(dest.rglob('*')) if p.is_file() and p.name!='artifacts.json'}
    write_json(dest/'artifacts.json',manifest)
    print('REPORT',dest/'RESULTS.md',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True);p.add_argument('--conditions',default='clean,heterogeneous_weak')
    p.add_argument('--stage',required=True,choices=('surfaces','assess','register','integrate','policies','select','policy-evaluate','report'))
    p.add_argument('--policy-variant',choices=('surface_geometry','geometry'),default='geometry')
    p.add_argument('--variants',default='raw,support')
    a=p.parse_args();out=Path(a.out).resolve();out.mkdir(exist_ok=True,parents=True)
    if out==SOURCE or SOURCE in out.parents or out==STUDY:raise ValueError('separate output required')
    cv2.setNumThreads(1);names=a.conditions.split(',')
    if a.stage=='surfaces':surfaces(out,names)
    elif a.stage=='assess':assess(out,names,a.variants.split(','))
    elif a.stage=='register':register(out,names)
    elif a.stage=='integrate':integrate(out,names)
    elif a.stage=='policies':policies(out,names,a.policy_variant)
    elif a.stage=='select':policies(out,names,a.policy_variant,False)
    elif a.stage=='policy-evaluate':policy_evaluate(out,names,a.policy_variant)
    elif a.stage=='report':report(out,names)
    else:raise NotImplementedError(a.stage)


if __name__=='__main__':main()
