#!/usr/bin/env python3
"""Paired final-goal changes and scene-cluster uncertainty; no policy feedback."""
import argparse,csv,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from covor.explore.runner import csv_write
csv.field_size_limit(sys.maxsize)


def counts(rows):
    def total(k):return sum(float(q[k]) for q in rows)
    return np.array([total(k) for k in ('valid_confirmed','reachable_denominator',
        'correct_blocked','blocked_denominator','invalid_paths','verified_path_denominator',
        'false_blocked','failed_drones')])


def statistics(x):
    v,r,c,b,i,p,fb,failed=np.moveaxis(x,-1,0)
    invalid_rate=np.full_like(i,np.nan,dtype=float)
    np.divide(i,p,out=invalid_rate,where=p>0)
    return np.stack((v/r,c/b,.5*(v/r+c/b),i,invalid_rate,fb,failed),axis=-1)


def main():
    p=argparse.ArgumentParser();p.add_argument('--report',default='reports/entry_optimization_20261109');a=p.parse_args()
    report=Path(a.report);data=report/'data';runs=list(csv.DictReader(open(data/'validation_runs.csv')))
    targets=list(csv.DictReader(open(data/'validation_targets.csv')))
    run_index={(q['scene_seed'],q['run_seed'],q['condition'],q['label']):q for q in runs}
    target_index={(q['scene'],q['seed'],q['condition'],q['label'],q['goal']):q for q in targets}
    changes=[];paired=[];intervals=[]
    for comparator,seeds in [('unknown',['2','3']),('legacy',['2'])]:
        for q in targets:
            if q['label']!='optimized' or q['seed'] not in seeds:continue
            b=target_index[q['scene'],q['seed'],q['condition'],comparator,q['goal']]
            assert q['reference_reachable']==b['reference_reachable']
            ev=q['path_status']=='valid';bv=b['path_status']=='valid'
            ec=q['reference_reachable']=='False' and q['verdict']=='blocked'
            bc=b['reference_reachable']=='False' and b['verdict']=='blocked'
            changes.append(dict(comparator=comparator,scene=q['scene'],seed=q['seed'],condition=q['condition'],
                split=q['split'],goal=q['goal'],reference_reachable=q['reference_reachable'],
                baseline_verdict=b['verdict'],entry_verdict=q['verdict'],baseline_path_status=b['path_status'],entry_path_status=q['path_status'],
                gained_valid=ev and not bv,lost_valid=bv and not ev,
                gained_correct_blocked=ec and not bc,lost_correct_blocked=bc and not ec,
                new_invalid=q['path_status']=='invalid' and b['path_status']!='invalid',
                removed_invalid=b['path_status']=='invalid' and q['path_status']!='invalid',
                entry_path_length_m=q['proposed_path_length_m'],baseline_path_length_m=b['proposed_path_length_m'],
                entry_failure=q['first_failure'],baseline_failure=b['first_failure'],
                entry_path=q['path'],baseline_path=b['path']))
        for condition in ['clean','strong']:
            for split in ['all','reserved_geometry','development_geometry_new_seed']:
                subset=[q for q in runs if q['label']=='optimized' and q['run_seed'] in seeds and q['condition']==condition and (split=='all' or q['split']==split)]
                scene_ids=sorted({q['scene_seed'] for q in subset},key=int)
                clusters=[]
                for scene in scene_ids:
                    e=[q for q in subset if q['scene_seed']==scene]
                    b=[run_index[scene,q['run_seed'],condition,comparator] for q in e]
                    clusters.append(np.stack([counts(b),counts(e)]))
                clusters=np.stack(clusters);baseline,entry=statistics(clusters.sum(axis=0))
                delta=entry-baseline
                # Resample entire paired building clusters, retaining all its seeds.
                # Goals, seeds of one building, and policy outcomes are not treated
                # as independent samples. This quantifies only this simulator set.
                rng=np.random.default_rng(20261109);draws=rng.integers(len(clusters),size=(10000,len(clusters)))
                boot=statistics(clusters[draws].sum(axis=1));boot_delta=boot[:,1]-boot[:,0]
                keys=['valid_rate','correct_blocked_rate','balanced_score','invalid_count','wrong_path_rate','false_blocked_count','failed_drone_count']
                for k,key in enumerate(keys):
                    values=boot_delta[:,k];finite=values[np.isfinite(values)]
                    lo,hi=np.quantile(finite,[.025,.975]) if len(finite) else (np.nan,np.nan)
                    intervals.append(dict(comparator=comparator,condition=condition,split=split,metric=key,
                        scene_clusters=len(scene_ids),paired_runs=len(subset),baseline=float(baseline[k]),entry=float(entry[k]),
                        delta=float(delta[k]),scene_cluster_bootstrap_lower=float(lo),scene_cluster_bootstrap_upper=float(hi),
                        bootstrap_draws=10000,finite_draws=len(finite),bootstrap_seed=20261109))
                for q in subset:
                    b=run_index[q['scene_seed'],q['run_seed'],condition,comparator]
                    paired.append(dict(comparator=comparator,condition=condition,split=split,scene=q['scene_seed'],seed=q['run_seed'],
                        valid_delta=int(q['valid_confirmed'])-int(b['valid_confirmed']),
                        blocked_delta=int(q['correct_blocked'])-int(b['correct_blocked']),
                        invalid_delta=int(q['invalid_paths'])-int(b['invalid_paths']),
                        failed_delta=int(q['failed_drones'])-int(b['failed_drones'])))
    csv_write(data/'final_goal_changes.csv',changes);csv_write(data/'final_paired_comparators.csv',paired)
    csv_write(data/'paired_scene_cluster_intervals.csv',intervals)
    lines=['# 최종 목표별 쌍 비교와 변동 범위','','정책을 선택한 뒤의 검증 결과만 사용했다. unknown은 두 실행 시드, 기존entry는 실행 시드2와만 비교한다. 같은 건물의 모든 시드와 정책 쌍을 유지한 채 건물 단위로10,000회 재표집한 구간이다. 전체 라이브러리는10개, 조정 제외 부분은6개 건물뿐이다. 실제 화재나 새로운 건물 모집단에 대한 보장을 뜻하지 않는다. 잘못 안내한 경로 비율은 유효·무효 검증 제안이0이면 계산하지 않는다.','','| 비교 기준 | 조건 | 구분 | 추가 유효 | 잃은 유효 | 추가 정차단 | 잃은 정차단 | 새 무효 | 무효 취소 |','|---|---|---|---|---|---|---|---|---|']
    for comp in ['unknown','legacy']:
        for condition in ['clean','strong']:
            for split in ['all','reserved_geometry','development_geometry_new_seed']:
                rr=[q for q in changes if q['comparator']==comp and q['condition']==condition and (split=='all' or q['split']==split)]
                values=[sum(q[k] for q in rr) for k in ['gained_valid','lost_valid','gained_correct_blocked','lost_correct_blocked','new_invalid','removed_invalid']]
                lines.append('| '+' | '.join(map(str,[comp,condition,split,*values]))+' |')
    lines+=['','구간과 정확한 분자·분모는 paired_scene_cluster_intervals.csv와 validation_summary.csv에 있다. 개선한 목표와 잃은 목표를 각각 공개했으며, 목표별 순서 있는 실제 경로·첫 실패 위치는 final_goal_changes.csv에 보존했다. 평가기가 다른 경로를 대신 찾지 않았다.','']
    (report/'OUTCOME_ANALYSIS.md').write_text('\n'.join(lines));print('PAIRED GOAL OUTCOMES COMPLETE',flush=True)


if __name__=='__main__':main()
