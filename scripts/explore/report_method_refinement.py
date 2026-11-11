#!/usr/bin/env python3
"""Paired continuation metrics, referenced to the actual fork time, not t=0."""
import argparse,collections,csv,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from covor.explore.runner import csv_write,write_json


def read_csv(path):
    csv.field_size_limit(sys.maxsize)
    return list(csv.DictReader(path.open()))


def final_targets(path):
    rows=read_csv(path);t=max(float(r['time']) for r in rows)
    return {r['goal']:r for r in rows if float(r['time'])==t}


def aggregate(rows,group):
    count_keys=['valid_confirmed','reachable_denominator','correct_blocked','blocked_denominator','invalid_paths','verified_path_denominator','unknown','decided','total_goals','false_blocked','proposed_paths','unverifiable_paths','failed_drones','new_failed_drones','extra_captures','extra_distance','new_collision_events','unknown_to_valid','unknown_to_correct_blocked','new_invalid','revoked_invalid','lost_valid','revoked_passable','ever_invalid_goals','invalid_onset_events']
    q=dict(group=group,runs=len(rows),**{k:sum(r.get(k,0) for r in rows) for k in count_keys})
    q['valid_path_confirmation_rate']=q['valid_confirmed']/q['reachable_denominator']
    q['correct_blocked_rate']=q['correct_blocked']/q['blocked_denominator']
    q['balanced_score']=.5*(q['valid_path_confirmation_rate']+q['correct_blocked_rate'])
    q['wrong_path_rate']=q['invalid_paths']/q['verified_path_denominator'] if q['verified_path_denominator'] else None
    q['decision_rate']=q['decided']/q['total_goals']
    q['mean_case_balanced_score']=sum(r['balanced_score'] for r in rows)/len(rows)
    q['mean_branch_wall_seconds']=sum(r.get('wall_seconds',0) for r in rows)/len(rows)
    return q


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('results/method_refinement_20261110'))
    p.add_argument('--out',type=Path,default=Path('reports/method_refinement_20261110'));a=p.parse_args()
    study=a.root/'paired32';spec=json.loads((study/'fixed_spec.json').read_text());a.out.mkdir(parents=True,exist_ok=True)
    runs=[];targets=[];series=[];fairness=[];assignments=[]
    for job in spec['jobs']:
        folder=Path(job['out']);parent=Path(job['parent'])
        summary=json.loads((folder/'summary.json').read_text());before=json.loads((parent/'summary.json').read_text())
        manifest=json.loads((folder/'branch_manifest.json').read_text())
        old=final_targets(parent/'target_paths.csv');new=final_targets(folder/'target_paths.csv')
        counters=collections.Counter();local=[]
        history=read_csv(folder/'target_paths.csv');ever_invalid=set();onsets=0;previous=dict(old)
        for r in history:
            if float(r['time'])<=180:continue
            if r['path_status']=='invalid':
                ever_invalid.add(r['goal']);onsets+=previous[r['goal']]['path_status']!='invalid'
            previous[r['goal']]=r
        for goal,r in new.items():
            b=old[goal];ref=r['reference_reachable']=='True'
            transitions=dict(unknown_to_valid=b['verdict']=='unverified' and r['path_status']=='valid' and ref,
                unknown_to_correct_blocked=b['verdict']=='unverified' and r['verdict']=='blocked' and not ref,
                new_invalid=r['path_status']=='invalid' and b['path_status']!='invalid',
                revoked_invalid=b['path_status']=='invalid' and r['verdict']!='passable',
                lost_valid=b['path_status']=='valid' and r['path_status']!='valid',
                revoked_passable=b['verdict']=='passable' and r['verdict']!='passable')
            counters.update({k:int(v) for k,v in transitions.items()})
            row=dict(variant=job['variant'],scene=job['scene'],condition=job['condition'],goal=goal,
                reference_reachable=ref,before_verdict=b['verdict'],before_path_status=b['path_status'],
                after_verdict=r['verdict'],after_path_status=r['path_status'],path_reason=r['path_reason'],
                first_failure=r['first_failure'],path=json.loads(r['path']),**transitions)
            targets.append(row);local.append(row)
        csv_write(folder/'branch_transitions.csv',local)
        initial=folder/'maps/initial.npz'
        if not initial.exists():initial.write_bytes((parent/'maps/final.npz').read_bytes())
        row=dict(summary,variant=job['variant'],scene=job['scene'],source=str(folder),
            initial_valid=before['valid_confirmed'],initial_blocked=before['correct_blocked'],
            extra_captures=summary['captures']-before['captures'],extra_distance=summary['travel_distance']-before['travel_distance'],
            new_failed_drones=summary['failed_drones']-before['failed_drones'],
            new_collision_events=summary['collision_events']-before['collision_events'],ever_invalid_goals=len(ever_invalid),invalid_onset_events=onsets,**counters)
        runs.append(row)
        for r in read_csv(folder/'metrics_time.csv'):
            if float(r['time'])>=manifest['start_time']:
                series.append(dict(variant=job['variant'],scene=job['scene'],condition=job['condition'],**r))
        first=[];first_t=None
        for line in (folder/'candidates.jsonl').open():
            c=json.loads(line)
            if c['time']<=manifest['start_time']:continue
            if first_t is None:first_t=c['time']
            if c['time']!=first_t:break
            first.append({k:c.get(k) for k in ['id','position','yaw','pitch','kind','target_query','orientation_source']})
        fairness.append(dict(variant=job['variant'],scene=job['scene'],condition=job['condition'],first_assignment_time=first_t,
            candidate_pool_sha256=hashlib.sha256(json.dumps(first,sort_keys=True).encode()).hexdigest(),
            initial_map_hashes=manifest['initial_map_hashes'],initial_states=manifest['initial_states']))
        for line in (folder/'assignments.jsonl').open():
            q=json.loads(line)
            if q['time']>manifest['start_time']:
                assignments.append(dict(variant=job['variant'],scene=job['scene'],condition=job['condition'],time=q['time'],drone=q['drone'],candidate=q['candidate']['id'],kind=q['candidate']['kind'],position=q['candidate']['position'],yaw=q['candidate']['yaw'],score=q['score'],fallback=q['fallback']))
    for scene in spec['scenes']:
        for condition in spec['conditions']:
            group=[q for q in fairness if q['scene']==scene and q['condition']==condition]
            assert len(group)==3
            for key in ['initial_map_hashes','initial_states','candidate_pool_sha256','first_assignment_time']:
                assert len({json.dumps(q[key],sort_keys=True) for q in group})==1,(scene,condition,key)
    aggregates=[]
    for variant in spec['variants']:
        aggregates.append(aggregate([r for r in runs if r['variant']==variant],variant))
        for condition in spec['conditions']:
            aggregates.append(aggregate([r for r in runs if r['variant']==variant and r['condition']==condition],variant+'/'+condition))
    csv_write(a.out/'runs.csv',runs);csv_write(a.out/'targets.csv',targets);csv_write(a.out/'metrics_time.csv',series)
    csv_write(a.out/'summary.csv',aggregates);csv_write(a.out/'selected_actions.csv',assignments)
    floors=[]
    for variant in spec['variants']:
        for condition in spec['conditions']:
            for level in sorted({r['goal'].split('_')[0] for r in targets}):
                rows=[r for r in targets if r['variant']==variant and r['condition']==condition and r['goal'].split('_')[0]==level]
                if not rows:continue
                floors.append(dict(variant=variant,condition=condition,floor=level,
                    reachable_denominator=sum(r['reference_reachable'] for r in rows),blocked_denominator=sum(not r['reference_reachable'] for r in rows),
                    valid_confirmed=sum(r['reference_reachable'] and r['after_path_status']=='valid' for r in rows),
                    correct_blocked=sum(not r['reference_reachable'] and r['after_verdict']=='blocked' for r in rows),
                    invalid_paths=sum(r['after_path_status']=='invalid' for r in rows),unknown=sum(r['after_verdict']=='unverified' for r in rows)))
    csv_write(a.out/'floors.csv',floors)
    write_json(a.out/'fairness.json',dict(paired_cases=8,identical_initial_map_and_states=True,identical_first_candidate_pools=True,cases=fairness))
    offline=[]
    for origin in ['unknown','precision','gate']:
        folder=a.root/('fixed_map_'+origin)
        for r in read_csv(folder/'summary.csv'):
            offline.append(dict(origin=origin,**r))
    csv_write(a.out/'fixed_map_runs.csv',offline)
    offline_groups=[]
    for origin in ['unknown','precision','gate']:
        for method in ['shortest','observed_quality','complete_goal_cut','quality_and_cut']:
            rows=[r for r in offline if r['origin']==origin and r['method']==method]
            converted=[]
            for r in rows:
                converted.append({k:float(v) if v else 0 for k,v in r.items() if k not in ['source','method','condition','origin']})
            offline_groups.append(aggregate(converted,origin+'/'+method))
    csv_write(a.out/'fixed_map_summary.csv',offline_groups)
    strict_runs=[];strict_targets=[]
    for job,row in zip(spec['jobs'],runs):
        folder=Path(job['out']);dest=a.root/'full_goal_reassessment'/job['variant']/f'scene{job["scene"]}_{job["condition"]}'
        strict=next(r for r in read_csv(dest/'summary.csv') if r['method']=='complete_goal_cut')
        values={k:float(v) if v else 0 for k,v in strict.items() if k not in ('source','method','condition')}
        old=final_targets(a.root/'fixed_map_precision'/Path(job['parent']).name/'complete_goal_cut_paths.csv')
        new=final_targets(dest/folder.name/'complete_goal_cut_paths.csv');counts=collections.Counter()
        for goal,r in new.items():
            b=old[goal]
            transitions=dict(unknown_to_valid=b['verdict']=='unverified' and r['path_status']=='valid',
                unknown_to_correct_blocked=b['verdict']=='unverified' and r['verdict']=='blocked' and r['reference_reachable']=='False',
                new_invalid=r['path_status']=='invalid' and b['path_status']!='invalid',
                lost_valid=b['path_status']=='valid' and r['path_status']!='valid',
                revoked_passable=b['verdict']=='passable' and r['verdict']!='passable')
            counts.update({k:int(v) for k,v in transitions.items()})
            strict_targets.append(dict(variant=job['variant'],scene=job['scene'],condition=job['condition'],goal=goal,
                reference_reachable=r['reference_reachable'],before_verdict=b['verdict'],before_path_status=b['path_status'],after_verdict=r['verdict'],after_path_status=r['path_status'],certificate=r['certificate'],**transitions))
        strict_runs.append(dict(row,**{k:v for k,v in values.items() if k not in ('scene','extra_captures','extra_movement')},**counts))
    strict_summary=[aggregate([r for r in strict_runs if r['variant']==variant],variant) for variant in spec['variants']]
    csv_write(a.out/'whole_goal_cut_runs.csv',strict_runs);csv_write(a.out/'whole_goal_cut_targets.csv',strict_targets);csv_write(a.out/'whole_goal_cut_summary.csv',strict_summary)
    write_json(a.out/'completion.json',dict(closed_loop_runs=len(runs),expected_runs=24,fixed_map_runs=len(offline),fixed_map_expected=68,
        primary_endpoint='212s, common180s prefix plus32s causal continuation',test_result=json.loads((a.root/'tests.json').read_text()),
        source_manifests=[str(study/'source_manifest.json'),str(a.root/'fixed_map_source_manifest.json'),str(a.root/'reassessment_source_manifest.json')],
        final_map_whole_goal_reassessments=len(strict_runs),human_function_regression=json.loads((a.root/'human_function_regression.json').read_text()),
        not_independent_validation=True,original_results_preserved=True))
    assert len(runs)==24 and len(offline)==68
    print(json.dumps(aggregates,indent=2))


if __name__=='__main__':main()
