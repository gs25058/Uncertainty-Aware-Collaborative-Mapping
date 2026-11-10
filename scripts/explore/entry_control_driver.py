#!/usr/bin/env python3
"""Separate control/utility factorial. Primary frozen trials are not modified."""
import ast,copy,csv,datetime,fcntl,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from covor.explore.execution_state import atomic_json
from covor.explore.runner import csv_write
root=Path('results/entry_control_20261109').resolve();report=Path('reports/entry_control_20261109').resolve()
if (root/'user_scope_finish_current.json').exists() and '--resume-deferred' not in sys.argv:
    sys.exit('User closed this experiment scope. Deferred rollouts require an explicit new request and --resume-deferred.')
lock=(root/'driver.lock').open('a+')
try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError:sys.exit('control study driver already active')
py=sys.executable


def status(phase,**extra):
    atomic_json(root/'pipeline_status.json',dict(phase=phase,updated=datetime.datetime.now().astimezone().isoformat(),**extra))


def wait_batch(name):
    while True:
        p=root/name/'status.json'
        if p.exists():
            q=json.load(open(p))
            if q['phase']=='failed':raise RuntimeError(q)
            if q['phase']=='complete':return
        time.sleep(10)


def run_batch(name,spec,source,workers=32):
    p=root/(name+'_spec.json')
    if p.exists():assert json.load(open(p))==spec
    else:atomic_json(p,spec)
    state=root/name/'status.json'
    if state.exists():
        existing=json.load(open(state))
        if existing['phase']=='complete':return
        process=Path('/proc')/str(existing.get('pid',-1))/'cmdline'
        if existing['phase']=='running' and process.exists():
            raw=process.read_bytes()
            if b'entry_optimization_batch.py' in raw and str(root/name).encode() in raw:
                wait_batch(name);return
    cmd=[py,'scripts/explore/entry_optimization_batch.py','--spec',str(p),'--out',str(root/name),
        '--workers',str(workers),'--global-limit','40','--source-from',str(source)]
    with (root/(name+'.log')).open('a') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)


def audit_gates():
    summaries=[];collisions=[]
    for p in sorted((root/'gates').glob('*/scene*/summary.json')):
        s=json.load(open(p));assert s['captures']>0 and s['travel_distance']>1.,(str(p),'no closed-loop movement')
        cfg=json.load(open(p.parent/'config.json'))['configuration'];trace=list(csv.DictReader(open(p.parent/'trajectories.csv')))
        previous={}
        for q in trace:
            d=int(q['drone']);t=float(q['time'])
            if d in previous:
                old=previous[d];dt=t-float(old['time'])
                dy=abs((float(q['yaw'])-float(old['yaw'])+np.pi)%(2*np.pi)-np.pi)
                dp=abs(float(q['gimbal_pitch'])-float(old['gimbal_pitch']))
                assert dy<=cfg['motion']['yaw_rate']*dt+1e-7,(p,d,t,'yaw rate')
                assert dp<=cfg['motion']['pitch_rate']*dt+1e-7,(p,d,t,'pitch rate')
            previous[d]=q
        summaries.append(dict(variant=p.parent.parent.name,**s))
        if s['failed_drones']:collisions.append(dict(run=str(p.parent),failed_drones=s['failed_drones']))
    assert len(summaries)==6
    csv_write(report/'gates.csv',summaries)
    atomic_json(root/'gates_audit.json',dict(runs=6,connected=True,camera_rate_limits=True,physical_failures=collisions))
    if collisions:
        review=root/'gates_failure_review.json'
        if not review.exists():
            status('gate_physical_failure_needs_diagnosis',cases=collisions)
            raise RuntimeError('Diagnose retained gate failures before extending the run')


def aggregate(rows):
    keys=['valid_confirmed','reachable_denominator','correct_blocked','blocked_denominator','invalid_paths','false_blocked','failed_drones','decided','total_goals']
    q={k:sum(r[k] for r in rows) for k in keys}
    q['balanced_score']=.5*(q['valid_confirmed']/q['reachable_denominator']+q['correct_blocked']/q['blocked_denominator'])
    return q


try:
    status('waiting_connection_gates');wait_batch('gates');audit_gates()
    spec=json.load(open(root/'development_spec.json'));status('development',expected=72)
    run_batch('development',spec,root/'gates/source_snapshot');wait_batch('development')
    cases={}
    for label in spec['variants']:
        cases[label]=[json.load(open(p)) for p in (root/'development'/label).glob('scene*/summary.json')]
        assert len(cases[label])==8
    candidates=[];summary=[]
    for label,rows in cases.items():
        q=aggregate(rows);q['variant']=label;backend=label.rsplit('_',1)[0];eligible=True
        for condition in spec['conditions']:
            own=aggregate([r for r in rows if r['condition']==condition])
            unknown=aggregate([r for r in cases[backend+'_unknown'] if r['condition']==condition])
            eligible&=own['invalid_paths']<=unknown['invalid_paths'] and own['false_blocked']<=unknown['false_blocked']
            summary.append(dict(variant=label,condition=condition,**own))
        q['eligible']=eligible;summary.append(dict(condition='pooled',**q))
        if not label.endswith('_unknown'):candidates.append(q)
    for q in candidates:q['batch']='development'
    csv_write(report/'development_summary.csv',summary)
    eligible=[q for q in candidates if q['eligible']]
    best=sorted(eligible or candidates,key=lambda q:(-q['balanced_score'],q['invalid_paths'],q['failed_drones'],-q['valid_confirmed'],q['variant']))[0]
    # Combine the selected controller with the independently chosen primary
    # DEVELOPMENT utility. Only winner_frozen/config/source are read, never
    # primary validation outcomes. This round was declared before any main
    # control-development outcome or seed4 validation was inspected.
    controller_cfg=json.load(open(root/'development/configs'/(best['variant']+'.json')))
    primary=Path('results/entry_optimization_20261109/winner_frozen.json')
    status('waiting_primary_development_selection',controller=best['variant'])
    while not primary.exists():time.sleep(10)
    selected_primary=json.load(open(primary));objective_source=Path(selected_primary['mathematical_source'])
    hybrid=root/'refinement_source';control_source=root/'development/source_snapshot'
    if not hybrid.exists():
        shutil.copytree(control_source,hybrid)
        for file in ['entry_objective.py','policy.py']:
            shutil.copy2(objective_source/'covor/explore'/file,hybrid/'covor/explore'/file)
        def candidate_ast(path):
            module=ast.parse(path.read_text());return ast.dump(next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='candidates'),include_attributes=False)
        assert candidate_ast(control_source/'covor/explore/policy.py')==candidate_ast(hybrid/'covor/explore/policy.py'),'candidate generator must remain common'
        atomic_json(root/'refinement_source_origin.json',dict(control=str(control_source),utility=str(objective_source),primary_selected_variant=selected_primary['variant'],primary_validation_not_read=True,overlays=['covor/explore/entry_objective.py','covor/explore/policy.py']))
    refined={};refined_policies={}
    for label in ['refined_unknown','refined_progress','refined_rate']:
        cfg=copy.deepcopy(controller_cfg);cfg['utility']=copy.deepcopy(json.load(open(selected_primary['config']))['utility'])
        if label=='refined_rate':cfg['utility']['offer_objective']='gain_rate'
        p=root/'configs'/(label+'.json')
        if p.exists():assert json.load(open(p))==cfg
        else:atomic_json(p,cfg)
        refined[label]=str(p);refined_policies[label]='unknown' if label.endswith('_unknown') else 'entry'
    refinement=dict(variants=refined,variant_policies=refined_policies,scenes=[0,1,4,8],seeds=[0],conditions=['clean','strong'],duration=180.,memoize_motion=True,defer_dense=True,selection_enabled=False,role='predeclared combination of independently selected development utility and selected control backend; no validation feedback')
    status('refinement',expected=24,controller=best['variant'],utility=selected_primary['variant']);run_batch('refinement',refinement,hybrid)
    refined_cases={label:[json.load(open(p)) for p in (root/'refinement'/label).glob('scene*/summary.json')] for label in refined}
    for label,rows in refined_cases.items():
        assert len(rows)==8;q=aggregate(rows);q.update(variant=label,batch='refinement');allowed=True
        for condition in ['clean','strong']:
            own=aggregate([r for r in rows if r['condition']==condition]);u=aggregate([r for r in refined_cases['refined_unknown'] if r['condition']==condition])
            allowed&=own['invalid_paths']<=u['invalid_paths'] and own['false_blocked']<=u['false_blocked']
            summary.append(dict(variant=label,condition=condition,**own))
        q['eligible']=allowed;summary.append(dict(condition='pooled',**q))
        if not label.endswith('_unknown'):candidates.append(q)
    csv_write(report/'development_summary.csv',summary);eligible=[q for q in candidates if q['eligible']]
    best=sorted(eligible or candidates,key=lambda q:(-q['balanced_score'],q['invalid_paths'],q['failed_drones'],-q['valid_confirmed'],q['variant']))[0]
    selected_batch=root/best['batch'];selected=selected_batch/'configs'/(best['variant']+'.json');new=json.load(open(selected))
    base=json.load(open('results/entry_optimization_20261109/round02/configs/contact_v3.json'))
    original=copy.deepcopy(base);original['utility']=copy.deepcopy(new['utility'])
    variants={};policies={}
    for label,cfg in [('original_unknown',original),('original_entry_selected',original),('new_unknown',new),('new_entry_selected',new)]:
        p=root/'configs'/('validation_'+label+'.json')
        if p.exists():assert json.load(open(p))==cfg
        else:atomic_json(p,cfg)
        variants[label]=str(p);policies[label]='unknown' if label.endswith('_unknown') else 'entry'
    frozen=dict(time=datetime.datetime.now().astimezone().isoformat(),selected=best,eligible=bool(eligible),
        no_eligible_label='characterization only, not a recommended winner' if not eligible else None,
        source=str(selected_batch/'source_snapshot'),source_manifest_sha256=hashlib.sha256((selected_batch/'source_manifest.json').read_bytes()).hexdigest(),
        configurations={k:dict(path=v,sha256=hashlib.sha256(Path(v).read_bytes()).hexdigest()) for k,v in variants.items()},
        validation_seed=4,validation_unopened=True,no_primary_validation_feedback=True)
    p=root/'winner_frozen.json'
    if p.exists():assert json.load(open(p))['selected']==best
    else:atomic_json(p,frozen)
    validation=dict(variants=variants,variant_policies=policies,scenes=list(range(10)),seeds=[4],conditions=['clean','strong'],
        duration=180.,memoize_motion=True,defer_dense=True,selection_enabled=False,role='frozen two-by-two control versus utility evaluation',
        development_scenes=[0,1,4,8],reserved_scenes=[2,3,5,6,7,9])
    status('validation',expected=80,selected=best['variant']);run_batch('validation',validation,selected_batch/'source_snapshot',workers=32)
    status('independent_audit')
    with (root/'final_report.log').open('w') as f:
        subprocess.run([py,'scripts/explore/report_entry_control.py','--root',str(root),'--report',str(report)],stdout=f,stderr=subprocess.STDOUT,check=True)
    status('complete',development_trials=96,validation_trials=80,gate_trials=6,independent_audit=True)
except Exception as exc:
    old=json.load(open(root/'pipeline_status.json'))
    if old['phase']!='gate_physical_failure_needs_diagnosis':status('execution_error',previous=old,error=str(exc))
    raise
