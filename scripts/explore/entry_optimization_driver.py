#!/usr/bin/env python3
"""Complete predeclared development, freeze a winner, then run untouched seeds.

Only completed development outcomes choose weights. No validation outcome is
fed back to selection, candidate generation, sensor, mapping, or decisions.
"""
import csv,datetime,fcntl,hashlib,json,os,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from covor.explore.execution_state import atomic_json
csv.field_size_limit(sys.maxsize)
root=Path('results/entry_optimization_20261109').resolve();report=Path('reports/entry_optimization_20261109').resolve();report.mkdir(exist_ok=True)
if (root/'user_scope_finish_current.json').exists() and '--resume-deferred' not in sys.argv:
 sys.exit('User closed this experiment scope. Deferred rollouts require an explicit new request and --resume-deferred.')
lock=(root/'driver.lock').open('a+')
try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError:sys.exit('optimization driver already running')
py=sys.executable
def status(phase,**kw):atomic_json(root/'pipeline_status.json',dict(phase=phase,updated=datetime.datetime.now().astimezone().isoformat(),**kw))
def wait_batch(name):
 while True:
  p=root/name/'status.json'
  if p.exists():
   q=json.load(open(p))
   if q['phase']=='failed':status('execution_error',batch=name,failures=q.get('failures'));raise RuntimeError(q)
   if q['phase']=='complete':return
  time.sleep(10)
def summarize(name,baseline=None):
 args=[py,'scripts/explore/report_entry_batch.py','--batch',str(root/name),'--out',str(report/name)]
 if baseline:args+=['--baseline',str(baseline)]
 with (root/(name+'_report.log')).open('w') as f:subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,check=True)
 return list(csv.DictReader(open(report/name/'summary.csv')))
def run_batch(name,spec,source=None,workers=24):
 path=root/(name+'_spec.json')
 if path.exists():assert json.load(open(path))==spec
 else:atomic_json(path,spec)
 args=[py,'scripts/explore/entry_optimization_batch.py','--spec',str(path),'--out',str(root/name),'--workers',str(workers),'--global-limit','44']
 if source:args+=['--source-from',str(source)]
 with (root/(name+'.log')).open('a') as f:subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,check=True)
def eligibility(rows,variant):
 parts=[q for q in rows if q['variant']==variant and q['condition']!='pooled']
 return bool(parts) and all(q.get('count_guardrail')=='True' for q in parts)
def key(q):return (-float(q['balanced_score']),int(q['invalid_paths']),int(q['failed_drones']),-int(q['decided']),q['variant'])

status('waiting_development_round02');wait_batch('round02');rows=summarize('round02');eligible=[q for q in rows if q['condition']=='pooled' and eligibility(rows,q['variant'])];pooled=[q for q in rows if q['condition']=='pooled'];parent=sorted(eligible or pooled,key=key)[0]['variant'];parent_cfg=json.load(open(root/'round02/configs'/f'{parent}.json'))
atomic_json(root/'round03_parent.json',dict(parent=parent,eligible=bool(eligible),selection_source='completed development round02 only',validation_unopened=True))
variants={}
for name,quality,precision,explore,scale in [
 ('anchored_v4',False,None,None,1.),('coverage_v4',True,None,None,1.),('precision_v4',True,.006,None,1.),
 ('minimal_mix_v4',True,.006,1.,.02),('explore_mix_v4',True,.006,1.,.1),('balanced_mix_v4',True,.006,1.,.3),('task_mix_v4',True,.006,1.,1.)]:
 cfg=json.loads(json.dumps(parent_cfg));cfg['utility']['entry_objective']='task_progress_v4';o=cfg['utility']['task_progress'];o.update(soft_route_progress=True,unknown_connection_penalty=4.,soft_path_max_expansions=50000,max_anchored_queries=8,anchored_weight=1.6,foot_sampling_quality=quality,foot_precision_scale_m=precision,mapping_sample_stride=4,effective_scan_views=4.,task_gain_scale=scale,contact_aware_foot_evidence=True)
 if explore is not None:o['exploration_weight']=explore
 path=root/'configs'/f'{name}.json'
 if path.exists():assert json.load(open(path))==cfg
 else:atomic_json(path,cfg)
 variants[name]=str(path)
spec=dict(variants=variants,scenes=[0,1,4,8],seeds=[0],conditions=['clean','strong'],duration=180.,memoize_motion=True,defer_dense=True)
status('development_round03',parent=parent,expected=56);run_batch('round03',spec,root/'planned_round03_source_v2');summarize('round03')
status('waiting_all_development');wait_batch('round01');summarize('round01');candidates=[]
for name in ['round01','round02','round03']:
 rows=summarize(name)
 for q in rows:
  if q['condition']=='pooled' and eligibility(rows,q['variant']):candidates.append(dict(q,batch=name))
if not candidates:status('no_eligible_new_variant');raise RuntimeError('Every candidate violated invalid/false-blocked count guardrails; retain all data and do not claim a winner')
best=sorted(candidates,key=key)[0];batch=root/best['batch'];cfg_path=batch/'configs'/f"{best['variant']}.json";cfg=json.load(open(cfg_path));source=batch/'source_snapshot'
winner=dict(selected=datetime.datetime.now().astimezone().isoformat(),batch=best['batch'],variant=best['variant'],development_metrics=best,config=str(cfg_path),config_sha256=hashlib.sha256(cfg_path.read_bytes()).hexdigest(),mathematical_source=str(source),source_manifest_sha256=hashlib.sha256((batch/'source_manifest.json').read_bytes()).hexdigest(),validation_seeds=[2,3],validation_unopened=True,no_validation_feedback_to_selection=True)
if (root/'winner_frozen.json').exists():
 previous=json.load(open(root/'winner_frozen.json'));assert previous['variant']==winner['variant'] and previous['config_sha256']==winner['config_sha256']
else:atomic_json(root/'winner_frozen.json',winner)
for label,c in [('unknown',cfg),('optimized',cfg),('legacy',json.load(open(root/'intake/source_snapshot/configs/fire_recon_20261107.json')))]:
 p=root/'configs'/f'validation_{label}.json'
 if p.exists():assert json.load(open(p))==c
 else:atomic_json(p,c)
for seed,labels in [(2,['unknown','legacy','optimized']),(3,['unknown','optimized'])]:
 name=f'validation_seed{seed}';spec=dict(variants={k:str(root/'configs'/f'validation_{k}.json') for k in labels},variant_policies={k:'unknown' if k=='unknown' else 'entry' for k in labels},scenes=list(range(10)),seeds=[seed],conditions=['clean','strong'],duration=180.,selection_enabled=False,memoize_motion=True,defer_dense=True,development_scenes=[0,1,4,8],reserved_scenes=[2,3,5,6,7,9],selected_variant=best['variant'],selected_batch=best['batch'])
 status(name,selected=best['variant'],expected=len(labels)*20);run_batch(name,spec,source,workers=28);summarize(name,root/name/'unknown')
status('simulation_complete_pending_final_audit',development_rounds=3,validation_trials=100,winner=best['variant']);print('ALL OPTIMIZATION AND SEALED VALIDATION SIMULATIONS COMPLETE',flush=True)
