#!/usr/bin/env python3
"""Paired utility-development tables; select only after all declared trials finish."""
import argparse,csv,json,sys
from functools import lru_cache
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from covor.explore.runner import csv_write,write_json
csv.field_size_limit(sys.maxsize)
p=argparse.ArgumentParser();p.add_argument('--batch',required=True);p.add_argument('--baseline',default='results/fire_recon_20261108_resume/main');p.add_argument('--out',required=True);a=p.parse_args();root=Path(a.batch);base=Path(a.baseline);out=Path(a.out);out.mkdir(parents=True,exist_ok=True);spec=json.load(open(root/'fixed_spec.json'));duration=spec.get('duration',180.);runs=[];details=[];paired=[]
@lru_cache(maxsize=None)
def baseline(scene,seed,condition,policy):
 folder=base/f'scene{scene}_seed{seed}_{condition}_{policy}'
 if not (folder/'summary.json').exists():return None
 q=json.load(open(folder/'summary.json'))
 if q['duration']!=duration:
  rows=[r for r in csv.DictReader(open(folder/'metrics_time.csv')) if abs(float(r['time'])-duration)<1e-8]
  if not rows:return None
  q.update({k:(None if v in ('','None') else float(v)) for k,v in rows[0].items()});q['duration']=duration
  trace=[r for r in csv.DictReader(open(folder/'trajectories.csv')) if float(r['time'])<=duration]
  q['failed_drones']=len({r['drone'] for r in trace if r['failed']=='True'})
  q['captures']=sum(float(r['time'])<=duration for r in csv.DictReader(open(folder/'frames.csv')))
 return q
for variant in spec['variants']:
 for scene in spec['scenes']:
  for seed in spec['seeds']:
   for condition in spec['conditions']:
    policy=spec.get('variant_policies',{}).get(variant,'entry');folder=root/variant/f'scene{scene}_seed{seed}_{condition}_{policy}'
    if not (folder/'summary.json').exists():continue
    s=json.load(open(folder/'summary.json'));runs.append(dict(variant=variant,run=folder.name,**s))
    for row in csv.DictReader(open(folder/'target_paths.csv')):
     if abs(float(row['time'])-duration)<1e-8:details.append(dict(variant=variant,run=folder.name,**row))
    b=baseline(scene,seed,condition,'unknown')
    if b:
     paired.append(dict(variant=variant,scene=scene,seed=seed,condition=condition,valid_delta=s['valid_confirmed']-b['valid_confirmed'],blocked_delta=s['correct_blocked']-b['correct_blocked'],invalid_delta=s['invalid_paths']-b['invalid_paths'],false_blocked_delta=s['false_blocked']-b['false_blocked'],unknown_delta=s['unknown']-b['unknown']))
groups=[]
for variant in spec['variants']:
 for condition in [*spec['conditions'],'pooled']:
  rows=[q for q in runs if q['variant']==variant and (condition=='pooled' or q['condition']==condition)]
  if not rows:continue
  keys=['valid_confirmed','correct_blocked','invalid_paths','false_blocked','unknown','decided','proposed_paths','unverifiable_paths','reachable_denominator','blocked_denominator','total_goals','captures','failed_drones','fallback_seconds']
  q={k:sum(r[k] for r in rows) for k in keys};q.update(variant=variant,condition=condition,runs=len(rows),expected_runs=len(spec['scenes'])*len(spec['seeds'])*(len(spec['conditions']) if condition=='pooled' else 1));R=q['reachable_denominator'];B=q['blocked_denominator'];P=q['proposed_paths']-q['unverifiable_paths']
  q.update(valid_rate=q['valid_confirmed']/R if R else None,blocked_rate=q['correct_blocked']/B if B else None,wrong_path_rate=q['invalid_paths']/P if P else None,decision_rate=q['decided']/q['total_goals'],balanced_score=.5*(q['valid_confirmed']/R+q['correct_blocked']/B) if R and B else None)
  matched=[baseline(r['scene_seed'],r['run_seed'],r['condition'],'unknown') for r in rows]
  if all(b is not None for b in matched):
   q.update(unknown_valid=sum(b['valid_confirmed'] for b in matched),unknown_blocked=sum(b['correct_blocked'] for b in matched),unknown_invalid=sum(b['invalid_paths'] for b in matched),unknown_false_blocked=sum(b['false_blocked'] for b in matched),unknown_proposed=sum(b['proposed_paths'] for b in matched),unknown_failed_drones=sum(b['failed_drones'] for b in matched))
   q['count_guardrail']=q['invalid_paths']<=q['unknown_invalid'] and q['false_blocked']<=q['unknown_false_blocked']
   q['unknown_balanced_score']=.5*(q['unknown_valid']/R+q['unknown_blocked']/B) if R and B else None
  groups.append(q)
csv_write(out/'runs.csv',runs);csv_write(out/'targets.csv',details);csv_write(out/'summary.csv',groups);csv_write(out/'paired_deltas.csv',paired)
expected=len(spec['variants'])*len(spec['scenes'])*len(spec['seeds'])*len(spec['conditions']);complete=len(runs)==expected
selection=dict(all_declared_trials_complete=complete,actual=len(runs),expected=expected,ranked=[],winner=None)
if complete and spec.get('selection_enabled',duration>=180.):
 qualified=[]
 for q in groups:
  if q['condition']!='pooled':continue
  parts=[v for v in groups if v['variant']==q['variant'] and v['condition']!='pooled']
  if all(v.get('count_guardrail',False) for v in parts):qualified.append(q)
 ranked=sorted(qualified,key=lambda q:(-q['balanced_score'],q['invalid_paths'],q['failed_drones'],-q['decided'],q['variant']))
 selection['ranked']=[dict(variant=q['variant'],balanced_score=q['balanced_score'],valid=q['valid_confirmed'],correct_blocked=q['correct_blocked'],invalid=q['invalid_paths'],failed_drones=q['failed_drones'],improves_unknown=q['balanced_score']>q['unknown_balanced_score']) for q in ranked]
 if ranked:selection['winner']=ranked[0]['variant']
write_json(out/'selection.json',selection)
header=['variant','condition','runs','valid/R','correct_blocked/B','balanced','invalid/checked','unknown/G','failed_drones']
lines=['# Utility batch actual results','',f'Completed {len(runs)}/{expected} declared trials. Partial tables do not select a winner.','', '| '+' | '.join(header)+' |','|'+'|'.join(['---']*len(header))+'|']
for q in groups:
 P=q['proposed_paths']-q['unverifiable_paths'];lines.append('| '+' | '.join(map(str,[q['variant'],q['condition'],str(q['runs'])+'/'+str(q['expected_runs']),f"{q['valid_confirmed']}/{q['reachable_denominator']}",f"{q['correct_blocked']}/{q['blocked_denominator']}",q['balanced_score'],f"{q['invalid_paths']}/{P}",f"{q['unknown']}/{q['total_goals']}",q['failed_drones']]))+' |')
(out/'RESULTS.md').write_text('\n'.join(lines)+'\n');print(json.dumps(selection,indent=2),flush=True)
