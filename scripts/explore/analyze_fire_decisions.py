#!/usr/bin/env python3
"""Post-hoc explanation of graph coverage and actual policy differences."""
import argparse,json,csv,sys
from pathlib import Path
from collections import Counter,defaultdict
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from covor.explore.runner import csv_write,write_json
p=argparse.ArgumentParser();p.add_argument('--root',default='results/fire_closed_loop_20261107/main');p.add_argument('--out',default='reports/fire_closed_loop_20261107/data');a=p.parse_args();root=Path(a.root);out=Path(a.out);out.mkdir(parents=True,exist_ok=True);rows=[];compact=[]
for folder in sorted(q.parent for q in root.glob('scene*/summary.json')):
 summary=json.load(open(folder/'summary.json'));epochs={}
 for line in open(folder/'candidates.jsonl'):
  q=json.loads(line);key=float(q['time']);stats=epochs.setdefault(key,dict(candidates=0,positive_candidates=0,max_target_score=0.,target_ids=set()))
  stats['candidates']+=1;positive={k:v for k,v in q['entry_target_scores'].items() if v>0};stats['positive_candidates']+=bool(positive);stats['max_target_score']=max(stats['max_target_score'],sum(positive.values()));stats['target_ids'].update(positive)
 for line in open(folder/'decisions.jsonl'):
  q=json.loads(line);ds=q['decisions'];stats=epochs.get(q['time']);possible=[d for d in ds if d['possible_path']];segments=set(s for d in possible for s in d['uncertain_segments']);cutsegments=set(d['segment'] for d in q['cut']);affected=set(g for goals in q['impact'].values() for g in goals)
  rows.append(dict(run=folder.name,scene=summary['scene_seed'],condition=summary['condition'],policy=summary['policy'],time=q['time'],role=q.get('role','assessment'),map_version=q['map_version'],total_goals=len(ds),confirmed_goals=sum(d['verdict']=='passable' for d in ds),possible_path_goals=len(possible),possible_but_unconfirmed_goals=sum(d['verdict']!='passable' for d in possible),unique_dependencies=len(segments),tested_dependency_segments=len(cutsegments),important_segments=len(q['impact']),affected_goals=len(affected),candidate_epoch=stats is not None,positive_entry_candidates=stats['positive_candidates'] if stats else '',entry_max_target_sum=stats['max_target_score'] if stats else '',goals_with_positive_visibility=sorted(stats['target_ids']) if stats else []))
  for d in ds:
   compact.append(dict(run=folder.name,time=q['time'],role=q.get('role','assessment'),goal=d['goal'],verdict=d['verdict'],reason=d['reason'],has_possible_path=bool(d['possible_path']),dependencies=d['uncertain_segments']))
csv_write(out/'decision_graph_coverage.csv',rows);csv_write(out/'target_state_history_compact.csv',compact);print('decision coverage states',len(rows),'target states',len(compact),flush=True)
