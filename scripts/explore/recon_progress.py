#!/usr/bin/env python3
"""Read-only common-checkpoint progress, never mix final and partial scores."""
import csv,json,re,statistics
from pathlib import Path
from collections import defaultdict
root=Path('results/fire_recon_20261107');main=root/'main';cfg=json.load(open(main/'fixed_config.json'));series={};fresh_times=[];fresh_complete=0;reused=0;errors=[]
for i in cfg['scene_seeds']:
 for seed in cfg['run_seeds']:
  for c in cfg['conditions']:
   for p in cfg['policies']:
    name=f'scene{i}_seed{seed}_{c}_{p}';folder=main/name;reuse=(folder/'reuse_provenance.json').exists();done=(folder/'summary.json').exists();reused+=reuse;fresh_complete+=done and not reuse
    if done:
     data={float(q['time']):dict(V=int(q['valid_confirmed']),C=int(q['correct_blocked'])) for q in csv.DictReader(open(folder/'metrics_time.csv'))}
    else:
     log=main/(name+'.log');s=log.read_text() if log.exists() else '';data={float(t):dict(V=int(v),C=int(b)) for t,v,b in re.findall(r'\bt=(\d+).* V=(\d+) B=(\d+)',s)}
     if 'Traceback' in s:errors.append(name)
    series[name]=data
    if not reuse and data:fresh_times.append(max(data))
common=set.intersection(*(set(q) for q in series.values())) if all(series.values()) else set();t=max(common) if common else None;groups=defaultdict(lambda:dict(V=0,C=0))
if t is not None:
 for name,seq in series.items():
  c,p=name.split('_')[-2:];groups[c+'_'+p]['V']+=seq[t]['V'];groups[c+'_'+p]['C']+=seq[t]['C']
summary=dict(reused_runs=reused,fresh_complete=fresh_complete,fresh_expected=40-reused,fresh_progress_min=min(fresh_times) if fresh_times else None,fresh_progress_median=statistics.median(fresh_times) if fresh_times else None,fresh_progress_max=max(fresh_times) if fresh_times else None,common_checkpoint=t,groups=dict(groups),errors=errors,original_complete=len(list(Path('results/fire_closed_loop_20261107/main').glob('scene*/summary.json'))),original_post_complete=Path('results/fire_closed_loop_20261107/post_complete.json').exists())
(root/'interim_comparison.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))
