#!/usr/bin/env python
"""Finish all fixed runs before postprocessing; never silently omit failures."""
import sys,time,subprocess,json
from pathlib import Path
root=Path('results/depth_model_20261106');deadline=time.time()+10800
while True:
 n=len(list((root/'main').glob('scene*/summary.json')));m=len(list((root/'rate1').glob('scene*/summary.json')))
 if n==32 and m==2:break
 if time.time()>deadline:raise TimeoutError(f'completed main={n}/32 rate={m}/2')
 for p in list((root/'main').glob('*.log'))+list((root/'rate1').glob('*.log')):
  if 'Traceback (most recent call last)' in p.read_text():raise RuntimeError(f'failed run, preserved: {p}')
 time.sleep(30)
commands=[
 ['scripts/explore/summarize.py','--root',str(root/'main'),'--out','reports/depth_model_20261106/data'],
 ['scripts/explore/summarize.py','--root',str(root/'rate1'),'--out','reports/depth_model_20261106/rate1_data'],
 ['scripts/explore/verify_depth_study.py','--root',str(root/'main'),'--out',str(root/'audit/main_checks.json'),'--expected','32'],
 ['scripts/explore/verify_depth_study.py','--root',str(root/'rate1'),'--out',str(root/'audit/rate_checks.json'),'--expected','2'],
 ['scripts/explore/analyze.py','--root',str(root/'main'),'--out',str(root/'analysis')]]
for args in commands:
 with (root/'audit'/('post_'+Path(args[0]).stem+'.log')).open('a') as f:subprocess.run([sys.executable,*args],stdout=f,stderr=subprocess.STDOUT,check=True)
(root/'post_complete.json').write_text(json.dumps(dict(main_runs=32,rate_runs=2,all_post_checks_passed=True))+'\n');print('ALL POSTPROCESSING COMPLETE',flush=True)
