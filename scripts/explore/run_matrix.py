#!/usr/bin/env python
"""Independent paired runs on fixed CPUs; no shared map between runs."""
import sys,json,subprocess,itertools,argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from covor.explore.runner import csv_write,write_json

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',default='configs/closed_loop_v4_20261106.json');p.add_argument('--out',required=True);p.add_argument('--workers',type=int,default=4);a=p.parse_args();cfg=json.load(open(a.config));out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 frozen=out/'fixed_config.json'
 if frozen.exists():assert json.load(open(frozen))==cfg,'config changed after matrix started'
 else:write_json(frozen,cfg)
 jobs=list(itertools.product(cfg['scene_seeds'],cfg['run_seeds'],cfg['conditions'],cfg['policies']))
 def task(job):
  scene,seed,condition,policy=job;name=f'scene{scene}_seed{seed}_{condition}_{policy}';target=out/name
  if not (target/'summary.json').exists():
   cmd=[sys.executable,'scripts/explore/run_closed_loop.py','--config',str(frozen),'--out',str(target),'--scene',str(scene),'--seed',str(seed),'--condition',condition,'--policy',policy]
   with open(out/(name+'.log'),'a') as log:
    ret=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
   if ret.returncode:raise RuntimeError(f'{name}: exit {ret.returncode}; see log; not excluded')
  return json.load(open(target/'summary.json'))
 rows=[]
 with ThreadPoolExecutor(max_workers=a.workers) as pool:
  futures={pool.submit(task,j):j for j in jobs}
  for f in as_completed(futures):
   r=f.result();rows.append(r);csv_write(out/'aggregate.csv',sorted(rows,key=lambda r:(r['scene_seed'],r['run_seed'],r['condition'],r['policy'])));print('FINISHED',futures[f],r['balanced_score'],r['travel_distance'],r['failed_drones'],flush=True)
if __name__=='__main__':main()
