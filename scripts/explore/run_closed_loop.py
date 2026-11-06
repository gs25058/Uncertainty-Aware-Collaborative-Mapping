#!/usr/bin/env python
"""Reproducible paired exploration; prior results never overwritten."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse,json
from covor.explore.runner import run,validate_building,csv_write,write_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/closed_loop_v4_20261106.json');p.add_argument('--out',required=True)
    p.add_argument('--stage',choices=['truth','single','pair','run','matrix'],default='run');p.add_argument('--scene',type=int,default=11);p.add_argument('--seed',type=int,default=0);p.add_argument('--policy',choices=['unknown','entry'],default='unknown');p.add_argument('--condition',choices=['clean','smoke'],default='clean');p.add_argument('--duration',type=float);p.add_argument('--ideal-depth',action='store_true')
    a=p.parse_args();cfg=json.load(open(a.config));out=Path(a.out)
    if a.stage=='truth':
        for seed in cfg['scene_seeds']:print(seed,validate_building(cfg,seed,out/f'scene{seed}'),flush=True)
    elif a.stage=='matrix':
        rows=[]
        for scene in cfg['scene_seeds']:
            for seed in cfg['run_seeds']:
                for condition in cfg['conditions']:
                    for policy in cfg['policies']:
                        target=out/f'scene{scene}_seed{seed}_{condition}_{policy}'
                        if (target/'summary.json').exists():r=json.load(open(target/'summary.json'))
                        else:r=run(cfg,scene,seed,condition,policy,target,duration=a.duration,ideal=a.ideal_depth)
                        rows.append(r);csv_write(out/'aggregate.csv',rows)
    else:run(cfg,a.scene,a.seed,a.condition,a.policy,out,n_drones={'single':1,'pair':2}.get(a.stage),duration=a.duration,ideal=a.ideal_depth)
if __name__=='__main__':main()
