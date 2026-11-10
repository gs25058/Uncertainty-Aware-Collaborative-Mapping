#!/usr/bin/env python3
"""Use the existing closed loop with a versioned fire-world factory only.

No utility, motion, integration or human evaluator is duplicated here. The
factory binding is process-local; running legacy experiments are unaffected.
"""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from covor.explore import runner
from covor.explore.fire_recon_scene import ReconFireWorld
p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True);p.add_argument('--scene',type=int,default=1);p.add_argument('--seed',type=int,default=0);p.add_argument('--condition',default='clean',choices=['clean','weak','medium','strong']);p.add_argument('--policy',default='unknown',choices=['unknown','entry']);p.add_argument('--duration',type=float);p.add_argument('--drones',type=int);a=p.parse_args();cfg=json.load(open(a.config))
assert cfg['scene']['type']=='multistorey_fire'
runner.make_world=lambda configuration,index:ReconFireWorld(index,configuration)
runner.run(cfg,a.scene,a.seed,a.condition,a.policy,a.out,n_drones=a.drones,duration=a.duration)
