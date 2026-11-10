#!/usr/bin/env python3
"""Same experiment kernel, with exact state recovery and current-frame caching."""
import argparse,fcntl,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from covor.explore import runner
from covor.explore.fire_recon_scene import ReconFireWorld
from covor.explore.execution_state import DurableExecution
p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True);p.add_argument('--scene',type=int,required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--condition',required=True);p.add_argument('--policy',required=True);p.add_argument('--cache');p.add_argument('--checkpoint-period',type=float,default=4.);a=p.parse_args();out=Path(a.out);(out/'.execution').mkdir(parents=True,exist_ok=True)
lock=(out/'.execution/worker.lock').open('a+')
try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError:raise RuntimeError('this trial already has a running worker')
if (out/'summary.json').exists() and (out/'maps/final.npz').exists():print('ALREADY COMPLETE',out);sys.exit(0)
cfg=json.load(open(a.config));runner.make_world=lambda c,i:ReconFireWorld(i,c);execution=DurableExecution(out,a.cache,a.checkpoint_period);runner.run(cfg,a.scene,a.seed,a.condition,a.policy,out,execution=execution)
