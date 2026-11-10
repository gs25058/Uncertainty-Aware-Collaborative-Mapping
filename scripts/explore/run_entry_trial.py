#!/usr/bin/env python3
"""Frozen-source, resumable utility trial; the physical/evaluation stack is shared."""
import argparse,fcntl,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from covor.explore import runner
from covor.explore.fire_recon_scene import ReconFireWorld
from covor.explore.execution_state import DurableExecution,CheckpointStop,atomic_json
p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True);p.add_argument('--scene',type=int,required=True);p.add_argument('--seed',type=int,default=0);p.add_argument('--condition',required=True);p.add_argument('--policy',default='entry',choices=['unknown','entry']);p.add_argument('--duration',type=float);p.add_argument('--stop-after',type=float);p.add_argument('--cache');p.add_argument('--memoize-motion',action='store_true');p.add_argument('--defer-dense',action='store_true');a=p.parse_args()
out=Path(a.out);(out/'.execution').mkdir(parents=True,exist_ok=True)
lock=(out/'.execution/worker.lock').open('a+')
try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError:sys.exit('trial already running')
if (out/'summary.json').exists():sys.exit('trial already complete')
cfg=json.load(open(a.config));runner.make_world=lambda c,i:ReconFireWorld(i,c)
accelerator=None
if a.memoize_motion:
 from covor.explore.execution_acceleration import enable_exact_motion_cache
 accelerator=enable_exact_motion_cache()
execution=DurableExecution(out,a.cache,period=8.,stop_after=a.stop_after)
deferred=None
if a.defer_dense:
 from covor.explore.execution_acceleration import DeferredDenseReads
 mode=dict(defer_dense=True,helper_sha256=hashlib.sha256(Path(__file__).resolve().parents[2].joinpath('covor/explore/execution_acceleration.py').read_bytes()).hexdigest())
 modefile=out/'.execution/dense_read_mode.json'
 if modefile.exists():assert json.load(open(modefile))==mode,'execution helper changed across checkpoint resume'
 else:
  assert not (out/'.execution/latest.json').exists(),'cannot switch dense execution mode inside an existing trial'
  atomic_json(modefile,mode)
 deferred=DeferredDenseReads(execution,runner.SharedMap)
try:runner.run(cfg,a.scene,a.seed,a.condition,a.policy,out,duration=a.duration,execution=execution)
except CheckpointStop:
 state=execution.folder/execution.record['directory'] if execution.record and 'directory' in execution.record else None
 atomic_json(out/'screen_complete.json',dict(time=a.stop_after,checkpoint=json.load(open(execution.folder/'latest.json')),not_a_final_mission=True));print('SCREEN COMPLETE',flush=True)
finally:
 if accelerator is not None:atomic_json(out/'execution_acceleration.json',dict(mode='bit-exact native-call memoization',hits=accelerator.hits,misses=accelerator.misses,capacity=accelerator.maxsize,physical_model_unchanged=True))
 if deferred is not None:atomic_json(out/'dense_read_acceleration.json',dict(full_reads=deferred.full_reads,deferred_unused_reads=deferred.deferred_reads,inner_occupancy_updates_preserved=True,all_sensor_integrations_preserved=True))
