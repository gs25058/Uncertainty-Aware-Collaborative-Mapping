#!/usr/bin/env python3
"""Frozen pre-interruption kernel vs cache-backed, stopped/resumed execution."""
import argparse,csv,importlib.util,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from covor.explore import runner
from covor.explore.fire_recon_scene import ReconFireWorld
from covor.explore.execution_state import DurableExecution,CheckpointStop
p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
frozen=Path('results/fire_recon_20261107/main/source_snapshot/covor/explore/runner.py');spec=importlib.util.spec_from_file_location('covor.explore.runner_frozen_validation',frozen);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
factory=lambda c,i:ReconFireWorld(i,c);old.make_world=factory;runner.make_world=factory;cfg=json.load(open('configs/fire_recon_20261107.preflight.json'));source=Path('results/fire_recon_20261107/preflight/pilot/scene1_seed0_clean_unknown')
old.run(cfg,1,0,'clean','unknown',out/'frozen',duration=4.)
try:runner.run(cfg,1,0,'clean','unknown',out/'resumed',duration=4.,execution=DurableExecution(out/'resumed',source,period=2.,stop_after=2.))
except CheckpointStop:pass
else:raise AssertionError('validation interruption did not occur')
runner.run(cfg,1,0,'clean','unknown',out/'resumed',duration=4.,execution=DurableExecution(out/'resumed',source,period=2.))
checks={}
with np.load(out/'frozen/maps/final.npz') as x,np.load(out/'resumed/maps/final.npz') as y:
 for k in x.files:checks['map_'+k]=np.array_equal(x[k],y[k],equal_nan=True)
for name in ['trajectories.csv','events.csv','target_paths.csv','metrics_time.csv','assignment_scores.csv','assignments.jsonl','decisions.jsonl','candidates.jsonl']:
 checks[name]=(out/'frozen'/name).read_bytes()==(out/'resumed'/name).read_bytes()
for f in (out/'frozen/frames').glob('*.npz'):
 with np.load(f) as x,np.load(out/'resumed/frames'/f.name) as y:
  assert all(np.array_equal(x[k],y[k],equal_nan=True) for k in x.files),f
x=json.load(open(out/'frozen/summary.json'));y=json.load(open(out/'resumed/summary.json'));checks['summary_without_wall_time']={k:v for k,v in x.items() if k!='wall_seconds'}=={k:v for k,v in y.items() if k!='wall_seconds'}
assert all(checks.values()),checks
(out/'verification.json').write_text(json.dumps(dict(all_exact=True,checks=checks,frames=x['captures'],frozen_kernel=str(frozen),checkpoint_restore_at_seconds=2.,no_policy_or_human_rule_change=True),indent=2)+'\n');print('FROZEN/CACHED/RESUMED EXACT PASS',checks,flush=True)
