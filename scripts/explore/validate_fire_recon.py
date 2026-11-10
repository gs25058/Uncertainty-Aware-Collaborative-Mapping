#!/usr/bin/env python3
"""Exhaustive known-map gate and independent path controls for a new revision."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from covor.explore.fire_recon_scene import ReconFireWorld
from covor.explore.human import Human,complete_support
from covor.explore.runner import evaluate,write_json,csv_write
from covor.entry.path_eval import ReferenceValidator,PathEvalCfg,motion_path
p=argparse.ArgumentParser();p.add_argument('--config',default='configs/fire_recon_20261107.preflight.json');p.add_argument('--scene',type=int,default=1);p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
if (out/'summary.json').exists():raise FileExistsError(out)
cfg=json.load(open(a.config));start=time.monotonic();b=ReconFireWorld(a.scene,cfg);truth,_,_=b.reference(cfg['map_resolution']);lab,ijk=b.labels(cfg['map_resolution']);b.render_scene(False);validator=ReferenceValidator(lab,ijk,cfg['map_resolution'],b.scene,cfg=PathEvalCfg(step_limit_m=cfg['human']['exact_step_limit'],contact_epsilon_m=cfg['human']['contact_epsilon_m']));controls=[]
for ref,goal in zip(truth,b.goals):
 if ref['reachable']:
  q=validator.validate(motion_path(ref['path']),b.entry,goal);assert q['status']=='valid',(ref['id'],q);controls.append(dict(goal=ref['id'],**q))
write_json(out/'reference_targets.json',truth);write_json(out/'prior.json',b.prior());write_json(out/'geometry.json',b.metadata());write_json(out/'config.json',cfg);csv_write(out/'positive_controls.csv',controls)
layer=complete_support(b,ijk,lab.shape,cfg['map_resolution']);analysis=Human(b.prior(),cfg).analyze(lab,layer,ijk,False);rows,summary=evaluate(analysis,truth,validator,b.prior(),0,0);csv_write(out/'targets.csv',rows)
assert not any(r['path_status']=='valid' and not r['reference_reachable'] for r in rows),'reference contradiction; inspect saved paths before any policy run'
upper=sum(r['path_status']=='valid' and r['goal'].startswith('L3_') for r in rows);assert summary['invalid_paths']==0;assert summary['false_blocked']==0;assert summary['valid_confirmed']==summary['reachable_denominator'];reference_upper=sum(t['reachable'] and t['id'].startswith('L3_') for t in truth);assert reference_upper==upper
summary.update(geometry_sha256=b.geometry_hash(),third_floor_valid=upper,reference_reachable_third_floor=reference_upper,repairs=b.geometry_repairs,wall_seconds=time.monotonic()-start);write_json(out/'summary.json',summary);print(summary,flush=True)
