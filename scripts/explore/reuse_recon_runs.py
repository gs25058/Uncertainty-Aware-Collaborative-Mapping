#!/usr/bin/env python3
"""Reuse only physically identical two-storey trials, including every failure.

Large immutable depth/map folders are linked; metadata are copied so subsequent
postprocessing cannot overwrite the original reports/replay manifests.
"""
import argparse,copy,hashlib,json,shutil,sys,csv
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from covor.explore.fire_recon_scene import ReconFireWorld
from covor.explore.depth_observation import RayDepthSensor
from covor.explore.runner import write_json,csv_write
p=argparse.ArgumentParser();p.add_argument('--config',default='configs/fire_recon_20261107.json');p.add_argument('--old',default='results/fire_closed_loop_20261107');p.add_argument('--out',default='results/fire_recon_20261107/main');a=p.parse_args();old=Path(a.old);out=Path(a.out);out.mkdir(parents=True,exist_ok=True);cfg=json.load(open(a.config));ocfg=json.load(open(old/'main/fixed_config.json'));manifest=json.load(open(old/'main/source_manifest.json'));checks=json.load(open('results/fire_recon_20261107/intake/asset_geometry_checks.json'));assert checks['all_base_ply_exact'];protected=['covor/occupancy.py','covor/explore/policy.py','covor/explore/motion.py','covor/explore/sensing.py','covor/explore/depth_observation.py','covor/explore/human.py','covor/explore/runner.py','covor/explore/fire_smoke.py','covor/explore/fire_world.py','covor/explore/fire_buildings.py','covor/entry/surfaces.py','covor/entry/decision.py','covor/entry/path_eval.py','covor/entry/traverse.py'];assert all(hashlib.sha256(Path(f).read_bytes()).hexdigest()==manifest[f] for f in protected)
keys=['duration','n_drones','dt','sensor_period','assessment_period','assignment_period','snapshot_period','entry_interval','minimum_task_time','capture_dwell','pose_mode','map_resolution','candidates','utility','motion','human','fault_policy','communication'];assert all(cfg[k]==ocfg[k] for k in keys)
def stripped_sensor(c):
 q=copy.deepcopy(c['sensor']);q['fire_smoke'].pop('cache',None);return q
assert stripped_sensor(cfg)==stripped_sensor(ocfg)
rows=[];replays=[]
for i in cfg['scene_seeds']:
 b=ReconFireWorld(i,cfg)
 if b.geometry_repairs:continue
 assert b.levels==2
 # The source-local field is identical because its geometry, source metadata
 # and all numeric parameters are unchanged. Copy rather than recompute it.
 src=Path(ocfg['sensor']['fire_smoke']['cache'])/(b.id+'.npz');dest=Path(cfg['sensor']['fire_smoke']['cache'])/(b.id+'.npz');dest.parent.mkdir(parents=True,exist_ok=True)
 if not dest.exists():shutil.copy2(src,dest)
 else:
  with np.load(src) as x,np.load(dest) as y:assert all(np.array_equal(x[k],y[k]) for k in x.files)
 for seed in cfg['run_seeds']:
  for condition in cfg['conditions']:
   sensor=RayDepthSensor(b,cfg,i*1000+seed,condition)
   for policy in cfg['policies']:
    name=f'scene{i}_seed{seed}_{condition}_{policy}';source=old/'main'/name;target=out/name;summary=json.load(open(source/'summary.json'));assert json.load(open(source/'prior.json'))==b.prior();assert json.load(open(source/'config.json'))['configuration']==ocfg
    if (target/'summary.json').exists():raise FileExistsError(target)
    frames=list(csv.DictReader(open(source/'frames.csv')))
    for f in [frames[0],frames[len(frames)//2],frames[-1]]:
     with np.load(source/'frames'/(f['frame_id']+'.npz')) as q:
      z,sigma,valid=sensor.measure(q['T'],float(q['time']),int(q['drone_id']),q['teammates']);assert np.array_equal(z,q['Z'],equal_nan=True);assert np.array_equal(sigma,q['sigma'],equal_nan=True);assert np.array_equal(valid,q['valid'])
     replays.append(dict(run=name,frame=f['frame_id'],exact_Z=True,exact_sigma=True,exact_valid=True))
    target.mkdir(parents=True,exist_ok=True)
    for item in source.iterdir():
     if item.name=='replay_manifest.json':continue
     if item.is_dir():(target/item.name).symlink_to(item.resolve(),target_is_directory=True)
     else:shutil.copy2(item,target/item.name)
    proof=dict(source_run=str(source),source_manifest=str(old/'main/source_manifest.json'),geometry_sha256=b.geometry_hash(),no_geometry_repairs=True,identical_prior=True,protected_code_unchanged=protected,identical_physics_and_numeric_sensor_configuration=True,sample_replays=[r for r in replays if r['run']==name],failures_retained=summary['failed_drones'],new_observations_generated=False,metadata_copied_immutable_input_folders_linked=True)
    write_json(target/'reuse_provenance.json',proof);rows.append(dict(run=name,scene=i,condition=condition,policy=policy,source=str(source),geometry_sha256=b.geometry_hash(),captures=summary['captures'],failed_drones=summary['failed_drones'],valid=summary['valid_confirmed']));print('REUSED RUN',name,'faults',summary['failed_drones'],flush=True)
csv_write(out/'reused_runs.csv',rows);csv_write(out/'reuse_observation_replays.csv',replays);write_json(out/'reuse_checks.json',dict(reused_runs=len(rows),exact_observation_replays=len(replays),protected_components_unchanged=True,numeric_configuration_unchanged=True,failures_not_excluded=True));print('REUSE COMPLETE',len(rows),flush=True)
