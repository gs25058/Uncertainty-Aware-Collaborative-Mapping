#!/usr/bin/env python3
"""Actual six-second viewpoint-capacity probe, separate from flight validation.

All candidates/scores are frozen from observations BEFORE this script opens the
scene. Each candidate starts from the identical full probability checkpoint.
This diagnostic does NOT claim the viewpoint was flown to or give oracle data
to an online policy.
"""
import argparse,copy,hashlib,json,pickle,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import octomap
from covor.occupancy import _sigmoid
from covor.explore.sensing import SharedMap,Sensor
from covor.explore.human import Human
from covor.explore.fire_recon_scene import ReconFireWorld
from covor.explore.runner import evaluate,write_json,csv_write
from covor.explore.execution_state import array_hashes
from covor.entry.path_eval import ReferenceValidator,PathEvalCfg


def main():
    p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--pool',type=Path,required=True);p.add_argument('--candidate',required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();metadata=json.loads((a.parent/'config.json').read_text());cfg=metadata['configuration']
    pool=json.loads(a.pool.read_text())['results'][0]['candidates'];candidate=next(c for c in pool if c['id']==a.candidate)
    prior=json.loads((a.parent/'prior.json').read_text());latest=json.loads((a.parent/'.execution/latest.json').read_text())
    checkpoint=a.parent/'.execution'/latest['directory'];commit=json.loads((checkpoint/'commit.json').read_text())
    for name,digest in commit['files'].items():assert hashlib.sha256((checkpoint/name).read_bytes()).hexdigest()==digest
    state=pickle.loads((checkpoint/'runtime.pkl').read_bytes());m=SharedMap(prior,cfg);data=state['map']
    m.__dict__.update(copy.deepcopy(data['attributes']));m.builder.__dict__.update(copy.deepcopy(data['builder']));m.surfaces.__dict__.update(copy.deepcopy(data['surfaces']))
    m.builder.tree=octomap.OcTree.read(str(checkpoint/'map.ot').encode());c=m.builder.cfg
    m.builder.tree.setClampingThresMin(_sigmoid(c.clamp_min));m.builder.tree.setClampingThresMax(_sigmoid(c.clamp_max));m.builder.tree.setOccupancyThres(_sigmoid(c.tau_occ))
    lab,layer=m.dense();assert array_hashes(lab,layer)==state['map_hashes']
    position=np.array(candidate['position']);states=state['runtime']['states']
    carrier=min((s for s in states if not s['failed']),key=lambda s:(np.linalg.norm(s['p']-position),s['id']))
    mates=[s['p'] for s in states if s['id']!=carrier['id']]
    write_json(a.out/'selection_before_sensor.json',dict(candidate=candidate,carrier=carrier['id'],initial_map_hashes=state['map_hashes'],stationary_viewpoint_only=True,not_a_flown_mission=True))
    b=ReconFireWorld(metadata['scene_seed'],cfg);b.render_scene(False)
    invalid=b.flight_collision(position,position,cfg['motion']['radius']) or any(np.linalg.norm(position-q)<2*(cfg['motion']['radius']+cfg['motion']['margin']) for q in mates)
    if invalid:
        baseline=json.loads((a.parent/'summary.json').read_text())
        write_json(a.out/'summary.json',dict(baseline,candidate=a.candidate,position=position.tolist(),stationary_capacity_only=True,additional_frames=0,invalid_viewpoint_collision=True))
        print(a.candidate,'INVALID VIEWPOINT',flush=True);return
    sensor=Sensor(b,cfg,metadata['scene_seed']*1000+metadata['run_seed'],metadata['condition']);frames=[]
    for tick in range(1,25):
        t=latest['time']+tick*.25;scan=cfg['sensor']['gimbal_scan'];phase=t%scan['period'];lo=scan['lower'];hi=scan['upper'];hold=scan['hold'];slew=scan['slew']
        pitch=lo if phase<=hold else lo+(hi-lo)*(phase-hold)/slew if phase<=hold+slew else hi if phase<=2*hold+slew else hi-(hi-lo)*(phase-2*hold-slew)/slew if phase<=2*hold+2*slew else lo
        T,z,sz,valid,_=sensor.capture(position,candidate['yaw'],pitch,t,mates,a.out/'frames',f't{tick:03d}',drone_id=carrier['id'])
        n=m.integrate(sensor.sd,T,z,sz,valid,mates);m.builder.tree.updateInnerOccupancy()
        frames.append(dict(time=t,pitch=pitch,valid=float(valid.mean()),support_points=n,map_version=m.version))
    lab,layer=m.dense();analysis=Human(prior,cfg).analyze(lab,layer,m.ijk,False)
    gt,ijk=b.labels(m.res);truth=json.loads((a.parent/'reference_targets.json').read_text())
    validator=ReferenceValidator(gt,ijk,m.res,b.scene,cfg=PathEvalCfg(step_limit_m=cfg['human']['exact_step_limit'],contact_epsilon_m=cfg['human']['contact_epsilon_m']))
    rows,summary=evaluate(analysis,truth,validator,prior,latest['time']+6,m.version)
    summary.update(candidate=a.candidate,position=position.tolist(),wall_seconds=time.monotonic()-start,stationary_capacity_only=True,additional_frames=24)
    write_json(a.out/'summary.json',summary);csv_write(a.out/'targets.csv',rows);csv_write(a.out/'frames.csv',frames)
    print(a.candidate,summary['valid_confirmed'],summary['correct_blocked'],summary['invalid_paths'],flush=True)


if __name__=='__main__':main()
