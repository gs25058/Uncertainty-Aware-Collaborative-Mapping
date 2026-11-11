#!/usr/bin/env python3
"""Map-only vertical-view selection, followed by actual first-hit sensor audit.

This is a viewpoint capability diagnostic, not a flown mission result.
"""
import argparse,csv,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from covor.explore.motion import FlightMap
from covor.explore.policy import candidates,predicted_unknown,interest_mask
from covor.explore.sensing import camera_pose,Sensor
from covor.explore.depth_observation import camera_for
from covor.explore.fire_recon_scene import ReconFireWorld
from covor.explore.runner import write_json,csv_write
from covor.occupancy import _dda_batch


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    metadata=json.loads((a.run/'config.json').read_text());cfg=metadata['configuration'];prior=json.loads((a.run/'prior.json').read_text())
    with np.load(a.run/'maps/final.npz') as q:
        lab=q['labels'];ijk=q['ijk'];res=float(q['res']);layer={k:q[k] for k in q.files if k not in ('labels','ijk','res','version')}
    rows=list(csv.DictReader((a.run/'trajectories.csv').open()));t=max(float(q['time']) for q in rows)
    states=[dict(id=int(q['drone']),p=np.array([float(q[k]) for k in ('x','y','z')]),failed=q['failed']=='True') for q in rows if float(q['time'])==t]
    flight=FlightMap(lab,ijk,res,cfg['motion']);sd=camera_for(cfg)
    pool=candidates(lab,ijk,res,layer,flight,states,cfg,round(t/cfg['assignment_period']),metadata['run_seed'],prior)
    floors=sorted(set(round(sum(g['floor_z_bounds'])/2,3) for g in prior['goals']))
    threshold=floors[1]-.15
    interests=interest_mask(lab,ijk,res,prior)&((np.arange(lab.shape[2])[None,None,:]+ijk[2]+.5)*res>=threshold)
    selections=[];predictions=[]
    for label,pitch in [('legacy_upper',.17),('wide_upward',1.17)]:
        scores=[]
        for c in pool:
            T=camera_pose(np.array(c['position']),c['yaw'],pitch,cfg['sensor']['baseline'])
            value=len(predicted_unknown(lab,ijk,res,T,sd,sd.cfg.z_max,interests))
            scores.append(dict(mode=label,pitch=pitch,candidate=c,predicted_vertical_unknown=value))
        best=sorted(scores,key=lambda q:(-q['predicted_vertical_unknown'],q['candidate']['id']))[0]
        selections.append(best);predictions.extend(scores)
    write_json(a.out/'selection_before_sensor.json',dict(map_only_selection=True,threshold_from_public_target_floors=threshold,selections=selections,predictions=predictions))
    # Geometry is first opened AFTER both choices have been frozen to disk.
    b=ReconFireWorld(metadata['scene_seed'],cfg);b.render_scene(False)
    sensor=Sensor(b,cfg,metadata['scene_seed']*1000+metadata['run_seed'],metadata['condition'])
    results=[]
    for selection in selections:
        c=selection['candidate'];position=np.array(c['position'])
        # A diagnostic observation at a proposed viewpoint, explicitly NOT a
        # flown capture. Actual current teammate positions still occlude rays.
        mates=[s['p'] for s in states if np.linalg.norm(s['p']-position)>cfg['motion']['radius']*2]
        view=camera_pose(position,c['yaw'],selection['pitch'],cfg['sensor']['baseline'])
        true_z,_=sensor.impl._cast(view[:3,3],sensor.impl.dirs@view[:3,:3].T,mates)
        finite=np.isfinite(true_z)
        # Private simulator diagnostics are written only AFTER selection. They
        # are not policy features or repairs to observed depth/map values.
        geometric_audit=dict(first_hit_below_min=int((finite&(true_z<cfg['sensor']['min_depth'])).sum()),
            first_hit_above_max=int((finite&(true_z>cfg['sensor']['range'])).sum()),
            first_hit_z_min=float(true_z[finite].min()) if finite.any() else None,
            first_hit_z_max=float(true_z[finite].max()) if finite.any() else None)
        T,z,sz,valid,_=sensor.capture(position,c['yaw'],selection['pitch'],t+.25,mates,a.out/'frames',selection['mode'],drone_id=0)
        points,_=sensor.sd.backproject(z,sz,valid,downsample=2);world=points@T[:3,:3].T+T[:3,3]
        _,voxels=_dda_batch(T[:3,3],world,res)
        free=np.unique(voxels,axis=0);above=free[(free[:,2]+.5)*res>=threshold]
        results.append(dict(mode=selection['mode'],candidate=c['id'],position=position.tolist(),pitch=selection['pitch'],predicted_vertical_unknown=selection['predicted_vertical_unknown'],actual_first_hit_above=int((world[:,2]>=threshold).sum()),actual_free_cells_above=len(above),valid_pixels=int(valid.sum()),diagnostic_not_flown=True,**geometric_audit))
    csv_write(a.out/'summary.csv',results)
    print(json.dumps(results),flush=True)


if __name__=='__main__':main()
