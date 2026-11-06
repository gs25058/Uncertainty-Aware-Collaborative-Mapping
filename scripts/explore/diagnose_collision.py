#!/usr/bin/env python
"""Posthoc raw-frame replay of a map used for a failed flight assignment."""
import sys,argparse,csv,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from covor.explore.scene import Building
from covor.explore.sensing import SharedMap
from covor.explore.motion import FlightMap,Trajectory
from covor.explore.runner import write_json
from covor.occupancy import StereoCalib,StereoDepth,DepthCfg
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--out',required=True);args=p.parse_args();folder=Path(args.run);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
config=json.load(open(folder/'config.json'));cfg=config['configuration'];c=cfg['sensor'];prior=json.load(open(folder/'prior.json'));events=list(csv.DictReader(open(folder/'events.csv')));event=next(e for e in events if e['event']=='geometry_collision_frozen');t=float(event['time']);drone=int(event['drone'])
assign=[json.loads(l) for l in open(folder/'assignments.jsonl')];a=[a for a in assign if a['drone']==drone and a['time']<=t][-1]
K=np.array([[c['fx'],0,c['width']/2],[0,c['fx'],c['height']/2],[0,0,1.]])
sd=StereoDepth(StereoCalib(K,np.zeros(4),K.copy(),np.zeros(4),np.eye(3),np.array([-c['baseline'],0.,0.]),(c['width'],c['height'])),DepthCfg(z_max=c['range'],num_disp=c['num_disp'],block=c['block']))
m=SharedMap(prior,cfg);frames=list(csv.DictReader(open(folder/'frames.csv')))
for f in frames:
 if int(f['map_version'])>a['map_version']:break
 z=np.load(folder/'frames'/(f['frame_id']+'.npz'));m.integrate(sd,np.asarray(json.loads(f['camera_pose'])),z['Z'],z['sigma'],z['valid'],z['teammates'])
assert m.version==a['map_version'];lab,layer=m.dense();m.save(out/'replayed_assignment_map.npz',lab,layer);flight=FlightMap(lab,m.ijk,m.res,cfg['motion'])
r=a['trajectory'];tr=Trajectory(r['points'],r['yaw'],r['pitch'],r['yaw']+r['dy'],r['pitch']+r['dp'],r['start'],cfg['motion'],r['wait']);b=Building(config['scene_seed']);bias=np.asarray(json.load(open(folder/'pose_error.json'))['biases'])[drone]
# Truth is first read HERE, strictly after the recorded assignment and its map.
findings=[]
for when in np.arange(t-cfg['dt'],t+.0001,.001):
 estimated=tr.sample(when);point=estimated-bias
 d=np.maximum(np.maximum(b.lo-point,point-b.hi),0);dist=np.linalg.norm(d,axis=1);hits=np.flatnonzero(dist<cfg['motion']['radius'])
 if len(hits):
  near=[]
  for k in hits:
   contact=np.clip(point,b.lo[k]+1e-5,b.hi[k]-1e-5);q=np.floor((contact+bias)/m.res).astype(int)-m.ijk
   value=int(lab[tuple(q)]) if ((q>=0)&(q<lab.shape)).all() else 0
   near.append(dict(box=b.boxes[k]['name'],body_distance=float(dist[k]),actual_surface=contact.tolist(),estimated_map_surface_label=value))
  findings.append(dict(time=float(when),actual_body_centre=point.tolist(),estimated_body_centre=estimated.tolist(),observed_flight_map_clear_at_assignment=flight.point_clear(estimated),colliding_geometry=near));break
write_json(out/'diagnosis.json',dict(run=str(folder),collision_event=event,selected_assignment=a,replayed_frames=m.version,first_exact_sphere_contact=findings[0] if findings else None,scope='post-selection evaluation only; no truth is used by the flight planner'))
print(findings,flush=True)
